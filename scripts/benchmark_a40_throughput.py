#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.util
import gc
import json
import math
import os
import sys
import time
from pathlib import Path


BASELINE_ROOT: Path | None = None
MSFFAT_ROOT = Path(__file__).resolve().parents[1]


def append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "framework",
                "seq_len",
                "num_classes",
                "max_feasible_batch_size",
                "peak_throughput_traces_s",
                "peak_batch_size",
                "latency_ms_at_peak",
                "device",
                "gpu_name",
                "tested_batches",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def torch_model(method: str, num_classes: int):
    def load_class(filename: str, class_name: str):
        if BASELINE_ROOT is None:
            raise ValueError("--baseline-root is required for external baseline methods")
        path = BASELINE_ROOT / "WFlib" / "models" / filename
        spec = importlib.util.spec_from_file_location(f"bench_{class_name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {class_name} from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, class_name)

    if method == "DF":
        return load_class("DF.py", "DF")(num_classes), (1,)
    if method == "Tik-Tok":
        return load_class("TikTok.py", "TikTok")(num_classes), (1,)
    if method == "Var-CNN":
        return load_class("VarCNN.py", "VarCNN")(num_classes), (2,)
    if method == "NetCLR":
        return load_class("NetCLR.py", "NetCLR")(num_classes), (1,)
    if method == "MSFFAT" and getattr(torch_model, "backend", "tensorflow") == "torch":
        return build_torch_msffat(num_classes), (1,)
    raise ValueError(method)


def build_torch_msffat(num_classes: int):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class SameConv1d(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, stride=1, bias=True):
            super().__init__()
            self.left = (kernel_size - 1) // 2
            self.right = kernel_size - 1 - self.left
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, bias=bias)

        def forward(self, x):
            return self.conv(F.pad(x, (self.left, self.right)))

    class CausalConv1d(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, bias=False):
            super().__init__()
            self.pad = dilation * (kernel_size - 1)
            self.conv = nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
            )

        def forward(self, x):
            return self.conv(F.pad(x, (self.pad, 0)))

    class MSFBlock(nn.Module):
        def __init__(self, in_ch, filters, residual):
            super().__init__()
            self.residual = residual
            self.k3 = SameConv1d(in_ch, filters[0], 3)
            self.k5 = SameConv1d(in_ch, filters[1], 5)
            self.k7 = SameConv1d(in_ch, filters[2], 7)
            self.k9 = SameConv1d(in_ch, filters[3], 9)
            self.bn = nn.BatchNorm1d(sum(filters))

        def forward(self, x):
            out = torch.cat(
                [
                    F.relu(self.k9(x)),
                    F.relu(self.k7(x)),
                    F.relu(self.k5(x)),
                    F.relu(self.k3(x)),
                ],
                dim=1,
            )
            if self.residual:
                out = out + x
            return self.bn(out)

    class LTFBlock(nn.Module):
        def __init__(self, in_ch, out_ch, dilations, stride=1, project=False):
            super().__init__()
            self.conv1 = CausalConv1d(in_ch, out_ch, 5, stride=stride, dilation=dilations[0], bias=False)
            self.bn1 = nn.BatchNorm1d(out_ch, eps=1e-5)
            self.conv2 = CausalConv1d(out_ch, out_ch, 5, dilation=dilations[1], bias=False)
            self.bn2 = nn.BatchNorm1d(out_ch, eps=1e-5)
            if project:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_ch, eps=1e-5),
                )
            else:
                self.shortcut = nn.Identity()

        def forward(self, x):
            y = F.relu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            return F.relu(y + self.shortcut(x))

    class ChannelAttention(nn.Module):
        def __init__(self, channels, ratio=16):
            super().__init__()
            hidden = max(channels // ratio, 1)
            self.fc1 = nn.Linear(channels, hidden)
            self.fc2 = nn.Linear(hidden, channels)

        def forward(self, x):
            avg = x.mean(dim=2)
            mx = x.amax(dim=2)
            weights = self.fc2(F.relu(self.fc1(avg))) + self.fc2(F.relu(self.fc1(mx)))
            weights = F.hardsigmoid(weights).unsqueeze(-1)
            return x * weights

    class TorchMSFFAT(nn.Module):
        def __init__(self, classes):
            super().__init__()
            self.sfed = nn.Sequential(
                SameConv1d(1, 32, 8),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.MaxPool1d(4, 2),
                SameConv1d(32, 64, 8),
                nn.BatchNorm1d(64),
                nn.ELU(),
                nn.MaxPool1d(4, 2),
                nn.Dropout(0.1),
            )
            self.msf = nn.Sequential(
                MSFBlock(64, (8, 8, 16, 32), True),
                MSFBlock(64, (16, 16, 32, 64), False),
                nn.MaxPool1d(8, 4),
                nn.Dropout(0.2),
                MSFBlock(128, (16, 16, 32, 64), True),
                MSFBlock(128, (32, 32, 64, 128), False),
                nn.MaxPool1d(8, 4),
                nn.Dropout(0.2),
                MSFBlock(256, (32, 32, 64, 128), True),
                MSFBlock(256, (64, 64, 128, 256), False),
                nn.MaxPool1d(8, 4),
                nn.Dropout(0.2),
            )
            ltf_layers = []
            in_ch = 64
            filters = 64
            for stage in range(4):
                stride = 2 if stage in {1, 2} else 1
                ltf_layers.append(LTFBlock(in_ch, filters, (1, 2), stride=stride, project=True))
                ltf_layers.append(LTFBlock(filters, filters, (4, 8), project=False))
                ltf_layers.append(nn.MaxPool1d(4, 2))
                in_ch = filters
                filters *= 2
            self.ltf = nn.Sequential(*ltf_layers)
            self.attn = ChannelAttention(1024)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(18 * 1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, classes),
            )

        def forward(self, x):
            x = self.sfed(x)
            ms = self.msf(x)
            lt = self.ltf(x)
            length = min(ms.shape[-1], lt.shape[-1])
            fused = torch.cat([ms[:, :, :length], lt[:, :, :length]], dim=1)
            return self.head(self.attn(fused))

    return TorchMSFFAT(num_classes)


def clear_torch(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def torch_feasible(torch, model, batch: int, channels: int, seq_len: int, device: str) -> tuple[bool, str | None]:
    try:
        clear_torch(torch)
        x = torch.randn(batch, channels, seq_len, device=device)
        with torch.inference_mode():
            _ = model(x)
        torch.cuda.synchronize()
        del x
        clear_torch(torch)
        return True, None
    except RuntimeError as exc:
        msg = str(exc)
        clear_torch(torch)
        if "out of memory" in msg.lower() or "cuda" in msg.lower():
            return False, msg.splitlines()[0]
        raise


def torch_latency_ms(torch, model, batch: int, channels: int, seq_len: int, device: str, warmup: int, min_time: float) -> float:
    clear_torch(torch)
    x = torch.randn(batch, channels, seq_len, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        count = 0
        start_wall = time.perf_counter()
        start.record()
        while time.perf_counter() - start_wall < min_time:
            _ = model(x)
            count += 1
        end.record()
        torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    del x
    clear_torch(torch)
    return elapsed_ms / max(count, 1)


def run_torch(args) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for PyTorch")
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    torch_model.backend = getattr(args, "msffat_backend", "tensorflow")

    device = "cuda"
    model, channel_shape = torch_model(args.method, args.num_classes)
    channels = channel_shape[0]
    model.eval().to(device)

    feasible_batches = []
    failed_batch = None
    batch = args.start_batch
    while batch <= args.max_batch:
        print(json.dumps({"method": args.method, "testing_batch": batch}), flush=True)
        ok, err = torch_feasible(torch, model, batch, channels, args.seq_len, device)
        if ok:
            feasible_batches.append(batch)
            batch *= 2
        else:
            failed_batch = batch
            print(json.dumps({"method": args.method, "failed_batch": batch, "error": err}), flush=True)
            break

    if not feasible_batches:
        raise RuntimeError(f"{args.method}: no feasible batch size found")

    low = feasible_batches[-1]
    high = failed_batch
    if high is not None and args.binary_search:
        while high - low > 1:
            mid = (low + high) // 2
            print(json.dumps({"method": args.method, "testing_batch": mid}), flush=True)
            ok, err = torch_feasible(torch, model, mid, channels, args.seq_len, device)
            if ok:
                low = mid
                feasible_batches.append(mid)
            else:
                high = mid
                print(json.dumps({"method": args.method, "failed_batch": mid, "error": err}), flush=True)

    max_feasible = low
    candidates = sorted(set(feasible_batches + [max_feasible]))
    peak = (0.0, None, math.inf)
    measured = []
    for b in candidates:
        try:
            latency_ms = torch_latency_ms(torch, model, b, channels, args.seq_len, device, args.warmup, args.min_time)
        except RuntimeError as exc:
            msg = str(exc)
            if "out of memory" in msg.lower() or "cuda" in msg.lower():
                clear_torch(torch)
                print(json.dumps({"method": args.method, "skipped_batch": b, "error": msg.splitlines()[0]}), flush=True)
                continue
            raise
        throughput = b * 1000.0 / latency_ms
        measured.append({"batch": b, "latency_ms": latency_ms, "throughput": throughput})
        print(json.dumps({"method": args.method, "batch": b, "latency_ms": latency_ms, "throughput": throughput}), flush=True)
        if throughput > peak[0]:
            peak = (throughput, b, latency_ms)

    return {
        "method": args.method,
        "framework": "pytorch-msffat-port" if args.method == "MSFFAT" else "pytorch",
        "seq_len": args.seq_len,
        "num_classes": args.num_classes,
        "max_feasible_batch_size": max_feasible,
        "peak_throughput_traces_s": f"{peak[0]:.3f}",
        "peak_batch_size": peak[1],
        "latency_ms_at_peak": f"{peak[2]:.3f}",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0),
        "tested_batches": json.dumps(measured),
    }


def tf_feasible(tf, model, batch: int, seq_len: int) -> tuple[bool, str | None]:
    try:
        gc.collect()
        x = tf.random.normal((batch, seq_len, 1))
        _ = model(x, training=False)
        _ = tf.experimental.async_wait() if hasattr(tf.experimental, "async_wait") else None
        del x
        gc.collect()
        return True, None
    except tf.errors.ResourceExhaustedError as exc:
        gc.collect()
        return False, str(exc).splitlines()[0]


def tf_latency_ms(tf, model, batch: int, seq_len: int, warmup: int, min_time: float) -> float:
    gc.collect()
    x = tf.random.normal((batch, seq_len, 1))

    @tf.function(jit_compile=False)
    def forward(inp):
        return model(inp, training=False)

    for _ in range(warmup):
        _ = forward(x)
    if hasattr(tf.experimental, "async_wait"):
        tf.experimental.async_wait()

    count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < min_time:
        y = forward(x)
        _ = y.numpy()
        count += 1
    elapsed = time.perf_counter() - start
    del x
    gc.collect()
    return elapsed * 1000.0 / max(count, 1)


def run_tf(args) -> dict:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    sys.path.insert(0, str(MSFFAT_ROOT))
    print(json.dumps({"method": args.method, "stage": "before_import_tensorflow"}), flush=True)
    import tensorflow as tf
    print(json.dumps({"method": args.method, "stage": "after_import_tensorflow", "tf_version": tf.__version__}), flush=True)
    from msffat.model import build_msffat

    gpus = tf.config.list_physical_devices("GPU")
    print(json.dumps({"method": args.method, "stage": "after_list_gpus", "gpus": [str(g) for g in gpus]}), flush=True)
    if not gpus:
        raise RuntimeError("CUDA GPU is not available for TensorFlow")

    model = build_msffat(input_shape=(args.seq_len, 1), num_classes=args.num_classes, mode="single")
    print(json.dumps({"method": args.method, "stage": "after_build_model", "params": model.count_params()}), flush=True)

    feasible_batches = []
    failed_batch = None
    batch = args.start_batch
    while batch <= args.max_batch:
        print(json.dumps({"method": args.method, "testing_batch": batch}), flush=True)
        ok, err = tf_feasible(tf, model, batch, args.seq_len)
        if ok:
            feasible_batches.append(batch)
            batch *= 2
        else:
            failed_batch = batch
            print(json.dumps({"method": args.method, "failed_batch": batch, "error": err}), flush=True)
            break

    if not feasible_batches:
        raise RuntimeError(f"{args.method}: no feasible batch size found")

    low = feasible_batches[-1]
    high = failed_batch
    if high is not None and args.binary_search:
        while high - low > 1:
            mid = (low + high) // 2
            print(json.dumps({"method": args.method, "testing_batch": mid}), flush=True)
            ok, err = tf_feasible(tf, model, mid, args.seq_len)
            if ok:
                low = mid
                feasible_batches.append(mid)
            else:
                high = mid
                print(json.dumps({"method": args.method, "failed_batch": mid, "error": err}), flush=True)

    max_feasible = low
    candidates = sorted(set(feasible_batches + [max_feasible]))
    peak = (0.0, None, math.inf)
    measured = []
    for b in candidates:
        try:
            latency_ms = tf_latency_ms(tf, model, b, args.seq_len, args.warmup, args.min_time)
        except tf.errors.ResourceExhaustedError as exc:
            print(json.dumps({"method": args.method, "skipped_batch": b, "error": str(exc).splitlines()[0]}), flush=True)
            continue
        throughput = b * 1000.0 / latency_ms
        measured.append({"batch": b, "latency_ms": latency_ms, "throughput": throughput})
        print(json.dumps({"method": args.method, "batch": b, "latency_ms": latency_ms, "throughput": throughput}), flush=True)
        if throughput > peak[0]:
            peak = (throughput, b, latency_ms)

    gpu_name = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU")
    return {
        "method": args.method,
        "framework": "tensorflow",
        "seq_len": args.seq_len,
        "num_classes": args.num_classes,
        "max_feasible_batch_size": max_feasible,
        "peak_throughput_traces_s": f"{peak[0]:.3f}",
        "peak_batch_size": peak[1],
        "latency_ms_at_peak": f"{peak[2]:.3f}",
        "device": "cuda",
        "gpu_name": gpu_name,
        "tested_batches": json.dumps(measured),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="GPU inference batch-size and throughput benchmark")
    parser.add_argument("--method", required=True, choices=["DF", "Tik-Tok", "Var-CNN", "NetCLR", "MSFFAT"])
    parser.add_argument("--msffat-backend", choices=["tensorflow", "torch"], default="tensorflow")
    parser.add_argument("--seq-len", type=int, default=5000)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--start-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--min-time", type=float, default=1.0)
    parser.add_argument("--binary-search", action="store_true")
    parser.add_argument(
        "--baseline-root", type=Path, default=None,
        help="Website-Fingerprinting-Library root; required for external baselines.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/a40_throughput.csv"))
    return parser.parse_args()


def main() -> None:
    global BASELINE_ROOT
    args = parse_args()
    BASELINE_ROOT = args.baseline_root.resolve() if args.baseline_root else None
    if args.method == "MSFFAT" and args.msffat_backend == "tensorflow":
        row = run_tf(args)
    else:
        row = run_torch(args)
    append_result(args.output, row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
