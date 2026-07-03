#!/usr/bin/env python
"""Unified A40 forward-pass benchmark for Table 10 PyTorch models."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn


WFL_ROOT: Path | None = None
MSFFAT_ROOT = Path(__file__).resolve().parents[1]

SPECS = {
    "AWF-CNN": ("AWF.py", "AWF", (1, 3000)),
    "AWF-LSTM": (None, "AWFLSTM", (1, 150)),
    "AWF-SDAE": (None, "AWFSDAE", (1, 5000)),
    "RF": ("RF.py", "RF", (1, 2, 1800)),
    "NetCLR": ("NetCLR.py", "NetCLR", (1, 5000)),
    "TF": ("TF.py", "TF", (1, 5000)),
    "DF": ("DF.py", "DF", (1, 5000)),
    "Tik-Tok": ("TikTok.py", "TikTok", (1, 5000)),
    "ARES": ("ARES.py", "ARES", (8, 8000)),
    "TMWF": ("TMWF.py", "TMWF", (1, 30720)),
    "GANDaLF": (None, "GANDaLF", (1, 5000)),
}


class GANDaLFDiscriminator(nn.Module):
    """Inference network from official wfi/cw/wfi-cw.py (generator excluded)."""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        channels = [1, 32, 32, 64, 64, 128, 128, 256, 256]
        strides = [2, 2, 2, 2, 2, 2, 1, 1]
        blocks = []
        for in_ch, out_ch, stride in zip(channels[:-1], channels[1:], strides):
            blocks.extend([
                nn.Conv1d(in_ch, out_ch, 5, stride=stride, padding=2),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(0.3),
            ])
        self.tail = nn.Sequential(*blocks, nn.Flatten())
        self.head = nn.Sequential(
            nn.Linear(79 * 256, 2048), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2048, 2048), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.tail(x))


def load_wfl(filename: str, class_name: str):
    if WFL_ROOT is None:
        raise ValueError("--wfl-root is required for Website-Fingerprinting-Library models")
    path = WFL_ROOT / "WFlib" / "models" / filename
    spec = importlib.util.spec_from_file_location(f"table10_{class_name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def build_model(method: str, classes: int):
    filename, class_name, shape = SPECS[method]
    if method in {"AWF-LSTM", "AWF-SDAE"}:
        sys.path.insert(0, str(MSFFAT_ROOT / "scripts"))
        from awf_variants import AWFLSTM, AWFSDAE
        model = AWFLSTM(classes) if method == "AWF-LSTM" else AWFSDAE(classes, 5000)
    elif method == "GANDaLF":
        model = GANDaLFDiscriminator(classes)
    else:
        cls = load_wfl(filename, class_name)
        model = cls(classes)
        if method == "TF":
            model = nn.Sequential(model, nn.Linear(64, classes))
    return model, shape


def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def timed_forward(model, x, warmup: int, target_seconds: float, min_iters: int):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()

        probe_start = torch.cuda.Event(enable_timing=True)
        probe_end = torch.cuda.Event(enable_timing=True)
        probe_start.record()
        for _ in range(min_iters):
            model(x)
        probe_end.record()
        torch.cuda.synchronize()
        probe_ms = probe_start.elapsed_time(probe_end)

        iterations = max(min_iters, math.ceil(target_seconds * 1000.0 * min_iters / probe_ms))
        iterations = min(iterations, 10000)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            model(x)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iterations, iterations


def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    model, sample_shape = build_model(args.method, args.classes)
    model.eval().cuda()
    params = sum(p.numel() for p in model.parameters())

    batches = []
    b = 1
    while b <= args.max_batch:
        batches.append(b)
        b *= 2
    if 128 not in batches:
        batches.append(128)
    batches.sort()

    measured = []
    for batch in batches:
        clear_cuda()
        try:
            x = torch.randn((batch, *sample_shape), dtype=torch.float32, device="cuda")
            latency_ms, iterations = timed_forward(
                model, x, args.warmup, args.target_seconds, args.min_iters
            )
            throughput = batch * 1000.0 / latency_ms
            item = {
                "batch": batch,
                "latency_ms": latency_ms,
                "throughput_traces_s": throughput,
                "iterations": iterations,
            }
            measured.append(item)
            print(json.dumps(item), flush=True)
            del x
        except torch.cuda.OutOfMemoryError as exc:
            print(json.dumps({"batch": batch, "oom": str(exc).splitlines()[0]}), flush=True)
            clear_cuda()
            break

    by_batch = {x["batch"]: x for x in measured}
    if 1 not in by_batch or 128 not in by_batch:
        raise RuntimeError("Required batch=1 or batch=128 measurement is missing")
    peak = max(measured, key=lambda x: x["throughput_traces_s"])
    row = {
        "method": args.method,
        "framework": "PyTorch",
        "input_shape": "x".join(map(str, sample_shape)),
        "params": params,
        "batch1_latency_ms_per_trace": by_batch[1]["latency_ms"],
        "batch128_throughput_traces_s": by_batch[128]["throughput_traces_s"],
        "peak_stable_throughput_traces_s": peak["throughput_traces_s"],
        "peak_batch": peak["batch"],
        "gpu": torch.cuda.get_device_name(0),
        "details_json": json.dumps(measured),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row), flush=True)


def main():
    global WFL_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(SPECS))
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--max-batch", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--target-seconds", type=float, default=3.0)
    parser.add_argument("--min-iters", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--wfl-root", type=Path, default=None,
        help="Website-Fingerprinting-Library root for external model definitions.",
    )
    args = parser.parse_args()
    WFL_ROOT = args.wfl_root.resolve() if args.wfl_root else None
    benchmark(args)


if __name__ == "__main__":
    main()
