"""MSFFAT model definition.

This file is a cleaned version of the recovered MSFFAT scripts.  It preserves
the core backbone: shallow downsampling, multi-scale burst extraction, causal
dilated temporal extraction, channel-attention fusion, and either a softmax
single-label head or a sigmoid multi-label head.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    ELU,
    Flatten,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Layer,
    MaxPooling1D,
    Multiply,
)
from tensorflow.keras.models import Model


def channel_attention(
    inputs, ratio: int = 16, name: str = "channel_attention", activation: str = "sigmoid",
    form: str = "proposed",
    mlp=None,
):
    """Generate branch gates, optionally reusing the supplied Dense layers."""
    if activation != "sigmoid":
        raise ValueError("The aligned method requires sigmoid attention")
    if ratio < 1 or form not in {"proposed", "standard"}:
        raise ValueError("Invalid attention ratio/form")
    channels = int(inputs.shape[-1])
    hidden = max(channels // ratio, 1)

    shared_dense_1 = mlp[0] if mlp is not None else Dense(
        hidden,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        name=f"{name}_shared_dense_1",
    )
    shared_dense_2 = mlp[1] if mlp is not None else Dense(
        channels,
        kernel_initializer="he_normal",
        use_bias=True,
        name=f"{name}_shared_dense_2",
    )

    avg_pool = GlobalAveragePooling1D(name=f"{name}_avg_pool")(inputs)
    max_pool = GlobalMaxPooling1D(name=f"{name}_max_pool")(inputs)
    if form == "proposed":
        statistics = Concatenate(name=f"{name}_statistics")([avg_pool, max_pool])
        weights = shared_dense_2(shared_dense_1(statistics))
    else:
        avg = shared_dense_2(shared_dense_1(avg_pool))
        maximum = shared_dense_2(shared_dense_1(max_pool))
        weights = Add(name=f"{name}_add")([avg, maximum])
    weights = Activation(activation, name=f"{name}_weights")(weights)
    return Multiply(name=f"{name}_multiply")([inputs, weights])


def surface_downsample(inputs, name: str = "sfed"):
    x = Conv1D(32, 8, strides=1, padding="same", name=f"{name}_conv1")(inputs)
    x = BatchNormalization(name=f"{name}_bn1")(x)
    x = ELU(name=f"{name}_elu1")(x)
    x = MaxPooling1D(pool_size=4, strides=2, name=f"{name}_pool1")(x)

    x = Conv1D(64, 8, strides=1, padding="same", name=f"{name}_conv2")(x)
    x = BatchNormalization(name=f"{name}_bn2")(x)
    x = ELU(name=f"{name}_elu2")(x)
    x = MaxPooling1D(pool_size=4, strides=2, name=f"{name}_pool2")(x)
    x = Dropout(0.1, name=f"{name}_dropout")(x)
    return x


def msf_block(x, filters, residual: bool, name: str):
    shortcut = x
    b3 = Conv1D(filters[0], 3, padding="same", activation="relu", name=f"{name}_k3")(x)
    b5 = Conv1D(filters[1], 5, padding="same", activation="relu", name=f"{name}_k5")(x)
    b7 = Conv1D(filters[2], 7, padding="same", activation="relu", name=f"{name}_k7")(x)
    b9 = Conv1D(filters[3], 9, padding="same", activation="relu", name=f"{name}_k9")(x)
    out = Concatenate(axis=-1, name=f"{name}_concat")([b9, b7, b5, b3])
    if residual:
        out = Add(name=f"{name}_residual")([out, shortcut])
    out = BatchNormalization(name=f"{name}_bn")(out)
    return out


def msf_branch(inputs, dropout: float = 0.2, name: str = "msf"):
    x = msf_block(inputs, (8, 8, 16, 32), residual=True, name=f"{name}_block1")
    x = msf_block(x, (16, 16, 32, 64), residual=False, name=f"{name}_block2")
    x = MaxPooling1D(pool_size=8, strides=4, name=f"{name}_pool1")(x)
    x = Dropout(dropout, name=f"{name}_dropout1")(x)

    x = msf_block(x, (16, 16, 32, 64), residual=True, name=f"{name}_block3")
    x = msf_block(x, (32, 32, 64, 128), residual=False, name=f"{name}_block4")
    x = MaxPooling1D(pool_size=8, strides=4, name=f"{name}_pool2")(x)
    x = Dropout(dropout, name=f"{name}_dropout2")(x)

    x = msf_block(x, (32, 32, 64, 128), residual=True, name=f"{name}_block5")
    x = msf_block(x, (64, 64, 128, 256), residual=False, name=f"{name}_block6")
    x = MaxPooling1D(pool_size=8, strides=4, name=f"{name}_pool3")(x)
    x = Dropout(dropout, name=f"{name}_dropout3")(x)
    return x


def ltf_residual_block(
    x,
    filters: int,
    dilation_pair,
    stage: int,
    block: int,
    stride: int = 1,
):
    name = f"ltf_s{stage}_b{block}"
    y = Conv1D(
        filters,
        5,
        padding="causal",
        strides=stride,
        dilation_rate=dilation_pair[0],
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_conv1",
    )(x)
    y = BatchNormalization(epsilon=1e-5, name=f"{name}_bn1")(y)
    y = Activation("relu", name=f"{name}_relu1")(y)

    y = Conv1D(
        filters,
        5,
        padding="causal",
        dilation_rate=dilation_pair[1],
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_conv2",
    )(y)
    y = BatchNormalization(epsilon=1e-5, name=f"{name}_bn2")(y)

    # The recovered TensorFlow implementation projects every first block in a
    # stage, including stage 0 where the channel count is unchanged.
    if block == 0:
        shortcut = Conv1D(
            filters,
            1,
            strides=stride,
            use_bias=False,
            name=f"{name}_shortcut_conv",
        )(x)
        shortcut = BatchNormalization(epsilon=1e-5, name=f"{name}_shortcut_bn")(shortcut)
    else:
        shortcut = x

    y = Add(name=f"{name}_add")([y, shortcut])
    y = Activation("relu", name=f"{name}_relu2")(y)
    return y


def ltf_branch(inputs, name: str = "ltf"):
    x = inputs
    filters = 64
    for stage in range(4):
        # Match ../WF/MSFFAT.py: stages 1 and 2 downsample in the first
        # residual convolution; stages 0 and 3 keep stride 1.
        stride = 2 if stage in {1, 2} else 1
        x = ltf_residual_block(x, filters, (1, 2), stage, 0, stride=stride)
        x = ltf_residual_block(x, filters, (4, 8), stage, 1)
        x = MaxPooling1D(pool_size=4, strides=2, name=f"{name}_s{stage}_pool")(x)
        filters *= 2
    return x


@tf.keras.utils.register_keras_serializable(package="msffat")
class TemporalCropToMatch(Layer):
    """Crop two temporal feature maps to their shared minimum length."""

    def call(self, inputs):
        left, right = inputs
        length = tf.minimum(tf.shape(left)[1], tf.shape(right)[1])
        return left[:, :length, :], right[:, :length, :]

    def compute_output_shape(self, input_shape):
        left_shape, right_shape = input_shape
        if left_shape[1] is None or right_shape[1] is None:
            length = None
        else:
            length = min(left_shape[1], right_shape[1])
        return (left_shape[0], length, left_shape[2]), (right_shape[0], length, right_shape[2])


def build_msffat(
    input_shape=(5000, 1),
    num_classes: int = 95,
    mode: str = "single",
    msf_dropout: float = 0.2,
    head_dropout: float = 0.5,
    attention_activation: str = "sigmoid",
    variant: str = "full",
    fusion: str = "proposed",
) -> Model:
    """Build MSFFAT.

    Args:
        input_shape: Fixed-length sequence shape, e.g. ``(5000, 1)``.
        num_classes: Number of monitored classes.
        mode: ``single`` for softmax, ``multi`` for sigmoid multi-label.
        msf_dropout: Dropout used inside the multi-scale branch.
        head_dropout: Dropout used in the dense classification head.
    """
    if mode not in {"single", "multi"}:
        raise ValueError("mode must be 'single' or 'multi'")
    if variant not in {"full", "ltf_only", "msf_only"} or fusion not in {"proposed", "independent", "standard", "concat"}:
        raise ValueError("Unknown branch/fusion ablation")

    inputs = tf.keras.Input(shape=input_shape, name="cells")
    x = surface_downsample(inputs)
    # Preserve the original main-model initialization order for frozen seeds.
    branches = {}
    if variant in {"full", "msf_only"}:
        branches["msf"] = msf_branch(x, dropout=msf_dropout)
    if variant in {"full", "ltf_only"}:
        branches["ltf"] = ltf_branch(x)
    features = []
    shared_mlp = None
    if fusion == "proposed":
        shared_mlp = (
            Dense(32, activation="relu", kernel_initializer="he_normal", name="shared_attention_dense_1"),
            Dense(512, kernel_initializer="he_normal", name="shared_attention_dense_2"),
        )
    for branch in ("ltf", "msf"):
        if variant != "full" and variant != f"{branch}_only":
            continue
        feature = branches[branch]
        if fusion != "concat":
            feature = channel_attention(feature, name=f"{branch}_attention", activation=attention_activation,
                                        form="proposed" if fusion == "independent" else fusion, mlp=shared_mlp)
        features.append(feature)
    fused = (Concatenate(axis=-1, name="feature_fusion_concat")(features) if len(features) == 2
             else Activation("linear", name="feature_fusion_single")(features[0]))

    x = Flatten(name="head_flatten")(fused)
    x = Dense(512, name="head_dense1")(x)
    x = BatchNormalization(name="head_bn1")(x)
    x = Activation("relu", name="head_relu1")(x)
    x = Dropout(head_dropout, name="head_dropout1")(x)

    if mode == "single":
        outputs = Dense(num_classes, activation="softmax", name="single_label_output")(x)
    else:
        outputs = Dense(num_classes, activation="sigmoid", name="multi_label_output")(x)

    name = f"MSFFAT_v3_{mode}" if (variant, fusion) == ("full", "proposed") else f"MSFFAT_v3_{variant}_{fusion}_{mode}"
    return Model(inputs=inputs, outputs=outputs, name=name)


def set_attention_only_trainable(model: Model, keyword: str = "attention") -> Model:
    """Freeze all layers except channel-attention layers.

    The legacy code froze layers by numeric index, which is brittle.  This
    helper freezes by layer name so that attention transfer remains stable after
    minor architecture edits.
    """
    return set_adaptation_scope(model, "attention")


def set_attention_and_output_trainable(model: Model, keyword: str = "attention") -> Model:
    """Compatibility helper for the explicit attention+head ablation."""
    return set_adaptation_scope(model, "attention_head")


def model_identity(model: Model) -> dict:
    """Infer actual graph identity, not a user-supplied display name."""
    names = {layer.name for layer in model.layers}
    has_ltf, has_msf = "ltf_s3_pool" in names, "msf_dropout3" in names
    if not (has_ltf or has_msf):
        raise ValueError("No MSFFAT feature branch")
    fusion = ("proposed" if "shared_attention_dense_1" in names else
              "independent" if any(n.endswith("_attention_statistics") for n in names) else
              "standard" if any(n.endswith("_attention_add") for n in names) else "concat")
    return {"variant": "full" if has_ltf and has_msf else "ltf_only" if has_ltf else "msf_only",
            "fusion": fusion, "mode": "multi" if "multi_label_output" in names else "single",
            "length": int(model.input_shape[1]), "classes": int(model.output_shape[-1])}


def validate_aligned_model(model: Model, *, allow_ablation: bool = False) -> dict:
    """Check backbone parameters and key graph connections, rejecting proxy graphs."""
    identity = model_identity(model)
    if not allow_ablation and (identity["variant"], identity["fusion"]) != ("full", "proposed"):
        raise ValueError("Ablation checkpoint cannot be labeled as the complete method")
    if model.input_shape[-1] != 1 or "fusion_attention_weights" in {l.name for l in model.layers}:
        raise ValueError("Legacy/incompatible checkpoint")

    def check(name, cls, **fields):
        layer = model.get_layer(name)
        if not isinstance(layer, cls):
            raise ValueError(f"Wrong layer type: {name}")
        cfg = layer.get_config()
        for key, value in fields.items():
            actual = cfg[key]
            if isinstance(value, tuple):
                actual = tuple(actual)
            if actual != value:
                raise ValueError(f"Architecture mismatch {name}.{key}: {actual} != {value}")
        return layer

    def parents(name):
        tensors = tf.nest.flatten(model.get_layer(name).input)
        return [getattr(t._keras_history, "operation", t._keras_history[0]).name for t in tensors]

    checked = {"cells"}
    original_check = check
    def check(name, cls, **fields):
        checked.add(name)
        return original_check(name, cls, **fields)

    def connected(name, expected):
        if parents(name) != expected:
            raise ValueError(f"Architecture wiring mismatch at {name}")

    previous = "cells"
    for i, channels in ((1, 32), (2, 64)):
        check(f"sfed_conv{i}", Conv1D, filters=channels, kernel_size=(8,), strides=(1,), dilation_rate=(1,), padding="same", activation="linear", use_bias=True)
        check(f"sfed_bn{i}", BatchNormalization, epsilon=.001, momentum=.99)
        check(f"sfed_elu{i}", ELU, alpha=1.)
        check(f"sfed_pool{i}", MaxPooling1D, pool_size=(4,), strides=(2,), padding="valid")
        for name in (f"sfed_conv{i}", f"sfed_bn{i}", f"sfed_elu{i}", f"sfed_pool{i}"):
            connected(name, [previous])
            previous = name
    check("sfed_dropout", Dropout, rate=.1)
    connected("sfed_dropout", [previous])
    branch_outputs = []
    for branch in ("ltf", "msf"):
        if identity["variant"] != "full" and identity["variant"] != f"{branch}_only":
            continue
        if branch == "ltf":
            previous = "sfed_dropout"
            for stage in range(4):
                for block, dilations in enumerate(((1, 2), (4, 8))):
                    prefix = f"ltf_s{stage}_b{block}"
                    residual_input = previous
                    for conv, dilation in enumerate(dilations, 1):
                        stride = 2 if conv == 1 and block == 0 and stage in (1, 2) else 1
                        check(f"ltf_s{stage}_b{block}_conv{conv}", Conv1D, filters=64 * 2**stage,
                              kernel_size=(5,), dilation_rate=(dilation,), strides=(stride,), padding="causal", use_bias=False, activation="linear")
                        connected(f"{prefix}_conv{conv}", [previous])
                        check(f"{prefix}_bn{conv}", BatchNormalization, epsilon=1e-5, momentum=.99)
                        connected(f"{prefix}_bn{conv}", [f"{prefix}_conv{conv}"])
                        previous = f"{prefix}_bn{conv}"
                        if conv == 1:
                            check(f"{prefix}_relu1", Activation, activation="relu")
                            connected(f"{prefix}_relu1", [previous])
                            previous = f"{prefix}_relu1"
                    shortcut = residual_input
                    if block == 0:
                        stride = 2 if stage in (1,2) else 1
                        check(f"{prefix}_shortcut_conv", Conv1D, filters=64*2**stage, kernel_size=(1,), strides=(stride,), use_bias=False, padding="valid", activation="linear")
                        check(f"{prefix}_shortcut_bn", BatchNormalization, epsilon=1e-5, momentum=.99)
                        connected(f"{prefix}_shortcut_conv", [residual_input])
                        connected(f"{prefix}_shortcut_bn", [f"{prefix}_shortcut_conv"])
                        shortcut = f"{prefix}_shortcut_bn"
                    check(f"{prefix}_add", Add)
                    connected(f"{prefix}_add", [previous, shortcut])
                    check(f"{prefix}_relu2", Activation, activation="relu")
                    connected(f"{prefix}_relu2", [f"{prefix}_add"])
                    previous = f"{prefix}_relu2"
                check(f"ltf_s{stage}_pool", MaxPooling1D, pool_size=(4,), strides=(2,), padding="valid")
                connected(f"ltf_s{stage}_pool", [previous])
                previous = f"ltf_s{stage}_pool"
            last = "ltf_s3_pool"
        else:
            widths = ((8,8,16,32),(16,16,32,64),(16,16,32,64),(32,32,64,128),(32,32,64,128),(64,64,128,256))
            previous = "sfed_dropout"
            for block, filters in enumerate(widths, 1):
                for kernel, channels in zip((3,5,7,9), filters):
                    name = f"msf_block{block}_k{kernel}"
                    check(name, Conv1D, filters=channels, kernel_size=(kernel,), strides=(1,), dilation_rate=(1,), use_bias=True, padding="same", activation="relu")
                    connected(name, [previous])
                prefix = f"msf_block{block}"
                check(f"{prefix}_concat", Concatenate, axis=-1)
                connected(f"{prefix}_concat", [f"{prefix}_k{k}" for k in (9,7,5,3)])
                if block % 2:
                    check(f"{prefix}_residual", Add)
                    connected(f"{prefix}_residual", [f"{prefix}_concat", previous])
                    previous = f"{prefix}_residual"
                else:
                    previous = f"{prefix}_concat"
                check(f"{prefix}_bn", BatchNormalization, epsilon=.001, momentum=.99)
                connected(f"{prefix}_bn", [previous])
                previous = f"{prefix}_bn"
                if block % 2 == 0:
                    i = block//2
                    check(f"msf_pool{i}", MaxPooling1D, pool_size=(8,), strides=(4,), padding="valid")
                    connected(f"msf_pool{i}", [previous])
                    check(f"msf_dropout{i}", Dropout, rate=.2)
                    connected(f"msf_dropout{i}", [f"msf_pool{i}"])
                    previous = f"msf_dropout{i}"
            last = "msf_dropout3"
        if identity["fusion"] != "concat":
            prefix = f"{branch}_attention"
            check(f"{prefix}_weights", Activation, activation="sigmoid")
            first_name = "shared_attention_dense_1" if identity["fusion"] == "proposed" else f"{prefix}_shared_dense_1"
            second_name = "shared_attention_dense_2" if identity["fusion"] == "proposed" else f"{prefix}_shared_dense_2"
            check(first_name, Dense, units=32, activation="relu", use_bias=True)
            check(second_name, Dense, units=512, activation="linear", use_bias=True)
            check(f"{prefix}_avg_pool", GlobalAveragePooling1D)
            check(f"{prefix}_max_pool", GlobalMaxPooling1D)
            check(f"{prefix}_multiply", Multiply)
            connected(f"{prefix}_multiply", [last, f"{prefix}_weights"])
            if identity["fusion"] in {"proposed", "independent"}:
                check(f"{prefix}_statistics", Concatenate, axis=-1)
                connected(f"{prefix}_statistics", [f"{prefix}_avg_pool", f"{prefix}_max_pool"])
                if identity["fusion"] == "independent":
                    connected(first_name, [f"{prefix}_statistics"])
                    connected(second_name, [first_name])
                connected(f"{prefix}_weights", [second_name])
            else:
                check(f"{prefix}_add", Add)
                connected(f"{prefix}_add", [f"{prefix}_shared_dense_2"]*2)
                connected(f"{prefix}_weights", [f"{prefix}_add"])
                # A shared MLP must be called once for each pooling path.
                first = model.get_layer(f"{prefix}_shared_dense_1")
                second = model.get_layer(f"{prefix}_shared_dense_2")
                if len(first._inbound_nodes) != 2 or len(second._inbound_nodes) != 2:
                    raise ValueError("Standard attention does not share its MLP over two paths")
                pooled = [tf.nest.flatten(n.input_tensors)[0]._keras_history[0].name for n in first._inbound_nodes]
                if pooled != [f"{prefix}_avg_pool",f"{prefix}_max_pool"]:
                    raise ValueError("Standard attention has the wrong pooling paths")
                inputs = [tf.nest.flatten(n.input_tensors)[0]._keras_history for n in second._inbound_nodes]
                if [(h[0].name,h[1]) for h in inputs] != [(first.name,0),(first.name,1)]:
                    raise ValueError("Standard attention has crossed/repeated shared paths")
            expected_in = 1024 if identity["fusion"] in {"proposed", "independent"} else 512
            if model.get_layer(first_name).kernel.shape[0] != expected_in:
                raise ValueError("Average/max statistics do not match the specified fusion")
            if parents(f"{prefix}_avg_pool") != [last] or parents(f"{prefix}_max_pool") != [last]:
                raise ValueError("Attention is not applied independently inside its branch")
            last = f"{prefix}_multiply"
        branch_outputs.append(last)
    if identity["fusion"] == "proposed":
        first = model.get_layer("shared_attention_dense_1")
        second = model.get_layer("shared_attention_dense_2")
        branches = [b for b in ("ltf", "msf") if identity["variant"] in {"full", b+"_only"}]
        if len(first._inbound_nodes) != len(branches) or len(second._inbound_nodes) != len(branches):
            raise ValueError("The shared attention MLP must be called exactly once per branch")
        for index, branch in enumerate(branches):
            first_input = tf.nest.flatten(first._inbound_nodes[index].input_tensors)[0]._keras_history
            second_input = tf.nest.flatten(second._inbound_nodes[index].input_tensors)[0]._keras_history
            gate_input = tf.nest.flatten(model.get_layer(branch+"_attention_weights").input)[0]._keras_history
            if first_input[0].name != branch+"_attention_statistics" or (second_input[0].name,second_input[1]) != (first.name,index) or (gate_input[0].name,gate_input[1]) != (second.name,index):
                raise ValueError("Shared MLP does not preserve the separate branch-to-gate paths")
    fusion_layer = "feature_fusion_concat" if len(branch_outputs) == 2 else "feature_fusion_single"
    check(fusion_layer, Concatenate if len(branch_outputs)==2 else Activation)
    if len(branch_outputs)==2:
        check(fusion_layer, Concatenate, axis=-1)
    else:
        check(fusion_layer, Activation, activation="linear")
    if parents(fusion_layer) != branch_outputs:
        raise ValueError("Wrong branch order or attention/fusion order")
    check("head_flatten", Flatten)
    check("head_dense1", Dense, units=512, activation="linear", use_bias=True)
    check("head_bn1", BatchNormalization, epsilon=.001, momentum=.99)
    check("head_relu1", Activation, activation="relu")
    check("head_dropout1", Dropout, rate=.5)
    output = "multi_label_output" if identity["mode"] == "multi" else "single_label_output"
    check(output, Dense, units=identity["classes"], activation="sigmoid" if identity["mode"] == "multi" else "softmax")
    previous = fusion_layer
    for name in ("head_flatten","head_dense1","head_bn1","head_relu1","head_dropout1",output):
        connected(name, [previous])
        previous = name
    if checked != {layer.name for layer in model.layers}:
        raise ValueError("Checkpoint contains unexpected/missing graph layers")
    if identity["length"] == 5000 and tuple(model.get_layer(fusion_layer).output.shape[1:]) != (18, 1024 if len(branch_outputs)==2 else 512):
        raise ValueError("Wrong temporal/channel dimensions for the 5000-cell input")
    return identity


def set_adaptation_scope(model: Model, scope: str = "attention") -> Model:
    """Set main-method attention scope or a named head/full ablation.

    Head means the entire classification head, including Dense(512), BN affine
    parameters and the output layer. Partial adaptation keeps BN running moments
    and Dropout in inference mode; see maintenance.attention_transfer_fit.
    """
    validate_aligned_model(model, allow_ablation=True)
    if scope not in {"attention", "head", "attention_head", "full"}:
        raise ValueError(f"Unknown adaptation scope: {scope}")
    for layer in model.layers:
        attention = layer.name.startswith(("ltf_attention_", "msf_attention_", "shared_attention_"))
        head = layer.name.startswith("head_") or layer.name in {"single_label_output", "multi_label_output"}
        layer.trainable = (scope == "full" or (attention and scope in {"attention", "attention_head"})
                           or (head and scope in {"head", "attention_head"}))
    if not model.trainable_variables:
        raise ValueError("Selected scope has no parameters (e.g. attention scope on direct concatenation)")
    return model
