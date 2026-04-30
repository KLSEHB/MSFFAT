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


def channel_attention(inputs, ratio: int = 16, name: str = "channel_attention"):
    channels = int(inputs.shape[-1])
    hidden = max(channels // ratio, 1)

    shared_dense_1 = Dense(
        hidden,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        name=f"{name}_shared_dense_1",
    )
    shared_dense_2 = Dense(
        channels,
        kernel_initializer="he_normal",
        use_bias=True,
        name=f"{name}_shared_dense_2",
    )

    avg_pool = GlobalAveragePooling1D(name=f"{name}_avg_pool")(inputs)
    avg_pool = shared_dense_2(shared_dense_1(avg_pool))

    max_pool = GlobalMaxPooling1D(name=f"{name}_max_pool")(inputs)
    max_pool = shared_dense_2(shared_dense_1(max_pool))

    weights = Add(name=f"{name}_add")([avg_pool, max_pool])
    weights = Activation("hard_sigmoid", name=f"{name}_weights")(weights)
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


def ltf_residual_block(x, filters: int, dilation_pair, stage: int, block: int):
    name = f"ltf_s{stage}_b{block}"
    y = Conv1D(
        filters,
        5,
        padding="causal",
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

    if int(x.shape[-1]) != filters:
        shortcut = Conv1D(filters, 1, use_bias=False, name=f"{name}_shortcut_conv")(x)
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
        x = ltf_residual_block(x, filters, (1, 2), stage, 0)
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

    inputs = tf.keras.Input(shape=input_shape, name="cells")
    x = surface_downsample(inputs)
    ms_features = msf_branch(x, dropout=msf_dropout)
    lt_features = ltf_branch(x)
    ms_features, lt_features = TemporalCropToMatch(name="align_temporal")([ms_features, lt_features])
    fused = Concatenate(axis=-1, name="feature_fusion_concat")([ms_features, lt_features])
    fused = channel_attention(fused, ratio=16, name="fusion_attention")

    x = Flatten(name="head_flatten")(fused)
    x = Dense(512, name="head_dense1")(x)
    x = BatchNormalization(name="head_bn1")(x)
    x = Activation("relu", name="head_relu1")(x)
    x = Dropout(head_dropout, name="head_dropout1")(x)

    if mode == "single":
        outputs = Dense(num_classes, activation="softmax", name="single_label_output")(x)
    else:
        outputs = Dense(num_classes, activation="sigmoid", name="multi_label_output")(x)

    return Model(inputs=inputs, outputs=outputs, name=f"MSFFAT_{mode}")


def set_attention_only_trainable(model: Model, keyword: str = "attention") -> Model:
    """Freeze all layers except channel-attention layers.

    The legacy code froze layers by numeric index, which is brittle.  This
    helper freezes by layer name so that attention transfer remains stable after
    minor architecture edits.
    """
    for layer in model.layers:
        layer.trainable = keyword in layer.name
    return model
