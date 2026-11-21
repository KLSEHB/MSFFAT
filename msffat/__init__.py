"""MSFFAT package.

This package provides reproducible entry points with configurable paths and no
import-time data loading.
"""

__all__ = ["build_msffat", "set_attention_only_trainable"]


def __getattr__(name):
    # Numerical detector/data helpers do not require importing TensorFlow.
    if name in __all__:
        from . import model
        return getattr(model, name)
    raise AttributeError(name)
