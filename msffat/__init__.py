"""MSFFAT package.

This package provides reproducible entry points with configurable paths and no
import-time data loading.
"""

from .model import build_msffat, set_attention_only_trainable

__all__ = ["build_msffat", "set_attention_only_trainable"]
