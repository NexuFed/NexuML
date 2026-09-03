"""NexuML - Modular deep learning pipeline framework."""

from importlib.metadata import version

from nexuml.core.torch_adapter import NnModuleLayer, nn_module

__version__ = version("nexuml")

__all__ = ["NnModuleLayer", "nn_module"]
