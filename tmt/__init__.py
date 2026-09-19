"""Test-Model-Thing (TMT): Byte-level recurrent language model."""

__version__ = "0.2.0"

from tmt.model import Model, count_params
from tmt.runtime import Runtime
from tmt.torch_model import TorchModel, is_torch_available

__all__ = ["Model", "Runtime", "TorchModel", "count_params", "is_torch_available", "__version__"]
