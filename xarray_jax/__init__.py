import xarray
import xarray_jax.register_pytrees as register_pytrees
from xarray_jax.custom_types import XjDataArray, XjDataset, XjVariable

xarray.set_options(keep_attrs=True)  # Necessary for preserving PyTree structure.

__all__ = ["XjDataArray", "XjDataset", "XjVariable"]
