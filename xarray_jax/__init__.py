import xarray
from xarray_jax.custom_types import (
    XjDataArray,
    XjDataset,
    XjVariable,
    from_xarray,
    to_xarray,
)
from xarray_jax.register_pytrees import var_change_on_unflatten

xarray.set_options(keep_attrs=True)  # Necessary for preserving PyTree structure.

__all__ = [
    "XjDataArray",
    "XjDataset",
    "XjVariable",
    "from_xarray",
    "to_xarray",
    "var_change_on_unflatten",
]
