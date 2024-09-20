import jax
import xarray
import xarray_jax.register_pytrees as register_pytrees

xarray.set_options(keep_attrs=True)  # Necessary for preserving PyTree structure.

__all__ = ["DataArray", "Dataset", "Variable"]
