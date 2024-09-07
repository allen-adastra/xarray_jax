# (Experiment): Simple Xarray + JAX Integration

This is a simple experiment at integrating Xarray + JAX, leveraging [equinox](https://github.com/patrick-kidger/equinox).

This involves two components:
1. Some minor changes to Xarray to prevent Xarray from automatically converting to numpy arrays. This will not be necessary starting with JAX v0.4.32 when the Array API gets adopted [relevant discussion](https://github.com/pydata/xarray/issues/7848#issuecomment-2336411994).
2. Registering `xr.Variable`, `xr.IndexVariable`, `xr.DataArray`, and `xr.Dataset` as PyTree nodes.

See the [notebook](./example.ipynb).

## Distinction from the GraphCast Implementation
This experiment is largely inspired by the [GraphCast implementation](https://github.com/google-deepmind/graphcast/blob/main/graphcast/xarray_jax.py), with a direct re-use of the `_HashableCoords` in that project.

However, this experiment aims to:
1. Take a more minimialist approach (and thus neglects some features such as support JAX arrays as coordinates).
2. Find a solution more compatible with common JAX PyTree manipulation patterns that trigger errors with Xarray types. For example, it's common to use boolean masks to filter out elements of a PyTree, but this tends to fail with Xarray types.