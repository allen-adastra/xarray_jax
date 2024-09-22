# (Experiment): Simple Xarray + JAX Integration

This is a simple experiment at integrating Xarray + JAX, leveraging [equinox](https://github.com/patrick-kidger/equinox).

``` python
import jax.numpy as jnp
import xarray as xr
import xarray_jax as xj

# Construct a DataArray.
da = xr.DataArray(
    xr.Variable(["x", "y"], jnp.ones((2, 3))),
    coords={"x": [1, 2], "y": [3, 4, 5]},
    name="foo",
    attrs={"attr1": "value1"},
)

# Do some operations inside a JIT compiled function.
@eqx.filter_jit
def some_function(data):
    neg_data = -1.0 * data
    return neg_data * neg_data.coords["y"] # Multiply data by coords.

da = some_function(da)

# Construct a xr.DataArray with dummy data (useful for tree manipulation).
da_mask = jax.tree.map(lambda _: True, data)

# Use jax.grad.
@eqx.filter_jit
def fn(data):
    return (data**2.0).sum().data

grad = jax.grad(fn)(da)
```


## Status
- [x] PyTree node registrations
  - [x] `xr.Variable`
  - [x] `xr.DataArray`
  - [x] `xr.Dataset`
- [x] Minimal shadow types implemented as [equinox modules](https://github.com/patrick-kidger/equinox) to handle edge cases (Note: these types are merely data structures that contain the data of these types. They don't have any of the methods of the xarray types).
  - [x] `XjVariable`
  - [x] `XjDataArray`
  - [x] `XjDataset`
- [x] `xj.from_xarray` and `xj.to_xarray` functions to go between `xj` and `xr` types.
- [x] Support for `xr` types with dummy data (useful for tree manipulation).
- [ ] Support for transformations that change the dimensionality of the data.

## Sharp Edges

### Prefer `eqx.filter_jit` over `jax.jit`
There are some edge cases with metadata that `eqx.filter_jit` handles but `jax.jit` does not.

### Operations that Increase the Dimensionality of the Data
Operations that increase the dimensionality of the data (e.g. `jnp.expand_dims`) will cause problems downstream.

``` python
var = xr.Variable(dims=("x", "y"), data=jnp.ones((3, 3)))

# This will not error.
var = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), var)

# The error from expanding the dimensionality will be triggered here.
var = var + 1 
```

### Dispatching to jnp is not supported yet
Pending resolution of https://github.com/pydata/xarray/issues/7848.
``` python
var = xr.Variable(dims=("x", "y"), data=jnp.ones((3, 3)))

# This will fail.
jnp.square(var)

# This will work.
xr.apply_ufunc(jnp.square, var)
```


## Distinction from the GraphCast Implementation
This experiment is largely inspired by the [GraphCast implementation](https://github.com/google-deepmind/graphcast/blob/main/graphcast/xarray_jax.py), with a direct re-use of the `_HashableCoords` in that project.

However, this experiment aims to:
1. Take a more minimialist approach (and thus neglects some features such as support JAX arrays as coordinates).
2. Find a solution more compatible with common JAX PyTree manipulation patterns that trigger errors with Xarray types. For example, it's common to use boolean masks to filter out elements of a PyTree, but this tends to fail with Xarray types.
