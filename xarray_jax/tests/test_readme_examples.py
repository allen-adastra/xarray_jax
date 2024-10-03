import xarray as xr
import equinox as eqx
import jax.numpy as jnp
import pytest
import jax

"""
Tests for the examples in the README.
"""


def test_main_example():
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
        return neg_data * neg_data.coords["y"]  # Multiply data by coords.

    da = some_function(da)

    # Construct a xr.DataArray with dummy data (useful for tree manipulation).
    da_mask = jax.tree.map(lambda _: True, da)

    # Take the gradient of a jitted function.    @eqx.filter_jit
    def fn(data):
        return (data**2.0).sum().data

    da_grad = jax.grad(fn)(da)

    # Convert to a custom XjDataArray, implemented as an equinox module.
    # (Useful for avoiding potentially weird xarray interactions with JAX).
    xj_da = xj.from_xarray(da)

    # Convert back to a xr.DataArray.
    da = xj.to_xarray(xj_da)

    # Use xj.var_change_on_unflatten to allow us to expand the dimensions of the DataArray.
    def add_dim_to_var(var):
        var._dims = ("new_dim", *var._dims)
        return var

    with xj.var_change_on_unflatten(add_dim_to_var):
        da = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), da)


def test_fail_example():
    var = xr.Variable(dims=("x", "y"), data=jnp.ones((3, 3)))

    # This will succeed.
    var = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), var)

    with pytest.raises(ValueError):
        # This will fail.
        var = var + 1

    var = xr.Variable(dims=("x", "y"), data=jnp.ones((3, 3)))

    # This will fail.
    with pytest.raises(TypeError):
        jnp.square(var)

    # This will work.
    xr.apply_ufunc(jnp.square, var)
