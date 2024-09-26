import xarray as xr
import equinox as eqx
import jax.numpy as jnp
import pytest
import jax

"""
Tests for the examples in the README.
"""


def test_dataarray_example():
    da = xr.DataArray(
        xr.Variable(["x", "y"], jnp.ones((2, 3))),
        coords={"x": [1, 2], "y": [3, 4, 5]},
        name="foo",
        attrs={"attr1": "value1"},
    )

    @eqx.filter_jit
    def some_function(data):
        neg_data = -1.0 * data  # Multiply data by -1.
        return neg_data * neg_data.coords["y"]  # Multiply data by coords.

    da_new = some_function(da)

    assert da_new.equals(-1.0 * da * da.coords["y"])

    @eqx.filter_jit
    def fn(data):
        return (data**2.0).sum().data

    grad = jax.grad(fn)(da)
    assert grad.equals(2.0 * da)


def test_fail_example():
    var = xr.Variable(dims=("x", "y"), data=jnp.ones((3, 3)))

    # This will succeed.
    var = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), var)

    with pytest.raises(ValueError):
        # This will fail.
        var = var + 1
