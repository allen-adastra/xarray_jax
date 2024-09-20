import xarray as xr
import equinox as eqx
import jax.numpy as jnp
import xarray_jax as xj
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
        neg_data = -1.0 * data # Multiply data by -1.
        return neg_data * neg_data.coords["y"] # Multiply data by coords.

    da_new = some_function(da)

    assert da_new.equals(-1.0 * da * da.coords["y"])

def test_fail_example():
    var = xr.Variable(dims=("x", "y"), data=jnp.ones((3, 3)))

    # This will succeed.
    var = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), var)

    with pytest.raises(ValueError):
        # This will fail.
        var = var + 1 

def test_dataset_example():
    ds = xr.tutorial.load_dataset("air_temperature")

    @eqx.filter_jit
    def some_function(xjds: xj.XjDataset):
        # Convert to xr.Dataset.
        xrds = xj.to_xarray(xjds)

        # Do some operation.
        xrds = -1.0 * xrds

        # Convert back to xj.Dataset.
        return xj.from_xarray(xrds)
    
    xjds = some_function(xj.from_xarray(ds))
    ds_new = xj.to_xarray(xjds)

    assert isinstance(ds, xr.Dataset)
    assert ds_new.equals(ds * -1.0)