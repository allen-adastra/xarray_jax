import xarray_jax
import jax
import xarray as xr


def test_variable():
    # Create a sample xarray Variable
    var = xr.Variable(
        data=jax.numpy.array([1, 2, 3]), dims=("x",), attrs={"units": "m"}
    )
    var_mask = jax.tree.map(lambda x: True, var)

    assert var_mask._data is True


def test_dataarray():
    # Create a sample xarray DataArray
    da = xr.DataArray(
        data=jax.numpy.array([[1, 2], [3, 4]]), dims=("x", "y"), attrs={"units": "m"}
    )
    da_mask = jax.tree.map(lambda x: True, da)

    assert da_mask._variable._data is True


def test_recursion_issue():
    import jax
    import xarray_jax
    import xarray as xr
    import jax.numpy as jnp

    da = xr.DataArray(jnp.arange(10), dims=["x"])

    da_mask = jax.tree.map(lambda x: True, da)

    print(da_mask)


def test_ds():
    ds = xr.tutorial.load_dataset("air_temperature")

    foo, bar = jax.tree.flatten(ds)

    back = jax.tree.unflatten(bar, foo)
    import pdb

    pdb.set_trace()

    # ds_mask = jax.tree.map(lambda x: True, ds)

    # v = ds_mask.data_vars
    # import pdb

    # pdb.set_trace()
