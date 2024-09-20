import xarray_jax
import jax
import jax.numpy as jnp
import xarray as xr

xr.set_options(keep_attrs=True)  # Necessary for preserving PyTree structure.


@jax.jit
def fn(x):
    # Test a couple operations we care about.
    foo = jax.tree.map(lambda x: 2.0 * x, x)
    bar = xr.apply_ufunc(jnp.square, foo)
    return bar


def test_variable():
    # Create a sample xarray Variable
    var = xr.Variable(
        data=jax.numpy.array([1, 2, 3]), dims=("x",), attrs={"units": "m"}
    )

    # Test creating a boolean mask tree.
    var_mask = jax.tree.map(lambda x: True, var)

    assert var_mask._data is True
    assert var_mask._dims == var._dims
    assert var_mask._attrs == var._attrs

    # Test applying a jitted function to a variable.
    var_fd = fn(var)
    assert jnp.equal(var_fd._data, jnp.square(2.0 * var._data)).all()
    assert var_fd._dims == var._dims
    assert var_fd._attrs == var._attrs


def test_dataarray():
    # Create a sample xarray DataArray
    da = xr.DataArray(
        data=jax.numpy.array([[1, 2], [3, 4]]), dims=("x", "y"), attrs={"units": "m"}
    )
    da_mask = jax.tree.map(lambda x: True, da)

    assert da_mask._variable._data is True
    assert da_mask._variable._dims == da._variable._dims
    assert da_mask._variable._attrs == da._variable._attrs

    da_fd = fn(da)
    assert jnp.equal(da_fd._variable._data, jnp.square(2.0 * da._variable._data)).all()
    assert da_fd._variable._dims == da._variable._dims
    assert da_fd._variable._attrs == da._variable._attrs


def test_ds():
    ds = xr.tutorial.load_dataset("air_temperature")

    ds_mask = jax.tree.map(lambda x: True, ds)

    for k, v in ds_mask.data_vars.items():
        assert v._data is True  # TODO(allenw): RecursionError!
        assert v._dims == ds[k]._dims
        assert v._attrs == ds[k]._attrs

    ds_fd = fn(ds)

    air_before = ds["air"]
    air_after = ds_fd["air"]
    assert jnp.equal(
        air_after._variable._data, jnp.square(2.0 * air_before._variable._data)
    ).all()
