import xarray as xr
from hypothesis import given, settings
import hypothesis.strategies as st
import jax.numpy as jnp
import jax
from hypothesis.extra.array_api import make_strategies_namespace
import xarray.testing.strategies as xrst
from hypothesis.strategies import sampled_from

import xarray_jax
import pytest
import equinox as eqx
from jaxtyping import PyTree
import numpy as np

jax.config.update("jax_enable_x64", True)

jnps = make_strategies_namespace(jnp)

xp_variables = xrst.variables(
    array_strategy_fn=jnps.arrays,
    dtype=jnps.scalar_dtypes(),
)

index_variables = xrst.variables(
    array_strategy_fn=jnps.arrays,
    dtype=jnps.scalar_dtypes(),
    dims=xrst.dimension_names(min_dims=1, max_dims=1),
)

# TODO(allenw): for now, just construct simple DataArrays from Variables.
# Pending xarray developers adding strategies for DataArrays.
# https://github.com/pydata/xarray/pull/6908
data_arrays = xp_variables.map(
    lambda var: xr.DataArray(
        var,
        coords={
            "dummy_coord": (var.dims, np.ones(var.data.shape)),
            "dummy_coord2": (var.dims, np.asarray(var.data)),
        },
    )
)

# Creating a strategy for ufuncs to test.
ufuncs = [jnp.abs, jnp.sin, jnp.cos, jnp.exp, jnp.log]
ufunc_strat = sampled_from(ufuncs)


@given(var=xp_variables, ufunc=ufunc_strat)
@settings(deadline=None)
def test_variables(var: xr.Variable, ufunc):
    # Test that we can wrap a jax array in a variable.
    assert isinstance(var.data, jax.Array)

    #
    # Test that we can flatten and unflatten the variable and get the same result.
    #
    leaves, treedef = jax.tree.flatten(var)
    var_unflattened = jax.tree.unflatten(treedef, leaves)
    assert var.equals(var_unflattened)

    #
    # Test that we can apply a jitted ufunc to the variable and get the correct result as if we applied it directly to the data.
    #
    @eqx.filter_jit
    def fn(v):
        return xr.apply_ufunc(ufunc, v)

    result_var = fn(var)
    assert isinstance(result_var.data, jax.Array)
    assert result_var.equals(xr.Variable(var.dims, ufunc(var.data), var.attrs))

    #
    # Test that we can create a boolean mask tree.
    #
    var_mask = jax.tree.map(lambda _: True, var)
    assert var_mask._data is True
    assert var_mask._dims == var._dims
    assert var_mask._attrs == var._attrs


@given(var=index_variables, ufunc=ufunc_strat)
@settings(deadline=None)
def test_index_variables(var: xr.Variable, ufunc):
    var = var.to_index_variable()

    # Index variables should not be wrapped in a jax array.
    assert not isinstance(var.data, jax.Array)


@given(da=data_arrays, ufunc=ufunc_strat)
@settings(deadline=None)
def test_dataarrays(da: xr.DataArray, ufunc):
    #
    # Test that the data is a jax array.
    #
    assert isinstance(da.variable.data, jax.Array)

    #
    # Test that we can flatten and unflatten the da and get the same result.
    #
    leaves, treedef = jax.tree.flatten(da)
    da_unflattened = jax.tree.unflatten(treedef, leaves)
    assert da.equals(da_unflattened)

    #
    # Test that we can apply a jitted ufunc to the da and get the correct result as if we applied it directly to the data.
    #
    @eqx.filter_jit
    def fn(da_):
        return xr.apply_ufunc(ufunc, da_)

    result_da = fn(da)

    expected_da = xr.DataArray(
        xr.Variable(da.variable.dims, ufunc(da.variable.data), da.variable.attrs),
        coords=da.coords,
    )
    assert result_da.equals(expected_da)


@given(xr_data=st.one_of(xp_variables, index_variables, data_arrays))
@settings(deadline=None)
def test_roundtrip(xr_data):
    #
    # Test that we can convert an xarray object to a xarray_jax object and back.
    #
    xj_data = xarray_jax.from_xarray(xr_data)
    assert isinstance(xj_data, PyTree)
    xr_data_roundtrip = xarray_jax.to_xarray(xj_data)
    assert xr_data.equals(xr_data_roundtrip)

    #
    # Test that we can go back and forth inside a jit-compiled function.
    #
    @eqx.filter_jit
    def fn(xj_data_):
        xr_data_ = xarray_jax.to_xarray(xj_data_)
        xr_data_ = -1.0 * xr_data_  # Some operation.
        return xarray_jax.from_xarray(xr_data_)

    xj_data_roundtrip_neg = fn(xj_data)
    xr_data_roundtrip_neg = xarray_jax.to_xarray(xj_data_roundtrip_neg)
    assert xr_data_roundtrip_neg.equals(-1.0 * xr_data)


@pytest.mark.skip(reason="Dataset is not yet supported.")
def test_ds():
    ds = xr.tutorial.load_dataset("air_temperature")

    ds_mask = jax.tree.map(lambda x: True, ds)

    for k, v in ds_mask.data_vars.items():
        assert v._data is True  # TODO(allenw): RecursionError!
        assert v._dims == ds[k]._dims
        assert v._attrs == ds[k]._attrs
