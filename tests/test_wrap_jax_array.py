import xarray as xr
from hypothesis import given
import jax.numpy as jnp
import jax
from hypothesis.extra.array_api import make_strategies_namespace
import xarray.testing.strategies as xrst
from hypothesis.strategies import sampled_from

jax.config.update("jax_enable_x64", True)

jnps = make_strategies_namespace(jnp)

xp_variables = xrst.variables(
    array_strategy_fn=jnps.arrays,
    dtype=jnps.scalar_dtypes(),
)


# Creating a strategy for ufuncs to test.
ufuncs = [jnp.abs, jnp.sin, jnp.cos, jnp.exp, jnp.log]
ufunc_strat = sampled_from(ufuncs)


@given(var=xp_variables, ufunc=ufunc_strat)
def test_arrays_are_jax(var: xr.Variable, ufunc):
    assert isinstance(var.data, jax.Array)

    # Test that we can apply a ufunc to the variable and get the correct result as if we applied it directly to the data.
    result_var = xr.apply_ufunc(ufunc, var)
    assert isinstance(result_var.data, jax.Array)
    assert result_var.equals(xr.Variable(var.dims, ufunc(var.data), var.attrs))
