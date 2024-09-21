import xarray as xr
import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis.extra.array_api import make_strategies_namespace
import xarray.testing.strategies as xrst
from hypothesis.strategies import sampled_from
import numpy as np
import xarray_jax
import equinox as eqx
import jax

jnps = make_strategies_namespace(jnp)

xp_variables = xrst.variables(
    array_strategy_fn=jnps.arrays,
    dtype=jnps.scalar_dtypes(),
)

xp_variables_float = xrst.variables(
    array_strategy_fn=jnps.arrays, dtype=jnps.floating_dtypes()
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

data_arrays_float = xp_variables_float.map(
    lambda var: xr.DataArray(
        var,
        coords={
            "dummy_coord": (var.dims, np.ones(var.data.shape)),
            "dummy_coord2": (var.dims, np.asarray(var.data)),
        },
    )
)

generic_xr_strat = st.one_of(
    xp_variables,
    data_arrays,
    st.just(xr.tutorial.load_dataset("air_temperature")),
)

float_vars_and_das = st.one_of(
    xp_variables_float,
    data_arrays_float,
)

# Creating a strategy for ufuncs to test.
ufuncs = [jnp.abs, jnp.sin, jnp.cos, jnp.exp, jnp.log]
ufunc_strat = sampled_from(ufuncs)


"""
Strategy for sampling from identity transformations.
"""


def xj_roundtrip(xr_data):
    xj_data = xarray_jax.from_xarray(xr_data)
    xj_data_roundtrip = xarray_jax.to_xarray(xj_data)
    return xj_data_roundtrip


def flatten_unflatten(x):
    leaves, treedef = jax.tree.flatten(x)
    return jax.tree.unflatten(treedef, leaves)


def jit_identity(x):
    @eqx.filter_jit
    def fn(x_):
        return x_

    return fn(x)


def vmap_identity(x):
    @eqx.filter_vmap
    def fn(x_):
        return x_

    return fn(x)


def partition(x):
    out, _ = eqx.partition(x, lambda _: True)
    return out


def filt(x):
    return eqx.filter(x, lambda _: True)


identity_transforms = sampled_from(
    [
        xj_roundtrip,
        flatten_unflatten,
        jit_identity,
        vmap_identity,
        partition,
        filt,
    ]
)
