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


def jax_jit_identity(x):
    return jax.jit(lambda x_: x_)(x)


def jax_jit_lowering_identity(x):
    lowered_fn = jax.jit(lambda x_: x_).lower(x)
    result = lowered_fn.compile()(x)
    return result


def eqx_jit_identity(x):
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


def random_partition_combine(x, seed=42):
    """Randomly partition and combine a tree."""

    # Generate a random mask on the input tree.
    leaves, treedef = jax.tree.flatten(x)
    uniform = jax.random.uniform(jax.random.PRNGKey(seed), len(leaves))
    keep = [
        (u > 0.5).item() for u in uniform
    ]  # For each leaf, 50/50 chance of keeping it.
    tree_mask = jax.tree.unflatten(treedef, keep)

    # Partition and combine the tree.
    tree0, tree1 = eqx.partition(x, tree_mask)
    return eqx.combine(tree0, tree1)


identity_transforms = sampled_from(
    [
        xj_roundtrip,
        flatten_unflatten,
        jax_jit_identity,
        jax_jit_lowering_identity,
        eqx_jit_identity,
        vmap_identity,
        partition,
        filt,
        random_partition_combine,
    ]
)
