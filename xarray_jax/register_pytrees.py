import xarray
import jax
from .structs import XJVariable, XJDataArray, XJDataset


def _flatten_variable(
    v: xarray.Variable,
):
    children, aux = jax.tree.flatten(XJVariable(v))
    return children, aux


def _flatten_data_array(
    da: xarray.DataArray,
):
    children, aux = jax.tree.flatten(XJDataArray(da))
    return children, aux


def _flatten_dataset(
    ds: xarray.Dataset,
):
    children, aux = jax.tree.flatten(XJDataset(ds))
    return children, aux


jax.tree_util.register_pytree_node(
    xarray.Variable, _flatten_variable, jax.tree.unflatten
)
jax.tree_util.register_pytree_node(
    xarray.IndexVariable, _flatten_variable, jax.tree.unflatten
)
jax.tree_util.register_pytree_node(
    xarray.DataArray, _flatten_data_array, jax.tree.unflatten
)
jax.tree_util.register_pytree_node(xarray.Dataset, _flatten_dataset, jax.tree.unflatten)
