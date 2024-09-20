import xarray
import jax
from .types import _HashableCoords
from typing import Tuple, Hashable, Mapping


def _flatten_variable(
    v: xarray.Variable,
) -> Tuple[Tuple[jax.typing.ArrayLike], Tuple[Hashable, ...]]:
    children = (v._data,)
    aux = (
        v._dims,
        v._attrs,
    )
    assert isinstance(aux, Hashable)
    return children, aux


def _unflatten_variable(
    aux: Tuple[Hashable, ...], children: Tuple[jax.typing.ArrayLike]
) -> xarray.Variable:
    """Unflattens a Variable for jax.tree_util."""
    (
        dims,
        attrs,
    ) = aux
    (data,) = children

    var = object.__new__(xarray.Variable)
    var._dims = dims
    var._data = data
    var._attrs = attrs
    var._encoding = None

    return var


def _flatten_data_array(
    da: xarray.DataArray,
):
    children = (da._variable,)
    aux = (da._name, da._coords, da._indexes)
    assert isinstance(aux, Hashable)
    return children, aux


def _unflatten_data_array(
    aux: Tuple[Hashable, ...], children: Tuple[jax.typing.ArrayLike]
) -> xarray.DataArray:
    """Unflattens a DataArray for jax.tree_util."""
    name, coords, indexes = aux
    (variable,) = children
    da = object.__new__(xarray.DataArray)
    da._variable = variable
    da._name = name
    da._coords = coords
    da._indexes = indexes
    return da


def _flatten_dataset(
    ds: xarray.Dataset,
):
    data_vars = {name: data_array.variable for name, data_array in ds.data_vars.items()}

    data_var_leaves, data_var_treedef = jax.tree.flatten(data_vars)
    children = data_var_leaves
    aux = (ds.coords, data_var_treedef, ds._indexes, ds._dims, ds._attrs)
    assert isinstance(aux, Hashable)
    return children, aux


def _unflatten_dataset(
    aux: _HashableCoords,
    children: Tuple[
        Mapping[Hashable, xarray.Variable], Mapping[Hashable, xarray.Variable]
    ],
) -> xarray.Dataset:
    """Unflattens a Dataset for jax.tree_util."""
    data_var_leaves = children
    coords, data_var_treedef, indexes, dims, attrs = aux

    data_vars = jax.tree.unflatten(data_var_treedef, data_var_leaves)

    ds = object.__new__(xarray.Dataset)
    ds._dims = dims
    ds._variables = data_vars | dict(coords)
    ds._coord_names = list(coords.keys())
    ds._attrs = attrs
    ds._indexes = indexes
    ds._encoding = None
    ds._close = None
    return ds


jax.tree_util.register_pytree_node(
    xarray.Variable, _flatten_variable, _unflatten_variable
)
jax.tree_util.register_pytree_node(
    xarray.IndexVariable, _flatten_variable, _unflatten_variable
)
jax.tree_util.register_pytree_node(
    xarray.DataArray, _flatten_data_array, _unflatten_data_array
)
jax.tree_util.register_pytree_node(xarray.Dataset, _flatten_dataset, _unflatten_dataset)
