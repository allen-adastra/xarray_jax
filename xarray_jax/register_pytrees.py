import xarray
import jax
from .structs import _HashableCoords
from typing import Tuple, Hashable, Mapping, TypeVar
from collections import OrderedDict

DatasetOrDataArray = TypeVar("DatasetOrDataArray", xarray.Dataset, xarray.DataArray)

_JAX_COORD_ATTR_NAME = "_jax_coord"


def _flatten_variable(
    v: xarray.Variable,
) -> Tuple[Tuple[jax.typing.ArrayLike], Tuple[Hashable, ...]]:
    children = (v.data,)
    aux = (v.dims,)
    assert isinstance(aux, Hashable)
    return children, aux


def _unflatten_variable(
    aux: Tuple[Hashable, ...], children: Tuple[jax.typing.ArrayLike]
) -> xarray.Variable:
    """Unflattens a Variable for jax.tree_util."""
    dims = aux
    data = children[0]
    var = object.__new__(xarray.Variable)
    var._dims = dims
    var._data = data
    return var


def _flatten_data_array(
    da: xarray.DataArray,
):
    children = (da.variable,)
    aux = (da.name, _HashableCoords(da._coords), da._indexes)
    assert isinstance(aux, Hashable)
    return children, aux


def _unflatten_data_array(
    aux: Tuple[Hashable, ...], children: Tuple[jax.typing.ArrayLike]
) -> xarray.DataArray:
    """Unflattens a DataArray for jax.tree_util."""
    name, coords, indexes = aux
    da = object.__new__(xarray.DataArray)
    da._variable = children[0]
    da._name = name
    da._coords = coords
    da._indexes = indexes
    return da


def _flatten_dataset(
    ds: xarray.Dataset,
):
    data_vars = {name: data_array.variable for name, data_array in ds.data_vars.items()}
    data_var_names = tuple(data_vars.keys())
    data_var_vals = tuple(data_vars.values())
    coords = _HashableCoords(ds.coords)
    children = (data_var_vals,)
    aux = coords
    assert isinstance(aux, Hashable)
    return children, aux


def _unflatten_dataset(
    aux: _HashableCoords,
    children: Tuple[
        Mapping[Hashable, xarray.Variable], Mapping[Hashable, xarray.Variable]
    ],
) -> xarray.Dataset:
    """Unflattens a Dataset for jax.tree_util."""
    data_vars = children
    coords = aux
    ds = object.__new__(xarray.Dataset)
    # ds._attrs = attrs
    ds._coord_names = coords.keys()
    # ds._dims = dims
    # ds._indexes = indexes
    ds._variables = data_vars | coords
    return ds


jax.tree_util.register_pytree_node(
    xarray.Variable, _flatten_variable, _unflatten_variable
)
jax.tree_util.register_pytree_node(
    xarray.IndexVariable, _flatten_variable, _unflatten_data_array
)
jax.tree_util.register_pytree_node(
    xarray.DataArray, _flatten_data_array, _unflatten_data_array
)
jax.tree_util.register_pytree_node(xarray.Dataset, _flatten_dataset, _unflatten_dataset)
