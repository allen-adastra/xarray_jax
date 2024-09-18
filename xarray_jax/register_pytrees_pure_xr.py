import xarray
import jax
from .structs import _HashableCoords
from typing import Tuple, Hashable, Mapping


def _flatten_variable(
    v: xarray.Variable,
) -> Tuple[Tuple[jax.typing.ArrayLike], Tuple[Hashable, ...]]:
    children = (v.data,)
    aux = v.dims
    return children, aux


def _unflatten_variable(
    aux: Tuple[Hashable, ...], children: Tuple[jax.typing.ArrayLike]
) -> xarray.Variable:
    """Unflattens a Variable for jax.tree_util."""
    dims = aux
    return xarray.Variable(dims=dims, data=children[0])


def _flatten_data_array(
    da: xarray.DataArray,
):
    children = (da.variable,)
    aux = (da.name, _HashableCoords(da.coords))
    return children, aux


def _unflatten_data_array(
    aux: Tuple[Hashable, ...], children: Tuple[jax.typing.ArrayLike]
) -> xarray.DataArray:
    """Unflattens a DataArray for jax.tree_util."""
    name, coords = aux
    return xarray.DataArray(data=children[0], coords=coords, name=name)


def _flatten_dataset(
    ds: xarray.Dataset,
):
    variables = {name: data_array.variable for name, data_array in ds.data_vars.items()}
    children = (variables,)
    aux = _HashableCoords(ds.coords)
    return children, aux


def _unflatten_dataset(
    aux: _HashableCoords,
    children: Tuple[
        Mapping[Hashable, xarray.Variable], Mapping[Hashable, xarray.Variable]
    ],
) -> xarray.Dataset:
    """Unflattens a Dataset for jax.tree_util."""
    (data_vars,) = children
    coords = aux
    dataset = xarray.Dataset(data_vars, coords=coords)
    return dataset


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
