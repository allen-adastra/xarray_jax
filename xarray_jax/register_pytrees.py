import xarray
from xarray.core.indexes import Indexes
import jax
from .custom_types import maybe_hash_coords, _HashableCoords
from typing import Tuple, Hashable, Mapping, Callable, Any
import contextlib
import contextvars

VarChangeFn = Callable[[xarray.Variable], xarray.Variable]
_VAR_CHANGE_ON_UNFLATTEN_FN: contextvars.ContextVar[VarChangeFn] = (
    contextvars.ContextVar("var_change_on_unflatten_fn")
)

XrAttrs = Mapping[Any, Any]


@contextlib.contextmanager
def var_change_on_unflatten(var_change_fn: VarChangeFn):
    """A context manager for modifying a variable on unflatten. This was inspired by the dims_change_on_unflatten function in the GraphCast project, but is ultimately a different approach."""
    token = _VAR_CHANGE_ON_UNFLATTEN_FN.set(var_change_fn)
    try:
        yield
    finally:
        _VAR_CHANGE_ON_UNFLATTEN_FN.reset(token)


def _flatten_variable(
    v: xarray.Variable,
) -> Tuple[Tuple[jax.typing.ArrayLike], Tuple[Hashable, XrAttrs]]:
    """Flattens a Variable for jax.tree_util."""
    children = (
        v._data,
    )  # Use the private interface for allowing tree manipulations such as tree masks.
    aux = (
        v.dims,
        v.attrs,
    )
    return children, aux


def _unflatten_variable(
    aux: Tuple[Hashable, XrAttrs], children: Tuple[jax.typing.ArrayLike]
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

    var_change_fn = _VAR_CHANGE_ON_UNFLATTEN_FN.get(None)
    if var_change_fn is not None:
        var = var_change_fn(var)

    return var


def _flatten_data_array(
    da: xarray.DataArray,
) -> Tuple[Tuple[xarray.Variable], Tuple[Hashable, _HashableCoords, Indexes]]:
    """Flattens a DataArray for jax.tree_util."""
    children = (da.variable,)
    aux = (
        da.name,
        maybe_hash_coords(da._coords),
        da._indexes,
    )  # TODO(allenw): tests break when using the public API for .coords and .indexes.
    return children, aux


def _unflatten_data_array(
    aux: Tuple[Hashable, _HashableCoords, Indexes],
    children: Tuple[xarray.Variable],
) -> xarray.DataArray:
    """Unflattens a DataArray for jax.tree_util."""
    name, coords, indexes = aux
    (variable,) = children
    da = object.__new__(xarray.DataArray)
    da._variable = variable
    da._name = name
    da._coords = dict(coords)
    da._indexes = indexes
    return da


def _flatten_dataset(
    ds: xarray.Dataset,
) -> Tuple[
    Tuple[Mapping[Hashable, xarray.Variable]],
    Tuple[_HashableCoords, Indexes, Hashable, XrAttrs],
]:
    """Flattens a Dataset for jax.tree_util."""
    coord_names = ds._coord_names
    variables = ds._variables

    coords = {name: variables[name] for name in coord_names}
    data_vars = {name: variables[name] for name in variables if name not in coord_names}

    children = (data_vars,)
    aux = (maybe_hash_coords(coords), ds._indexes, ds.dims, ds.attrs)
    assert isinstance(aux, Hashable)
    return children, aux


def _unflatten_dataset(
    aux: Tuple[_HashableCoords, Indexes, Hashable, XrAttrs],
    children: Tuple[Mapping[Hashable, xarray.Variable]],
) -> xarray.Dataset:
    """Unflattens a Dataset for jax.tree_util."""
    data_vars = children[0]
    coords, indexes, dims, attrs = aux

    ds = object.__new__(xarray.Dataset)
    ds._dims = dims
    ds._variables = data_vars | dict(coords)
    ds._coord_names = set(coords.keys())
    ds._attrs = attrs
    ds._indexes = indexes
    ds._encoding = None
    ds._close = None
    return ds


jax.tree_util.register_pytree_node(
    xarray.Variable, _flatten_variable, _unflatten_variable
)
jax.tree_util.register_static(xarray.IndexVariable)
jax.tree_util.register_pytree_node(
    xarray.DataArray, _flatten_data_array, _unflatten_data_array
)
jax.tree_util.register_pytree_node(xarray.Dataset, _flatten_dataset, _unflatten_dataset)
