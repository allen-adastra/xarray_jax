import xarray
import jax
from .structs import _HashableCoords
from typing import Tuple, Hashable, Mapping, TypeVar, Callable
from collections import OrderedDict
import contextlib
import contextvars

DatasetOrDataArray = TypeVar("DatasetOrDataArray", xarray.Dataset, xarray.DataArray)

DimsChangeFn = Callable[[Tuple[Hashable, ...]], Tuple[Hashable, ...]]
_DIMS_CHANGE_ON_UNFLATTEN_FN: contextvars.ContextVar[DimsChangeFn] = (
    contextvars.ContextVar("dims_change_on_unflatten_fn")
)


@contextlib.contextmanager
def dims_change_on_unflatten(dims_change_fn: DimsChangeFn):
    """Can be used to change the dims used when unflattening arrays into xarrays.

    This is useful when some axes were added to / removed from the underlying jax
    arrays after they were flattened using jax.tree_util.tree_flatten, and you
    want to unflatten them again afterwards using the original treedef but
    adjusted for the added/removed dimensions.

    It can also be used with jax.tree_util.tree_map, when it's called with a
    function that adds/removes axes or otherwise changes the axis order.

    When dimensions are removed, any coordinates using those removed dimensions
    will also be removed on unflatten.

    This is implemented as a context manager that sets some thread-local state
    affecting the behaviour of our unflatten functions, because it's not possible
    to directly modify the treedef to change the dims/coords in it (and with
    tree_map, the treedef isn't exposed to you anyway).

    Args:
      dims_change_fn: Maps a tuple of dimension names for the original
        Variable/DataArray/Dataset that was flattened, to an updated tuple of
        dimensions which should be used when unflattening.

    Yields:
      To a context manager in whose scope jax.tree_util.tree_unflatten and
      jax.tree_util.tree_map will apply the dims_change_fn before reconstructing
      xarrays from jax arrays.
    """
    token = _DIMS_CHANGE_ON_UNFLATTEN_FN.set(dims_change_fn)
    try:
        yield
    finally:
        _DIMS_CHANGE_ON_UNFLATTEN_FN.reset(token)


def _flatten_variable(
    v: xarray.Variable,
) -> Tuple[Tuple[jax.typing.ArrayLike], Tuple[Hashable, ...]]:
    children = (v._data,)
    aux = (v._dims, v._attrs, v._encoding)
    assert isinstance(aux, Hashable)
    return children, aux


def _unflatten_variable(
    aux: Tuple[Hashable, ...], children: Tuple[jax.typing.ArrayLike]
) -> xarray.Variable:
    """Unflattens a Variable for jax.tree_util."""
    dims, attrs, encoding = aux
    (data,) = children
    dims_change_fn = _DIMS_CHANGE_ON_UNFLATTEN_FN.get(None)
    if dims_change_fn:
        dims = dims_change_fn(dims)

    var = object.__new__(xarray.Variable)
    var._dims = dims
    var._data = data
    var._attrs = attrs
    var._encoding = encoding
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
    coords = _HashableCoords(ds.coords)
    children = data_var_leaves
    aux = (coords, data_var_treedef, ds._indexes, ds._dims, ds._attrs)
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

    ds = xarray.Dataset()

    import pdb

    pdb.set_trace()

    ds._attrs = attrs
    ds._coord_names = list(coords.keys())
    ds._dims = dims
    ds._indexes = indexes
    ds._variables = data_vars | dict(coords)
    import pdb

    pdb.set_trace()
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
