import collections
from typing import (
    Hashable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
)

import jax
import xarray
import equinox as eqx


class _HashableCoords(collections.abc.Mapping):
    """
    Originally implemented in the GraphCast project:
    https://github.com/google-deepmind/graphcast/blob/main/graphcast/xarray_jax.py

    Wraps a dict of xarray Variables as hashable, used for static coordinates.

    This needs to be hashable so that when an xarray.Dataset is passed to a
    jax.jit'ed function, jax can check whether it's seen an array with the
    same static coordinates(*) before or whether it needs to recompile the
    function for the new values of the static coordinates.

    (*) note jax_coords are not included in this; their value can be different
    on different calls without triggering a recompile.
    """

    def __init__(self, coord_vars: Mapping[Hashable, xarray.Variable]):
        self._variables = coord_vars

    def __repr__(self) -> str:
        return f"_HashableCoords({repr(self._variables)})"

    def __getitem__(self, key: Hashable) -> xarray.Variable:
        return self._variables[key]

    def __len__(self) -> int:
        return len(self._variables)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._variables)

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash(
                frozenset(
                    (name, var.data.tobytes()) for name, var in self._variables.items()
                )
            )
        return self._hash

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, type(self)):
            return NotImplemented
        elif self._variables is other._variables:
            return True
        else:
            return self._variables.keys() == other._variables.keys() and all(
                variable.equals(other._variables[name])
                for name, variable in self._variables.items()
            )


class XJVariable(eqx.Module):
    data: jax.Array
    dims: Tuple[str, ...] = eqx.field(static=True)
    attrs: Optional[Mapping] = eqx.field(static=True)

    def __init__(self, var: xarray.Variable):
        self.data = var.data
        self.dims = var.dims
        self.attrs = var.attrs

    def to_xarray(self) -> xarray.Variable:
        if self.data is None:
            return None
        return xarray.Variable(dims=self.dims, data=self.data, attrs=self.attrs)


class XJDataArray(eqx.Module):
    variable: XJVariable
    coords: _HashableCoords = eqx.field(static=True)
    name: Optional[str] = eqx.field(static=True)

    def __init__(self, da: xarray.DataArray):
        self.variable = XJVariable(da.variable)
        self.coords = _HashableCoords(da.coords)
        self.name = da.name

    def to_xarray(self) -> xarray.DataArray:
        var = self.variable.to_xarray()
        if var is None:
            return None
        return xarray.DataArray(var, name=self.name, coords=self.coords)


class XJDataset(eqx.Module):
    variables: dict[Hashable, XJVariable]
    coords: _HashableCoords = eqx.field(static=True)
    attrs: Optional[Mapping] = eqx.field(static=True)

    def __init__(self, ds: xarray.Dataset):
        self.variables = {
            name: XJVariable(da.variable) for name, da in ds.data_vars.items()
        }
        self.coords = _HashableCoords(ds.coords)
        self.attrs = ds.attrs

    def to_xarray(self) -> xarray.Dataset:
        data_vars = {name: var.to_xarray() for name, var in self.variables.items()}

        data_vars = {name: var for name, var in data_vars.items() if var is not None}

        return xarray.Dataset(
            data_vars,
            coords=self.coords,
            attrs=self.attrs,
        )
