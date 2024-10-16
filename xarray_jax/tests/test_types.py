import xarray as xr
from hypothesis import given, settings
import jax
from xarray_jax.tests.strategies import (
    generic_xr_strat,
    ufunc_strat,
    xp_variables,
    identity_transforms,
    xj_roundtrip,
    float_vars_and_das,
)
import equinox as eqx
from xarray_jax.register_pytrees import var_change_on_unflatten
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@given(var=xp_variables, ufunc=ufunc_strat)
@settings(deadline=None)
def test_variables(var: xr.Variable, ufunc):
    #
    # Test that we can create a boolean mask tree.
    #
    var_mask = jax.tree.map(lambda _: True, var)
    assert var_mask._data is True
    assert var_mask._dims == var._dims
    assert var_mask.attrs == var.attrs


@given(
    xr_data=generic_xr_strat,
    ufunc=ufunc_strat,
)
@settings(deadline=None)
def test_ufunc(xr_data, ufunc):
    """
    Test that we can apply a jitted ufunc to the da and get the correct result as if we applied it directly to the data.
    """

    @eqx.filter_jit
    def fn(data):
        return xr.apply_ufunc(ufunc, data)

    result = fn(xr_data)
    expected = xr.apply_ufunc(ufunc, xr_data)
    assert result.equals(expected)


@given(
    xr_data=generic_xr_strat,
)
@settings(deadline=None)
def test_boolean_mask(xr_data):
    """
    Simple test that we can construct a boolean mask tree with all trues and apply eqx.filter.
    """

    @eqx.filter_jit
    def fn(data):
        mask = jax.tree.map(lambda _: True, data)
        return eqx.filter(data, mask)

    result = fn(xr_data)
    assert xr_data.equals(result)


@given(
    xr_data=generic_xr_strat,
    transform=identity_transforms,
)
@settings(deadline=None)
def test_identity_transforms(xr_data, transform):
    """
    Test for identity JAX transformations.
    """
    # Run the transform without JIT.
    out = transform(xr_data)
    assert xr_data.equals(out)
    # Do another round trip to test that we can call the xr constructor on the output.
    reconstructed = xj_roundtrip(out)
    assert xr_data.equals(reconstructed)


@given(
    xr_data=float_vars_and_das,
)
@settings(deadline=None)
def test_grads(xr_data):
    # Test the gradient of sum(x**2), which is 2*x.
    @eqx.filter_jit
    def fn(data):
        return (data**2.0).sum().data  # Requires .data to get the value.

    grad = jax.grad(fn)(xr_data)
    expected = 2 * xr_data
    xr.testing.assert_allclose(grad, expected)

    val, grad = eqx.filter_value_and_grad(fn)(xr_data)
    assert val == (xr_data**2.0).sum().data
    xr.testing.assert_allclose(grad, expected)


@given(xr_data=float_vars_and_das)
@settings(deadline=None)
def test_dims_change(xr_data):
    def add_one_n_times(data, n):
        # Use lax.scan to add 1 n times
        initial_carry = data

        def add_one_scan(carry, _):
            carry_out = carry + 1.0

            return carry_out, carry_out

        _, history = jax.lax.scan(add_one_scan, initial_carry, None, length=n)
        return history

    def var_change_fn(var):
        if var._data.ndim == len(var._dims):
            return var
        elif var._data.ndim == len(var._dims) + 1:
            var._dims = ("time", *var._dims)
            return var
        else:
            raise ValueError("Invalid dims.")

    n_steps = 10
    with var_change_on_unflatten(var_change_fn):
        history = add_one_n_times(xr_data, n_steps)

    assert history.dims == ("time", *xr_data.dims)
    assert history.shape == (n_steps, *xr_data.shape)

    reference = xr_data
    for i in range(n_steps):
        reference = reference + 1
        assert history.isel(time=i).equals(reference)
