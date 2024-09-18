import diffrax
import jax
import jax.numpy as jnp
import xarray as xr
import xarray_jax


def test_diffrax():
    da = xr.DataArray(jnp.arange(10), dims=["x"])

    @jax.jit
    def fn(t, y, args):
        da = y.to_xarray()
        y_dot = -1.0 * xr.apply_ufunc(jnp.square, da)
        return y_dot

    term = diffrax.ODETerm(fn)
    solver = diffrax.Dopri5()
    ts = jnp.linspace(0, 1, 100)
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=1.0, dt0=0.01, y0=da, saveat=diffrax.SaveAt(ts=ts)
    )
