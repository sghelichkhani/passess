# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 2D Cartesian: gravitational potential of a depth layer
#
# This example uses `passess` to evaluate the analytical solution of
# the 2D Cartesian gravitational Poisson equation for a single lateral
# Fourier mode of the density confined to a horizontal layer
# $z_1 \le z \le z_2$.
#
# The derivation is given in
# [`solution-dim2-cartesian.md`](https://github.com/sghelichkhani/passess/blob/main/solution-dim2-cartesian.md).
# Here we simply recall
#
# $$ \nabla^2 \psi(x,z) = -4\pi \gamma\, \rho(x,z), $$
#
# with the modal density used below:
#
# $$ \rho(x,z) = \rho_k\, e^{i k x}
#    \quad \text{for } z_1 \le z \le z_2, \quad 0 \text{ otherwise}. $$

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from passess.cartesian import PoissonCartesian2D

# %% [markdown]
# ## Problem setup
#
# Lengths are non-dimensionalised by the mantle thickness
# $D \approx 2890~\mathrm{km}$. The mantle occupies $z \in [0, 1]$,
# with $z = 0$ at the surface and $z = 1$ at the CMB.
#
# The density anomaly is a thin slab of physical thickness
# $100~\mathrm{km}$, i.e. $\Delta = 100/2890 \approx 0.0346$ in non-dim
# units, sitting at mid-mantle and carrying a single lateral Fourier
# mode with wavenumber $k = 2\pi$ (one wavelength per unit of $x$).

# %%
# Non-dimensionalisation: unit length = mantle thickness D = 2890 km.
km_nondim = 1.0 / 2890.0

# Mantle geometry.
z_surf, z_cmb = 0.0, 1.0

# Density anomaly: 100 km thick slab at mid-mantle.
z_mid = 0.5 * (z_surf + z_cmb)
delta = 100.0 * km_nondim
z1 = z_mid - 0.5 * delta
z2 = z_mid + 0.5 * delta

k = 2.0 * np.pi
rho_k = 1.0
gamma = 1.0

solver = PoissonCartesian2D(k=k, rho_k=rho_k, z1=z1, z2=z2, gamma=gamma)

# Extended domain: several wavelengths in x, and above/below the mantle in z.
x_extent = 2.0  # two wavelengths since k = 2 pi
z_min, z_max = z_surf - 0.4, z_cmb + 0.4

# Fine grid so the 100 km anomaly slab is resolved.
nx, nz = 401, 801
xs = np.linspace(-x_extent / 2.0, x_extent / 2.0, nx)
zs = np.linspace(z_min, z_max, nz)
X, Z = np.meshgrid(xs, zs)

# The solver iterates over a flat array of z values, so flatten and
# reshape back to the 2D grid.
shape = X.shape
rho = solver.rho_to_spatial(X.ravel(), Z.ravel()).real.reshape(shape)
psi = solver.to_spatial(X.ravel(), Z.ravel()).real.reshape(shape)

# %% [markdown]
# ## Visualise density and potential
#
# Left panel: the density — non-zero only inside the thin 100 km
# anomaly slab. Right panel: the potential on the full extended domain.
# In both panels the rectangle marks the mantle ($0 \le z \le 1$).

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

def _outline_mantle(ax):
    ax.add_patch(Rectangle((xs[0], z_surf), xs[-1] - xs[0], z_cmb - z_surf,
                            fill=False, edgecolor="k", linewidth=1.2))

ax = axes[0]
im0 = ax.pcolormesh(X, Z, rho, shading="auto", cmap="RdBu_r",
                    vmin=-abs(rho_k), vmax=abs(rho_k))
_outline_mantle(ax)
ax.invert_yaxis()
ax.set_title(r"Density $\rho(x,z)$ (real part)")
ax.set_xlabel("x")
ax.set_ylabel("z (depth)")
fig.colorbar(im0, ax=ax, shrink=0.85)

ax = axes[1]
vmax = np.max(np.abs(psi))
im1 = ax.pcolormesh(X, Z, psi, shading="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax)
_outline_mantle(ax)
ax.invert_yaxis()
ax.set_title(r"Potential $\psi(x,z)$ (real part)")
ax.set_xlabel("x")
ax.set_ylabel("z (depth)")
fig.colorbar(im1, ax=ax, shrink=0.85)

plt.show()

# %% [markdown]
# Outside the layer the potential decays as $e^{-|k||z - z_{\rm layer}|}$,
# which is the expected behaviour for a compactly supported source in a
# single lateral Fourier mode.
