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
# # 2D polar: gravitational potential of a mass anomaly in a shell
#
# This example shows how to use `passess` to evaluate the analytical
# solution of the 2D gravitational Poisson equation in polar coordinates
# for a single azimuthal Fourier mode of the density restricted to a
# cylindrical shell $r_1 \le r \le r_2$.
#
# The full derivation lives in
# [`solution-dim2-polar.md`](https://github.com/sghelichkhani/passess/blob/main/solution-dim2-polar.md).
# Here we only recall the governing equation,
#
# $$ \nabla^2 \psi(r,\phi) = -4\pi \gamma\, \rho(r,\phi), $$
#
# and the modal density we will visualise:
#
# $$ \rho(r,\phi) = \rho_m\, e^{i m \phi}
#    \quad \text{for } r_1 \le r \le r_2, \quad 0 \text{ otherwise}. $$
#
# `passess.polar.PoissonPolar2D` returns the matching modal potential
# $\psi_m(r)$ evaluated on an arbitrary grid, including well outside the
# shell where the source vanishes but the potential still decays.

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from passess.polar import PoissonPolar2D

# %% [markdown]
# ## Problem setup
#
# Lengths are non-dimensionalised by the mantle thickness
# $D \approx 2890~\mathrm{km}$. The mantle occupies the annular shell
# $r \in [r_\mathrm{cmb}, r_\mathrm{surf}] = [1.22, 2.22]$ (CMB radius in
# non-dim units is $r_\mathrm{cmb} = R_\mathrm{cmb}/D \approx 1.2$).
#
# The density anomaly is a thin shell of physical thickness
# $100~\mathrm{km}$, i.e. $\Delta = 100/2890 \approx 0.0346$ in non-dim
# units, sitting at mid-mantle and carrying an $m = 4$ azimuthal pattern.

# %%
# Non-dimensionalisation: unit length = mantle thickness D = 2890 km.
km_nondim = 1.0 / 2890.0

# Mantle geometry.
r_cmb, r_surf = 1.22, 2.22

# Density anomaly: 100 km thick shell at mid-mantle.
r_mid = 0.5 * (r_cmb + r_surf)
delta = 100.0 * km_nondim
r1 = r_mid - 0.5 * delta
r2 = r_mid + 0.5 * delta

m = 4
rho_m = 1.0
gamma = 1.0

solver = PoissonPolar2D(m=m, rho_m=rho_m, r1=r1, r2=r2, gamma=gamma)

# Extended square domain: reach beyond the surface so far-field decay is visible.
extent = 1.2 * r_surf
# Use a fine grid so the 100 km thick anomaly shell is resolved.
n = 801
xs = np.linspace(-extent, extent, n)
ys = np.linspace(-extent, extent, n)
X, Y = np.meshgrid(xs, ys)
R = np.sqrt(X**2 + Y**2)
PHI = np.arctan2(Y, X)

# Avoid evaluating exactly at r = 0 (outside the domain of the solver).
R_eval = np.where(R == 0.0, 1e-12, R)

# The solver iterates over a flat array of radii, so we flatten for
# evaluation and reshape the result back to the 2D grid.
shape = R_eval.shape
rho = solver.rho_to_spatial(R_eval.ravel(), PHI.ravel()).real.reshape(shape)
psi = solver.to_spatial(R_eval.ravel(), PHI.ravel()).real.reshape(shape)

# %% [markdown]
# ## Visualise density and potential
#
# Left panel: the density — non-zero only in the thin 100 km anomaly
# shell. Right panel: the gravitational potential on the full extended
# domain. In both panels the two black circles mark the CMB and surface
# so the reader can place where the mantle actually lives.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

def _outline_mantle(ax):
    for radius in (r_cmb, r_surf):
        ax.add_patch(Circle((0, 0), radius, fill=False,
                             edgecolor="k", linewidth=1.2))

ax = axes[0]
im0 = ax.pcolormesh(X, Y, rho, shading="auto", cmap="RdBu_r",
                    vmin=-abs(rho_m), vmax=abs(rho_m))
_outline_mantle(ax)
ax.set_aspect("equal")
ax.set_title(r"Density $\rho(r,\phi)$ (real part)")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(im0, ax=ax, shrink=0.85)

ax = axes[1]
vmax = np.max(np.abs(psi))
im1 = ax.pcolormesh(X, Y, psi, shading="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax)
_outline_mantle(ax)
ax.set_aspect("equal")
ax.set_title(r"Potential $\psi(r,\phi)$ (real part)")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(im1, ax=ax, shrink=0.85)

plt.show()

# %% [markdown]
# The potential is smooth across the shell boundaries and decays as
# $r^{-|m|}$ in the exterior, as expected from the $m$-th multipole of
# a compactly supported source.
