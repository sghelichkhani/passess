"""Analytical solutions to the 3D gravitational Poisson equation in spherical coordinates.

Solves nabla^2 psi = -4 pi G rho in 3D spherical coordinates (r, theta, phi),
where the density is expanded in spherical harmonics Y_l^m(theta, phi).

For a single mode (l, m) with constant density coefficient rho_lm on a radial
shell [r1, r2], the potential mode psi_lm(r) is computed via the radial Green's
function g_l(r, r') = (1/(2l+1)) r_<^l / r_>^{l+1}.
"""
import numpy as np
from scipy.special import sph_harm_y


class PoissonSpherical3D:
    """Gravitational potential for a single spherical harmonic mode of a radial shell.

    Given a density field rho(r, theta, phi) = rho_lm * Y_l^m(theta, phi) that
    is nonzero only for r1 <= r' <= r2, computes the potential mode psi_lm(r)
    satisfying the radial ODE from spherical harmonic decomposition of the 3D
    Poisson equation.

    Parameters
    ----------
    l : int
        Spherical harmonic degree (l >= 0).
    m : int
        Spherical harmonic order (|m| <= l).
    rho_lm : float or complex
        Constant spherical harmonic coefficient of the density on the shell.
    r1, r2 : float
        Inner and outer radii of the density shell (0 < r1 < r2).
    G_grav : float
        Gravitational constant.
    """

    def __init__(self, l, m, rho_lm, r1, r2, G_grav):
        if l < 0:
            raise ValueError(f"Require l >= 0, got l={l}")
        if abs(m) > l:
            raise ValueError(f"Require |m| <= l, got l={l}, m={m}")
        if r1 <= 0 or r2 <= 0:
            raise ValueError(f"Shell radii must be positive, got r1={r1}, r2={r2}")
        if r1 >= r2:
            raise ValueError(f"Require r1 < r2, got r1={r1}, r2={r2}")

        self.l = l
        self.m = m
        self.rho_lm = rho_lm
        self.r1 = r1
        self.r2 = r2
        self.G_grav = G_grav

    def _unit(self, r):
        """Geometric factor: psi_lm = G_grav * rho_lm * _unit(r).

        Evaluates (4*pi/(2l+1)) * [r^{-(l+1)} I_inner + r^l I_outer] where:
          I_inner = int_{r1}^{min(r,r2)} r'^{l+2} dr'
          I_outer = int_{max(r,r1)}^{r2} r'^{1-l} dr'
        """
        l = self.l
        r1, r2 = self.r1, self.r2
        prefactor = 4.0 * np.pi / (2.0 * l + 1.0)

        # Inner integral antiderivative: r'^{l+3}/(l+3)
        exp_inner = l + 3

        # Outer integral antiderivative:
        #   r'^{2-l}/(2-l)  when l != 2
        #   ln(r')           when l == 2
        log_outer = (l == 2)

        if r <= r1:
            inner = 0.0
            if log_outer:
                outer = np.log(r2) - np.log(r1)
            else:
                exp_out = 2 - l
                outer = (r2**exp_out - r1**exp_out) / exp_out
            return prefactor * r**l * outer

        elif r >= r2:
            inner = (r2**exp_inner - r1**exp_inner) / exp_inner
            return prefactor * r**(-(l + 1)) * inner

        else:
            inner = (r**exp_inner - r1**exp_inner) / exp_inner
            if log_outer:
                outer = np.log(r2) - np.log(r)
            else:
                exp_out = 2 - l
                outer = (r2**exp_out - r**exp_out) / exp_out
            return prefactor * (r**(-(l + 1)) * inner + r**l * outer)

    def psi_lm(self, r):
        """Evaluate the radial potential mode at r.

        Parameters
        ----------
        r : float or array_like
            Radial coordinate(s). Must be positive.

        Returns
        -------
        float, complex, or ndarray
            The potential mode psi_lm(r). Complex if rho_lm is complex.
        """
        r_arr = np.asarray(r, dtype=float)
        scalar = r_arr.ndim == 0
        r_arr = np.atleast_1d(r_arr)

        unit = np.array([self._unit(ri) for ri in r_arr])
        result = self.G_grav * self.rho_lm * unit

        if scalar:
            return result.item()
        return result

    def to_spatial(self, r, theta, phi):
        """Evaluate the spatial potential psi_lm(r) * Y_l^m(theta, phi).

        Parameters
        ----------
        r : float or array_like
            Radial coordinate(s).
        theta : float or array_like
            Colatitude(s) in [0, pi].
        phi : float or array_like
            Longitude(s) in [0, 2*pi].

        Returns
        -------
        complex or ndarray
            The complex spatial field contribution from this mode.
        """
        Y_lm = sph_harm_y(self.l, self.m, theta, phi)
        return self.psi_lm(r) * Y_lm

    def rho_to_spatial(self, r, theta, phi):
        """Evaluate the spatial density rho_lm * Y_l^m(theta, phi).

        Returns rho_lm * Y_l^m for r1 <= r <= r2, zero otherwise.

        Parameters
        ----------
        r : float or array_like
            Radial coordinate(s).
        theta : float or array_like
            Colatitude(s).
        phi : float or array_like
            Longitude(s).

        Returns
        -------
        complex or ndarray
            The complex spatial density contribution from this mode.
        """
        r_arr = np.asarray(r, dtype=float)
        mask = (r_arr >= self.r1) & (r_arr <= self.r2)
        Y_lm = sph_harm_y(self.l, self.m, theta, phi)
        rho = np.where(mask, self.rho_lm, 0.0)
        return rho * Y_lm
