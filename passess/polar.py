"""Analytical solutions to the 2D gravitational Poisson equation in polar coordinates.

Solves nabla^2 psi = -4 pi gamma rho for a single azimuthal Fourier mode m,
with constant density coefficient rho_m on a radial shell [r1, r2].

The potential mode psi_m(r) is computed via Green's function convolution,
yielding closed-form expressions involving power-law and logarithmic integrals.
"""
import numpy as np


class PoissonPolar2D:
    """Gravitational potential for a single azimuthal mode of a cylindrical shell.

    Given a density field rho(r, phi) = rho_m * exp(i m phi) that is nonzero
    only for r1 <= r' <= r2, computes the potential mode psi_m(r) satisfying
    the radial ODE that arises from Fourier decomposition of the 2D Poisson
    equation.

    Parameters
    ----------
    m : int
        Azimuthal wavenumber.
    rho_m : float or complex
        Constant Fourier coefficient of the density on the shell.
    r1, r2 : float
        Inner and outer radii of the density shell (r1 < r2, both > 0).
    gamma : float
        Gravitational constant.
    r_ref : float, optional
        Reference radius where psi_0 = 0 (gauge fixing). Only used for m = 0.
    """

    def __init__(self, m, rho_m, r1, r2, gamma, r_ref=None):
        if r1 <= 0 or r2 <= 0:
            raise ValueError(f"Shell radii must be positive, got r1={r1}, r2={r2}")
        if r1 >= r2:
            raise ValueError(f"Require r1 < r2, got r1={r1}, r2={r2}")

        self.m = m
        self.abs_m = abs(m)
        self.rho_m = rho_m
        self.r1 = r1
        self.r2 = r2
        self.gamma = gamma
        self.r_ref = r_ref

        if m == 0:
            self._gauge_offset = self._unit_m0(r_ref) if r_ref is not None else 0.0

    def _unit_mneq0(self, r):
        """Geometric factor for m != 0: psi_m = gamma * rho_m * _unit_mneq0(r).

        Evaluates (1/|m|) [r^{-|m|} I_inner + r^{|m|} I_outer] where:
          I_inner = int_{r1}^{min(r,r2)} r'^{|m|+1} dr'
          I_outer = int_{max(r,r1)}^{r2} r'^{1-|m|} dr'
        """
        am = self.abs_m
        r1, r2 = self.r1, self.r2

        # Antiderivative for the inner integral: int r'^{|m|+1} dr' = r'^{|m|+2}/(|m|+2)
        exp_inner = am + 2

        # Antiderivative for the outer integral: int r'^{1-|m|} dr'
        #   = r'^{2-|m|}/(2-|m|)  when |m| != 2
        #   = ln(r')              when |m| == 2
        log_outer = (am == 2)

        if r <= r1:
            inner = 0.0
            if log_outer:
                outer = np.log(r2) - np.log(r1)
            else:
                exp_out = 2 - am
                outer = (r2**exp_out - r1**exp_out) / exp_out
            return 2 * np.pi * (r**am * outer) / am

        elif r >= r2:
            inner = (r2**exp_inner - r1**exp_inner) / exp_inner
            return 2 * np.pi * (r**(-am) * inner) / am

        else:
            inner = (r**exp_inner - r1**exp_inner) / exp_inner
            if log_outer:
                outer = np.log(r2) - np.log(r)
            else:
                exp_out = 2 - am
                outer = (r2**exp_out - r**exp_out) / exp_out
            return 2 * np.pi * (r**(-am) * inner + r**am * outer) / am

    def _unit_m0(self, r):
        """Geometric factor for m = 0: psi_0 = gamma * rho_m * (_unit_m0(r) - gauge_offset).

        Evaluates -2 [ln(r) * int_{r1}^{min(r,r2)} r' dr'
                      + int_{max(r,r1)}^{r2} r' ln(r') dr']

        where int r' ln(r') dr' = r'^2/2 ln(r') - r'^2/4.
        """
        r1, r2 = self.r1, self.r2

        def F_rln(x):
            return x**2 / 2.0 * np.log(x) - x**2 / 4.0

        if r <= r1:
            term1 = 0.0
            term2 = F_rln(r2) - F_rln(r1)
        elif r >= r2:
            mass = (r2**2 - r1**2) / 2.0
            term1 = np.log(r) * mass
            term2 = 0.0
        else:
            mass_inner = (r**2 - r1**2) / 2.0
            term1 = np.log(r) * mass_inner
            term2 = F_rln(r2) - F_rln(r)

        return -4.0 * np.pi * (term1 + term2)

    def psi_m(self, r):
        """Evaluate the radial potential mode at r.

        Parameters
        ----------
        r : float or array_like
            Radial coordinate(s). Must be positive.

        Returns
        -------
        float, complex, or ndarray
            The potential mode psi_m(r). Complex if rho_m is complex.
        """
        r_arr = np.asarray(r, dtype=float)
        scalar = r_arr.ndim == 0
        r_arr = np.atleast_1d(r_arr)

        if self.m == 0:
            unit = np.array([self._unit_m0(ri) for ri in r_arr])
            result = self.gamma * self.rho_m * (unit - self._gauge_offset)
        else:
            unit = np.array([self._unit_mneq0(ri) for ri in r_arr])
            result = self.gamma * self.rho_m * unit

        if scalar:
            return result.item()
        return result

    def to_spatial(self, r, phi):
        """Evaluate the spatial potential psi_m(r) * exp(i m phi).

        Parameters
        ----------
        r : float or array_like
            Radial coordinate(s).
        phi : float or array_like
            Azimuthal angle(s).

        Returns
        -------
        complex or ndarray
            The complex spatial field contribution from this mode.
        """
        return self.psi_m(r) * np.exp(1j * self.m * phi)

    def rho_to_spatial(self, r, phi):
        """Evaluate the spatial density rho_m * exp(i m phi).

        Returns rho_m for r1 <= r <= r2, zero otherwise.

        Parameters
        ----------
        r : float or array_like
            Radial coordinate(s).
        phi : float or array_like
            Azimuthal angle(s).

        Returns
        -------
        complex or ndarray
            The complex spatial density contribution from this mode.
        """
        r_arr = np.asarray(r, dtype=float)
        mask = (r_arr >= self.r1) & (r_arr <= self.r2)
        rho = np.where(mask, self.rho_m, 0.0)
        return rho * np.exp(1j * self.m * phi)
