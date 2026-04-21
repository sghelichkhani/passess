"""Analytical solutions to the 2D gravitational Poisson equation in Cartesian coordinates.

Solves nabla^2 psi = -4 pi gamma rho in 2D Cartesian coordinates (x, z),
where x is the lateral (infinite/periodic) direction and z is the depth
direction. The density is expanded in lateral Fourier modes.

For a single mode k with constant density coefficient rho_k on a depth
layer [z1, z2], the potential mode psi_k(z) is computed via Green's function
convolution, yielding closed-form expressions involving exponential integrals.
"""
import numpy as np


class PoissonCartesian2D:
    """Gravitational potential for a single lateral Fourier mode of a depth layer.

    Given a density field rho(x, z) = rho_k * exp(i k x) that is nonzero
    only for z1 <= z' <= z2, computes the potential mode psi_k(z) satisfying
    the vertical ODE that arises from Fourier decomposition of the 2D Poisson
    equation.

    Parameters
    ----------
    k : float
        Lateral wavenumber.
    rho_k : float or complex
        Constant Fourier coefficient of the density on the layer.
    z1, z2 : float
        Top and bottom depths of the density layer (z1 < z2).
    gamma : float
        Gravitational constant.
    z_ref : float, optional
        Reference depth where psi_0 = 0 (gauge fixing). Only used for k = 0.
    """

    def __init__(self, k, rho_k, z1, z2, gamma, z_ref=None):
        if z1 >= z2:
            raise ValueError(f"Require z1 < z2, got z1={z1}, z2={z2}")

        self.k = k
        self.abs_k = abs(k)
        self.rho_k = rho_k
        self.z1 = z1
        self.z2 = z2
        self.gamma = gamma
        self.z_ref = z_ref

        if k == 0:
            self._gauge_offset = self._unit_k0(z_ref) if z_ref is not None else 0.0

    def _unit_kneq0(self, z):
        """Geometric factor for k != 0: psi_k = gamma * rho_k * _unit_kneq0(z).

        Uses the numerically stable form where all exponential arguments
        are non-positive:
          z < z1:  (2pi/k^2)(e^{-|k|(z1-z)} - e^{-|k|(z2-z)})
          inside:  (2pi/k^2)(2 - e^{-|k|(z-z1)} - e^{-|k|(z2-z)})
          z > z2:  (2pi/k^2)(e^{-|k|(z-z2)} - e^{-|k|(z-z1)})
        """
        ak = self.abs_k
        ak2 = ak * ak
        z1, z2 = self.z1, self.z2

        if z <= z1:
            val = np.exp(-ak * (z1 - z)) - np.exp(-ak * (z2 - z))
        elif z >= z2:
            val = np.exp(-ak * (z - z2)) - np.exp(-ak * (z - z1))
        else:
            val = 2.0 - np.exp(-ak * (z - z1)) - np.exp(-ak * (z2 - z))

        return 2.0 * np.pi * val / ak2

    def _unit_k0(self, z):
        """Geometric factor for k = 0: psi_0 = gamma * rho_k * (_unit_k0(z) - gauge_offset).

        Evaluates -2pi * int_{z1}^{z2} |z - z'| dz'.
        """
        z1, z2 = self.z1, self.z2

        if z <= z1:
            I = (z2 - z1) * ((z2 + z1) / 2.0 - z)
        elif z >= z2:
            I = (z2 - z1) * (z - (z2 + z1) / 2.0)
        else:
            I = (z - z1)**2 / 2.0 + (z2 - z)**2 / 2.0

        return -2.0 * np.pi * I

    def psi_k(self, z):
        """Evaluate the potential mode at depth z.

        Parameters
        ----------
        z : float or array_like
            Depth coordinate(s).

        Returns
        -------
        float, complex, or ndarray
            The potential mode psi_k(z). Complex if rho_k is complex.
        """
        z_arr = np.asarray(z, dtype=float)
        scalar = z_arr.ndim == 0
        z_arr = np.atleast_1d(z_arr)

        if self.k == 0:
            unit = np.array([self._unit_k0(zi) for zi in z_arr])
            result = self.gamma * self.rho_k * (unit - self._gauge_offset)
        else:
            unit = np.array([self._unit_kneq0(zi) for zi in z_arr])
            result = self.gamma * self.rho_k * unit

        if scalar:
            return result.item()
        return result

    def to_spatial(self, x, z):
        """Evaluate the spatial potential psi_k(z) * exp(i k x).

        Parameters
        ----------
        x : float or array_like
            Lateral coordinate(s).
        z : float or array_like
            Depth coordinate(s).

        Returns
        -------
        complex or ndarray
            The complex spatial field contribution from this mode.
        """
        return self.psi_k(z) * np.exp(1j * self.k * x)

    def rho_to_spatial(self, x, z):
        """Evaluate the spatial density rho_k * exp(i k x).

        Returns rho_k for z1 <= z <= z2, zero otherwise.

        Parameters
        ----------
        x : float or array_like
            Lateral coordinate(s).
        z : float or array_like
            Depth coordinate(s).

        Returns
        -------
        complex or ndarray
            The complex spatial density contribution from this mode.
        """
        z_arr = np.asarray(z, dtype=float)
        mask = (z_arr >= self.z1) & (z_arr <= self.z2)
        rho = np.where(mask, self.rho_k, 0.0)
        return rho * np.exp(1j * self.k * x)
