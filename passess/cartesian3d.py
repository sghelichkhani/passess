"""Analytical solutions to the 3D gravitational Poisson equation in Cartesian coordinates.

Solves nabla^2 psi = -4 pi G rho in 3D Cartesian coordinates (x, y, z),
where (x, y) are the horizontal directions and z is the vertical (depth)
direction. The density is expanded in 2D horizontal Fourier modes with
wavevector k = (kx, ky).

For a single mode k with constant density coefficient rho_k on a depth
layer [z1, z2], the potential mode psi_k(z) is computed via Green's function
convolution. The vertical structure is identical to the 2D Cartesian case
with |k| replaced by k_h = sqrt(kx^2 + ky^2).
"""
import numpy as np


class PoissonCartesian3D:
    """Gravitational potential for a single 2D Fourier mode of a depth layer.

    Given a density field rho(x, y, z) = rho_k * exp(i(kx*x + ky*y)) that is
    nonzero only for z1 <= z' <= z2, computes the potential mode psi_k(z)
    satisfying the vertical ODE from 2D Fourier decomposition of the 3D
    Poisson equation.

    Parameters
    ----------
    kx, ky : float
        Horizontal wavevector components.
    rho_k : float or complex
        Constant Fourier coefficient of the density on the layer.
    z1, z2 : float
        Top and bottom depths of the density layer (z1 < z2).
    G_grav : float
        Gravitational constant.
    z_ref : float, optional
        Reference depth where psi_0 = 0 (gauge fixing). Only used for k_h = 0.
    """

    def __init__(self, kx, ky, rho_k, z1, z2, G_grav, z_ref=None):
        if z1 >= z2:
            raise ValueError(f"Require z1 < z2, got z1={z1}, z2={z2}")

        self.kx = kx
        self.ky = ky
        self.k_h = np.sqrt(kx**2 + ky**2)
        self.rho_k = rho_k
        self.z1 = z1
        self.z2 = z2
        self.G_grav = G_grav
        self.z_ref = z_ref

        if self.k_h == 0:
            self._gauge_offset = self._unit_kh0(z_ref) if z_ref is not None else 0.0

    def _unit_kh_neq0(self, z):
        """Geometric factor for k_h != 0: psi_k = G_grav * rho_k * _unit(z).

        Uses the numerically stable form where all exponential arguments
        are non-positive.
        """
        kh = self.k_h
        kh2 = kh * kh
        z1, z2 = self.z1, self.z2

        if z <= z1:
            val = np.exp(-kh * (z1 - z)) - np.exp(-kh * (z2 - z))
        elif z >= z2:
            val = np.exp(-kh * (z - z2)) - np.exp(-kh * (z - z1))
        else:
            val = 2.0 - np.exp(-kh * (z - z1)) - np.exp(-kh * (z2 - z))

        return 2.0 * np.pi * val / kh2

    def _unit_kh0(self, z):
        """Geometric factor for k_h = 0: psi_0 = G_grav * rho_k * (unit - gauge_offset).

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

        if self.k_h == 0:
            unit = np.array([self._unit_kh0(zi) for zi in z_arr])
            result = self.G_grav * self.rho_k * (unit - self._gauge_offset)
        else:
            unit = np.array([self._unit_kh_neq0(zi) for zi in z_arr])
            result = self.G_grav * self.rho_k * unit

        if scalar:
            return result.item()
        return result

    def to_spatial(self, x, y, z):
        """Evaluate the spatial potential psi_k(z) * exp(i(kx*x + ky*y)).

        Parameters
        ----------
        x, y : float or array_like
            Horizontal coordinates.
        z : float or array_like
            Depth coordinate(s).

        Returns
        -------
        complex or ndarray
            The complex spatial field contribution from this mode.
        """
        return self.psi_k(z) * np.exp(1j * (self.kx * x + self.ky * y))

    def rho_to_spatial(self, x, y, z):
        """Evaluate the spatial density rho_k * exp(i(kx*x + ky*y)).

        Returns rho_k for z1 <= z <= z2, zero otherwise.

        Parameters
        ----------
        x, y : float or array_like
            Horizontal coordinates.
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
        return rho * np.exp(1j * (self.kx * x + self.ky * y))
