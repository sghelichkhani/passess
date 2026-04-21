"""
Tests for the 3D gravitational potential solver in Cartesian coordinates.

These tests verify the mathematical correctness of the Green's function
convolution for solving:
    nabla^2 psi = -4 pi G rho
in 3D Cartesian coordinates (x, y, z), where the density is expanded in
2D horizontal Fourier modes with wavevector k = (kx, ky):
    rho(x,y,z) = (1/(2pi)^2) int rho_hat_k(z) e^{i(kx*x + ky*y)} dkx dky.

For a single mode k with CONSTANT density coefficient rho_hat_k on a
depth layer [z1, z2], the potential mode psi_hat_k(z) is computed
analytically via exponential integrals. The vertical structure is
IDENTICAL to the 2D Cartesian case with |k| replaced by
k_h = sqrt(kx^2 + ky^2).

For k_h != 0, the three regions give:
  z < z1:         psi_k = (2*pi*G*rho_k / k_h^2) * (e^{-k_h(z1-z)} - e^{-k_h(z2-z)})
  z1 <= z <= z2:  psi_k = (2*pi*G*rho_k / k_h^2) * (2 - e^{-k_h(z-z1)} - e^{-k_h(z2-z)})
  z > z2:         psi_k = (2*pi*G*rho_k / k_h^2) * (e^{-k_h(z-z2)} - e^{-k_h(z-z1)})

For k_h = 0:
  psi_0(z) = -2*pi*G*rho_0 * int_{z1}^{z2} |z - z'| dz' + gauge

The modal ODE is: psi_k'' - k_h^2 psi_k = -4*pi*G*rho_k (inside),
                                          = 0                (outside).

The 1D modal Green's function is g_k(z,z') = (1/(2*k_h)) e^{-k_h|z-z'|}
for k_h != 0.

Key difference from 2D Cartesian: the input is a 2D wavevector (kx, ky)
with k_h = sqrt(kx^2 + ky^2), and to_spatial multiplies by
e^{i(kx*x + ky*y)}.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import integrate

from passess.cartesian3d import PoissonCartesian3D


# ---------------------------------------------------------------------------
# A. Direct integral verification for k_h != 0 (two parameter sets)
# ---------------------------------------------------------------------------
# Parameter set 1: kx=3, ky=4 => k_h=5, rho_k=1, z1=0, z2=1, G_grav=1.
#
# k_h = sqrt(9 + 16) = sqrt(25) = 5
#
# Prefactor = 2*pi*G*rho_k / k_h^2 = 2*pi*1*1 / 25 = 2*pi/25
#
# Region 1: z = -1 < z1 = 0
#   psi_k = (2*pi/25) * (e^{-5*(0-(-1))} - e^{-5*(1-(-1))})
#         = (2*pi/25) * (e^{-5} - e^{-10})
#
#   Verify via integral form:
#     Prefactor_integral = 2*pi*G / k_h = 2*pi/5
#     inner integral (z' in [z1, min(z,z2)] = [0, -1] => empty since z < z1)
#     outer integral (z' in [max(z,z1), z2] = [0, 1]):
#       int_0^1 e^{-5*z'} dz' = [-e^{-5*z'}/5]_0^1 = (1 - e^{-5})/5
#     e^{k_h*z} * outer = e^{-5} * (1 - e^{-5})/5 = (e^{-5} - e^{-10})/5
#     psi_k = (2*pi/5) * (e^{-5} - e^{-10})/5 = (2*pi/25)*(e^{-5} - e^{-10}) [matches]
#
# Region 2: z = 0.5 (inside layer [0, 1])
#   psi_k = (2*pi/25) * (2 - e^{-5*(0.5-0)} - e^{-5*(1-0.5)})
#         = (2*pi/25) * (2 - e^{-2.5} - e^{-2.5})
#         = (2*pi/25) * (2 - 2*e^{-2.5})
#
#   Verify via integral form:
#     inner: int_0^{0.5} e^{5*z'} dz' = [e^{5*z'}/5]_0^{0.5} = (e^{2.5} - 1)/5
#     e^{-k_h*z} * inner = e^{-2.5} * (e^{2.5} - 1)/5 = (1 - e^{-2.5})/5
#     outer: int_{0.5}^{1} e^{-5*z'} dz' = [-e^{-5*z'}/5]_{0.5}^{1} = (e^{-2.5} - e^{-5})/5
#     e^{k_h*z} * outer = e^{2.5} * (e^{-2.5} - e^{-5})/5 = (1 - e^{-2.5})/5
#     psi_k = (2*pi/5) * 2*(1 - e^{-2.5})/5 = (2*pi/25)*(2 - 2*e^{-2.5}) [matches]
#
# Region 3: z = 2 > z2 = 1
#   psi_k = (2*pi/25) * (e^{-5*(2-1)} - e^{-5*(2-0)})
#         = (2*pi/25) * (e^{-5} - e^{-10})
#
#   Note symmetry: psi_k(-1) = psi_k(2), same as 2D case (equidistant
#   from nearest/farthest boundaries).

class TestDirectIntegralKh5:
    """Category A: hand-computed integrals for kx=3, ky=4, k_h=5,
    rho_k=1, z1=0, z2=1, G_grav=1."""

    @pytest.fixture
    def solver(self):
        return PoissonCartesian3D(kx=3.0, ky=4.0, rho_k=1.0, z1=0.0, z2=1.0, G_grav=1.0)

    def test_k_h_property(self, solver):
        assert_allclose(solver.k_h, 5.0, rtol=1e-14)

    def test_region1_below_layer(self, solver):
        # z = -1 < z1 = 0
        # psi_k = (2*pi/25) * (e^{-5} - e^{-10})
        z = -1.0
        expected = (2.0 * np.pi / 25.0) * (np.exp(-5.0) - np.exp(-10.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2_inside_layer(self, solver):
        # z = 0.5, inside [0, 1]
        # psi_k = (2*pi/25) * (2 - 2*e^{-2.5})
        z = 0.5
        expected = (2.0 * np.pi / 25.0) * (2.0 - 2.0 * np.exp(-2.5))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region3_above_layer(self, solver):
        # z = 2 > z2 = 1
        # psi_k = (2*pi/25) * (e^{-5} - e^{-10})
        z = 2.0
        expected = (2.0 * np.pi / 25.0) * (np.exp(-5.0) - np.exp(-10.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_at_z1_boundary(self, solver):
        # z = 0 = z1 (at left boundary, use inside formula)
        # psi_k = (2*pi/25) * (2 - e^{0} - e^{-5})
        #       = (2*pi/25) * (1 - e^{-5})
        z = 0.0
        expected = (2.0 * np.pi / 25.0) * (1.0 - np.exp(-5.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_at_z2_boundary(self, solver):
        # z = 1 = z2 (at right boundary, use inside formula)
        # psi_k = (2*pi/25) * (2 - e^{-5} - e^{0})
        #       = (2*pi/25) * (1 - e^{-5})
        z = 1.0
        expected = (2.0 * np.pi / 25.0) * (1.0 - np.exp(-5.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_array_input(self, solver):
        """Verify that psi_k works on numpy arrays."""
        z = np.array([-1.0, 0.5, 2.0])
        results = solver.psi_k(z)
        assert results.shape == (3,)
        coeff = 2.0 * np.pi / 25.0
        assert_allclose(results[0], coeff * (np.exp(-5.0) - np.exp(-10.0)), rtol=1e-12)
        assert_allclose(results[1], coeff * (2.0 - 2.0 * np.exp(-2.5)), rtol=1e-12)
        assert_allclose(results[2], coeff * (np.exp(-5.0) - np.exp(-10.0)), rtol=1e-12)


# ---------------------------------------------------------------------------
# A2. Direct integral verification with different parameters:
#     kx=1, ky=0 => k_h=1, rho_k=3, z1=1, z2=3, G_grav=0.5
# ---------------------------------------------------------------------------
# k_h = sqrt(1 + 0) = 1
# Prefactor = 2*pi*G*rho_k / k_h^2 = 2*pi*0.5*3 / 1 = 3*pi
#
# This gives k_h=1, which is identical to the 2D case with k=1.
#
# Region 1: z = 0 < z1 = 1
#   psi_k = 3*pi * (e^{-1*(1-0)} - e^{-1*(3-0)})
#         = 3*pi * (e^{-1} - e^{-3})
#
# Region 2: z = 2 (inside [1, 3], at midpoint)
#   psi_k = 3*pi * (2 - e^{-1*(2-1)} - e^{-1*(3-2)})
#         = 3*pi * (2 - e^{-1} - e^{-1})
#         = 3*pi * (2 - 2*e^{-1})
#
# Region 2b: z = 1.5 (inside [1, 3], asymmetric)
#   psi_k = 3*pi * (2 - e^{-1*(1.5-1)} - e^{-1*(3-1.5)})
#         = 3*pi * (2 - e^{-0.5} - e^{-1.5})
#
# Region 3: z = 5 > z2 = 3
#   psi_k = 3*pi * (e^{-1*(5-3)} - e^{-1*(5-1)})
#         = 3*pi * (e^{-2} - e^{-4})

class TestDirectIntegralKh1:
    """Category A2: kx=1, ky=0 (k_h=1), rho_k=3, z1=1, z2=3, G_grav=0.5."""

    @pytest.fixture
    def solver(self):
        return PoissonCartesian3D(kx=1.0, ky=0.0, rho_k=3.0, z1=1.0, z2=3.0, G_grav=0.5)

    def test_k_h_property(self, solver):
        assert_allclose(solver.k_h, 1.0, rtol=1e-14)

    def test_region1(self, solver):
        z = 0.0
        expected = 3.0 * np.pi * (np.exp(-1.0) - np.exp(-3.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2(self, solver):
        z = 2.0
        expected = 3.0 * np.pi * (2.0 - 2.0 * np.exp(-1.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2_asymmetric(self, solver):
        z = 1.5
        expected = 3.0 * np.pi * (2.0 - np.exp(-0.5) - np.exp(-1.5))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region3(self, solver):
        z = 5.0
        expected = 3.0 * np.pi * (np.exp(-2.0) - np.exp(-4.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# B. k_h = 0 mode (kx=0, ky=0)
# ---------------------------------------------------------------------------
# For k_h=0:
#   psi_0(z) = -2*pi*G*rho_0 * I(z) + gauge
# where I(z) = int_{z1}^{z2} |z - z'| dz'
#
# Parameters: rho_0=1, z1=0, z2=1, G_grav=1, z_ref=0 (gauge: psi_0(0)=0).
#
# Three regions for I(z):
#
# z < z1 = 0:
#   I(z) = (z2-z1)*[(z2+z1)/2 - z] = 1*(0.5 - z)
#   At z=-1: I = 0.5 + 1 = 1.5
#
# z1 <= z <= z2:
#   I(z) = (z-z1)^2/2 + (z2-z)^2/2 = z^2/2 + (1-z)^2/2
#   At z=0: I = 0 + 0.5 = 0.5
#   At z=0.5: I = 0.125 + 0.125 = 0.25
#   At z=1: I = 0.5 + 0 = 0.5
#
# z > z2 = 1:
#   I(z) = (z2-z1)*(z - (z2+z1)/2) = 1*(z - 0.5)
#   At z=2: I = 1.5
#
# With G_grav=1, rho_0=1: psi_0(z) = -2*pi * I(z) + gauge
# Gauge: psi_0(z_ref=0) = 0 => -2*pi * I(0) + gauge = 0
#   I(0) = 0.5 => gauge = pi
#
# psi_0(z) = -2*pi * I(z) + pi
#
# z = -1: psi_0 = -2*pi*1.5 + pi = -3*pi + pi = -2*pi
# z = 0.5: psi_0 = -2*pi*0.25 + pi = -0.5*pi + pi = 0.5*pi
# z = 2: psi_0 = -2*pi*1.5 + pi = -2*pi
# z = 0: psi_0 = 0 (by gauge)
# z = 1: psi_0 = -2*pi*0.5 + pi = 0

class TestKh0Mode:
    """Category B: k_h=0 mode (kx=0, ky=0) with |z-z'| integral."""

    @pytest.fixture
    def solver(self):
        return PoissonCartesian3D(kx=0.0, ky=0.0, rho_k=1.0, z1=0.0, z2=1.0,
                                  G_grav=1.0, z_ref=0.0)

    def test_k_h_property(self, solver):
        assert_allclose(solver.k_h, 0.0, atol=1e-16)

    def test_region1_below(self, solver):
        # z=-1: I=1.5, psi = -2*pi*1.5 + pi = -2*pi
        z = -1.0
        expected = -2.0 * np.pi
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2_inside(self, solver):
        # z=0.5: I=0.25, psi = -0.5*pi + pi = pi/2
        z = 0.5
        expected = 0.5 * np.pi
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region3_above(self, solver):
        # z=2: I=1.5, psi = -2*pi
        z = 2.0
        expected = -2.0 * np.pi
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_at_z1(self, solver):
        # z=0 = z_ref: psi_0(0) = 0 by gauge
        z = 0.0
        result = solver.psi_k(z)
        assert_allclose(result, 0.0, atol=1e-14)

    def test_at_z2(self, solver):
        # z=1: I=0.5, psi = -pi + pi = 0
        z = 1.0
        expected = 0.0
        result = solver.psi_k(z)
        assert_allclose(result, expected, atol=1e-14)

    def test_gauge_reference(self, solver):
        # psi_0(z_ref=0) = 0 by construction
        result = solver.psi_k(0.0)
        assert_allclose(result, 0.0, atol=1e-14)

    def test_different_gauge(self):
        # Two gauges differ only by a constant
        s1 = PoissonCartesian3D(kx=0.0, ky=0.0, rho_k=1.0, z1=0.0, z2=1.0,
                                G_grav=1.0, z_ref=0.0)
        s2 = PoissonCartesian3D(kx=0.0, ky=0.0, rho_k=1.0, z1=0.0, z2=1.0,
                                G_grav=1.0, z_ref=2.0)
        z = np.linspace(-3.0, 5.0, 100)
        diff = s1.psi_k(z) - s2.psi_k(z)
        assert_allclose(diff, diff[0], rtol=1e-12)


# ---------------------------------------------------------------------------
# B2. k_h=0 with different parameters: rho_0=2, z1=1, z2=3, G_grav=0.5, z_ref=2
# ---------------------------------------------------------------------------
# Prefactor: -2*pi*G*rho_0 = -2*pi*0.5*2 = -2*pi
# Thickness: z2-z1 = 2, midpoint: (z1+z2)/2 = 2
#
# z < z1=1:
#   I(z) = (z2-z1)*[(z2+z1)/2 - z] = 2*(2 - z)
#   At z=0: I = 2*2 = 4
#
# z1 <= z <= z2:
#   I(z) = (z-1)^2/2 + (3-z)^2/2
#   At z=2: I = 1/2 + 1/2 = 1
#
# z > z2=3:
#   I(z) = 2*(z - 2)
#   At z=4: I = 2*2 = 4
#
# Gauge: psi_0(z_ref=2) = 0 => -2*pi*I(2) + gauge = 0 => gauge = 2*pi
#
# psi_0(0) = -2*pi*4 + 2*pi = -6*pi
# psi_0(2) = 0  (by gauge)
# psi_0(1) = -2*pi*2 + 2*pi = -2*pi
# psi_0(3) = -2*pi*2 + 2*pi = -2*pi
# psi_0(4) = -2*pi*4 + 2*pi = -6*pi

class TestKh0Mode2:
    """Category B2: k_h=0 with different parameters."""

    @pytest.fixture
    def solver(self):
        return PoissonCartesian3D(kx=0.0, ky=0.0, rho_k=2.0, z1=1.0, z2=3.0,
                                  G_grav=0.5, z_ref=2.0)

    def test_below(self, solver):
        result = solver.psi_k(0.0)
        assert_allclose(result, -6.0 * np.pi, rtol=1e-12)

    def test_at_midpoint(self, solver):
        result = solver.psi_k(2.0)
        assert_allclose(result, 0.0, atol=1e-14)

    def test_at_z1(self, solver):
        result = solver.psi_k(1.0)
        assert_allclose(result, -2.0 * np.pi, rtol=1e-12)

    def test_at_z2(self, solver):
        result = solver.psi_k(3.0)
        assert_allclose(result, -2.0 * np.pi, rtol=1e-12)

    def test_above(self, solver):
        result = solver.psi_k(4.0)
        assert_allclose(result, -6.0 * np.pi, rtol=1e-12)

    def test_symmetry_about_midpoint(self, solver):
        # The layer [1,3] is symmetric about z=2, so I(2-d) = I(2+d)
        d_vals = np.array([0.3, 0.7, 1.0, 1.5, 3.0])
        left = solver.psi_k(2.0 - d_vals)
        right = solver.psi_k(2.0 + d_vals)
        assert_allclose(left, right, rtol=1e-12)


# ---------------------------------------------------------------------------
# C. Delta function limit (thin layer)
# ---------------------------------------------------------------------------
# For thin layer z2 = zp + dz/2, z1 = zp - dz/2 with dz -> 0:
#   psi_k(z) -> 4*pi*G * g_k(z, zp) * rho_k * dz
# where g_k(z,zp) = (1/(2*k_h)) e^{-k_h|z-zp|} for k_h != 0.
#
# So expected = 4*pi*G * (1/(2*k_h)) * e^{-k_h|z-zp|} * rho_k * dz
#             = (2*pi*G*rho_k*dz / k_h) * e^{-k_h|z-zp|}

class TestDeltaFunctionLimit:
    """Category C: thin layer approaches point-source Green's function."""

    @pytest.mark.parametrize("kx,ky", [(3.0, 4.0), (1.0, 0.0), (2.0, 2.0), (0.0, 5.0)])
    def test_thin_layer_kh_neq0(self, kx, ky):
        k_h = np.sqrt(kx**2 + ky**2)
        zp = 2.0
        dz = 1e-6
        rho_k = 1.0
        G_grav = 1.0
        z1 = zp - dz / 2.0
        z2 = zp + dz / 2.0
        solver = PoissonCartesian3D(kx=kx, ky=ky, rho_k=rho_k, z1=z1, z2=z2, G_grav=G_grav)

        for z in [-1.0, 0.0, 3.0, 5.0]:
            g_k = (1.0 / (2.0 * k_h)) * np.exp(-k_h * abs(z - zp))
            expected = 4.0 * np.pi * G_grav * g_k * rho_k * dz
            result = solver.psi_k(z)
            assert_allclose(result, expected, rtol=1e-4,
                            err_msg=f"Failed for kx={kx}, ky={ky}, z={z}")

    def test_thin_layer_kh0(self):
        # For k_h=0: g_0(z,zp) = -(1/2)|z-zp| + C
        # psi_0(z) = 4*pi*G * [-(1/2)|z-zp|] * rho_0 * dz + gauge
        #          = -2*pi*G*rho_0*|z-zp|*dz + gauge
        # Test differences to eliminate gauge.
        zp = 2.0
        dz = 1e-6
        rho_k = 1.0
        G_grav = 1.0
        z1 = zp - dz / 2.0
        z2 = zp + dz / 2.0
        solver = PoissonCartesian3D(kx=0.0, ky=0.0, rho_k=rho_k, z1=z1, z2=z2,
                                    G_grav=G_grav, z_ref=10.0)

        z_test = [0.0, 1.0, 3.0, 5.0]
        for z in z_test:
            expected = -2.0 * np.pi * G_grav * rho_k * dz * (
                abs(z - zp) - abs(10.0 - zp)
            )
            result = solver.psi_k(z)
            assert_allclose(result, expected, rtol=1e-3, atol=1e-15,
                            err_msg=f"Failed for k_h=0, z={z}")


# ---------------------------------------------------------------------------
# D. Laplacian verification (finite differences)
# ---------------------------------------------------------------------------
# Outside the layer: psi_k'' - k_h^2 psi_k = 0
# Inside the layer:  psi_k'' - k_h^2 psi_k = -4*pi*G*rho_k
#
# The ODE comes from substituting psi(x,y,z) = psi_k(z) e^{i(kx*x+ky*y)}
# into nabla^2 psi = -4*pi*G*rho: the x,y derivatives give -k_h^2 psi_k,
# the z-derivative gives psi_k''.

class TestLaplacianOutsideLayer:
    """Category D: Laplacian is zero outside the source layer."""

    @pytest.mark.parametrize("kx,ky", [(3.0, 4.0), (1.0, 0.0), (0.0, 2.0)])
    def test_homogeneous_outside(self, kx, ky):
        k_h = np.sqrt(kx**2 + ky**2)
        solver = PoissonCartesian3D(kx=kx, ky=ky, rho_k=1.0, z1=2.0, z2=4.0, G_grav=1.0)
        h = 1e-5

        # Points outside the layer [2, 4]
        test_points = [-2.0, 0.0, 1.0, 5.0, 7.0, 10.0]
        for z in test_points:
            psi_m = solver.psi_k(z - h)
            psi_c = solver.psi_k(z)
            psi_p = solver.psi_k(z + h)

            d2psi = (psi_p - 2.0 * psi_c + psi_m) / h**2
            # Modal equation: psi_k'' - k_h^2 * psi_k = 0
            residual = d2psi - k_h**2 * psi_c
            assert abs(residual) < 1e-4, \
                f"Laplacian not zero at z={z}, kx={kx}, ky={ky}: residual={residual}"

    def test_homogeneous_kh0_outside(self):
        # For k_h=0 outside the layer, psi_0'' = 0 (linear in z).
        solver = PoissonCartesian3D(kx=0.0, ky=0.0, rho_k=1.0, z1=2.0, z2=4.0,
                                    G_grav=1.0, z_ref=0.0)
        h = 1e-5

        for z in [-1.0, 0.0, 1.0, 5.0, 7.0]:
            psi_m = solver.psi_k(z - h)
            psi_c = solver.psi_k(z)
            psi_p = solver.psi_k(z + h)
            d2psi = (psi_p - 2.0 * psi_c + psi_m) / h**2
            assert abs(d2psi) < 1e-4, \
                f"Second derivative not zero at z={z}: d2psi={d2psi}"


class TestLaplacianInsideLayer:
    """Category D: Laplacian equals source inside the layer."""

    @pytest.mark.parametrize("kx,ky", [(3.0, 4.0), (1.0, 0.0), (0.0, 2.0)])
    def test_source_inside(self, kx, ky):
        k_h = np.sqrt(kx**2 + ky**2)
        rho_k = 1.0
        G_grav = 1.0
        z1, z2 = 1.0, 3.0
        solver = PoissonCartesian3D(kx=kx, ky=ky, rho_k=rho_k, z1=z1, z2=z2, G_grav=G_grav)

        h = 1e-5
        for z in [1.5, 2.0, 2.5]:
            psi_m = solver.psi_k(z - h)
            psi_c = solver.psi_k(z)
            psi_p = solver.psi_k(z + h)

            d2psi = (psi_p - 2.0 * psi_c + psi_m) / h**2
            # psi_k'' - k_h^2 * psi_k = -4*pi*G*rho_k
            residual = d2psi - k_h**2 * psi_c
            expected_source = -4.0 * np.pi * G_grav * rho_k
            assert_allclose(residual, expected_source, rtol=1e-3, atol=1e-6,
                            err_msg=f"Source mismatch at z={z}, kx={kx}, ky={ky}")

    def test_source_inside_kh0(self):
        # For k_h=0: psi_0'' = -4*pi*G*rho_0
        rho_k = 2.0
        G_grav = 0.5
        z1, z2 = 1.0, 3.0
        solver = PoissonCartesian3D(kx=0.0, ky=0.0, rho_k=rho_k, z1=z1, z2=z2,
                                    G_grav=G_grav, z_ref=0.0)

        h = 1e-5
        for z in [1.5, 2.0, 2.5]:
            psi_m = solver.psi_k(z - h)
            psi_c = solver.psi_k(z)
            psi_p = solver.psi_k(z + h)

            d2psi = (psi_p - 2.0 * psi_c + psi_m) / h**2
            expected_source = -4.0 * np.pi * G_grav * rho_k  # = -4*pi
            assert_allclose(d2psi, expected_source, rtol=1e-3, atol=1e-6,
                            err_msg=f"Source mismatch at z={z}")


# ---------------------------------------------------------------------------
# E. Continuity at layer boundaries
# ---------------------------------------------------------------------------
# psi_k(z) and psi_k'(z) are both continuous at z1 and z2.

class TestContinuityAtBoundaries:
    """Category E: psi_k and its derivative are continuous at z1 and z2."""

    @pytest.mark.parametrize("kx,ky", [(0.0, 0.0), (3.0, 4.0), (1.0, 0.0), (0.0, 3.0)])
    def test_continuity_psi(self, kx, ky):
        kwargs = dict(kx=kx, ky=ky, rho_k=1.0, z1=1.0, z2=3.0, G_grav=1.0)
        if kx == 0.0 and ky == 0.0:
            kwargs['z_ref'] = 5.0
        solver = PoissonCartesian3D(**kwargs)

        eps = 1e-10
        for z_bnd in [1.0, 3.0]:
            left = solver.psi_k(z_bnd - eps)
            right = solver.psi_k(z_bnd + eps)
            assert_allclose(left, right, rtol=1e-6, atol=1e-12,
                            err_msg=f"Discontinuity at z={z_bnd}, kx={kx}, ky={ky}")

    @pytest.mark.parametrize("kx,ky", [(0.0, 0.0), (3.0, 4.0), (1.0, 0.0), (0.0, 3.0)])
    def test_continuity_derivative(self, kx, ky):
        kwargs = dict(kx=kx, ky=ky, rho_k=1.0, z1=1.0, z2=3.0, G_grav=1.0)
        if kx == 0.0 and ky == 0.0:
            kwargs['z_ref'] = 5.0
        solver = PoissonCartesian3D(**kwargs)

        h = 1e-5
        for z_bnd in [1.0, 3.0]:
            eps = 1e-8
            dpsi_left = (solver.psi_k(z_bnd - eps + h) - solver.psi_k(z_bnd - eps - h)) / (2 * h)
            dpsi_right = (solver.psi_k(z_bnd + eps + h) - solver.psi_k(z_bnd + eps - h)) / (2 * h)
            assert_allclose(dpsi_left, dpsi_right, rtol=1e-2, atol=1e-6,
                            err_msg=f"Derivative discontinuity at z={z_bnd}, kx={kx}, ky={ky}")


# ---------------------------------------------------------------------------
# F. Far-field behavior, symmetry, and linearity
# ---------------------------------------------------------------------------

class TestFarFieldBehavior:
    """Category F: asymptotic, symmetry, and linearity checks."""

    def test_exponential_decay_above(self):
        """For k_h != 0 and z >> z2, psi_k ~ C * e^{-k_h*z}."""
        kx, ky = 3.0, 4.0  # k_h = 5
        k_h = 5.0
        solver = PoissonCartesian3D(kx=kx, ky=ky, rho_k=1.0, z1=0.0, z2=1.0, G_grav=1.0)

        z1_far, z2_far = 50.0, 51.0
        psi1 = solver.psi_k(z1_far)
        psi2 = solver.psi_k(z2_far)

        # For z >> z2, dominant term is e^{-k_h*(z-z2)}, so
        # psi(z+1)/psi(z) ~ e^{-k_h}
        ratio = psi2 / psi1
        expected_ratio = np.exp(-k_h)
        assert_allclose(ratio, expected_ratio, rtol=1e-8)

    def test_exponential_decay_below(self):
        """For k_h != 0 and z << z1, psi_k ~ C * e^{k_h*z}."""
        kx, ky = 3.0, 4.0  # k_h = 5
        k_h = 5.0
        solver = PoissonCartesian3D(kx=kx, ky=ky, rho_k=1.0, z1=0.0, z2=1.0, G_grav=1.0)

        z1_far, z2_far = -51.0, -50.0
        psi1 = solver.psi_k(z1_far)
        psi2 = solver.psi_k(z2_far)

        # Below layer: dominant term is e^{-k_h*(z1-z)} = e^{k_h*z} * e^{-k_h*z1}
        # psi(z+1)/psi(z) ~ e^{k_h}
        ratio = psi2 / psi1
        expected_ratio = np.exp(k_h)
        assert_allclose(ratio, expected_ratio, rtol=1e-8)

    def test_linearity_in_rho(self):
        """Scaling rho_k scales psi_k proportionally."""
        s1 = PoissonCartesian3D(kx=3.0, ky=4.0, rho_k=1.0, z1=0.0, z2=1.0, G_grav=1.0)
        s2 = PoissonCartesian3D(kx=3.0, ky=4.0, rho_k=3.5, z1=0.0, z2=1.0, G_grav=1.0)
        z = np.linspace(-3.0, 5.0, 50)
        assert_allclose(s2.psi_k(z), 3.5 * s1.psi_k(z), rtol=1e-14)

    def test_linearity_in_G(self):
        """Scaling G_grav scales psi_k proportionally."""
        s1 = PoissonCartesian3D(kx=3.0, ky=4.0, rho_k=1.0, z1=0.0, z2=1.0, G_grav=1.0)
        s2 = PoissonCartesian3D(kx=3.0, ky=4.0, rho_k=1.0, z1=0.0, z2=1.0, G_grav=2.0)
        z = np.linspace(-3.0, 5.0, 50)
        assert_allclose(s2.psi_k(z), 2.0 * s1.psi_k(z), rtol=1e-14)

    def test_to_spatial(self):
        """to_spatial(x, y, z) = psi_k(z) * exp(i*(kx*x + ky*y))."""
        kx, ky = 3.0, 4.0
        solver = PoissonCartesian3D(kx=kx, ky=ky, rho_k=1.0, z1=0.0, z2=1.0, G_grav=1.0)
        z = 1.5
        x = np.pi / 4.0
        y = np.pi / 6.0
        spatial = solver.to_spatial(x, y, z)
        expected = solver.psi_k(z) * np.exp(1j * (kx * x + ky * y))
        assert_allclose(spatial, expected, rtol=1e-14)

    def test_to_spatial_real_part(self):
        """For real rho_k, psi_k(z) is real, so Re[to_spatial] = psi_k*cos(kx*x+ky*y)."""
        kx, ky = 2.0, 1.0
        rho_k = 3.0
        solver = PoissonCartesian3D(kx=kx, ky=ky, rho_k=rho_k, z1=0.0, z2=1.0, G_grav=1.0)
        z = 0.5
        x = np.pi / 3.0
        y = np.pi / 5.0
        spatial = solver.to_spatial(x, y, z)
        phase = kx * x + ky * y
        expected_real = solver.psi_k(z) * np.cos(phase)
        assert_allclose(np.real(spatial), expected_real, rtol=1e-14)

    def test_kh0_far_field_linear(self):
        """For k_h=0, z >> z2: psi_0 grows linearly with |z|."""
        # psi_0(z) for z > z2: -2*pi*G*rho_0 * (z2-z1)*(z - (z1+z2)/2) + gauge
        # Slope: -2*pi*G*rho_0*(z2-z1)
        solver = PoissonCartesian3D(kx=0.0, ky=0.0, rho_k=1.0, z1=1.0, z2=3.0,
                                    G_grav=1.0, z_ref=0.0)
        z_far = np.array([10.0, 20.0, 50.0])
        psi_vals = solver.psi_k(z_far)

        # Expected slope: -2*pi*1*1*2 = -4*pi
        slope = (psi_vals[-1] - psi_vals[0]) / (z_far[-1] - z_far[0])
        assert_allclose(slope, -4.0 * np.pi, rtol=1e-8)


# ---------------------------------------------------------------------------
# G. Comparison with direct numerical integration of 1D Green's function
# ---------------------------------------------------------------------------
# Compute psi_k(z) by numerically integrating:
#   psi_k(z) = (2*pi*G/k_h) * int_{z1}^{z2} e^{-k_h|z-z'|} * rho_k dz'
# using scipy.integrate.quad. Completely independent of piecewise forms.

class TestNumericalIntegration1D:
    """Category G: compare closed-form with 1D numerical quadrature."""

    def _numerical_psi_kh(self, z, k_h, rho_k, z1, z2, G_grav):
        """Compute psi_k(z) via numerical integration of the 1D Green's function."""
        def integrand(zp):
            return np.exp(-k_h * abs(z - zp)) * rho_k

        result, _ = integrate.quad(integrand, z1, z2, epsabs=1e-12, epsrel=1e-12)
        return (2.0 * np.pi * G_grav / k_h) * result

    def _numerical_psi_0(self, z, rho_k, z1, z2, G_grav, z_ref):
        """Compute psi_0(z) via numerical integration of |z-z'|."""
        def integrand(zp):
            return abs(z - zp) * rho_k

        def integrand_ref(zp):
            return abs(z_ref - zp) * rho_k

        val, _ = integrate.quad(integrand, z1, z2, epsabs=1e-12, epsrel=1e-12)
        ref, _ = integrate.quad(integrand_ref, z1, z2, epsabs=1e-12, epsrel=1e-12)
        return -2.0 * np.pi * G_grav * (val - ref)

    @pytest.mark.parametrize("kx,ky", [(3.0, 4.0), (1.0, 0.0), (0.0, 2.0), (5.0, 0.0)])
    def test_kh_neq0_vs_numerical(self, kx, ky):
        k_h = np.sqrt(kx**2 + ky**2)
        rho_k = 2.0
        z1, z2, G_grav = 1.0, 3.0, 0.5
        solver = PoissonCartesian3D(kx=kx, ky=ky, rho_k=rho_k, z1=z1, z2=z2, G_grav=G_grav)

        test_z = [-2.0, 0.0, 1.5, 2.0, 2.8, 4.0, 6.0]
        for z in test_z:
            numerical = self._numerical_psi_kh(z, k_h, rho_k, z1, z2, G_grav)
            closed = solver.psi_k(z)
            assert_allclose(closed, numerical, rtol=1e-10,
                            err_msg=f"Mismatch at z={z}, kx={kx}, ky={ky}")

    def test_kh0_vs_numerical(self):
        rho_k = 2.0
        z1, z2, G_grav = 1.0, 3.0, 0.5
        z_ref = 5.0
        solver = PoissonCartesian3D(kx=0.0, ky=0.0, rho_k=rho_k, z1=z1, z2=z2,
                                    G_grav=G_grav, z_ref=z_ref)

        test_z = [-2.0, 0.0, 1.5, 2.0, 2.8, 4.0, 6.0]
        for z in test_z:
            numerical = self._numerical_psi_0(z, rho_k, z1, z2, G_grav, z_ref)
            closed = solver.psi_k(z)
            assert_allclose(closed, numerical, rtol=1e-10,
                            err_msg=f"Mismatch at z={z}, k_h=0")


# ---------------------------------------------------------------------------
# H. Input validation and edge cases
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Category H: input validation."""

    def test_z1_equals_z2(self):
        with pytest.raises(ValueError):
            PoissonCartesian3D(kx=1.0, ky=0.0, rho_k=1.0, z1=2.0, z2=2.0, G_grav=1.0)

    def test_z1_greater_than_z2(self):
        with pytest.raises(ValueError):
            PoissonCartesian3D(kx=1.0, ky=0.0, rho_k=1.0, z1=3.0, z2=2.0, G_grav=1.0)


class TestComplexDensity:
    """Category H: complex density coefficients."""

    def test_complex_rho_k(self):
        rho_k = 1.0 + 2.0j
        s_real = PoissonCartesian3D(kx=3.0, ky=4.0, rho_k=1.0, z1=0.0, z2=1.0, G_grav=1.0)
        s_complex = PoissonCartesian3D(kx=3.0, ky=4.0, rho_k=rho_k, z1=0.0, z2=1.0, G_grav=1.0)

        z = np.linspace(-3.0, 5.0, 20)
        psi_unit = s_real.psi_k(z)
        expected = rho_k * psi_unit
        result = s_complex.psi_k(z)
        assert_allclose(result, expected, rtol=1e-14)

    def test_pure_imaginary_rho(self):
        rho_k = 2.0j
        s_real = PoissonCartesian3D(kx=3.0, ky=4.0, rho_k=1.0, z1=1.0, z2=3.0, G_grav=1.0)
        s_imag = PoissonCartesian3D(kx=3.0, ky=4.0, rho_k=rho_k, z1=1.0, z2=3.0, G_grav=1.0)

        z = np.linspace(-2.0, 6.0, 20)
        expected = rho_k * s_real.psi_k(z)
        result = s_imag.psi_k(z)
        assert_allclose(result, expected, rtol=1e-14)


class TestSuperposition:
    """Category H: superposition of two non-overlapping layers."""

    def test_two_layers(self):
        # Two non-overlapping layers: [0,1] and [3,4]
        # At z=2 (between the layers), both contribute as "outside" terms.
        kx, ky = 3.0, 4.0  # k_h = 5
        k_h = 5.0
        G_grav = 1.0
        rho_k = 1.0
        coeff = 2.0 * np.pi * G_grav * rho_k / k_h**2  # = 2*pi/25

        s1 = PoissonCartesian3D(kx=kx, ky=ky, rho_k=rho_k, z1=0.0, z2=1.0, G_grav=G_grav)
        s2 = PoissonCartesian3D(kx=kx, ky=ky, rho_k=rho_k, z1=3.0, z2=4.0, G_grav=G_grav)

        # z = -1 (below both layers):
        # s1: coeff * (e^{-5*(0-(-1))} - e^{-5*(1-(-1))}) = coeff*(e^{-5} - e^{-10})
        # s2: coeff * (e^{-5*(3-(-1))} - e^{-5*(4-(-1))}) = coeff*(e^{-20} - e^{-25})
        z = -1.0
        total = s1.psi_k(z) + s2.psi_k(z)
        expected = coeff * ((np.exp(-5.0) - np.exp(-10.0))
                            + (np.exp(-20.0) - np.exp(-25.0)))
        assert_allclose(total, expected, rtol=1e-12)

        # z = 2 (between layers):
        # s1 above: coeff * (e^{-5*(2-1)} - e^{-5*(2-0)}) = coeff*(e^{-5} - e^{-10})
        # s2 below: coeff * (e^{-5*(3-2)} - e^{-5*(4-2)}) = coeff*(e^{-5} - e^{-10})
        z = 2.0
        total = s1.psi_k(z) + s2.psi_k(z)
        expected = coeff * 2.0 * (np.exp(-5.0) - np.exp(-10.0))
        assert_allclose(total, expected, rtol=1e-12)

        # z = 5 (above both layers):
        # s1: coeff * (e^{-5*(5-1)} - e^{-5*(5-0)}) = coeff*(e^{-20} - e^{-25})
        # s2: coeff * (e^{-5*(5-4)} - e^{-5*(5-3)}) = coeff*(e^{-5} - e^{-10})
        z = 5.0
        total = s1.psi_k(z) + s2.psi_k(z)
        expected = coeff * ((np.exp(-20.0) - np.exp(-25.0))
                            + (np.exp(-5.0) - np.exp(-10.0)))
        assert_allclose(total, expected, rtol=1e-12)

    def test_two_layers_kh0_superposition(self):
        # For k_h=0, superposition of two layers should sum (up to gauge).
        G_grav = 1.0
        rho_k = 1.0

        s1 = PoissonCartesian3D(kx=0.0, ky=0.0, rho_k=rho_k, z1=0.0, z2=1.0,
                                G_grav=G_grav, z_ref=5.0)
        s2 = PoissonCartesian3D(kx=0.0, ky=0.0, rho_k=rho_k, z1=3.0, z2=4.0,
                                G_grav=G_grav, z_ref=5.0)

        z_vals = np.array([-2.0, 0.5, 2.0, 3.5, 6.0])
        total = s1.psi_k(z_vals) + s2.psi_k(z_vals)

        def numerical_I(z, z1, z2):
            val, _ = integrate.quad(lambda zp: abs(z - zp), z1, z2)
            return val

        for i, z in enumerate(z_vals):
            I1 = numerical_I(z, 0.0, 1.0)
            I1_ref = numerical_I(5.0, 0.0, 1.0)
            I2 = numerical_I(z, 3.0, 4.0)
            I2_ref = numerical_I(5.0, 3.0, 4.0)
            expected = -2.0 * np.pi * G_grav * rho_k * ((I1 - I1_ref) + (I2 - I2_ref))
            assert_allclose(total[i], expected, rtol=1e-10,
                            err_msg=f"Superposition mismatch at z={z}")


# ---------------------------------------------------------------------------
# I. k_h isotropy -- CRITICAL test unique to 3D
# ---------------------------------------------------------------------------
# Different (kx, ky) pairs with the same k_h must produce identical psi_k(z).
# The vertical structure depends ONLY on k_h = sqrt(kx^2 + ky^2).
#
# Test triplet: (kx=3, ky=4), (kx=5, ky=0), (kx=0, ky=5) all give k_h=5.
# Also: (kx=4, ky=3) gives k_h=5.
# And: (kx=5/sqrt(2), ky=5/sqrt(2)) gives k_h=5.
#
# All must produce identical psi_k(z) for every z, since psi_k depends
# only on k_h through the Green's function g_k(z,z') = (1/(2k_h))*e^{-k_h|z-z'|}.

class TestKhIsotropy:
    """Category I: k_h isotropy -- different (kx,ky) with same k_h give same psi_k."""

    def test_isotropy_kh5(self):
        """(kx=3,ky=4), (kx=5,ky=0), (kx=0,ky=5), (kx=4,ky=3) all give k_h=5."""
        kxy_pairs = [(3.0, 4.0), (5.0, 0.0), (0.0, 5.0), (4.0, 3.0),
                     (5.0 / np.sqrt(2), 5.0 / np.sqrt(2))]
        solvers = [
            PoissonCartesian3D(kx=kx, ky=ky, rho_k=1.0, z1=0.0, z2=1.0, G_grav=1.0)
            for kx, ky in kxy_pairs
        ]

        # Verify k_h = 5 for all
        for s in solvers:
            assert_allclose(s.k_h, 5.0, rtol=1e-14)

        z = np.linspace(-3.0, 5.0, 100)
        ref = solvers[0].psi_k(z)
        for i, s in enumerate(solvers[1:], start=1):
            assert_allclose(s.psi_k(z), ref, rtol=1e-14,
                            err_msg=f"Isotropy broken for pair {kxy_pairs[i]}")

    def test_isotropy_kh1(self):
        """(kx=1,ky=0), (kx=0,ky=1), (kx=1/sqrt(2),ky=1/sqrt(2)) all give k_h=1."""
        kxy_pairs = [(1.0, 0.0), (0.0, 1.0),
                     (1.0 / np.sqrt(2), 1.0 / np.sqrt(2))]
        solvers = [
            PoissonCartesian3D(kx=kx, ky=ky, rho_k=2.0, z1=1.0, z2=3.0, G_grav=0.5)
            for kx, ky in kxy_pairs
        ]

        for s in solvers:
            assert_allclose(s.k_h, 1.0, rtol=1e-14)

        z = np.linspace(-2.0, 6.0, 80)
        ref = solvers[0].psi_k(z)
        for i, s in enumerate(solvers[1:], start=1):
            assert_allclose(s.psi_k(z), ref, rtol=1e-14,
                            err_msg=f"Isotropy broken for pair {kxy_pairs[i]}")

    def test_isotropy_different_psi_values(self):
        """Isotropy holds at specific hand-computed values.

        For k_h=5, rho_k=1, z1=0, z2=1, G_grav=1, at z=0.5 (inside):
        psi_k = (2*pi/25) * (2 - 2*e^{-2.5})

        This value must be identical whether kx=3,ky=4 or kx=5,ky=0 or kx=0,ky=5.
        """
        expected = (2.0 * np.pi / 25.0) * (2.0 - 2.0 * np.exp(-2.5))

        for kx, ky in [(3.0, 4.0), (5.0, 0.0), (0.0, 5.0)]:
            s = PoissonCartesian3D(kx=kx, ky=ky, rho_k=1.0, z1=0.0, z2=1.0, G_grav=1.0)
            result = s.psi_k(0.5)
            assert_allclose(result, expected, rtol=1e-12,
                            err_msg=f"Isotropy value mismatch for ({kx},{ky})")

    def test_isotropy_to_spatial_differs(self):
        """Even though psi_k is identical, to_spatial differs because of
        the different phase e^{i(kx*x + ky*y)}.

        For (kx=3, ky=4) vs (kx=5, ky=0) at x=1, y=1:
          phase1 = 3*1 + 4*1 = 7
          phase2 = 5*1 + 0*1 = 5
        Same psi_k but different spatial field.
        """
        s1 = PoissonCartesian3D(kx=3.0, ky=4.0, rho_k=1.0, z1=0.0, z2=1.0, G_grav=1.0)
        s2 = PoissonCartesian3D(kx=5.0, ky=0.0, rho_k=1.0, z1=0.0, z2=1.0, G_grav=1.0)

        x, y, z = 1.0, 1.0, 0.5
        # psi_k values are the same
        assert_allclose(s1.psi_k(z), s2.psi_k(z), rtol=1e-14)

        # to_spatial values differ (different phases)
        sp1 = s1.to_spatial(x, y, z)
        sp2 = s2.to_spatial(x, y, z)
        psi = s1.psi_k(z)
        assert_allclose(sp1, psi * np.exp(1j * 7.0), rtol=1e-14)
        assert_allclose(sp2, psi * np.exp(1j * 5.0), rtol=1e-14)
        # They should NOT be equal in general
        assert abs(sp1 - sp2) > 1e-10


# ---------------------------------------------------------------------------
# rho_to_spatial
# ---------------------------------------------------------------------------
class TestRhoToSpatial:
    """Verify the spatial density reconstruction."""

    def test_inside_layer(self):
        kx, ky, rho_k = 3.0, 4.0, 3.0
        solver = PoissonCartesian3D(kx=kx, ky=ky, rho_k=rho_k, z1=1.0, z2=3.0, G_grav=1.0)
        x, y, z = np.pi / 4, np.pi / 6, 2.0
        result = solver.rho_to_spatial(x, y, z)
        expected = rho_k * np.exp(1j * (kx * x + ky * y))
        assert_allclose(result, expected, rtol=1e-14)

    def test_outside_layer(self):
        solver = PoissonCartesian3D(kx=3.0, ky=4.0, rho_k=3.0, z1=1.0, z2=3.0, G_grav=1.0)
        for z in [0.0, 4.0]:
            result = solver.rho_to_spatial(np.pi / 3, np.pi / 5, z)
            assert_allclose(result, 0.0, atol=1e-16)

    def test_at_boundaries(self):
        solver = PoissonCartesian3D(kx=1.0, ky=2.0, rho_k=1.0, z1=1.0, z2=3.0, G_grav=1.0)
        x, y = 0.0, 0.0
        for z in [1.0, 3.0]:
            result = solver.rho_to_spatial(x, y, z)
            assert_allclose(result, np.exp(1j * (1.0 * x + 2.0 * y)), rtol=1e-14)

    def test_array_input(self):
        solver = PoissonCartesian3D(kx=1.0, ky=0.0, rho_k=2.0, z1=1.0, z2=3.0, G_grav=1.0)
        z = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        x, y = 0.0, 0.0
        result = solver.rho_to_spatial(x, y, z)
        expected = np.array([0.0, 2.0, 2.0, 2.0, 0.0])
        assert_allclose(result, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# k_h=0 without gauge reference
# ---------------------------------------------------------------------------
class TestKh0NoGauge:
    """Verify k_h=0 works without z_ref (gauge_offset=0)."""

    def test_no_zref_produces_valid_output(self):
        solver = PoissonCartesian3D(kx=0.0, ky=0.0, rho_k=1.0, z1=0.0, z2=1.0, G_grav=1.0)
        z = np.linspace(-3.0, 5.0, 20)
        result = solver.psi_k(z)
        assert result.shape == (20,)
        assert np.all(np.isfinite(result))

    def test_no_zref_vs_zref_differ_by_constant(self):
        s_no_ref = PoissonCartesian3D(kx=0.0, ky=0.0, rho_k=1.0, z1=0.0, z2=1.0, G_grav=1.0)
        s_with_ref = PoissonCartesian3D(kx=0.0, ky=0.0, rho_k=1.0, z1=0.0, z2=1.0,
                                        G_grav=1.0, z_ref=3.0)
        z = np.linspace(-3.0, 5.0, 50)
        diff = s_no_ref.psi_k(z) - s_with_ref.psi_k(z)
        assert_allclose(diff, diff[0], rtol=1e-12)


# ---------------------------------------------------------------------------
# Boundary-exact evaluation
# ---------------------------------------------------------------------------
class TestBoundaryExactValues:
    """Verify psi_k at z = z1 and z = z2 matches from both piecewise branches."""

    @pytest.mark.parametrize("kx,ky", [(3.0, 4.0), (1.0, 0.0), (0.0, 3.0)])
    def test_at_z1(self, kx, ky):
        solver = PoissonCartesian3D(kx=kx, ky=ky, rho_k=1.0, z1=1.0, z2=3.0, G_grav=1.0)
        val = solver.psi_k(1.0)
        val_just_above = solver.psi_k(1.0 + 1e-14)
        assert_allclose(val, val_just_above, rtol=1e-10)

    @pytest.mark.parametrize("kx,ky", [(3.0, 4.0), (1.0, 0.0), (0.0, 3.0)])
    def test_at_z2(self, kx, ky):
        solver = PoissonCartesian3D(kx=kx, ky=ky, rho_k=1.0, z1=1.0, z2=3.0, G_grav=1.0)
        val = solver.psi_k(3.0)
        val_just_below = solver.psi_k(3.0 - 1e-14)
        assert_allclose(val, val_just_below, rtol=1e-10)

    def test_at_boundaries_kh0(self):
        solver = PoissonCartesian3D(kx=0.0, ky=0.0, rho_k=1.0, z1=1.0, z2=3.0,
                                    G_grav=1.0, z_ref=5.0)
        for z_bnd in [1.0, 3.0]:
            val = solver.psi_k(z_bnd)
            val_near = solver.psi_k(z_bnd + 1e-14)
            assert_allclose(val, val_near, rtol=1e-10)


# ---------------------------------------------------------------------------
# Detailed hand-computed k_h=0 integrals
# ---------------------------------------------------------------------------
# For k_h=0, rho_0=1, z1=2, z2=5, G_grav=1, z_ref=0:
#
# I(z) for each region:
#
# z < 2: I = (5-2)*((5+2)/2 - z) = 3*(3.5 - z)
#   At z=0: I = 3*3.5 = 10.5
#   At z=1: I = 3*2.5 = 7.5
#
# 2 <= z <= 5: I = (z-2)^2/2 + (5-z)^2/2
#   At z=2: I = 0 + 4.5 = 4.5
#   At z=3: I = 0.5 + 2.0 = 2.5
#   At z=3.5 (midpoint): I = 1.125 + 1.125 = 2.25
#   At z=5: I = 4.5 + 0 = 4.5
#
# z > 5: I = 3*(z - 3.5)
#   At z=6: I = 3*2.5 = 7.5
#   At z=7: I = 3*3.5 = 10.5
#
# psi_0(z) = -2*pi * I(z) + gauge
# gauge: psi_0(0) = 0 => -2*pi*10.5 + gauge = 0 => gauge = 21*pi
#
# psi_0(1) = -2*pi*7.5 + 21*pi = 6*pi
# psi_0(2) = -2*pi*4.5 + 21*pi = 12*pi
# psi_0(3) = -2*pi*2.5 + 21*pi = 16*pi
# psi_0(3.5) = -2*pi*2.25 + 21*pi = 16.5*pi
# psi_0(5) = -2*pi*4.5 + 21*pi = 12*pi
# psi_0(6) = -2*pi*7.5 + 21*pi = 6*pi
# psi_0(7) = -2*pi*10.5 + 21*pi = 0

class TestKh0DetailedArithmetic:
    """Detailed hand-computed k_h=0 values for rho_0=1, z1=2, z2=5, G_grav=1, z_ref=0."""

    @pytest.fixture
    def solver(self):
        return PoissonCartesian3D(kx=0.0, ky=0.0, rho_k=1.0, z1=2.0, z2=5.0,
                                  G_grav=1.0, z_ref=0.0)

    def test_at_zref(self, solver):
        assert_allclose(solver.psi_k(0.0), 0.0, atol=1e-14)

    def test_below_layer(self, solver):
        assert_allclose(solver.psi_k(1.0), 6.0 * np.pi, rtol=1e-12)

    def test_at_z1(self, solver):
        assert_allclose(solver.psi_k(2.0), 12.0 * np.pi, rtol=1e-12)

    def test_inside_layer(self, solver):
        assert_allclose(solver.psi_k(3.0), 16.0 * np.pi, rtol=1e-12)

    def test_at_midpoint(self, solver):
        assert_allclose(solver.psi_k(3.5), 16.5 * np.pi, rtol=1e-12)

    def test_at_z2(self, solver):
        assert_allclose(solver.psi_k(5.0), 12.0 * np.pi, rtol=1e-12)

    def test_above_layer(self, solver):
        assert_allclose(solver.psi_k(6.0), 6.0 * np.pi, rtol=1e-12)

    def test_far_above(self, solver):
        # z=7: I = 3*3.5 = 10.5, psi = -21*pi + 21*pi = 0
        assert_allclose(solver.psi_k(7.0), 0.0, atol=1e-12)

    def test_symmetry_about_midpoint(self, solver):
        # The layer [2,5] has midpoint 3.5.
        # I(3.5-d) = I(3.5+d), so psi(3.5-d) = psi(3.5+d)
        d_vals = [0.5, 1.0, 1.5, 2.5, 3.5]
        for d in d_vals:
            left = solver.psi_k(3.5 - d)
            right = solver.psi_k(3.5 + d)
            assert_allclose(left, right, rtol=1e-12,
                            err_msg=f"k_h=0 symmetry broken at d={d}")


# ---------------------------------------------------------------------------
# Additional: midpoint symmetry for k_h != 0
# ---------------------------------------------------------------------------
# For k_h=5, rho_k=1, z1=0, z2=2, G_grav=1:
# Prefactor: 2*pi*G*rho_k/k_h^2 = 2*pi/25
#
# At z=1 (midpoint of [0,2]):
# Inside formula: psi = (2*pi/25) * (2 - e^{-5*(1-0)} - e^{-5*(2-1)})
#                     = (2*pi/25) * (2 - 2*e^{-5})

class TestMidpointSymmetryKhNeq0:
    """Verify potential at layer midpoint has expected symmetric form."""

    def test_midpoint_kh5(self):
        solver = PoissonCartesian3D(kx=3.0, ky=4.0, rho_k=1.0, z1=0.0, z2=2.0, G_grav=1.0)
        z = 1.0  # midpoint
        # psi = (2*pi/25) * (2 - 2*e^{-5})
        expected = (2.0 * np.pi / 25.0) * (2.0 - 2.0 * np.exp(-5.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_symmetric_about_midpoint(self):
        """For a layer [z1, z2], psi_k(zmid-d) = psi_k(zmid+d) when inside."""
        solver = PoissonCartesian3D(kx=3.0, ky=4.0, rho_k=1.0, z1=1.0, z2=5.0, G_grav=1.0)
        zmid = 3.0
        d_vals = [0.3, 0.7, 1.0, 1.5, 1.9]
        for d in d_vals:
            left = solver.psi_k(zmid - d)
            right = solver.psi_k(zmid + d)
            assert_allclose(left, right, rtol=1e-12,
                            err_msg=f"Symmetry broken at d={d}")

    def test_symmetric_outside_layer(self):
        """Points equidistant from nearest boundary (opposite sides) give equal psi_k.

        For layer [1, 3] with k_h=5:
        z=0 is distance 1 from z1=1, z=4 is distance 1 from z2=3.
        z<z1: psi = C*(e^{-5*(z1-z)} - e^{-5*(z2-z)})
        z>z2: psi = C*(e^{-5*(z-z2)} - e^{-5*(z-z1)})
        At z=0: C*(e^{-5} - e^{-15})
        At z=4: C*(e^{-5} - e^{-15})
        """
        solver = PoissonCartesian3D(kx=3.0, ky=4.0, rho_k=1.0, z1=1.0, z2=3.0, G_grav=1.0)
        psi_below = solver.psi_k(0.0)
        psi_above = solver.psi_k(4.0)
        assert_allclose(psi_below, psi_above, rtol=1e-12)
