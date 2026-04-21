"""
Tests for the 2D gravitational potential solver in Cartesian coordinates.

These tests verify the mathematical correctness of the Green's function
convolution for solving:
    nabla^2 psi = -4 pi gamma rho
in 2D Cartesian coordinates (x, z), where the density is expanded in
lateral Fourier modes rho(x,z) = (1/2pi) int rho_hat_k(z) e^{ikx} dk.

For a single mode k with CONSTANT density coefficient rho_hat_k on a
depth layer [z1, z2], the potential mode psi_hat_k(z) is computed
analytically via exponential integrals.

For k != 0, the three regions give:
  z < z1:         psi_k = (2*pi*gamma*rho_k / k^2) * (e^{-|k|(z1-z)} - e^{-|k|(z2-z)})
  z1 <= z <= z2:  psi_k = (2*pi*gamma*rho_k / k^2) * (2 - e^{-|k|(z-z1)} - e^{-|k|(z2-z)})
  z > z2:         psi_k = (2*pi*gamma*rho_k / k^2) * (e^{-|k|(z-z2)} - e^{-|k|(z-z1)})

For k = 0:
  psi_0(z) = -2*pi*gamma*rho_0 * int_{z1}^{z2} |z - z'| dz' + gauge

The modal ODE is: psi_k'' - k^2 psi_k = -4*pi*gamma*rho_k (inside),
                                       = 0                  (outside).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import integrate

from passess.cartesian import PoissonCartesian2D


# ---------------------------------------------------------------------------
# A. Direct integral verification for k != 0 (generic)
# ---------------------------------------------------------------------------
# Parameters set 1: k=1, rho_k=1, z1=0, z2=1, gamma=1.
#
# psi_k(z) = (2*pi*gamma / |k|) * [
#     e^{-|k|z} * int_{z1}^{min(z,z2)} rho_k e^{|k|z'} dz'
#   + e^{|k|z}  * int_{max(z,z1)}^{z2}  rho_k e^{-|k|z'} dz'
# ]
#
# For k=1 (|k|=1):
#   int e^{z'} dz' = e^{z'}
#   int e^{-z'} dz' = -e^{-z'}
#
# Prefactor = 2*pi / 1 = 2*pi
#
# Using the stable closed forms:
#
# Region 1: z = -1 < z1 = 0
#   psi_k = (2*pi*1*1/1) * (e^{-1*(0-(-1))} - e^{-1*(1-(-1))})
#         = 2*pi * (e^{-1} - e^{-2})
#
#   Verify via integral form:
#     inner integral (z' < z): int_0^{min(-1,1)} = int_0^{-1} = 0 (empty, since -1 < 0 = z1)
#     outer integral (z' > z): int_{max(-1,0)}^{1} = int_0^1 e^{-z'} dz'
#       = [-e^{-z'}]_0^1 = -e^{-1} + 1 = 1 - e^{-1}
#     e^{|k|z} * outer = e^{-1} * (1 - e^{-1}) = e^{-1} - e^{-2}
#     psi_k = 2*pi * (0 + e^{-1} - e^{-2}) = 2*pi*(e^{-1} - e^{-2})  [matches]
#
# Region 2: z = 0.5 (inside layer)
#   psi_k = (2*pi/1) * (2 - e^{-1*(0.5-0)} - e^{-1*(1-0.5)})
#         = 2*pi * (2 - e^{-0.5} - e^{-0.5})
#         = 2*pi * (2 - 2*e^{-0.5})
#
#   Verify via integral form:
#     inner: int_0^{0.5} e^{z'} dz' = e^{0.5} - 1
#     e^{-z} * inner = e^{-0.5} * (e^{0.5} - 1) = 1 - e^{-0.5}
#     outer: int_{0.5}^{1} e^{-z'} dz' = -e^{-1} + e^{-0.5} = e^{-0.5} - e^{-1}
#     e^{z} * outer = e^{0.5} * (e^{-0.5} - e^{-1}) = 1 - e^{-0.5}
#     psi_k = 2*pi * ((1 - e^{-0.5}) + (1 - e^{-0.5})) = 2*pi*(2 - 2*e^{-0.5})  [matches]
#
# Region 3: z = 2 > z2 = 1
#   psi_k = (2*pi/1) * (e^{-1*(2-1)} - e^{-1*(2-0)})
#         = 2*pi * (e^{-1} - e^{-2})
#
#   Verify via integral form:
#     inner: int_0^1 e^{z'} dz' = e^{1} - 1
#     e^{-z} * inner = e^{-2} * (e - 1) = e^{-1} - e^{-2}
#     outer: int_{max(2,0)}^{1} = int_2^1 = 0 (empty, since 2 > 1 = z2)
#     psi_k = 2*pi * (e^{-1} - e^{-2})  [matches]
#
# Note the symmetry: psi_k(-1) = psi_k(2) since both are distance 1 from
# the nearest boundary but the layer [0,1] is asymmetric about z=0.5.
# Actually for z=-1: distance to z1=0 is 1, distance to z2=1 is 2.
# For z=2: distance to z2=1 is 1, distance to z1=0 is 2. Same exponentials.

class TestDirectIntegralK1:
    """Category A: hand-computed integrals for k=1, rho_k=1, z1=0, z2=1, gamma=1."""

    @pytest.fixture
    def solver(self):
        return PoissonCartesian2D(k=1.0, rho_k=1.0, z1=0.0, z2=1.0, gamma=1.0)

    def test_region1_below_layer(self, solver):
        # z = -1 < z1 = 0
        # psi_k = 2*pi * (e^{-|0-(-1)|} - e^{-|1-(-1)|})
        #       = 2*pi * (e^{-1} - e^{-2})
        z = -1.0
        expected = 2.0 * np.pi * (np.exp(-1.0) - np.exp(-2.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2_inside_layer(self, solver):
        # z = 0.5, inside [0, 1]
        # psi_k = 2*pi * (2 - e^{-0.5} - e^{-0.5})
        #       = 2*pi * (2 - 2*e^{-0.5})
        z = 0.5
        expected = 2.0 * np.pi * (2.0 - 2.0 * np.exp(-0.5))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region3_above_layer(self, solver):
        # z = 2 > z2 = 1
        # psi_k = 2*pi * (e^{-(2-1)} - e^{-(2-0)})
        #       = 2*pi * (e^{-1} - e^{-2})
        z = 2.0
        expected = 2.0 * np.pi * (np.exp(-1.0) - np.exp(-2.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_at_z1_boundary(self, solver):
        # z = 0 = z1 (at left boundary, use inside formula)
        # psi_k = 2*pi * (2 - e^{0} - e^{-1})
        #       = 2*pi * (2 - 1 - e^{-1})
        #       = 2*pi * (1 - e^{-1})
        z = 0.0
        expected = 2.0 * np.pi * (1.0 - np.exp(-1.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_at_z2_boundary(self, solver):
        # z = 1 = z2 (at right boundary, use inside formula)
        # psi_k = 2*pi * (2 - e^{-1} - e^{0})
        #       = 2*pi * (1 - e^{-1})
        z = 1.0
        expected = 2.0 * np.pi * (1.0 - np.exp(-1.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_array_input(self, solver):
        """Verify that psi_k works on numpy arrays."""
        z = np.array([-1.0, 0.5, 2.0])
        results = solver.psi_k(z)
        assert results.shape == (3,)
        assert_allclose(results[0], 2.0 * np.pi * (np.exp(-1.0) - np.exp(-2.0)), rtol=1e-12)
        assert_allclose(results[1], 2.0 * np.pi * (2.0 - 2.0 * np.exp(-0.5)), rtol=1e-12)
        assert_allclose(results[2], 2.0 * np.pi * (np.exp(-1.0) - np.exp(-2.0)), rtol=1e-12)


# ---------------------------------------------------------------------------
# A2. Direct integral verification for k=2, rho_k=3, z1=1, z2=3, gamma=0.5
# ---------------------------------------------------------------------------
# Prefactor = 2*pi*gamma / |k| = 2*pi*0.5 / 2 = pi/2
# Stable form coefficient = 2*pi*gamma*rho_k / k^2 = 2*pi*0.5*3 / 4 = 3*pi/4
#
# Region 1: z = 0 < z1 = 1
#   psi_k = (3*pi/4) * (e^{-2*(1-0)} - e^{-2*(3-0)})
#         = (3*pi/4) * (e^{-2} - e^{-6})
#
# Region 2: z = 2 (inside [1, 3])
#   psi_k = (3*pi/4) * (2 - e^{-2*(2-1)} - e^{-2*(3-2)})
#         = (3*pi/4) * (2 - e^{-2} - e^{-2})
#         = (3*pi/4) * (2 - 2*e^{-2})
#
# Region 3: z = 5 > z2 = 3
#   psi_k = (3*pi/4) * (e^{-2*(5-3)} - e^{-2*(5-1)})
#         = (3*pi/4) * (e^{-4} - e^{-8})

class TestDirectIntegralK2:
    """Category A2: k=2, rho_k=3, z1=1, z2=3, gamma=0.5."""

    @pytest.fixture
    def solver(self):
        return PoissonCartesian2D(k=2.0, rho_k=3.0, z1=1.0, z2=3.0, gamma=0.5)

    def test_region1(self, solver):
        z = 0.0
        expected = (3.0 * np.pi / 4.0) * (np.exp(-2.0) - np.exp(-6.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2(self, solver):
        z = 2.0
        expected = (3.0 * np.pi / 4.0) * (2.0 - 2.0 * np.exp(-2.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region3(self, solver):
        z = 5.0
        expected = (3.0 * np.pi / 4.0) * (np.exp(-4.0) - np.exp(-8.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2_asymmetric_point(self, solver):
        # z = 1.5 (inside [1, 3])
        # psi_k = (3*pi/4) * (2 - e^{-2*(1.5-1)} - e^{-2*(3-1.5)})
        #       = (3*pi/4) * (2 - e^{-1} - e^{-3})
        z = 1.5
        expected = (3.0 * np.pi / 4.0) * (2.0 - np.exp(-1.0) - np.exp(-3.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# B. k = 0 mode
# ---------------------------------------------------------------------------
# For k=0:
#   psi_0(z) = -2*pi*gamma*rho_0 * I(z) + gauge
# where I(z) = int_{z1}^{z2} |z - z'| dz'
#
# Parameters: rho_0=1, z1=0, z2=1, gamma=1, z_ref=0 (gauge: psi_0(0)=0).
#
# Three regions for I(z):
#
# z < z1 = 0:
#   I(z) = int_0^1 (z' - z) dz' = [z'^2/2 - z*z']_0^1 = 1/2 - z
#   This equals (z2-z1)*[(z2+z1)/2 - z] = 1*(0.5 - z)  [matches]
#
# z1 <= z <= z2:
#   I(z) = (z - z1)^2/2 + (z2 - z)^2/2 = z^2/2 + (1-z)^2/2
#
#   Verify at z=0.5: I = 0.25/2 + 0.25/2 = 0.25
#   Verify at z=0: I = 0 + 0.5 = 0.5  (which matches z<z1 formula at z=0: 0.5-0=0.5)
#   Verify at z=1: I = 0.5 + 0 = 0.5  (matches z>z2 formula at z=1: 1-0.5=0.5)
#
# z > z2 = 1:
#   I(z) = int_0^1 (z - z') dz' = [z*z' - z'^2/2]_0^1 = z - 1/2
#   This equals (z2-z1)*[z - (z2+z1)/2] = 1*(z - 0.5)  [matches]
#
# With gamma=1, rho_0=1: psi_0(z) = -2*pi * I(z) + gauge
# Gauge: psi_0(z_ref=0) = 0 => -2*pi * I(0) + gauge = 0
#   I(0) = 0.5 (at boundary, use either formula)
#   gauge = 2*pi * 0.5 = pi
#
# psi_0(z) = -2*pi * I(z) + pi
#
# z = -1: I(-1) = 0.5 - (-1) = 1.5
#   psi_0(-1) = -2*pi*1.5 + pi = -3*pi + pi = -2*pi
#
# z = 0.5: I(0.5) = 0.25/2 + 0.25/2 = 0.25
#   psi_0(0.5) = -2*pi*0.25 + pi = -0.5*pi + pi = 0.5*pi
#
# z = 2: I(2) = 2 - 0.5 = 1.5
#   psi_0(2) = -2*pi*1.5 + pi = -3*pi + pi = -2*pi

class TestK0Mode:
    """Category B: k=0 mode with |z-z'| integral."""

    @pytest.fixture
    def solver(self):
        return PoissonCartesian2D(k=0.0, rho_k=1.0, z1=0.0, z2=1.0, gamma=1.0, z_ref=0.0)

    def test_region1_below(self, solver):
        # z=-1: I(-1) = 0.5 - (-1) = 1.5, psi = -2*pi*1.5 + pi = -2*pi
        z = -1.0
        expected = -2.0 * np.pi
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2_inside(self, solver):
        # z=0.5: I(0.5) = 0.125 + 0.125 = 0.25, psi = -0.5*pi + pi = pi/2
        z = 0.5
        expected = 0.5 * np.pi
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region3_above(self, solver):
        # z=2: I(2) = 1.5, psi = -2*pi
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
        # z=1: I(1) = 0.5 + 0 = 0.5, psi = -pi + pi = 0
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
        s1 = PoissonCartesian2D(k=0.0, rho_k=1.0, z1=0.0, z2=1.0, gamma=1.0, z_ref=0.0)
        s2 = PoissonCartesian2D(k=0.0, rho_k=1.0, z1=0.0, z2=1.0, gamma=1.0, z_ref=2.0)
        z = np.linspace(-3.0, 5.0, 100)
        diff = s1.psi_k(z) - s2.psi_k(z)
        assert_allclose(diff, diff[0], rtol=1e-12)


# ---------------------------------------------------------------------------
# B2. k=0 with different parameters: rho_0=2, z1=1, z2=3, gamma=0.5, z_ref=2
# ---------------------------------------------------------------------------
# Prefactor: -2*pi*gamma*rho_0 = -2*pi*0.5*2 = -2*pi
# Thickness: z2-z1 = 2, midpoint: (z1+z2)/2 = 2
#
# z < z1=1:
#   I(z) = (z2-z1)*[(z2+z1)/2 - z] = 2*(2 - z)
#   At z=0: I = 2*2 = 4
#
# z1 <= z <= z2:
#   I(z) = (z-1)^2/2 + (3-z)^2/2
#   At z=2: I = 1/2 + 1/2 = 1
#   At z=1: I = 0 + 2 = 2
#   At z=3: I = 2 + 0 = 2
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

class TestK0Mode2:
    """Category B2: k=0 with different parameters."""

    @pytest.fixture
    def solver(self):
        return PoissonCartesian2D(k=0.0, rho_k=2.0, z1=1.0, z2=3.0, gamma=0.5, z_ref=2.0)

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
        # and with gauge at z=2, psi_0(2-d) = psi_0(2+d)
        d_vals = np.array([0.3, 0.7, 1.0, 1.5, 3.0])
        left = solver.psi_k(2.0 - d_vals)
        right = solver.psi_k(2.0 + d_vals)
        assert_allclose(left, right, rtol=1e-12)


# ---------------------------------------------------------------------------
# C. Delta function limit (thin layer)
# ---------------------------------------------------------------------------
# For thin layer z2 = zp + dz/2, z1 = zp - dz/2 with dz -> 0:
#   psi_k(z) -> 4*pi*gamma * g_k(z, zp) * rho_k * dz
# where g_k(z,zp) = (1/2|k|) e^{-|k||z-zp|} for k != 0.
#
# So expected = 4*pi*gamma * (1/(2|k|)) * e^{-|k||z-zp|} * rho_k * dz
#             = (2*pi*gamma*rho_k*dz / |k|) * e^{-|k||z-zp|}

class TestDeltaFunctionLimit:
    """Category C: thin layer approaches point-source Green's function."""

    @pytest.mark.parametrize("k", [1.0, 2.0, 3.0, 5.0])
    def test_thin_layer_kneq0(self, k):
        zp = 2.0
        dz = 1e-6
        rho_k = 1.0
        gamma = 1.0
        z1 = zp - dz / 2.0
        z2 = zp + dz / 2.0
        solver = PoissonCartesian2D(k=k, rho_k=rho_k, z1=z1, z2=z2, gamma=gamma)

        for z in [-1.0, 0.0, 3.0, 5.0]:
            g_k = (1.0 / (2.0 * abs(k))) * np.exp(-abs(k) * abs(z - zp))
            expected = 4.0 * np.pi * gamma * g_k * rho_k * dz
            result = solver.psi_k(z)
            assert_allclose(result, expected, rtol=1e-4,
                            err_msg=f"Failed for k={k}, z={z}")

    def test_thin_layer_k0(self):
        # For k=0: g_0(z,zp) = -(1/2)|z-zp| + C
        # psi_0(z) = 4*pi*gamma * [-(1/2)|z-zp|] * rho_0 * dz + gauge
        #          = -2*pi*gamma*rho_0*|z-zp|*dz + gauge
        # Test differences to eliminate gauge.
        zp = 2.0
        dz = 1e-6
        rho_k = 1.0
        gamma = 1.0
        z1 = zp - dz / 2.0
        z2 = zp + dz / 2.0
        solver = PoissonCartesian2D(k=0.0, rho_k=rho_k, z1=z1, z2=z2, gamma=gamma,
                                    z_ref=10.0)

        z_test = [0.0, 1.0, 3.0, 5.0]
        for z in z_test:
            # Expected with gauge at z_ref=10:
            # psi_0(z) = -2*pi*gamma*rho_0 * |z-zp|*dz + gauge
            # psi_0(z_ref) = -2*pi*gamma*rho_0 * |z_ref-zp|*dz + gauge = 0
            # => gauge = 2*pi*gamma*rho_0 * |z_ref-zp|*dz
            # => psi_0(z) = -2*pi*gamma*rho_0 * dz * (|z-zp| - |z_ref-zp|)
            expected = -2.0 * np.pi * gamma * rho_k * dz * (
                abs(z - zp) - abs(10.0 - zp)
            )
            result = solver.psi_k(z)
            assert_allclose(result, expected, rtol=1e-3, atol=1e-15,
                            err_msg=f"Failed for k=0, z={z}")


# ---------------------------------------------------------------------------
# D. Laplacian verification
# ---------------------------------------------------------------------------
# Outside the layer: psi_k'' - k^2 psi_k = 0
# Inside the layer:  psi_k'' - k^2 psi_k = -4*pi*gamma*rho_k
#
# The ODE comes from substituting psi(x,z) = psi_k(z) e^{ikx} into
# nabla^2 psi = -4*pi*gamma*rho: the x-derivative gives -k^2 psi_k,
# the z-derivative gives psi_k''.

class TestLaplacianOutsideLayer:
    """Category D: Laplacian is zero outside the source layer."""

    @pytest.mark.parametrize("k", [1.0, 2.0, 3.0])
    def test_homogeneous_outside(self, k):
        solver = PoissonCartesian2D(k=k, rho_k=1.0, z1=2.0, z2=4.0, gamma=1.0)
        h = 1e-5

        # Points outside the layer [2, 4]
        test_points = [-2.0, 0.0, 1.0, 5.0, 7.0, 10.0]
        for z in test_points:
            psi_m = solver.psi_k(z - h)
            psi_c = solver.psi_k(z)
            psi_p = solver.psi_k(z + h)

            d2psi = (psi_p - 2.0 * psi_c + psi_m) / h**2
            # Modal equation: psi_k'' - k^2 * psi_k = 0
            residual = d2psi - k**2 * psi_c
            assert abs(residual) < 1e-4, \
                f"Laplacian not zero at z={z}, k={k}: residual={residual}"

    def test_homogeneous_k0_outside(self):
        # For k=0 outside the layer, psi_0'' = 0 (linear in z).
        solver = PoissonCartesian2D(k=0.0, rho_k=1.0, z1=2.0, z2=4.0, gamma=1.0, z_ref=0.0)
        h = 1e-5

        # z < z1: psi_0 = -2*pi*gamma*rho_0 * (z2-z1)*((z1+z2)/2 - z) + gauge
        # This is linear in z, so psi_0'' = 0.
        # z > z2: similarly linear.
        for z in [-1.0, 0.0, 1.0, 5.0, 7.0]:
            psi_m = solver.psi_k(z - h)
            psi_c = solver.psi_k(z)
            psi_p = solver.psi_k(z + h)
            d2psi = (psi_p - 2.0 * psi_c + psi_m) / h**2
            assert abs(d2psi) < 1e-4, \
                f"Second derivative not zero at z={z}: d2psi={d2psi}"


class TestLaplacianInsideLayer:
    """Category D: Laplacian equals source inside the layer."""

    @pytest.mark.parametrize("k", [1.0, 2.0, 3.0])
    def test_source_inside(self, k):
        rho_k = 1.0
        gamma = 1.0
        z1, z2 = 1.0, 3.0
        solver = PoissonCartesian2D(k=k, rho_k=rho_k, z1=z1, z2=z2, gamma=gamma)

        h = 1e-5
        for z in [1.5, 2.0, 2.5]:
            psi_m = solver.psi_k(z - h)
            psi_c = solver.psi_k(z)
            psi_p = solver.psi_k(z + h)

            d2psi = (psi_p - 2.0 * psi_c + psi_m) / h**2
            # psi_k'' - k^2 * psi_k = -4*pi*gamma*rho_k
            residual = d2psi - k**2 * psi_c
            expected_source = -4.0 * np.pi * gamma * rho_k
            assert_allclose(residual, expected_source, rtol=1e-3, atol=1e-6,
                            err_msg=f"Source mismatch at z={z}, k={k}")

    def test_source_inside_k0(self):
        # For k=0: psi_0'' = -4*pi*gamma*rho_0
        rho_k = 2.0
        gamma = 0.5
        z1, z2 = 1.0, 3.0
        solver = PoissonCartesian2D(k=0.0, rho_k=rho_k, z1=z1, z2=z2, gamma=gamma, z_ref=0.0)

        h = 1e-5
        for z in [1.5, 2.0, 2.5]:
            psi_m = solver.psi_k(z - h)
            psi_c = solver.psi_k(z)
            psi_p = solver.psi_k(z + h)

            d2psi = (psi_p - 2.0 * psi_c + psi_m) / h**2
            expected_source = -4.0 * np.pi * gamma * rho_k  # = -4*pi
            assert_allclose(d2psi, expected_source, rtol=1e-3, atol=1e-6,
                            err_msg=f"Source mismatch at z={z}")


# ---------------------------------------------------------------------------
# E. Continuity at layer boundaries
# ---------------------------------------------------------------------------
# psi_k(z) and psi_k'(z) are both continuous at z1 and z2.

class TestContinuityAtBoundaries:
    """Category E: psi_k and its derivative are continuous at z1 and z2."""

    @pytest.mark.parametrize("k", [0.0, 1.0, 2.0, 3.0])
    def test_continuity_psi(self, k):
        kwargs = dict(k=k, rho_k=1.0, z1=1.0, z2=3.0, gamma=1.0)
        if k == 0.0:
            kwargs['z_ref'] = 5.0
        solver = PoissonCartesian2D(**kwargs)

        eps = 1e-10
        for z_bnd in [1.0, 3.0]:
            left = solver.psi_k(z_bnd - eps)
            right = solver.psi_k(z_bnd + eps)
            assert_allclose(left, right, rtol=1e-6, atol=1e-12,
                            err_msg=f"Discontinuity at z={z_bnd}, k={k}")

    @pytest.mark.parametrize("k", [0.0, 1.0, 2.0, 3.0])
    def test_continuity_derivative(self, k):
        kwargs = dict(k=k, rho_k=1.0, z1=1.0, z2=3.0, gamma=1.0)
        if k == 0.0:
            kwargs['z_ref'] = 5.0
        solver = PoissonCartesian2D(**kwargs)

        h = 1e-5
        for z_bnd in [1.0, 3.0]:
            eps = 1e-8
            dpsi_left = (solver.psi_k(z_bnd - eps + h) - solver.psi_k(z_bnd - eps - h)) / (2 * h)
            dpsi_right = (solver.psi_k(z_bnd + eps + h) - solver.psi_k(z_bnd + eps - h)) / (2 * h)
            assert_allclose(dpsi_left, dpsi_right, rtol=1e-2, atol=1e-6,
                            err_msg=f"Derivative discontinuity at z={z_bnd}, k={k}")


# ---------------------------------------------------------------------------
# F. Far-field behavior and symmetry
# ---------------------------------------------------------------------------

class TestFarFieldBehavior:
    """Category F: asymptotic, symmetry, and linearity checks."""

    def test_exponential_decay_above(self):
        """For k != 0 and z >> z2, psi_k ~ C * e^{-|k|z}."""
        k = 2.0
        solver = PoissonCartesian2D(k=k, rho_k=1.0, z1=0.0, z2=1.0, gamma=1.0)

        z1_far, z2_far = 50.0, 51.0
        psi1 = solver.psi_k(z1_far)
        psi2 = solver.psi_k(z2_far)

        # psi_k(z) ~ (2*pi*gamma*rho_k/k^2)*(e^{-|k|(z-z2)} - e^{-|k|(z-z1)})
        # For z >> z2, the dominant term is e^{-|k|(z-z2)}, so
        # psi(z+1)/psi(z) ~ e^{-|k|}
        ratio = psi2 / psi1
        expected_ratio = np.exp(-abs(k))
        assert_allclose(ratio, expected_ratio, rtol=1e-8)

    def test_exponential_decay_below(self):
        """For k != 0 and z << z1, psi_k ~ C * e^{|k|z}."""
        k = 2.0
        solver = PoissonCartesian2D(k=k, rho_k=1.0, z1=0.0, z2=1.0, gamma=1.0)

        z1_far, z2_far = -51.0, -50.0
        psi1 = solver.psi_k(z1_far)
        psi2 = solver.psi_k(z2_far)

        # Below layer: dominant term is e^{-|k|(z1-z)} = e^{|k|z} * e^{-|k|z1}
        # psi(z+1)/psi(z) ~ e^{|k|} for z << z1
        ratio = psi2 / psi1
        expected_ratio = np.exp(abs(k))
        assert_allclose(ratio, expected_ratio, rtol=1e-8)

    def test_negative_k_symmetry(self):
        """Negative k gives same psi_k(z) as positive k (depends on |k|)."""
        s_pos = PoissonCartesian2D(k=2.0, rho_k=1.0, z1=1.0, z2=3.0, gamma=1.0)
        s_neg = PoissonCartesian2D(k=-2.0, rho_k=1.0, z1=1.0, z2=3.0, gamma=1.0)
        z = np.linspace(-2.0, 6.0, 50)
        assert_allclose(s_pos.psi_k(z), s_neg.psi_k(z), rtol=1e-14)

    def test_linearity_in_rho(self):
        """Scaling rho_k scales psi_k proportionally."""
        s1 = PoissonCartesian2D(k=1.0, rho_k=1.0, z1=0.0, z2=1.0, gamma=1.0)
        s2 = PoissonCartesian2D(k=1.0, rho_k=3.5, z1=0.0, z2=1.0, gamma=1.0)
        z = np.linspace(-3.0, 5.0, 50)
        assert_allclose(s2.psi_k(z), 3.5 * s1.psi_k(z), rtol=1e-14)

    def test_linearity_in_gamma(self):
        """Scaling gamma scales psi_k proportionally."""
        s1 = PoissonCartesian2D(k=1.0, rho_k=1.0, z1=0.0, z2=1.0, gamma=1.0)
        s2 = PoissonCartesian2D(k=1.0, rho_k=1.0, z1=0.0, z2=1.0, gamma=2.0)
        z = np.linspace(-3.0, 5.0, 50)
        assert_allclose(s2.psi_k(z), 2.0 * s1.psi_k(z), rtol=1e-14)

    def test_to_spatial(self):
        """to_spatial(x, z) = psi_k(z) * exp(ikx)."""
        k = 3.0
        solver = PoissonCartesian2D(k=k, rho_k=1.0, z1=0.0, z2=1.0, gamma=1.0)
        z = 1.5
        x = np.pi / 4.0
        spatial = solver.to_spatial(x, z)
        expected = solver.psi_k(z) * np.exp(1j * k * x)
        assert_allclose(spatial, expected, rtol=1e-14)

    def test_to_spatial_real_part(self):
        """For real rho_k, psi_k(z) is real, so Re[to_spatial] = psi_k*cos(kx)."""
        k = 2.0
        rho_k = 3.0
        solver = PoissonCartesian2D(k=k, rho_k=rho_k, z1=0.0, z2=1.0, gamma=1.0)
        z = 0.5
        x = np.pi / 3.0
        spatial = solver.to_spatial(x, z)
        expected_real = solver.psi_k(z) * np.cos(k * x)
        assert_allclose(np.real(spatial), expected_real, rtol=1e-14)

    def test_k0_far_field_linear(self):
        """For k=0, z >> z2: psi_0 grows linearly with |z|."""
        # psi_0(z) for z > z2: -2*pi*gamma*rho_0 * (z2-z1)*(z - (z1+z2)/2) + gauge
        # This is linear in z with slope -2*pi*gamma*rho_0*(z2-z1)
        solver = PoissonCartesian2D(k=0.0, rho_k=1.0, z1=1.0, z2=3.0, gamma=1.0, z_ref=0.0)
        z_far = np.array([10.0, 20.0, 50.0])
        psi_vals = solver.psi_k(z_far)

        # Expected slope: -2*pi*1*1*2 = -4*pi
        slope = (psi_vals[-1] - psi_vals[0]) / (z_far[-1] - z_far[0])
        assert_allclose(slope, -4.0 * np.pi, rtol=1e-8)


# ---------------------------------------------------------------------------
# G. Comparison with direct numerical integration of 1D Green's function
# ---------------------------------------------------------------------------
# Compute psi_k(z) by numerically integrating:
#   psi_k(z) = (2*pi*gamma/|k|) * int_{z1}^{z2} e^{-|k||z-z'|} * rho_k dz'
# using scipy.integrate.quad. This is completely independent of piecewise forms.

class TestNumericalIntegration1D:
    """Category G: compare closed-form with 1D numerical quadrature."""

    def _numerical_psi_k(self, z, k, rho_k, z1, z2, gamma):
        """Compute psi_k(z) via numerical integration of the 1D Green's function."""
        abs_k = abs(k)

        def integrand(zp):
            return np.exp(-abs_k * abs(z - zp)) * rho_k

        result, _ = integrate.quad(integrand, z1, z2, epsabs=1e-12, epsrel=1e-12)
        return (2.0 * np.pi * gamma / abs_k) * result

    def _numerical_psi_0(self, z, rho_k, z1, z2, gamma, z_ref):
        """Compute psi_0(z) via numerical integration of |z-z'|."""
        def integrand(zp):
            return abs(z - zp) * rho_k

        def integrand_ref(zp):
            return abs(z_ref - zp) * rho_k

        val, _ = integrate.quad(integrand, z1, z2, epsabs=1e-12, epsrel=1e-12)
        ref, _ = integrate.quad(integrand_ref, z1, z2, epsabs=1e-12, epsrel=1e-12)
        return -2.0 * np.pi * gamma * (val - ref)

    @pytest.mark.parametrize("k", [0.5, 1.0, 2.0, 5.0])
    def test_kneq0_vs_numerical(self, k):
        rho_k = 2.0
        z1, z2, gamma = 1.0, 3.0, 0.5
        solver = PoissonCartesian2D(k=k, rho_k=rho_k, z1=z1, z2=z2, gamma=gamma)

        test_z = [-2.0, 0.0, 1.5, 2.0, 2.8, 4.0, 6.0]
        for z in test_z:
            numerical = self._numerical_psi_k(z, k, rho_k, z1, z2, gamma)
            closed = solver.psi_k(z)
            assert_allclose(closed, numerical, rtol=1e-10,
                            err_msg=f"Mismatch at z={z}, k={k}")

    def test_k0_vs_numerical(self):
        rho_k = 2.0
        z1, z2, gamma = 1.0, 3.0, 0.5
        z_ref = 5.0
        solver = PoissonCartesian2D(k=0.0, rho_k=rho_k, z1=z1, z2=z2, gamma=gamma,
                                    z_ref=z_ref)

        test_z = [-2.0, 0.0, 1.5, 2.0, 2.8, 4.0, 6.0]
        for z in test_z:
            numerical = self._numerical_psi_0(z, rho_k, z1, z2, gamma, z_ref)
            closed = solver.psi_k(z)
            assert_allclose(closed, numerical, rtol=1e-10,
                            err_msg=f"Mismatch at z={z}, k=0")


# ---------------------------------------------------------------------------
# H. Input validation and edge cases
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Category H: input validation."""

    def test_z1_equals_z2(self):
        with pytest.raises(ValueError):
            PoissonCartesian2D(k=1.0, rho_k=1.0, z1=2.0, z2=2.0, gamma=1.0)

    def test_z1_greater_than_z2(self):
        with pytest.raises(ValueError):
            PoissonCartesian2D(k=1.0, rho_k=1.0, z1=3.0, z2=2.0, gamma=1.0)


class TestComplexDensity:
    """Category H: complex density coefficients."""

    def test_complex_rho_k(self):
        rho_k = 1.0 + 2.0j
        s_real = PoissonCartesian2D(k=1.0, rho_k=1.0, z1=0.0, z2=1.0, gamma=1.0)
        s_complex = PoissonCartesian2D(k=1.0, rho_k=rho_k, z1=0.0, z2=1.0, gamma=1.0)

        z = np.linspace(-3.0, 5.0, 20)
        psi_unit = s_real.psi_k(z)
        expected = rho_k * psi_unit
        result = s_complex.psi_k(z)
        assert_allclose(result, expected, rtol=1e-14)

    def test_pure_imaginary_rho(self):
        rho_k = 2.0j
        s_real = PoissonCartesian2D(k=2.0, rho_k=1.0, z1=1.0, z2=3.0, gamma=1.0)
        s_imag = PoissonCartesian2D(k=2.0, rho_k=rho_k, z1=1.0, z2=3.0, gamma=1.0)

        z = np.linspace(-2.0, 6.0, 20)
        expected = rho_k * s_real.psi_k(z)
        result = s_imag.psi_k(z)
        assert_allclose(result, expected, rtol=1e-14)


class TestSuperposition:
    """Category H: superposition of two non-overlapping layers."""

    def test_two_layers_k1(self):
        # Two non-overlapping layers: [0,1] and [3,4]
        # At z=2 (between the layers), both contribute as "outside" terms.
        k = 1.0
        gamma = 1.0
        rho_k = 1.0

        s1 = PoissonCartesian2D(k=k, rho_k=rho_k, z1=0.0, z2=1.0, gamma=gamma)
        s2 = PoissonCartesian2D(k=k, rho_k=rho_k, z1=3.0, z2=4.0, gamma=gamma)

        # z = -1 (below both layers):
        # s1: (2*pi/1) * (e^{-1*(0-(-1))} - e^{-1*(1-(-1))})
        #   = 2*pi * (e^{-1} - e^{-2})
        # s2: (2*pi/1) * (e^{-1*(3-(-1))} - e^{-1*(4-(-1))})
        #   = 2*pi * (e^{-4} - e^{-5})
        z = -1.0
        total = s1.psi_k(z) + s2.psi_k(z)
        expected = 2.0 * np.pi * ((np.exp(-1.0) - np.exp(-2.0))
                                  + (np.exp(-4.0) - np.exp(-5.0)))
        assert_allclose(total, expected, rtol=1e-12)

        # z = 2 (between layers):
        # s1 above: (2*pi/1) * (e^{-1*(2-1)} - e^{-1*(2-0)})
        #         = 2*pi * (e^{-1} - e^{-2})
        # s2 below: (2*pi/1) * (e^{-1*(3-2)} - e^{-1*(4-2)})
        #         = 2*pi * (e^{-1} - e^{-2})
        z = 2.0
        total = s1.psi_k(z) + s2.psi_k(z)
        expected = 2.0 * np.pi * 2.0 * (np.exp(-1.0) - np.exp(-2.0))
        assert_allclose(total, expected, rtol=1e-12)

        # z = 5 (above both layers):
        # s1: 2*pi * (e^{-(5-1)} - e^{-(5-0)}) = 2*pi*(e^{-4} - e^{-5})
        # s2: 2*pi * (e^{-(5-4)} - e^{-(5-3)}) = 2*pi*(e^{-1} - e^{-2})
        z = 5.0
        total = s1.psi_k(z) + s2.psi_k(z)
        expected = 2.0 * np.pi * ((np.exp(-4.0) - np.exp(-5.0))
                                  + (np.exp(-1.0) - np.exp(-2.0)))
        assert_allclose(total, expected, rtol=1e-12)

    def test_two_layers_k0_superposition(self):
        # For k=0, superposition of two layers should sum (up to gauge).
        # Use differences to avoid gauge issues.
        gamma = 1.0
        rho_k = 1.0

        s1 = PoissonCartesian2D(k=0.0, rho_k=rho_k, z1=0.0, z2=1.0, gamma=gamma, z_ref=5.0)
        s2 = PoissonCartesian2D(k=0.0, rho_k=rho_k, z1=3.0, z2=4.0, gamma=gamma, z_ref=5.0)

        # Verify by numerical integration that the sum matches
        z_vals = np.array([-2.0, 0.5, 2.0, 3.5, 6.0])
        total = s1.psi_k(z_vals) + s2.psi_k(z_vals)

        # Compute numerically from scratch
        def numerical_I(z, z1, z2):
            val, _ = integrate.quad(lambda zp: abs(z - zp), z1, z2)
            return val

        # Combined: I_total(z) = I(z, 0, 1) + I(z, 3, 4)
        # psi_total(z) = -2*pi*gamma*rho_k * I_total(z) + gauge_1 + gauge_2
        # With z_ref=5: gauge_i chosen so psi_i(5) = 0
        for i, z in enumerate(z_vals):
            I1 = numerical_I(z, 0.0, 1.0)
            I1_ref = numerical_I(5.0, 0.0, 1.0)
            I2 = numerical_I(z, 3.0, 4.0)
            I2_ref = numerical_I(5.0, 3.0, 4.0)
            expected = -2.0 * np.pi * gamma * rho_k * ((I1 - I1_ref) + (I2 - I2_ref))
            assert_allclose(total[i], expected, rtol=1e-10,
                            err_msg=f"Superposition mismatch at z={z}")


# ---------------------------------------------------------------------------
# rho_to_spatial
# ---------------------------------------------------------------------------
class TestRhoToSpatial:
    """Verify the spatial density reconstruction."""

    def test_inside_layer(self):
        k, rho_k = 2.0, 3.0
        solver = PoissonCartesian2D(k=k, rho_k=rho_k, z1=1.0, z2=3.0, gamma=1.0)
        x, z = np.pi / 4, 2.0
        result = solver.rho_to_spatial(x, z)
        expected = rho_k * np.exp(1j * k * x)
        assert_allclose(result, expected, rtol=1e-14)

    def test_outside_layer(self):
        solver = PoissonCartesian2D(k=2.0, rho_k=3.0, z1=1.0, z2=3.0, gamma=1.0)
        for z in [0.0, 4.0]:
            result = solver.rho_to_spatial(np.pi / 3, z)
            assert_allclose(result, 0.0, atol=1e-16)

    def test_at_boundaries(self):
        solver = PoissonCartesian2D(k=1.0, rho_k=1.0, z1=1.0, z2=3.0, gamma=1.0)
        x = 0.0
        for z in [1.0, 3.0]:
            result = solver.rho_to_spatial(x, z)
            assert_allclose(result, np.exp(1j * 1.0 * x), rtol=1e-14)

    def test_array_input(self):
        solver = PoissonCartesian2D(k=1.0, rho_k=2.0, z1=1.0, z2=3.0, gamma=1.0)
        z = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        x = 0.0
        result = solver.rho_to_spatial(x, z)
        expected = np.array([0.0, 2.0, 2.0, 2.0, 0.0])
        assert_allclose(result, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# k=0 without gauge reference
# ---------------------------------------------------------------------------
class TestK0NoGauge:
    """Verify k=0 works without z_ref (gauge_offset=0)."""

    def test_no_zref_produces_valid_output(self):
        solver = PoissonCartesian2D(k=0.0, rho_k=1.0, z1=0.0, z2=1.0, gamma=1.0)
        z = np.linspace(-3.0, 5.0, 20)
        result = solver.psi_k(z)
        assert result.shape == (20,)
        assert np.all(np.isfinite(result))

    def test_no_zref_vs_zref_differ_by_constant(self):
        s_no_ref = PoissonCartesian2D(k=0.0, rho_k=1.0, z1=0.0, z2=1.0, gamma=1.0)
        s_with_ref = PoissonCartesian2D(k=0.0, rho_k=1.0, z1=0.0, z2=1.0, gamma=1.0, z_ref=3.0)
        z = np.linspace(-3.0, 5.0, 50)
        diff = s_no_ref.psi_k(z) - s_with_ref.psi_k(z)
        assert_allclose(diff, diff[0], rtol=1e-12)


# ---------------------------------------------------------------------------
# Boundary-exact evaluation
# ---------------------------------------------------------------------------
class TestBoundaryExactValues:
    """Verify psi_k at z = z1 and z = z2 matches from both piecewise branches."""

    @pytest.mark.parametrize("k", [1.0, 2.0, 3.0])
    def test_at_z1(self, k):
        solver = PoissonCartesian2D(k=k, rho_k=1.0, z1=1.0, z2=3.0, gamma=1.0)
        val = solver.psi_k(1.0)
        val_just_above = solver.psi_k(1.0 + 1e-14)
        assert_allclose(val, val_just_above, rtol=1e-10)

    @pytest.mark.parametrize("k", [1.0, 2.0, 3.0])
    def test_at_z2(self, k):
        solver = PoissonCartesian2D(k=k, rho_k=1.0, z1=1.0, z2=3.0, gamma=1.0)
        val = solver.psi_k(3.0)
        val_just_below = solver.psi_k(3.0 - 1e-14)
        assert_allclose(val, val_just_below, rtol=1e-10)

    def test_at_boundaries_k0(self):
        solver = PoissonCartesian2D(k=0.0, rho_k=1.0, z1=1.0, z2=3.0, gamma=1.0, z_ref=5.0)
        for z_bnd in [1.0, 3.0]:
            val = solver.psi_k(z_bnd)
            val_near = solver.psi_k(z_bnd + 1e-14)
            assert_allclose(val, val_near, rtol=1e-10)


# ---------------------------------------------------------------------------
# Additional: verify specific hand-computed k=0 integrals in detail
# ---------------------------------------------------------------------------
# For k=0, rho_0=1, z1=2, z2=5, gamma=1, z_ref=0:
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
# psi_0(1) = -2*pi*7.5 + 21*pi = (-15 + 21)*pi = 6*pi
# psi_0(2) = -2*pi*4.5 + 21*pi = (-9 + 21)*pi = 12*pi
# psi_0(3) = -2*pi*2.5 + 21*pi = (-5 + 21)*pi = 16*pi
# psi_0(3.5) = -2*pi*2.25 + 21*pi = (-4.5 + 21)*pi = 16.5*pi
# psi_0(5) = -2*pi*4.5 + 21*pi = 12*pi
# psi_0(6) = -2*pi*7.5 + 21*pi = 6*pi
# psi_0(7) = -2*pi*10.5 + 21*pi = 0

class TestK0DetailedArithmetic:
    """Detailed hand-computed k=0 values for rho_0=1, z1=2, z2=5, gamma=1, z_ref=0."""

    @pytest.fixture
    def solver(self):
        return PoissonCartesian2D(k=0.0, rho_k=1.0, z1=2.0, z2=5.0, gamma=1.0, z_ref=0.0)

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
                            err_msg=f"k=0 symmetry broken at d={d}")


# ---------------------------------------------------------------------------
# Additional: verify k != 0 at the layer midpoint in detail
# ---------------------------------------------------------------------------
# For k=1, rho_k=1, z1=0, z2=2, gamma=1:
# Prefactor: 2*pi*gamma*rho_k/k^2 = 2*pi
#
# At z=1 (midpoint of [0,2]):
# Inside formula: psi = 2*pi * (2 - e^{-(1-0)} - e^{-(2-1)})
#                     = 2*pi * (2 - e^{-1} - e^{-1})
#                     = 2*pi * (2 - 2/e)
#
# This is symmetric because z is equidistant from both boundaries.

class TestMidpointSymmetryKNeq0:
    """Verify that the potential at the layer midpoint has the expected symmetric form."""

    def test_midpoint_k1(self):
        solver = PoissonCartesian2D(k=1.0, rho_k=1.0, z1=0.0, z2=2.0, gamma=1.0)
        z = 1.0  # midpoint
        # psi = 2*pi * (2 - 2*e^{-1})
        expected = 2.0 * np.pi * (2.0 - 2.0 * np.exp(-1.0))
        result = solver.psi_k(z)
        assert_allclose(result, expected, rtol=1e-12)

    def test_symmetric_about_midpoint(self):
        """For a layer [z1, z2], psi_k(zmid-d) = psi_k(zmid+d) when inside."""
        solver = PoissonCartesian2D(k=2.0, rho_k=1.0, z1=1.0, z2=5.0, gamma=1.0)
        zmid = 3.0
        d_vals = [0.3, 0.7, 1.0, 1.5, 1.9]
        for d in d_vals:
            left = solver.psi_k(zmid - d)
            right = solver.psi_k(zmid + d)
            assert_allclose(left, right, rtol=1e-12,
                            err_msg=f"Symmetry broken at d={d}")

    def test_symmetric_outside_layer(self):
        """Points equidistant from the nearest boundary (on opposite sides)
        should give equal potential, by the symmetry of the outside formulas."""
        # For layer [1, 3], midpoint 2. Outside:
        # z=0 is distance 1 from z1=1, z=4 is distance 1 from z2=3.
        # z<z1: psi = C*(e^{-|k|(z1-z)} - e^{-|k|(z2-z)})
        # z>z2: psi = C*(e^{-|k|(z-z2)} - e^{-|k|(z-z1)})
        # At z=0: C*(e^{-1} - e^{-3})
        # At z=4: C*(e^{-1} - e^{-3})
        solver = PoissonCartesian2D(k=1.0, rho_k=1.0, z1=1.0, z2=3.0, gamma=1.0)
        psi_below = solver.psi_k(0.0)
        psi_above = solver.psi_k(4.0)
        assert_allclose(psi_below, psi_above, rtol=1e-12)
