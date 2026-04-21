"""
Tests for the 3D gravitational potential solver in spherical coordinates.

These tests verify the mathematical correctness of the Green's function
convolution for solving:
    nabla^2 psi = -4 pi G rho
in 3D spherical coordinates (r, theta, phi), where the density is expanded
in spherical harmonics rho(r,theta,phi) = sum_{l,m} rho_{lm}(r) Y_l^m(theta,phi).

For a constant density coefficient rho_{lm} on a radial shell [r1, r2],
the potential mode psi_{lm}(r) is computed analytically via the radial
Green's function:

    psi_{lm}(r) = (4 pi G / (2l+1)) * [
        r^{-(l+1)} * int_{r1}^{min(r,r2)} rho_{lm} r'^{l+2} dr'
      + r^l        * int_{max(r,r1)}^{r2} rho_{lm} r'^{1-l} dr'
    ]

Each test derives expected values from first principles and compares
them against the package output.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import integrate
from scipy.special import sph_harm_y

from passess.spherical import PoissonSpherical3D


# ---------------------------------------------------------------------------
# A. Direct integral verification for l != 0, l != 2
# ---------------------------------------------------------------------------
# Parameters: l=1, m=0, rho_lm=1, r1=1, r2=2, G_grav=1.
#
# psi_{lm}(r) = (4*pi/(2*1+1)) * [
#     r^{-2} * int_{r1}^{min(r,r2)} r'^3 dr'
#   + r^1    * int_{max(r,r1)}^{r2} r'^0 dr'
# ]
#
# Prefactor = 4*pi/3
# Inner integral: int r'^3 dr' = r'^4/4
# Outer integral: int r'^0 dr' = r'
#
# Region 1: r = 0.5 < r1 = 1 (entire shell outside)
#   inner = 0
#   outer = int_1^2 dr' = 2 - 1 = 1
#   psi = (4*pi/3) * 0.5 * 1 = 2*pi/3
#
# Region 2: r = 1.5 (inside shell)
#   inner = int_1^{1.5} r'^3 dr' = (1.5^4 - 1)/4 = (5.0625 - 1)/4 = 4.0625/4
#   outer = int_{1.5}^2 dr' = 0.5
#   psi = (4*pi/3) * [1.5^{-2} * 4.0625/4 + 1.5 * 0.5]
#       = (4*pi/3) * [4.0625/9 + 0.75]
#       = (4*pi/3) * [0.451389... + 0.75]
#       = (4*pi/3) * 1.201389...
#
# Region 3: r = 3 > r2 = 2 (entire shell inside)
#   inner = int_1^2 r'^3 dr' = (16 - 1)/4 = 15/4
#   outer = 0
#   psi = (4*pi/3) * 3^{-2} * 15/4 = (4*pi/3) * 15/36 = (4*pi/3) * 5/12 = 5*pi/9

class TestDirectIntegralL1:
    """Category A: hand-computed integrals for l=1, m=0, rho_lm=1, r1=1, r2=2, G=1."""

    @pytest.fixture
    def solver(self):
        return PoissonSpherical3D(l=1, m=0, rho_lm=1.0, r1=1.0, r2=2.0, G_grav=1.0)

    def test_region1_outside_shell(self, solver):
        # r = 0.5 < r1 = 1: only the outer integral contributes.
        r = 0.5
        # outer = int_1^2 r'^0 dr' = 1
        expected = (4.0 * np.pi / 3.0) * 0.5 * 1.0  # = 2*pi/3
        assert_allclose(expected, 2.0 * np.pi / 3.0, rtol=1e-14)  # sanity
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2_inside_shell(self, solver):
        # r = 1.5, between r1=1 and r2=2.
        r = 1.5
        inner = (r**4 - 1.0) / 4.0      # (5.0625 - 1)/4 = 4.0625/4
        outer = 2.0 - r                   # 0.5
        expected = (4.0 * np.pi / 3.0) * (r**(-2) * inner + r * outer)
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region3_outside_shell(self, solver):
        # r = 3 > r2 = 2: only the inner integral contributes.
        r = 3.0
        inner = (2.0**4 - 1.0**4) / 4.0  # 15/4
        expected = (4.0 * np.pi / 3.0) * r**(-2) * inner  # = 5*pi/9
        assert_allclose(expected, 5.0 * np.pi / 9.0, rtol=1e-14)  # sanity
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_array_input(self, solver):
        """Verify that psi_lm works on numpy arrays."""
        r = np.array([0.5, 1.5, 3.0])
        results = solver.psi_lm(r)
        assert results.shape == (3,)
        assert_allclose(results[0], 2.0 * np.pi / 3.0, rtol=1e-12)
        assert_allclose(results[2], 5.0 * np.pi / 9.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# A2. Direct integral verification for l=3, rho_lm=2, r1=2, r2=4, G=0.5
# ---------------------------------------------------------------------------
# Prefactor = 4*pi*G/(2*3+1) = 4*pi*0.5/7 = 2*pi/7
# Inner exponent: l+2 = 5  =>  int r'^5 dr' = r'^6/6
# Outer exponent: 1-l = -2  =>  int r'^{-2} dr' = -r'^{-1}
#
# Region 1: r = 1 < r1 = 2
#   outer = rho_lm * int_2^4 r'^{-2} dr' = 2 * (-1/4 + 1/2) = 2 * 1/4 = 0.5
#   psi = (2*pi/7) * r^3 * 0.5 = (2*pi/7) * 0.5 = pi/7
#
# Region 3: r = 6 > r2 = 4
#   inner = rho_lm * int_2^4 r'^5 dr' = 2 * (4^6 - 2^6)/6 = 2*(4096-64)/6 = 2*4032/6 = 1344
#   psi = (2*pi/7) * r^{-4} * 1344 = (2*pi/7) * 1344/1296 = (2*pi/7) * 112/108 = (2*pi/7) * 28/27
#
# Region 2: r = 3 (inside shell)
#   inner = rho_lm * (3^6 - 2^6)/6 = 2*(729-64)/6 = 2*665/6 = 665/3
#   outer = rho_lm * (-1/4 + 1/3) = 2 * 1/12 = 1/6
#   psi = (2*pi/7) * [3^{-4} * 665/3 + 3^3 * 1/6]
#       = (2*pi/7) * [665/243 + 27/6]
#       = (2*pi/7) * [665/243 + 9/2]

class TestDirectIntegralL3:
    """Category A: l=3, m=1, rho_lm=2, r1=2, r2=4, G=0.5."""

    @pytest.fixture
    def solver(self):
        return PoissonSpherical3D(l=3, m=1, rho_lm=2.0, r1=2.0, r2=4.0, G_grav=0.5)

    def test_region1(self, solver):
        r = 1.0
        # outer = 2 * (-1/4 + 1/2) = 0.5
        outer_integral = 2.0 * (-1.0 / 4.0 + 1.0 / 2.0)
        expected = (2.0 * np.pi / 7.0) * r**3 * outer_integral
        assert_allclose(expected, np.pi / 7.0, rtol=1e-14)  # sanity
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region3(self, solver):
        r = 6.0
        inner_integral = 2.0 * (4.0**6 - 2.0**6) / 6.0  # 1344
        expected = (2.0 * np.pi / 7.0) * r**(-4) * inner_integral
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2(self, solver):
        r = 3.0
        inner_integral = 2.0 * (r**6 - 2.0**6) / 6.0   # 2*(729-64)/6 = 665/3
        outer_integral = 2.0 * (-1.0 / 4.0 + 1.0 / r)   # 2*(-1/4+1/3) = 1/6
        expected = (2.0 * np.pi / 7.0) * (r**(-4) * inner_integral + r**3 * outer_integral)
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# B. Special exponent l=2 (logarithmic outer integral)
# ---------------------------------------------------------------------------
# For l=2: outer integral exponent is 1-l = -1.
# int r'^{-1} dr' = ln(r')
#
# Prefactor = 4*pi*G/(2*2+1) = 4*pi/5
# Inner exponent: l+2 = 4  =>  int r'^4 dr' = r'^5/5
# Outer: int r'^{-1} dr' = ln(r')
#
# Parameters: l=2, m=0, rho_lm=1, r1=1, r2=3, G=1.
#
# Region 1: r = 0.5 < 1
#   inner = 0
#   outer = ln(3) - ln(1) = ln(3)
#   psi = (4*pi/5) * 0.5^2 * ln(3) = (4*pi/5) * ln(3)/4 = pi*ln(3)/5
#
# Region 2: r = 2
#   inner = (2^5 - 1)/5 = 31/5
#   outer = ln(3) - ln(2)
#   psi = (4*pi/5) * [2^{-3} * 31/5 + 2^2 * (ln3 - ln2)]
#       = (4*pi/5) * [31/40 + 4*(ln3-ln2)]
#
# Region 3: r = 4 > 3
#   inner = (3^5 - 1)/5 = 242/5
#   outer = 0
#   psi = (4*pi/5) * 4^{-3} * 242/5 = (4*pi/5) * 242/320 = (4*pi/5) * 121/160

class TestLogExponentL2:
    """Category B: l=2 produces logarithmic integrals in the outer part."""

    @pytest.fixture
    def solver(self):
        return PoissonSpherical3D(l=2, m=0, rho_lm=1.0, r1=1.0, r2=3.0, G_grav=1.0)

    def test_region1_log_integral(self, solver):
        r = 0.5
        expected = (4.0 * np.pi / 5.0) * r**2 * np.log(3.0)
        assert_allclose(expected, np.pi * np.log(3.0) / 5.0, rtol=1e-14)  # sanity
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2_log_integral(self, solver):
        r = 2.0
        inner = (r**5 - 1.0) / 5.0                # 31/5
        outer = np.log(3.0) - np.log(r)            # ln(3) - ln(2)
        expected = (4.0 * np.pi / 5.0) * (r**(-3) * inner + r**2 * outer)
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region3_log_integral(self, solver):
        r = 4.0
        inner = (3.0**5 - 1.0) / 5.0              # 242/5
        expected = (4.0 * np.pi / 5.0) * r**(-3) * inner
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-12)


# Additional test: m=+2 and m=-2 should give the same psi_lm(r) since
# the radial part is m-independent.
class TestMIndependence:
    """The radial potential mode depends only on l, not on m."""

    def test_different_m_same_l(self):
        r = np.linspace(0.3, 5.0, 50)
        s0 = PoissonSpherical3D(l=2, m=0, rho_lm=1.0, r1=1.0, r2=3.0, G_grav=1.0)
        s1 = PoissonSpherical3D(l=2, m=1, rho_lm=1.0, r1=1.0, r2=3.0, G_grav=1.0)
        s2 = PoissonSpherical3D(l=2, m=2, rho_lm=1.0, r1=1.0, r2=3.0, G_grav=1.0)
        s_neg = PoissonSpherical3D(l=2, m=-2, rho_lm=1.0, r1=1.0, r2=3.0, G_grav=1.0)
        assert_allclose(s0.psi_lm(r), s1.psi_lm(r), rtol=1e-14)
        assert_allclose(s0.psi_lm(r), s2.psi_lm(r), rtol=1e-14)
        assert_allclose(s0.psi_lm(r), s_neg.psi_lm(r), rtol=1e-14)


# ---------------------------------------------------------------------------
# C. l=0 mode (shell theorem)
# ---------------------------------------------------------------------------
# For l=0: prefactor = 4*pi*G/(2*0+1) = 4*pi*G
# Inner exponent: l+2 = 2  =>  int r'^2 dr' = r'^3/3
# Outer exponent: 1-l = 1  =>  int r' dr' = r'^2/2
#
# Parameters: l=0, m=0, rho_lm=1, r1=1, r2=2, G=1.
#
# Region 1: r < r1 = 1 (shell theorem: CONSTANT potential inside)
#   inner = 0
#   outer = int_1^2 r' dr' = (4-1)/2 = 3/2
#   psi = 4*pi * r^0 * 3/2 = 6*pi
#   This is constant (independent of r). Shell theorem!
#
# Region 3: r > r2 = 2
#   inner = int_1^2 r'^2 dr' = (8-1)/3 = 7/3
#   outer = 0
#   psi = 4*pi * r^{-1} * 7/3 = 28*pi/(3*r)
#   At r=3: psi = 28*pi/9
#   This is the total-mass/r behavior (Gauss's law).
#
# Region 2: r1 <= r <= r2, e.g. r = 1.5
#   inner = int_1^{1.5} r'^2 dr' = (1.5^3 - 1)/3 = (3.375 - 1)/3 = 2.375/3
#   outer = int_{1.5}^{2} r' dr' = (4 - 2.25)/2 = 1.75/2 = 0.875
#   psi = 4*pi * [1.5^{-1} * 2.375/3 + 1 * 0.875]
#       = 4*pi * [2.375/4.5 + 0.875]
#       = 4*pi * [0.527778 + 0.875]
#       = 4*pi * 1.402778

class TestL0Mode:
    """Category C: l=0 mode and the shell theorem."""

    @pytest.fixture
    def solver(self):
        return PoissonSpherical3D(l=0, m=0, rho_lm=1.0, r1=1.0, r2=2.0, G_grav=1.0)

    def test_region1_constant(self, solver):
        """For r < r1, the potential is constant (shell theorem)."""
        r_vals = np.array([0.1, 0.3, 0.5, 0.8, 0.99])
        results = solver.psi_lm(r_vals)
        assert_allclose(results, results[0], rtol=1e-12)

    def test_region1_value(self, solver):
        """psi = 4*pi * (r2^2 - r1^2)/2 = 4*pi * 3/2 = 6*pi."""
        r = 0.5
        expected = 4.0 * np.pi * (2.0**2 - 1.0**2) / 2.0
        assert_allclose(expected, 6.0 * np.pi, rtol=1e-14)  # sanity
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region3_value(self, solver):
        """psi = 4*pi/r * (r2^3 - r1^3)/3 = 28*pi/(3*r)."""
        r = 3.0
        inner = (2.0**3 - 1.0**3) / 3.0  # 7/3
        expected = 4.0 * np.pi * r**(-1) * inner
        assert_allclose(expected, 28.0 * np.pi / 9.0, rtol=1e-14)  # sanity
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2_value(self, solver):
        r = 1.5
        inner = (r**3 - 1.0**3) / 3.0    # (3.375 - 1)/3
        outer = (2.0**2 - r**2) / 2.0     # (4 - 2.25)/2 = 0.875
        expected = 4.0 * np.pi * (r**(-1) * inner + outer)
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_no_gauge_ambiguity(self):
        """In 3D, l=0 has no gauge issue. The potential decays as 1/r at infinity."""
        solver = PoissonSpherical3D(l=0, m=0, rho_lm=1.0, r1=1.0, r2=2.0, G_grav=1.0)
        # Far away, psi should approach 0 (unlike the 2D case).
        r_far = 1e6
        result = solver.psi_lm(r_far)
        assert abs(result) < 1e-2  # goes as 1/r, so ~ 28*pi/3 * 1e-6

    def test_shell_theorem_physical(self):
        """The potential is truly constant for r < r1 (like Newton's shell theorem)."""
        solver = PoissonSpherical3D(l=0, m=0, rho_lm=1.0, r1=2.0, r2=4.0, G_grav=1.0)
        # Region 1: psi = 4*pi * (r2^2 - r1^2)/2 = 4*pi * (16 - 4)/2 = 24*pi
        r_vals = np.array([0.01, 0.1, 0.5, 1.0, 1.5, 1.99])
        results = solver.psi_lm(r_vals)
        expected = 4.0 * np.pi * (4.0**2 - 2.0**2) / 2.0
        assert_allclose(results, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# D. Delta function (thin shell) limit
# ---------------------------------------------------------------------------
# For a thin shell at r=rp with width dr:
#   psi_{lm}(r) -> 4*pi*G * g_l(r,rp) * rho_lm * rp^2 * dr
#
# where g_l(r,r') = (1/(2l+1)) * r_<^l / r_>^{l+1}
#
# The rp^2 comes from the volume element r'^2 dr' integrated over the thin shell.

class TestDeltaFunctionLimit:
    """Category D: thin shell approaches point-source Green's function."""

    @pytest.mark.parametrize("l", [0, 1, 2, 3, 5])
    def test_thin_shell(self, l):
        rp = 2.0
        dr = 1e-6
        rho_lm = 1.0
        G_grav = 1.0
        r1 = rp - dr / 2.0
        r2 = rp + dr / 2.0
        solver = PoissonSpherical3D(l=l, m=0, rho_lm=rho_lm, r1=r1, r2=r2, G_grav=G_grav)

        for r in [0.5, 1.0, 3.0, 5.0]:
            r_less = min(r, rp)
            r_greater = max(r, rp)
            g_l = (1.0 / (2.0 * l + 1.0)) * r_less**l / r_greater**(l + 1)
            expected = 4.0 * np.pi * G_grav * g_l * rho_lm * rp**2 * dr
            result = solver.psi_lm(r)
            assert_allclose(result, expected, rtol=1e-4,
                            err_msg=f"Failed for l={l}, r={r}")

    @pytest.mark.parametrize("l", [1, 3])
    def test_thin_shell_complex_rho(self, l):
        """Delta function limit with complex density."""
        rp = 3.0
        dr = 1e-6
        rho_lm = 1.0 + 2.0j
        G_grav = 1.0
        r1 = rp - dr / 2.0
        r2 = rp + dr / 2.0
        solver = PoissonSpherical3D(l=l, m=0, rho_lm=rho_lm, r1=r1, r2=r2, G_grav=G_grav)

        r = 5.0
        r_less = min(r, rp)
        r_greater = max(r, rp)
        g_l = (1.0 / (2.0 * l + 1.0)) * r_less**l / r_greater**(l + 1)
        expected = 4.0 * np.pi * G_grav * g_l * rho_lm * rp**2 * dr
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# E. Laplacian verification (modal ODE)
# ---------------------------------------------------------------------------
# The modal radial ODE is:
#   (1/r^2) d/dr(r^2 d psi_{lm}/dr) - l(l+1)/r^2 psi_{lm} = source
#
# Equivalently:
#   psi'' + (2/r) psi' - l(l+1)/r^2 psi = source
#
# where source = -4*pi*G*rho_lm  inside the shell, 0 outside.

class TestLaplacianOutsideShell:
    """Category E: Laplacian is zero outside the source shell."""

    @pytest.mark.parametrize("l", [0, 1, 2, 3])
    def test_homogeneous_equation_outside(self, l):
        solver = PoissonSpherical3D(l=l, m=0, rho_lm=1.0, r1=2.0, r2=4.0, G_grav=1.0)
        h = 1e-5

        test_points = [1.0, 1.5, 5.0, 7.0, 10.0]
        for r in test_points:
            psi_minus = solver.psi_lm(r - h)
            psi_center = solver.psi_lm(r)
            psi_plus = solver.psi_lm(r + h)

            # psi'' + (2/r) psi' - l(l+1)/r^2 psi = 0
            d2psi = (psi_plus - 2.0 * psi_center + psi_minus) / h**2
            dpsi = (psi_plus - psi_minus) / (2.0 * h)

            laplacian_mode = d2psi + 2.0 * dpsi / r - l * (l + 1) * psi_center / r**2
            # Tolerance is 5e-4 for l=0 where function values are large
            # (O(100)) causing cancellation in the FD stencil.
            tol = 5e-4 if l == 0 else 1e-4
            assert abs(laplacian_mode) < tol, \
                f"Laplacian not zero at r={r}, l={l}: {laplacian_mode}"


class TestLaplacianInsideShell:
    """Verify that the Laplacian inside the shell equals the source term."""

    @pytest.mark.parametrize("l", [0, 1, 2, 3])
    def test_laplacian_equals_source(self, l):
        rho_lm = 1.0
        G_grav = 1.0
        r1, r2 = 1.0, 3.0
        solver = PoissonSpherical3D(l=l, m=0, rho_lm=rho_lm, r1=r1, r2=r2, G_grav=G_grav)

        h = 1e-5
        for r in [1.5, 2.0, 2.5]:
            psi_minus = solver.psi_lm(r - h)
            psi_center = solver.psi_lm(r)
            psi_plus = solver.psi_lm(r + h)

            d2psi = (psi_plus - 2.0 * psi_center + psi_minus) / h**2
            dpsi = (psi_plus - psi_minus) / (2.0 * h)

            # (1/r^2) d/dr(r^2 psi') - l(l+1)/r^2 psi = -4*pi*G*rho_lm
            # Expanding: psi'' + (2/r) psi' - l(l+1)/r^2 psi
            laplacian_mode = d2psi + 2.0 * dpsi / r - l * (l + 1) * psi_center / r**2
            expected_source = -4.0 * np.pi * G_grav * rho_lm
            assert_allclose(laplacian_mode, expected_source, rtol=1e-3, atol=1e-6,
                            err_msg=f"Source mismatch at r={r}, l={l}")


# ---------------------------------------------------------------------------
# F. Continuity at shell boundaries
# ---------------------------------------------------------------------------
# psi_{lm}(r) and its derivative must be continuous at r1 and r2.

class TestContinuityAtBoundaries:
    """Category F: psi_{lm} and its derivative are continuous at r1 and r2."""

    @pytest.mark.parametrize("l", [0, 1, 2, 3])
    def test_continuity_psi(self, l):
        solver = PoissonSpherical3D(l=l, m=0, rho_lm=1.0, r1=1.0, r2=3.0, G_grav=1.0)

        eps = 1e-10
        for r_bnd in [1.0, 3.0]:
            left = solver.psi_lm(r_bnd - eps)
            right = solver.psi_lm(r_bnd + eps)
            assert_allclose(left, right, rtol=1e-6, atol=1e-12,
                            err_msg=f"Discontinuity at r={r_bnd}, l={l}")

    @pytest.mark.parametrize("l", [0, 1, 2, 3])
    def test_continuity_derivative(self, l):
        solver = PoissonSpherical3D(l=l, m=0, rho_lm=1.0, r1=1.0, r2=3.0, G_grav=1.0)

        h = 1e-5
        for r_bnd in [1.0, 3.0]:
            eps = 1e-8
            dpsi_left = (solver.psi_lm(r_bnd - eps + h) - solver.psi_lm(r_bnd - eps - h)) / (2.0 * h)
            dpsi_right = (solver.psi_lm(r_bnd + eps + h) - solver.psi_lm(r_bnd + eps - h)) / (2.0 * h)
            assert_allclose(dpsi_left, dpsi_right, rtol=1e-2, atol=1e-6,
                            err_msg=f"Derivative discontinuity at r={r_bnd}, l={l}")


# ---------------------------------------------------------------------------
# G. Far-field, symmetry, linearity
# ---------------------------------------------------------------------------

class TestFarFieldBehavior:
    """Category G: asymptotic and symmetry checks."""

    def test_l0_far_field_1_over_r(self):
        """For r >> r2, psi_{00} ~ (4*pi*G*M)/r where M = int rho r'^2 dr' = (r2^3-r1^3)/3."""
        solver = PoissonSpherical3D(l=0, m=0, rho_lm=1.0, r1=1.0, r2=2.0, G_grav=1.0)
        M = (2.0**3 - 1.0**3) / 3.0  # = 7/3

        r1_far, r2_far = 100.0, 200.0
        psi1 = solver.psi_lm(r1_far)
        psi2 = solver.psi_lm(r2_far)

        # psi ~ C/r => psi1/psi2 = r2/r1 = 2
        ratio = psi1 / psi2
        assert_allclose(ratio, r2_far / r1_far, rtol=1e-8)

        # Also check the coefficient: psi(r) = 4*pi*G * M / r
        expected = 4.0 * np.pi * 1.0 * M / r1_far
        assert_allclose(psi1, expected, rtol=1e-8)

    @pytest.mark.parametrize("l", [1, 2, 3])
    def test_far_field_decay_power_law(self, l):
        """For l != 0 and r >> r2, psi_{lm}(r) ~ C * r^{-(l+1)}."""
        solver = PoissonSpherical3D(l=l, m=0, rho_lm=1.0, r1=1.0, r2=2.0, G_grav=1.0)

        r1_far, r2_far = 100.0, 200.0
        psi1 = solver.psi_lm(r1_far)
        psi2 = solver.psi_lm(r2_far)

        # psi ~ C * r^{-(l+1)} => psi2/psi1 = (r2/r1)^{-(l+1)}
        ratio = psi2 / psi1
        expected_ratio = (r2_far / r1_far)**(-(l + 1))
        assert_allclose(ratio, expected_ratio, rtol=1e-8)

    def test_linearity_in_rho(self):
        """Scaling rho_lm by a factor should scale psi_lm by the same factor."""
        s1 = PoissonSpherical3D(l=1, m=0, rho_lm=1.0, r1=1.0, r2=2.0, G_grav=1.0)
        s2 = PoissonSpherical3D(l=1, m=0, rho_lm=3.5, r1=1.0, r2=2.0, G_grav=1.0)
        r = np.linspace(0.1, 5.0, 50)
        assert_allclose(s2.psi_lm(r), 3.5 * s1.psi_lm(r), rtol=1e-14)

    def test_linearity_in_G(self):
        """Scaling G_grav should scale psi_lm proportionally."""
        s1 = PoissonSpherical3D(l=1, m=0, rho_lm=1.0, r1=1.0, r2=2.0, G_grav=1.0)
        s2 = PoissonSpherical3D(l=1, m=0, rho_lm=1.0, r1=1.0, r2=2.0, G_grav=2.0)
        r = np.linspace(0.1, 5.0, 50)
        assert_allclose(s2.psi_lm(r), 2.0 * s1.psi_lm(r), rtol=1e-14)

    def test_spatial_conversion_with_sph_harm(self):
        """to_spatial(r, theta, phi) should equal psi_lm(r) * Y_l^m(theta, phi)."""
        l, m = 2, 1
        solver = PoissonSpherical3D(l=l, m=m, rho_lm=1.0, r1=1.0, r2=2.0, G_grav=1.0)

        r = 1.5
        theta = np.pi / 4.0
        phi = np.pi / 3.0
        spatial = solver.to_spatial(r, theta, phi)
        # scipy.special.sph_harm_y(l, m, theta, phi)
        Y_lm = sph_harm_y(l, m, theta, phi)
        expected = solver.psi_lm(r) * Y_lm
        assert_allclose(spatial, expected, rtol=1e-12)

    def test_spatial_real_for_real_rho(self):
        """For real rho_lm and specific angular combination, check consistency."""
        l, m = 3, 2
        rho_lm = 2.0
        solver = PoissonSpherical3D(l=l, m=m, rho_lm=rho_lm, r1=1.0, r2=2.0, G_grav=1.0)

        r = 1.5
        theta = np.pi / 3.0
        phi = np.pi / 6.0
        spatial = solver.to_spatial(r, theta, phi)
        Y_lm = sph_harm_y(l, m, theta, phi)
        expected = solver.psi_lm(r) * Y_lm
        assert_allclose(spatial, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# H. Numerical integration comparison
# ---------------------------------------------------------------------------
# Compute psi_{lm}(r) = (4*pi*G/(2l+1)) * int_{r1}^{r2} (r_<^l/r_>^{l+1}) * rho_lm * r'^2 dr'
# via scipy.quad and compare to the analytical result.

class TestNumericalIntegration:
    """Category H: compare modal result with direct radial numerical integration."""

    @pytest.mark.parametrize("l", [0, 1, 2, 3, 5])
    def test_quad_vs_analytical(self, l):
        rho_lm = 1.0
        r1, r2, G_grav = 1.0, 3.0, 1.0
        solver = PoissonSpherical3D(l=l, m=0, rho_lm=rho_lm, r1=r1, r2=r2, G_grav=G_grav)

        def integrand(rp, r):
            r_less = min(r, rp)
            r_greater = max(r, rp)
            g_l = r_less**l / r_greater**(l + 1)
            return g_l * rho_lm * rp**2

        prefactor = 4.0 * np.pi * G_grav / (2.0 * l + 1.0)

        for r in [0.5, 1.5, 2.0, 2.5, 4.0, 10.0]:
            numerical, _ = integrate.quad(integrand, r1, r2, args=(r,),
                                          epsabs=1e-12, epsrel=1e-12)
            expected = prefactor * numerical
            result = solver.psi_lm(r)
            assert_allclose(result, expected, rtol=1e-10,
                            err_msg=f"Mismatch at r={r}, l={l}")

    def test_quad_complex_rho(self):
        """Numerical integration with complex density coefficient."""
        l = 2
        rho_lm = 1.0 + 3.0j
        r1, r2, G_grav = 1.0, 2.0, 1.0
        solver = PoissonSpherical3D(l=l, m=1, rho_lm=rho_lm, r1=r1, r2=r2, G_grav=G_grav)

        def integrand_real(rp, r):
            r_less = min(r, rp)
            r_greater = max(r, rp)
            g_l = r_less**l / r_greater**(l + 1)
            return g_l * np.real(rho_lm) * rp**2

        def integrand_imag(rp, r):
            r_less = min(r, rp)
            r_greater = max(r, rp)
            g_l = r_less**l / r_greater**(l + 1)
            return g_l * np.imag(rho_lm) * rp**2

        prefactor = 4.0 * np.pi * G_grav / (2.0 * l + 1.0)

        for r in [0.5, 1.5, 3.0]:
            real_part, _ = integrate.quad(integrand_real, r1, r2, args=(r,),
                                          epsabs=1e-12, epsrel=1e-12)
            imag_part, _ = integrate.quad(integrand_imag, r1, r2, args=(r,),
                                          epsabs=1e-12, epsrel=1e-12)
            expected = prefactor * (real_part + 1j * imag_part)
            result = solver.psi_lm(r)
            assert_allclose(result, expected, rtol=1e-10,
                            err_msg=f"Complex mismatch at r={r}")


# ---------------------------------------------------------------------------
# H2. Full 3D spatial integration comparison
# ---------------------------------------------------------------------------
# Compute psi(r,theta,phi) by brute-force 3D numerical integration:
#   psi(r,theta,phi) = G * int rho(r',theta',phi') / |r - r'| r'^2 sin(theta') dr' dtheta' dphi'
#
# with rho(r',theta',phi') = rho_lm * Y_l^m(theta',phi')
# and |r - r'| = sqrt(r^2 + r'^2 - 2*r*r'*cos(gamma))
# where cos(gamma) = cos(theta)*cos(theta') + sin(theta)*sin(theta')*cos(phi-phi')

class TestFull3DIntegration:
    """Category H2: compare modal result with full 3D numerical integration."""

    def _numerical_potential(self, r, theta, phi, l, m, rho_lm, r1, r2, G_grav):
        """Compute psi(r,theta,phi) by 3D numerical integration using the
        free-space Green's function G(r,r') = 1/(4*pi*|r-r'|)."""

        def integrand_real(phi_p, theta_p, r_p):
            cos_gamma = (np.cos(theta) * np.cos(theta_p)
                         + np.sin(theta) * np.sin(theta_p) * np.cos(phi - phi_p))
            dist_sq = r**2 + r_p**2 - 2.0 * r * r_p * cos_gamma
            dist_sq = max(dist_sq, 1e-30)
            dist = np.sqrt(dist_sq)
            Y = sph_harm_y(l, m, theta_p, phi_p)
            density = rho_lm * Y
            # Poisson: nabla^2 psi = -4*pi*G*rho => psi = G * int rho/|r-r'| dV
            return np.real(G_grav * density / dist * r_p**2 * np.sin(theta_p))

        def integrand_imag(phi_p, theta_p, r_p):
            cos_gamma = (np.cos(theta) * np.cos(theta_p)
                         + np.sin(theta) * np.sin(theta_p) * np.cos(phi - phi_p))
            dist_sq = r**2 + r_p**2 - 2.0 * r * r_p * cos_gamma
            dist_sq = max(dist_sq, 1e-30)
            dist = np.sqrt(dist_sq)
            Y = sph_harm_y(l, m, theta_p, phi_p)
            density = rho_lm * Y
            return np.imag(G_grav * density / dist * r_p**2 * np.sin(theta_p))

        real_part, _ = integrate.tplquad(
            integrand_real, r1, r2,
            0.0, np.pi,
            0.0, 2.0 * np.pi,
            epsabs=1e-8, epsrel=1e-8
        )
        imag_part, _ = integrate.tplquad(
            integrand_imag, r1, r2,
            0.0, np.pi,
            0.0, 2.0 * np.pi,
            epsabs=1e-8, epsrel=1e-8
        )
        return real_part + 1j * imag_part

    @pytest.mark.parametrize("l,m", [(1, 0), (1, 1), (2, 0)])
    def test_spatial_vs_numerical_3d(self, l, m):
        rho_lm = 1.0
        r1, r2, G_grav = 1.5, 2.5, 1.0
        solver = PoissonSpherical3D(l=l, m=m, rho_lm=rho_lm, r1=r1, r2=r2, G_grav=G_grav)

        # Test at points outside the shell to avoid near-singularity issues.
        test_cases = [
            (0.5, np.pi / 4.0, 0.0),
            (4.0, np.pi / 3.0, np.pi / 2.0),
        ]
        for r, theta, phi in test_cases:
            numerical = self._numerical_potential(r, theta, phi, l, m, rho_lm, r1, r2, G_grav)
            modal = solver.to_spatial(r, theta, phi)
            assert_allclose(modal, numerical, rtol=1e-4, atol=1e-10,
                            err_msg=f"Mismatch at r={r}, theta={theta}, phi={phi}, l={l}, m={m}")


# ---------------------------------------------------------------------------
# I. Input validation and edge cases
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Category I: verify that invalid inputs are rejected."""

    def test_r1_negative(self):
        with pytest.raises(ValueError, match="positive"):
            PoissonSpherical3D(l=1, m=0, rho_lm=1.0, r1=-1.0, r2=2.0, G_grav=1.0)

    def test_r2_zero(self):
        with pytest.raises(ValueError, match="positive"):
            PoissonSpherical3D(l=1, m=0, rho_lm=1.0, r1=0.0, r2=2.0, G_grav=1.0)

    def test_r1_equals_r2(self):
        with pytest.raises(ValueError, match="r1 < r2"):
            PoissonSpherical3D(l=1, m=0, rho_lm=1.0, r1=2.0, r2=2.0, G_grav=1.0)

    def test_r1_greater_than_r2(self):
        with pytest.raises(ValueError, match="r1 < r2"):
            PoissonSpherical3D(l=1, m=0, rho_lm=1.0, r1=3.0, r2=2.0, G_grav=1.0)

    def test_abs_m_greater_than_l(self):
        with pytest.raises(ValueError, match="|m| <= l"):
            PoissonSpherical3D(l=2, m=3, rho_lm=1.0, r1=1.0, r2=2.0, G_grav=1.0)

    def test_abs_m_negative_greater_than_l(self):
        with pytest.raises(ValueError, match="|m| <= l"):
            PoissonSpherical3D(l=1, m=-2, rho_lm=1.0, r1=1.0, r2=2.0, G_grav=1.0)

    def test_l_negative(self):
        with pytest.raises(ValueError, match="l >= 0"):
            PoissonSpherical3D(l=-1, m=0, rho_lm=1.0, r1=1.0, r2=2.0, G_grav=1.0)


class TestComplexDensity:
    """Verify that complex density coefficients are handled correctly."""

    def test_complex_rho_linearity(self):
        rho_lm = 1.0 + 2.0j
        s_unit = PoissonSpherical3D(l=1, m=0, rho_lm=1.0, r1=1.0, r2=2.0, G_grav=1.0)
        s_complex = PoissonSpherical3D(l=1, m=0, rho_lm=rho_lm, r1=1.0, r2=2.0, G_grav=1.0)

        r = np.linspace(0.1, 5.0, 20)
        psi_unit = s_unit.psi_lm(r)
        expected = rho_lm * psi_unit
        result = s_complex.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-14)


# ---------------------------------------------------------------------------
# Superposition: sum of two non-overlapping shells
# ---------------------------------------------------------------------------

class TestSuperposition:
    """Two non-overlapping shells should give potential = sum of individual potentials."""

    def test_two_shells_l1(self):
        l = 1
        G_grav = 1.0
        rho_lm = 1.0

        s1 = PoissonSpherical3D(l=l, m=0, rho_lm=rho_lm, r1=1.0, r2=2.0, G_grav=G_grav)
        s2 = PoissonSpherical3D(l=l, m=0, rho_lm=rho_lm, r1=3.0, r2=4.0, G_grav=G_grav)

        # Test r = 0.5 (below both shells):
        # s1: (4*pi/3) * r * int_1^2 dr' = (4*pi/3) * 0.5 * 1 = 2*pi/3
        # s2: (4*pi/3) * r * int_3^4 dr' = (4*pi/3) * 0.5 * 1 = 2*pi/3
        # total = 4*pi/3
        r = 0.5
        total = s1.psi_lm(r) + s2.psi_lm(r)
        expected = (4.0 * np.pi / 3.0) * 0.5 * 1.0 + (4.0 * np.pi / 3.0) * 0.5 * 1.0
        assert_allclose(total, expected, rtol=1e-12)

        # Test r = 5 (above both shells):
        # s1: (4*pi/3) * r^{-2} * int_1^2 r'^3 dr' = (4*pi/3) * (1/25) * 15/4
        # s2: (4*pi/3) * r^{-2} * int_3^4 r'^3 dr' = (4*pi/3) * (1/25) * (256-81)/4
        r = 5.0
        total = s1.psi_lm(r) + s2.psi_lm(r)
        s1_inner = (2.0**4 - 1.0**4) / 4.0   # 15/4
        s2_inner = (4.0**4 - 3.0**4) / 4.0   # (256-81)/4 = 175/4
        expected = (4.0 * np.pi / 3.0) * r**(-2) * (s1_inner + s2_inner)
        assert_allclose(total, expected, rtol=1e-12)

    def test_two_shells_l0(self):
        """For l=0, superposition of shells must add correctly."""
        s1 = PoissonSpherical3D(l=0, m=0, rho_lm=1.0, r1=1.0, r2=2.0, G_grav=1.0)
        s2 = PoissonSpherical3D(l=0, m=0, rho_lm=1.0, r1=3.0, r2=4.0, G_grav=1.0)

        # r = 0.5: inside both shells -> sum of two constants
        # s1: 4*pi*(r2^2-r1^2)/2 = 4*pi*(4-1)/2 = 6*pi
        # s2: 4*pi*(16-9)/2 = 14*pi
        r = 0.5
        total = s1.psi_lm(r) + s2.psi_lm(r)
        expected = 4.0 * np.pi * (2.0**2 - 1.0**2) / 2.0 + 4.0 * np.pi * (4.0**2 - 3.0**2) / 2.0
        assert_allclose(total, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Boundary-exact evaluation
# ---------------------------------------------------------------------------

class TestBoundaryExactValues:
    """Verify psi_{lm} at r = r1 and r = r2 matches from both piecewise branches."""

    @pytest.mark.parametrize("l", [0, 1, 2, 3])
    def test_at_r1(self, l):
        solver = PoissonSpherical3D(l=l, m=0, rho_lm=1.0, r1=1.0, r2=3.0, G_grav=1.0)
        val = solver.psi_lm(1.0)
        val_just_above = solver.psi_lm(1.0 + 1e-14)
        assert_allclose(val, val_just_above, rtol=1e-10)

    @pytest.mark.parametrize("l", [0, 1, 2, 3])
    def test_at_r2(self, l):
        solver = PoissonSpherical3D(l=l, m=0, rho_lm=1.0, r1=1.0, r2=3.0, G_grav=1.0)
        val = solver.psi_lm(3.0)
        val_just_below = solver.psi_lm(3.0 - 1e-14)
        assert_allclose(val, val_just_below, rtol=1e-10)


# ---------------------------------------------------------------------------
# rho_to_spatial
# ---------------------------------------------------------------------------

class TestRhoToSpatial:
    """Verify the spatial density reconstruction."""

    def test_inside_shell(self):
        l, m, rho_lm = 2, 1, 3.0
        solver = PoissonSpherical3D(l=l, m=m, rho_lm=rho_lm, r1=1.0, r2=3.0, G_grav=1.0)
        r, theta, phi = 2.0, np.pi / 4.0, np.pi / 3.0
        result = solver.rho_to_spatial(r, theta, phi)
        Y_lm = sph_harm_y(l, m, theta, phi)
        expected = rho_lm * Y_lm
        assert_allclose(result, expected, rtol=1e-12)

    def test_outside_shell(self):
        solver = PoissonSpherical3D(l=2, m=1, rho_lm=3.0, r1=1.0, r2=3.0, G_grav=1.0)
        for r in [0.5, 4.0]:
            result = solver.rho_to_spatial(r, np.pi / 4.0, np.pi / 3.0)
            assert_allclose(result, 0.0, atol=1e-16)

    def test_at_boundaries(self):
        l, m, rho_lm = 1, 0, 1.0
        solver = PoissonSpherical3D(l=l, m=m, rho_lm=rho_lm, r1=1.0, r2=3.0, G_grav=1.0)
        theta, phi = np.pi / 3.0, 0.0
        Y_lm = sph_harm_y(l, m, theta, phi)
        for r in [1.0, 3.0]:
            result = solver.rho_to_spatial(r, theta, phi)
            expected = rho_lm * Y_lm
            assert_allclose(result, expected, rtol=1e-12)

    def test_array_input(self):
        l, m, rho_lm = 1, 0, 2.0
        solver = PoissonSpherical3D(l=l, m=m, rho_lm=rho_lm, r1=1.0, r2=3.0, G_grav=1.0)
        r = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        theta, phi = np.pi / 2.0, 0.0
        Y_lm = sph_harm_y(l, m, theta, phi)
        result = solver.rho_to_spatial(r, theta, phi)
        expected = np.where((r >= 1.0) & (r <= 3.0), rho_lm * Y_lm, 0.0)
        assert_allclose(result, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# Additional: explicit l=4 to cover another generic case
# ---------------------------------------------------------------------------

class TestDirectIntegralL4:
    """Extra generic case: l=4, rho_lm=1, r1=1, r2=2, G=1."""

    @pytest.fixture
    def solver(self):
        return PoissonSpherical3D(l=4, m=0, rho_lm=1.0, r1=1.0, r2=2.0, G_grav=1.0)

    def test_region1(self, solver):
        # Prefactor = 4*pi/9
        # Outer: int r'^{-3} dr' = r'^{-2}/(-2)
        # outer = (-1/2) * (2^{-2} - 1^{-2}) = (-1/2)*(-3/4) = 3/8
        r = 0.5
        outer_integral = 0.5 * (1.0 - 1.0 / 4.0)  # (1/(-2))*(1/4 - 1) = (1/(-2))*(-3/4) = 3/8
        # More carefully: int_1^2 r'^{1-4} dr' = int_1^2 r'^{-3} dr'
        #   = [r'^{-2}/(-2)]_1^2 = -1/2 * (1/4 - 1) = -1/2 * (-3/4) = 3/8
        expected = (4.0 * np.pi / 9.0) * r**4 * (3.0 / 8.0)
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region3(self, solver):
        # Inner: int r'^6 dr' = r'^7/7
        # inner = (2^7 - 1)/7 = 127/7
        r = 3.0
        inner_integral = (2.0**7 - 1.0) / 7.0
        expected = (4.0 * np.pi / 9.0) * r**(-5) * inner_integral
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2(self, solver):
        r = 1.5
        inner_integral = (r**7 - 1.0) / 7.0
        # outer: int_{1.5}^2 r'^{-3} dr' = [r'^{-2}/(-2)]_{1.5}^2
        #       = (-1/2)*(1/4 - 1/2.25) = (-1/2)*(1/4 - 4/9) = (-1/2)*(-7/36) = 7/72
        outer_integral = (-0.5) * (2.0**(-2) - r**(-2))
        expected = (4.0 * np.pi / 9.0) * (r**(-5) * inner_integral + r**4 * outer_integral)
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Additional: l=0 with various rho and G values
# ---------------------------------------------------------------------------

class TestL0VariousParameters:
    """Check l=0 shell theorem with non-unit parameters."""

    def test_rho_and_G_scaling(self):
        """The l=0 potential inside scales as G * rho_lm."""
        rho_lm = 5.0
        G_grav = 2.0
        r1, r2 = 2.0, 3.0
        solver = PoissonSpherical3D(l=0, m=0, rho_lm=rho_lm, r1=r1, r2=r2, G_grav=G_grav)

        # Inside: psi = 4*pi*G*rho_lm * (r2^2 - r1^2)/2
        r = 1.0
        expected = 4.0 * np.pi * G_grav * rho_lm * (r2**2 - r1**2) / 2.0
        result = solver.psi_lm(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_l0_continuity_at_r2(self):
        """The transition from interior to 1/r behavior at r2 must be smooth."""
        solver = PoissonSpherical3D(l=0, m=0, rho_lm=1.0, r1=1.0, r2=3.0, G_grav=1.0)
        # At r2=3, from inside:
        #   psi = 4*pi * [r^{-1} * (r^3-1)/3 + (9-r^2)/2] at r=3
        #       = 4*pi * [3^{-1}*(27-1)/3 + (9-9)/2]
        #       = 4*pi * [26/9]
        # From outside:
        #   psi = 4*pi * r^{-1} * (27-1)/3 = 4*pi * (26/9)
        val_inside = solver.psi_lm(3.0 - 1e-12)
        val_outside = solver.psi_lm(3.0 + 1e-12)
        assert_allclose(val_inside, val_outside, rtol=1e-8)


# ---------------------------------------------------------------------------
# Near-field behavior for l >= 1: psi ~ r^l as r -> 0
# ---------------------------------------------------------------------------

class TestNearOriginBehavior:
    """For l >= 1, the potential should grow as r^l near the origin."""

    @pytest.mark.parametrize("l", [1, 2, 3])
    def test_origin_power_law(self, l):
        solver = PoissonSpherical3D(l=l, m=0, rho_lm=1.0, r1=2.0, r2=4.0, G_grav=1.0)

        r1_near, r2_near = 0.01, 0.02
        psi1 = solver.psi_lm(r1_near)
        psi2 = solver.psi_lm(r2_near)

        # psi ~ C * r^l => psi2/psi1 = (r2/r1)^l
        ratio = psi2 / psi1
        expected_ratio = (r2_near / r1_near)**l
        assert_allclose(ratio, expected_ratio, rtol=1e-8)
