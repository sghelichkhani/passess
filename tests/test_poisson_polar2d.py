"""
Tests for the 2D gravitational potential solver in polar coordinates.

These tests verify the mathematical correctness of the Green's function
convolution for solving:
    nabla^2 psi = -4 pi gamma rho
in 2D polar coordinates (r, phi), where the density is expanded in
azimuthal Fourier modes rho(r,phi) = sum_m rho_m(r) e^{im phi}.

For a constant density coefficient rho_m on a radial shell [r1, r2],
the potential mode psi_m(r) is computed analytically via power-law
integrals.  Each test derives expected values from first principles
and compares them against the package output.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import integrate

from passess.polar import PoissonPolar2D


# ---------------------------------------------------------------------------
# A. Direct integral verification for m != 0 (generic)
# ---------------------------------------------------------------------------
# Parameters: m=1, rho_m=1, r1=1, r2=2, gamma=1.
#
# psi_m(r) = (2*pi*gamma/|m|) * [
#     r^{-|m|} * integral_{r1}^{min(r,r2)} rho_m r'^{|m|+1} dr'
#   + r^{|m|}  * integral_{max(r,r1)}^{r2} rho_m r'^{1-|m|} dr'
# ]
#
# NOTE: The factor 2*pi arises from the azimuthal orthogonality integral
# when converting the spatial Green's representation to modal form.
#
# For m=1: exponent in first integral is |m|+1 = 2,
#           exponent in second integral is 1-|m| = 0.
#
# First integral:  int r'^2 dr' = r'^3/3
# Second integral: int r'^0 dr' = r'
#
# Region 1: r < r1 = 1 (entire shell outside, e.g. r = 0.5)
#   psi_1(0.5) = 2*pi * [0 + 0.5^1 * 1] = pi
#
# Region 2: r1 <= r <= r2 (inside shell, e.g. r = 1.5)
#   First integral: from 1 to 1.5: r'^3/3 |_1^1.5 = (3.375 - 1)/3 = 2.375/3
#   Second integral: from 1.5 to 2: r' |_1.5^2 = 2 - 1.5 = 0.5
#   psi_1(1.5) = 2*pi * [1.5^{-1} * 2.375/3 + 1.5^1 * 0.5]
#              = 2*pi * 1.277778 = 2*pi * 23/18
#
# Region 3: r > r2 = 2 (entire shell inside, e.g. r = 3)
#   psi_1(3) = 2*pi * [3^{-1} * 7/3] = 14*pi/9

class TestDirectIntegralM1:
    """Category A: hand-computed integrals for m=1, rho_m=1, r1=1, r2=2, gamma=1."""

    @pytest.fixture
    def solver(self):
        return PoissonPolar2D(m=1, rho_m=1.0, r1=1.0, r2=2.0, gamma=1.0)

    def test_region1_outside_shell(self, solver):
        # r = 0.5 < r1 = 1: only the outer integral contributes.
        r = 0.5
        expected = 2 * np.pi * 0.5  # 2*pi * r^1 * (2 - 1) = pi
        result = solver.psi_m(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2_inside_shell(self, solver):
        # r = 1.5, between r1=1 and r2=2.
        r = 1.5
        inner = (1.5**3 - 1.0**3) / 3.0   # 2.375/3
        outer = 2.0 - 1.5                   # 0.5
        expected = 2 * np.pi * (r**(-1) * inner + r * outer)
        assert_allclose(expected, 2 * np.pi * 23.0 / 18.0, rtol=1e-12)  # sanity
        result = solver.psi_m(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region3_inside_all(self, solver):
        # r = 3 > r2 = 2: only the inner integral contributes.
        r = 3.0
        inner = (2.0**3 - 1.0**3) / 3.0   # 7/3
        expected = 2 * np.pi * r**(-1) * inner  # 14*pi/9
        result = solver.psi_m(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_array_input(self, solver):
        """Verify that psi_m works on numpy arrays."""
        r = np.array([0.5, 1.5, 3.0])
        results = solver.psi_m(r)
        assert results.shape == (3,)
        assert_allclose(results[0], 2 * np.pi * 0.5, rtol=1e-12)
        assert_allclose(results[2], 2 * np.pi * 7.0 / 9.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# A2. Direct integral verification for m=3, rho_m=2, r1=2, r2=4, gamma=0.5
# ---------------------------------------------------------------------------
# psi_m(r) = (2*pi*gamma/|m|) * [...]  = (2*pi*0.5/3) * [...]
#
# First integral exponent: |m|+1 = 4  =>  int r'^4 dr' = r'^5/5
# Second integral exponent: 1-|m| = -2  =>  int r'^{-2} dr' = -r'^{-1}

class TestDirectIntegralM3:
    """Category A: m=3, rho_m=2, r1=2, r2=4, gamma=0.5."""

    @pytest.fixture
    def solver(self):
        return PoissonPolar2D(m=3, rho_m=2.0, r1=2.0, r2=4.0, gamma=0.5)

    def test_region3(self, solver):
        r = 6.0
        # 2*pi*gamma/|m| = 2*pi*0.5/3 = pi/3
        # rho_m * int_2^4 r'^4 dr' = 2 * (4^5 - 2^5)/5 = 2 * 992/5
        inner_integral = 2.0 * (4.0**5 - 2.0**5) / 5.0
        expected = (np.pi / 3.0) * r**(-3) * inner_integral
        result = solver.psi_m(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region1(self, solver):
        r = 1.0
        # Second integral: rho_m * int_2^4 r'^{-2} dr' = 2*(-1/4 + 1/2) = 0.5
        outer_integral = 2.0 * (-1.0 / 4.0 + 1.0 / 2.0)
        expected = (np.pi / 3.0) * r**3 * outer_integral
        result = solver.psi_m(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2(self, solver):
        r = 3.0
        inner_integral = 2.0 * (3.0**5 - 2.0**5) / 5.0  # 2*(243-32)/5 = 422/5
        outer_integral = 2.0 * (-1.0 / 4.0 + 1.0 / 3.0)  # 2*(1/12) = 1/6
        expected = (np.pi / 3.0) * (r**(-3) * inner_integral + r**3 * outer_integral)
        result = solver.psi_m(r)
        assert_allclose(result, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# B. Special exponent case |m| = 2
# ---------------------------------------------------------------------------
# For |m|=2, the second integral has exponent 1-|m| = -1.
# int r'^{-1} dr' = ln(r'), so we get a logarithmic term.
#
# Parameters: m=2, rho_m=1, r1=1, r2=3, gamma=1.
#
# psi_2(r) = (2*pi/2) * [
#     r^{-2} * int_{r1}^{min(r,r2)} r'^3 dr'
#   + r^2    * int_{max(r,r1)}^{r2} r'^{-1} dr'
# ] = pi * [...]
#
# First integral:  int r'^3 dr' = r'^4/4
# Second integral: int r'^{-1} dr' = ln(r')

class TestLogExponentM2:
    """Category B: |m|=2 produces logarithmic integrals in the outer part."""

    @pytest.fixture
    def solver(self):
        return PoissonPolar2D(m=2, rho_m=1.0, r1=1.0, r2=3.0, gamma=1.0)

    def test_region1_log_integral(self, solver):
        r = 0.5
        expected = np.pi * r**2 * np.log(3.0)
        result = solver.psi_m(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2_log_integral(self, solver):
        r = 2.0
        inner = (r**4 - 1.0) / 4.0          # 15/4
        outer = np.log(3.0) - np.log(r)      # ln(3) - ln(2)
        expected = np.pi * (r**(-2) * inner + r**2 * outer)
        result = solver.psi_m(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region3_log_integral(self, solver):
        r = 4.0
        inner = (3.0**4 - 1.0) / 4.0        # 80/4 = 20
        expected = np.pi * r**(-2) * inner   # pi * 20/16 = 5*pi/4
        result = solver.psi_m(r)
        assert_allclose(result, expected, rtol=1e-12)


# Additional test: m=-2 should give the same psi_m(r) as m=+2 (depends on |m|).
class TestNegativeModeSymmetry:
    """The radial potential mode depends on |m|, so m and -m give same psi_m(r)."""

    def test_m_plus_minus(self):
        s_pos = PoissonPolar2D(m=2, rho_m=1.0, r1=1.0, r2=3.0, gamma=1.0)
        s_neg = PoissonPolar2D(m=-2, rho_m=1.0, r1=1.0, r2=3.0, gamma=1.0)
        r = np.linspace(0.3, 5.0, 50)
        assert_allclose(s_pos.psi_m(r), s_neg.psi_m(r), rtol=1e-14)


# ---------------------------------------------------------------------------
# C. m=0 mode (axisymmetric, logarithmic Green's function)
# ---------------------------------------------------------------------------
# psi_0(r) = -4*pi*gamma * [
#     ln(r) * int_{r1}^{min(r,r2)} rho_0 r' dr'
#   + int_{max(r,r1)}^{r2} rho_0 r' ln(r') dr'
# ] + gauge_const
#
# Parameters: m=0, rho_0=1, r1=1, r2=2, gamma=1.
#
# Needed integrals:
#   int r' dr' = r'^2/2
#   int r' ln(r') dr' = r'^2/2 * ln(r') - r'^2/4
#     (integration by parts: u=ln(r'), dv=r' dr')
#
# Region 1: r = 0.5 < 1
#   inner = 0
#   outer = int_1^2 r' ln(r') dr' = [r'^2/2 ln(r') - r'^2/4]_1^2
#         = (4/2 * ln2 - 4/4) - (1/2 * 0 - 1/4)
#         = (2*ln2 - 1) - (- 1/4)
#         = 2*ln2 - 3/4
#   psi_0(0.5) = -2 * [0 + (2*ln2 - 3/4)] + gauge
#              = -4*ln2 + 3/2 + gauge
#
# Region 3: r = 3 > 2
#   inner = int_1^2 r' dr' = (4 - 1)/2 = 3/2
#   outer = 0
#   psi_0(3) = -2 * [ln(3) * 3/2 + 0] + gauge
#            = -3*ln(3) + gauge
#
# Region 2: r = 1.5
#   inner = int_1^{1.5} r' dr' = (2.25 - 1)/2 = 0.625
#   outer = int_{1.5}^{2} r' ln(r') dr'
#         = [r'^2/2 ln(r') - r'^2/4]_{1.5}^{2}
#         = (2*ln2 - 1) - (1.125*ln(1.5) - 0.5625)
#   psi_0(1.5) = -2 * [ln(1.5)*0.625 + outer] + gauge
#
# Gauge: set psi_0(r_ref) = 0. We'll use r_ref = 1.
# At r_ref = 1 (which is at the inner boundary):
#   inner = int_1^1 ... = 0
#   outer = int_1^2 r' ln(r') dr' = 2*ln2 - 3/4  (computed above)
#   psi_0(1) = -2*(0 + 2*ln2 - 3/4) + gauge = 0
#   => gauge = 4*ln2 - 3/2
#
# Now we can evaluate:
# psi_0(0.5) = -4*ln2 + 3/2 + 4*ln2 - 3/2 = 0
#   Wait, that means psi_0(0.5) = 0 with this gauge, since
#   at r=0.5 < r1: outer = same as at r=1 (both below r1, outer goes from r1 to r2).
#   Actually: at r=0.5: outer is int_1^2 r' ln(r') dr' same as at r=1,
#   and inner is 0 for both. So psi_0(0.5) = psi_0(1) = 0 with this gauge.
#   Hmm, that's correct! For m=0, when r < r1, psi_0(r) = -2*gamma*int_{r1}^{r2} rho_0 r' ln(r') dr' + gauge,
#   which is constant (independent of r). Wait, no:
#   psi_0(r) = -2*gamma*[ln(r)*int_{r1}^{min(r,r2)} rho_0 r' dr' + int_{max(r,r1)}^{r2} rho_0 r' ln(r') dr']
#   For r < r1: min(r,r2)=r, but the integral goes from r1 to r which is r1 to 0.5 => 0 when r < r1.
#   Wait, the integral limits are from r1 to min(r,r2). If r < r1, then min(r,r2) < r1, so the integral is 0.
#   The second integral: from max(r,r1)=r1 to r2. So it doesn't depend on r at all!
#   So for r < r1, psi_0(r) = -2*gamma*(0 + int_{r1}^{r2} rho_0 r' ln(r') dr') + gauge = constant.
#   This is correct: outside the shell on the inner side, the m=0 solution
#   should be constant (like ln(r) with zero coefficient because no enclosed mass... wait).
#
#   Actually, the m=0 Green's function is g_0 = -(1/2pi)*ln(r_>).
#   For r < r1 < r' (all source points outside), r_> = r', so
#   psi_0 = 4*pi*gamma * int_{r1}^{r2} [-(1/2pi)*ln(r')] rho_0 r' dr'
#         = -2*gamma * int_{r1}^{r2} rho_0 r' ln(r') dr' + gauge
#   This is indeed r-independent. The potential is constant for r < r1 in m=0.
#   Makes sense: in 2D, the potential inside a cylindrical shell is constant.
#
# For r > r2: psi_0(r) = -2*gamma*[ln(r)*int_{r1}^{r2} rho_0 r' dr' + 0] + gauge
#   = -2*gamma*ln(r)*M + gauge, where M = int_{r1}^{r2} rho_0 r' dr'
#   This is the 2D "line mass" behavior: -2*gamma*M*ln(r) + const.
#
# Let's use r_ref = 3 (outside the shell) for gauge: psi_0(3) = 0.
#   gauge = 2*gamma*ln(3)*M = 2*1*ln(3)*3/2 = 3*ln(3)
#   where M = int_1^2 r' dr' = 3/2.
#
# Then:
#   psi_0(0.5) = -2*(2*ln2 - 3/4) + 3*ln(3)  (constant for r < 1)
#   psi_0(5)   = -2*ln(5)*3/2 + 3*ln(3) = -3*ln(5) + 3*ln(3) = 3*ln(3/5)

class TestM0Mode:
    """Category C: m=0 axisymmetric mode with logarithmic integrals."""

    @pytest.fixture
    def solver(self):
        return PoissonPolar2D(m=0, rho_m=1.0, r1=1.0, r2=2.0, gamma=1.0, r_ref=3.0)

    def _int_rp_ln_rp(self, a, b):
        """Evaluate int_a^b r' ln(r') dr' = [r'^2/2 ln(r') - r'^2/4]_a^b."""
        def F(x):
            return x**2 / 2.0 * np.log(x) - x**2 / 4.0
        return F(b) - F(a)

    def test_region3_outside(self, solver):
        # r = 5 > r2 = 2
        M = (2.0**2 - 1.0**2) / 2.0  # 3/2
        # psi_0(r) = -4*pi*ln(r)*M + gauge, psi_0(3) = 0 => gauge = 4*pi*ln(3)*M
        expected = -4.0 * np.pi * np.log(5.0) * M + 4.0 * np.pi * np.log(3.0) * M
        result = solver.psi_m(5.0)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region1_constant(self, solver):
        # For r < r1 = 1, the potential is constant (r-independent).
        r_vals = np.array([0.1, 0.3, 0.5, 0.8, 0.99])
        results = solver.psi_m(r_vals)
        # All values should be identical.
        assert_allclose(results, results[0], rtol=1e-12)

    def test_region1_value(self, solver):
        # psi_0(r < 1) = -4*pi * int_1^2 r' ln(r') dr' + gauge
        outer = self._int_rp_ln_rp(1.0, 2.0)
        M = 1.5
        gauge = 4.0 * np.pi * np.log(3.0) * M
        expected = -4.0 * np.pi * outer + gauge
        result = solver.psi_m(0.5)
        assert_allclose(result, expected, rtol=1e-12)

    def test_region2_inside(self, solver):
        r = 1.5
        inner = (r**2 - 1.0) / 2.0  # int_1^{1.5} r' dr'
        outer = self._int_rp_ln_rp(r, 2.0)
        M = 1.5
        gauge = 4.0 * np.pi * np.log(3.0) * M
        expected = -4.0 * np.pi * (np.log(r) * inner + outer) + gauge
        result = solver.psi_m(r)
        assert_allclose(result, expected, rtol=1e-12)

    def test_gauge_reference(self, solver):
        # By construction, psi_0(r_ref=3) should be exactly zero.
        result = solver.psi_m(3.0)
        assert_allclose(result, 0.0, atol=1e-14)

    def test_different_gauge(self):
        # Two gauges should differ only by a constant.
        s1 = PoissonPolar2D(m=0, rho_m=1.0, r1=1.0, r2=2.0, gamma=1.0, r_ref=3.0)
        s2 = PoissonPolar2D(m=0, rho_m=1.0, r1=1.0, r2=2.0, gamma=1.0, r_ref=5.0)
        r = np.linspace(0.1, 10.0, 100)
        diff = s1.psi_m(r) - s2.psi_m(r)
        # Should be a constant everywhere.
        assert_allclose(diff, diff[0], rtol=1e-12)


# ---------------------------------------------------------------------------
# D. Delta function (thin shell) limit
# ---------------------------------------------------------------------------
# For an infinitesimally thin shell at r=rp with width dr,
# the result should approach:
#   psi_m(r) = 8*pi^2*gamma * g_m(r, rp) * rho_m * rp * dr
#
# (The extra 2*pi relative to the spatial formula comes from the azimuthal
# orthogonality integral in the modal decomposition.)
#
# For m != 0: g_m(r,rp) = 1/(4*pi*|m|) * (r_</r_>)^|m|
# For m = 0:  g_0(r,rp) = -(1/(2*pi)) * ln(r_>) + C
#
# We test with a thin shell and compare to the Green's function directly.

class TestDeltaFunctionLimit:
    """Category D: thin shell approaches point-source Green's function."""

    @pytest.mark.parametrize("m", [1, 2, 3, 5])
    def test_thin_shell_mneq0(self, m):
        rp = 2.0
        dr = 1e-6
        rho_m = 1.0
        gamma = 1.0
        r1 = rp - dr / 2.0
        r2 = rp + dr / 2.0
        solver = PoissonPolar2D(m=m, rho_m=rho_m, r1=r1, r2=r2, gamma=gamma)

        # Test at r outside the shell.
        for r in [0.5, 1.0, 3.0, 5.0]:
            r_less = min(r, rp)
            r_greater = max(r, rp)
            g_m = (1.0 / (4.0 * np.pi * abs(m))) * (r_less / r_greater)**abs(m)
            expected = 8.0 * np.pi**2 * gamma * g_m * rho_m * rp * dr
            result = solver.psi_m(r)
            assert_allclose(result, expected, rtol=1e-4,
                            err_msg=f"Failed for m={m}, r={r}")

    def test_thin_shell_m0(self):
        rp = 2.0
        dr = 1e-6
        rho_m = 1.0
        gamma = 1.0
        r1 = rp - dr / 2.0
        r2 = rp + dr / 2.0
        r_ref = 10.0
        solver = PoissonPolar2D(m=0, rho_m=rho_m, r1=r1, r2=r2, gamma=gamma,
                                r_ref=r_ref)

        for r in [0.5, 1.0, 3.0, 5.0]:
            r_greater = max(r, rp)
            r_greater_ref = max(r_ref, rp)
            # g_0 with gauge: -(1/2pi)*ln(r_>) - [-(1/2pi)*ln(r_>_ref)]
            g_0_gauged = -(1.0 / (2.0 * np.pi)) * (np.log(r_greater) - np.log(r_greater_ref))
            expected = 8.0 * np.pi**2 * gamma * g_0_gauged * rho_m * rp * dr
            result = solver.psi_m(r)
            assert_allclose(result, expected, rtol=1e-4,
                            err_msg=f"Failed for m=0, r={r}")


# ---------------------------------------------------------------------------
# E. Laplacian verification (homogeneous equation outside the shell)
# ---------------------------------------------------------------------------
# Outside the source region (r < r1 or r > r2), psi_m satisfies:
#   psi_m'' + (1/r)*psi_m' - m^2/r^2 * psi_m = 0
# We verify this numerically with finite differences.

class TestLaplacianOutsideShell:
    """Category E: Laplacian is zero outside the source shell."""

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_homogeneous_equation_outside(self, m):
        solver = PoissonPolar2D(m=m, rho_m=1.0, r1=2.0, r2=4.0, gamma=1.0)
        h = 1e-5

        # Test points in region 1 (r < r1) and region 3 (r > r2)
        test_points = [1.0, 1.5, 5.0, 7.0, 10.0]
        for r in test_points:
            psi_minus = solver.psi_m(r - h)
            psi_center = solver.psi_m(r)
            psi_plus = solver.psi_m(r + h)

            # Second derivative: (psi(r+h) - 2*psi(r) + psi(r-h)) / h^2
            d2psi = (psi_plus - 2.0 * psi_center + psi_minus) / h**2
            # First derivative: (psi(r+h) - psi(r-h)) / (2*h)
            dpsi = (psi_plus - psi_minus) / (2.0 * h)

            laplacian_mode = d2psi + dpsi / r - m**2 * psi_center / r**2
            assert abs(laplacian_mode) < 1e-4, \
                f"Laplacian not zero at r={r}, m={m}: {laplacian_mode}"

    def test_homogeneous_m0_outside(self):
        solver = PoissonPolar2D(m=0, rho_m=1.0, r1=2.0, r2=4.0, gamma=1.0, r_ref=1.0)
        # Use h=1e-4 for m=0 to reduce catastrophic cancellation: the function
        # values are O(100) and the FD numerator for the second derivative is
        # O(h^2 * |psi''|), so too-small h loses digits in the subtraction.
        h = 1e-4

        # For m=0 outside: psi_0'' + (1/r)*psi_0' = 0
        # Only test r > r2 (in region 1, psi_0 is constant so trivially satisfied).
        for r in [5.0, 7.0, 10.0]:
            psi_minus = solver.psi_m(r - h)
            psi_center = solver.psi_m(r)
            psi_plus = solver.psi_m(r + h)

            d2psi = (psi_plus - 2.0 * psi_center + psi_minus) / h**2
            dpsi = (psi_plus - psi_minus) / (2.0 * h)

            laplacian_mode = d2psi + dpsi / r
            assert abs(laplacian_mode) < 1e-4, \
                f"Laplacian not zero at r={r}, m=0: {laplacian_mode}"


# ---------------------------------------------------------------------------
# F. Continuity and derivative continuity at shell boundaries
# ---------------------------------------------------------------------------
# psi_m(r) must be continuous at r1 and r2.
# Its radial derivative must also be continuous (the source is extended,
# unlike a delta function which gives a derivative jump).

class TestContinuityAtBoundaries:
    """Category F: psi_m and its derivative are continuous at r1 and r2."""

    @pytest.mark.parametrize("m", [0, 1, 2, 3])
    def test_continuity_psi(self, m):
        kwargs = dict(m=m, rho_m=1.0, r1=1.0, r2=3.0, gamma=1.0)
        if m == 0:
            kwargs['r_ref'] = 5.0
        solver = PoissonPolar2D(**kwargs)

        eps = 1e-10
        for r_bnd in [1.0, 3.0]:
            left = solver.psi_m(r_bnd - eps)
            right = solver.psi_m(r_bnd + eps)
            assert_allclose(left, right, rtol=1e-6, atol=1e-12,
                            err_msg=f"Discontinuity at r={r_bnd}, m={m}")

    @pytest.mark.parametrize("m", [0, 1, 2, 3])
    def test_continuity_derivative(self, m):
        kwargs = dict(m=m, rho_m=1.0, r1=1.0, r2=3.0, gamma=1.0)
        if m == 0:
            kwargs['r_ref'] = 5.0
        solver = PoissonPolar2D(**kwargs)

        # Use centered differences offset slightly from the boundary
        # to avoid O(h) truncation error from one-sided stencils at
        # the piecewise junction.
        h = 1e-5
        for r_bnd in [1.0, 3.0]:
            eps = 1e-8
            # Centered derivative just below boundary
            dpsi_left = (solver.psi_m(r_bnd - eps + h) - solver.psi_m(r_bnd - eps - h)) / (2 * h)
            # Centered derivative just above boundary
            dpsi_right = (solver.psi_m(r_bnd + eps + h) - solver.psi_m(r_bnd + eps - h)) / (2 * h)
            assert_allclose(dpsi_left, dpsi_right, rtol=1e-2, atol=1e-6,
                            err_msg=f"Derivative discontinuity at r={r_bnd}, m={m}")


# ---------------------------------------------------------------------------
# G. Symmetry and far-field behavior
# ---------------------------------------------------------------------------

class TestFarFieldBehavior:
    """Category G: asymptotic and symmetry checks."""

    def test_m0_far_field_line_mass(self):
        """For r >> r2, psi_0 ~ -2*gamma*M*ln(r) + const."""
        solver = PoissonPolar2D(m=0, rho_m=1.0, r1=1.0, r2=2.0, gamma=1.0, r_ref=1.0)
        M = (2.0**2 - 1.0**2) / 2.0  # = 1.5

        r_far = np.array([100.0, 200.0, 500.0])
        psi_vals = solver.psi_m(r_far)

        # psi_0(r) = -4*pi*gamma*M*ln(r) + const
        # Fit slope: d(psi)/d(ln r) should be -4*pi*gamma*M = -6*pi
        ln_r = np.log(r_far)
        # Use finite differences on the log scale
        slope = (psi_vals[-1] - psi_vals[0]) / (ln_r[-1] - ln_r[0])
        assert_allclose(slope, -4.0 * np.pi * 1.0 * M, rtol=1e-8)

    def test_mneq0_far_field_decay(self):
        """For m != 0 and r >> r2, psi_m(r) ~ C * r^{-|m|}."""
        m = 2
        solver = PoissonPolar2D(m=m, rho_m=1.0, r1=1.0, r2=2.0, gamma=1.0)

        r1_far, r2_far = 100.0, 200.0
        psi1 = solver.psi_m(r1_far)
        psi2 = solver.psi_m(r2_far)

        # psi_m(r) = A * r^{-|m|} => psi2/psi1 = (r2/r1)^{-|m|}
        ratio = psi2 / psi1
        expected_ratio = (r2_far / r1_far)**(-abs(m))
        assert_allclose(ratio, expected_ratio, rtol=1e-8)

    def test_spatial_conversion(self):
        """to_spatial(r, phi) should equal psi_m(r) * exp(i*m*phi)."""
        m = 3
        solver = PoissonPolar2D(m=m, rho_m=1.0, r1=1.0, r2=2.0, gamma=1.0)

        r = 1.5
        phi = np.pi / 4.0
        spatial = solver.to_spatial(r, phi)
        expected = solver.psi_m(r) * np.exp(1j * m * phi)
        assert_allclose(spatial, expected, rtol=1e-14)

    def test_spatial_real_part_physical(self):
        """For a real density rho(r,phi) = rho_m*cos(m*phi), the physical
        potential is Re[psi_m(r)*exp(i*m*phi)] = psi_m(r)*cos(m*phi)
        when rho_m is real."""
        m = 2
        rho_m = 3.0
        solver = PoissonPolar2D(m=m, rho_m=rho_m, r1=1.0, r2=2.0, gamma=1.0)

        r = 1.5
        phi = np.pi / 3.0
        spatial = solver.to_spatial(r, phi)
        # For real rho_m, psi_m(r) is real, so Re(spatial) = psi_m(r)*cos(m*phi)
        expected_real = solver.psi_m(r) * np.cos(m * phi)
        assert_allclose(np.real(spatial), expected_real, rtol=1e-14)

    def test_linearity_in_rho(self):
        """Scaling rho_m by a factor should scale psi_m by the same factor."""
        s1 = PoissonPolar2D(m=1, rho_m=1.0, r1=1.0, r2=2.0, gamma=1.0)
        s2 = PoissonPolar2D(m=1, rho_m=3.5, r1=1.0, r2=2.0, gamma=1.0)
        r = np.linspace(0.1, 5.0, 50)
        assert_allclose(s2.psi_m(r), 3.5 * s1.psi_m(r), rtol=1e-14)

    def test_linearity_in_gamma(self):
        """Scaling gamma should scale psi_m proportionally."""
        s1 = PoissonPolar2D(m=1, rho_m=1.0, r1=1.0, r2=2.0, gamma=1.0)
        s2 = PoissonPolar2D(m=1, rho_m=1.0, r1=1.0, r2=2.0, gamma=2.0)
        r = np.linspace(0.1, 5.0, 50)
        assert_allclose(s2.psi_m(r), 2.0 * s1.psi_m(r), rtol=1e-14)


# ---------------------------------------------------------------------------
# H. Comparison with direct numerical integration of the 2D Green's function
# ---------------------------------------------------------------------------
# Compute psi(r,phi) by brute-force 2D numerical integration:
#   psi(r,phi) = 4*pi*gamma * int_{r1}^{r2} int_0^{2pi}
#       [-(1/(2*pi))*ln|R-R'|] * rho(r',phi') * r' dphi' dr'
# with rho(r',phi') = rho_m * exp(i*m*phi')
# and |R-R'| = sqrt(r^2 + r'^2 - 2*r*r'*cos(phi - phi'))
#
# This integration is completely independent of the Fourier decomposition.

class TestGreensFunctionNumericalIntegration:
    """Category H: compare modal result with direct 2D numerical integration."""

    def _numerical_potential(self, r, phi, m, rho_m, r1, r2, gamma):
        """Compute psi(r, phi) by direct 2D numerical integration."""
        def integrand_real(phi_prime, r_prime):
            dist_sq = r**2 + r_prime**2 - 2.0 * r * r_prime * np.cos(phi - phi_prime)
            dist_sq = max(dist_sq, 1e-30)  # avoid log(0)
            log_dist = 0.5 * np.log(dist_sq)
            density = rho_m * np.exp(1j * m * phi_prime)
            kernel = -(1.0 / (2.0 * np.pi)) * log_dist
            return np.real(4.0 * np.pi * gamma * kernel * density * r_prime)

        def integrand_imag(phi_prime, r_prime):
            dist_sq = r**2 + r_prime**2 - 2.0 * r * r_prime * np.cos(phi - phi_prime)
            dist_sq = max(dist_sq, 1e-30)
            log_dist = 0.5 * np.log(dist_sq)
            density = rho_m * np.exp(1j * m * phi_prime)
            kernel = -(1.0 / (2.0 * np.pi)) * log_dist
            return np.imag(4.0 * np.pi * gamma * kernel * density * r_prime)

        real_part, _ = integrate.dblquad(
            integrand_real, r1, r2, 0.0, 2.0 * np.pi,
            epsabs=1e-10, epsrel=1e-10
        )
        imag_part, _ = integrate.dblquad(
            integrand_imag, r1, r2, 0.0, 2.0 * np.pi,
            epsabs=1e-10, epsrel=1e-10
        )
        return real_part + 1j * imag_part

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_mneq0_vs_numerical(self, m):
        rho_m = 1.0
        r1, r2, gamma = 1.0, 2.0, 1.0
        solver = PoissonPolar2D(m=m, rho_m=rho_m, r1=r1, r2=r2, gamma=gamma)

        # Test at several (r, phi) points outside the shell to avoid singularity.
        test_cases = [
            (0.5, 0.0),
            (0.5, np.pi / 4.0),
            (3.0, np.pi / 3.0),
            (3.0, np.pi),
        ]
        for r, phi in test_cases:
            numerical = self._numerical_potential(r, phi, m, rho_m, r1, r2, gamma)
            modal = solver.to_spatial(r, phi)
            assert_allclose(modal, numerical, rtol=1e-6, atol=1e-10,
                            err_msg=f"Mismatch at r={r}, phi={phi}, m={m}")

    def test_m0_vs_numerical(self):
        """For m=0, the numerical integral gives the potential up to a constant.
        We compare differences to eliminate the gauge ambiguity."""
        rho_m = 1.0
        r1, r2, gamma = 1.0, 2.0, 1.0
        solver = PoissonPolar2D(m=0, rho_m=rho_m, r1=r1, r2=r2, gamma=gamma, r_ref=5.0)

        # For m=0, rho = rho_m (constant in phi), and the phi integral of
        # ln|R-R'| can be done analytically:
        #   int_0^{2pi} ln|R-R'| dphi' = 2*pi*ln(r_>)
        # so the numerical integration should give:
        #   psi(r) = 4*pi*gamma * int_{r1}^{r2} [-(1/2pi)*ln(r_>)] * rho_m * r' * 2pi * (1/2pi) dr'
        # Wait, let's just compute numerically for safety.

        r_vals = [0.5, 1.5, 3.0, 5.0]
        phi = 0.0  # m=0 is phi-independent

        numerical_vals = []
        for r in r_vals:
            val = self._numerical_potential(r, phi, 0, rho_m, r1, r2, gamma)
            numerical_vals.append(np.real(val))

        modal_vals = [np.real(solver.to_spatial(r, phi)) for r in r_vals]

        # Compare differences (eliminates gauge constant).
        for i in range(1, len(r_vals)):
            num_diff = numerical_vals[i] - numerical_vals[0]
            mod_diff = modal_vals[i] - modal_vals[0]
            assert_allclose(mod_diff, num_diff, rtol=1e-6, atol=1e-10,
                            err_msg=f"m=0 difference mismatch at r={r_vals[i]}")


# ---------------------------------------------------------------------------
# Additional edge case: m=1 with complex rho_m
# ---------------------------------------------------------------------------
class TestComplexDensity:
    """Verify that complex density coefficients are handled correctly."""

    def test_complex_rho_m1(self):
        rho_m = 1.0 + 2.0j
        s_real = PoissonPolar2D(m=1, rho_m=1.0, r1=1.0, r2=2.0, gamma=1.0)
        s_imag = PoissonPolar2D(m=1, rho_m=2.0, r1=1.0, r2=2.0, gamma=1.0)
        s_complex = PoissonPolar2D(m=1, rho_m=rho_m, r1=1.0, r2=2.0, gamma=1.0)

        r = np.linspace(0.1, 5.0, 20)
        # By linearity: psi_m with rho_m = 1+2j should equal
        # psi_m(rho=1) + j * psi_m(rho=2)/2 * 2... No:
        # psi_m is linear in rho_m, so psi(rho_m = a+bj) = a*psi(1) + b*j*psi(1)
        # Actually psi_m ~ rho_m * (integral), so psi_complex = rho_m * (psi_unit)
        # where psi_unit is the potential with rho_m=1.
        psi_unit = s_real.psi_m(r)
        expected = rho_m * psi_unit
        result = s_complex.psi_m(r)
        assert_allclose(result, expected, rtol=1e-14)


# ---------------------------------------------------------------------------
# Supplementary: Poisson equation inside the shell (Laplacian = source)
# ---------------------------------------------------------------------------
class TestLaplacianInsideShell:
    """Verify that the Laplacian inside the shell equals the source term."""

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_laplacian_equals_source(self, m):
        rho_m = 1.0
        gamma = 1.0
        r1, r2 = 1.0, 3.0
        solver = PoissonPolar2D(m=m, rho_m=rho_m, r1=r1, r2=r2, gamma=gamma)

        h = 1e-5
        # Points strictly inside the shell
        for r in [1.5, 2.0, 2.5]:
            psi_minus = solver.psi_m(r - h)
            psi_center = solver.psi_m(r)
            psi_plus = solver.psi_m(r + h)

            d2psi = (psi_plus - 2.0 * psi_center + psi_minus) / h**2
            dpsi = (psi_plus - psi_minus) / (2.0 * h)

            # The modal Laplacian: psi'' + (1/r)*psi' - m^2/r^2 * psi
            # should equal -4*pi*gamma*rho_m (the source for this mode,
            # noting the azimuthal integral of e^{im phi} * e^{-im phi} = 2pi,
            # and the factor 1/(2pi) from the Fourier transform convention
            # means the radial ODE source is -4*pi*gamma*rho_m).
            # Actually the ODE satisfied by psi_m is:
            #   psi_m'' + (1/r)*psi_m' - m^2/r^2 * psi_m = -4*pi*gamma*rho_m
            laplacian_mode = d2psi + dpsi / r - m**2 * psi_center / r**2
            expected_source = -4.0 * np.pi * gamma * rho_m
            assert_allclose(laplacian_mode, expected_source, rtol=1e-3, atol=1e-6,
                            err_msg=f"Source mismatch at r={r}, m={m}")

    def test_laplacian_m0_inside(self):
        rho_m = 1.0
        gamma = 1.0
        r1, r2 = 1.0, 3.0
        solver = PoissonPolar2D(m=0, rho_m=rho_m, r1=r1, r2=r2, gamma=gamma, r_ref=5.0)

        h = 1e-5
        for r in [1.5, 2.0, 2.5]:
            psi_minus = solver.psi_m(r - h)
            psi_center = solver.psi_m(r)
            psi_plus = solver.psi_m(r + h)

            d2psi = (psi_plus - 2.0 * psi_center + psi_minus) / h**2
            dpsi = (psi_plus - psi_minus) / (2.0 * h)

            laplacian_mode = d2psi + dpsi / r
            expected_source = -4.0 * np.pi * gamma * rho_m
            assert_allclose(laplacian_mode, expected_source, rtol=1e-3, atol=1e-6,
                            err_msg=f"Source mismatch at r={r}, m=0")


# ---------------------------------------------------------------------------
# Superposition: sum of two non-overlapping shells
# ---------------------------------------------------------------------------
class TestSuperposition:
    """Two non-overlapping shells should give potential = sum of individual potentials."""

    def test_two_shells_m1(self):
        m = 1
        gamma = 1.0
        rho_m = 1.0

        s1 = PoissonPolar2D(m=m, rho_m=rho_m, r1=1.0, r2=2.0, gamma=gamma)
        s2 = PoissonPolar2D(m=m, rho_m=rho_m, r1=3.0, r2=4.0, gamma=gamma)

        # Test outside both shells
        for r in [0.5, 5.0, 10.0]:
            total = s1.psi_m(r) + s2.psi_m(r)
            # For r=0.5 (below both shells), prefactor 2*pi*gamma/|m| = 2*pi:
            #   s1: 2*pi * r^1 * int_1^2 dr' = 2*pi * 0.5
            #   s2: 2*pi * r^1 * int_3^4 dr' = 2*pi * 0.5
            #   total = 2*pi
            if r == 0.5:
                expected = 2 * np.pi * (0.5 * (2.0 - 1.0) + 0.5 * (4.0 - 3.0))
                assert_allclose(total, expected, rtol=1e-12)

        # For r=5 (above both shells):
        # s1: 2*pi * r^{-1} * int_1^2 r'^2 dr' = 2*pi*(1/5)*(7/3)
        # s2: 2*pi * r^{-1} * int_3^4 r'^2 dr' = 2*pi*(1/5)*(37/3)
        r = 5.0
        total = s1.psi_m(r) + s2.psi_m(r)
        expected = 2 * np.pi * ((1.0 / 5.0) * (7.0 / 3.0) + (1.0 / 5.0) * (37.0 / 3.0))
        assert_allclose(total, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
class TestInputValidation:
    """Verify that invalid inputs are rejected."""

    def test_r1_negative(self):
        with pytest.raises(ValueError, match="positive"):
            PoissonPolar2D(m=1, rho_m=1.0, r1=-1.0, r2=2.0, gamma=1.0)

    def test_r1_equals_r2(self):
        with pytest.raises(ValueError, match="r1 < r2"):
            PoissonPolar2D(m=1, rho_m=1.0, r1=2.0, r2=2.0, gamma=1.0)

    def test_r1_greater_than_r2(self):
        with pytest.raises(ValueError, match="r1 < r2"):
            PoissonPolar2D(m=1, rho_m=1.0, r1=3.0, r2=2.0, gamma=1.0)


# ---------------------------------------------------------------------------
# Boundary-exact evaluation
# ---------------------------------------------------------------------------
class TestBoundaryExactValues:
    """Verify psi_m at r = r1 and r = r2 matches from both piecewise branches."""

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_at_r1(self, m):
        solver = PoissonPolar2D(m=m, rho_m=1.0, r1=1.0, r2=3.0, gamma=1.0)
        # At r = r1, the inner integral is zero, so psi_m(r1) = prefactor * r1^|m| * full_outer
        # This is the same expression from either the r<=r1 or middle branch.
        val = solver.psi_m(1.0)
        val_just_above = solver.psi_m(1.0 + 1e-14)
        assert_allclose(val, val_just_above, rtol=1e-10)

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_at_r2(self, m):
        solver = PoissonPolar2D(m=m, rho_m=1.0, r1=1.0, r2=3.0, gamma=1.0)
        val = solver.psi_m(3.0)
        val_just_below = solver.psi_m(3.0 - 1e-14)
        assert_allclose(val, val_just_below, rtol=1e-10)

    def test_at_boundaries_m0(self):
        solver = PoissonPolar2D(m=0, rho_m=1.0, r1=1.0, r2=3.0, gamma=1.0, r_ref=5.0)
        for r_bnd in [1.0, 3.0]:
            val = solver.psi_m(r_bnd)
            val_near = solver.psi_m(r_bnd + 1e-14)
            assert_allclose(val, val_near, rtol=1e-10)


# ---------------------------------------------------------------------------
# rho_to_spatial
# ---------------------------------------------------------------------------
class TestRhoToSpatial:
    """Verify the spatial density reconstruction."""

    def test_inside_shell(self):
        m, rho_m = 2, 3.0
        solver = PoissonPolar2D(m=m, rho_m=rho_m, r1=1.0, r2=3.0, gamma=1.0)
        r, phi = 2.0, np.pi / 4
        result = solver.rho_to_spatial(r, phi)
        expected = rho_m * np.exp(1j * m * phi)
        assert_allclose(result, expected, rtol=1e-14)

    def test_outside_shell(self):
        solver = PoissonPolar2D(m=2, rho_m=3.0, r1=1.0, r2=3.0, gamma=1.0)
        # Outside the shell, density is zero
        for r in [0.5, 4.0]:
            result = solver.rho_to_spatial(r, np.pi / 3)
            assert_allclose(result, 0.0, atol=1e-16)

    def test_at_boundaries(self):
        solver = PoissonPolar2D(m=1, rho_m=1.0, r1=1.0, r2=3.0, gamma=1.0)
        phi = 0.0
        # At r1 and r2, density should be nonzero (closed interval)
        for r in [1.0, 3.0]:
            result = solver.rho_to_spatial(r, phi)
            assert_allclose(result, np.exp(1j * phi), rtol=1e-14)

    def test_array_input(self):
        solver = PoissonPolar2D(m=1, rho_m=2.0, r1=1.0, r2=3.0, gamma=1.0)
        r = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        phi = 0.0
        result = solver.rho_to_spatial(r, phi)
        expected = np.array([0.0, 2.0, 2.0, 2.0, 0.0])
        assert_allclose(result, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# m=0 without gauge reference
# ---------------------------------------------------------------------------
class TestM0NoGauge:
    """Verify m=0 works without r_ref (gauge_offset=0)."""

    def test_no_rref_produces_valid_output(self):
        solver = PoissonPolar2D(m=0, rho_m=1.0, r1=1.0, r2=2.0, gamma=1.0)
        # Should not raise, and differences should be gauge-independent
        r = np.linspace(0.1, 5.0, 20)
        result = solver.psi_m(r)
        assert result.shape == (20,)
        assert np.all(np.isfinite(result))

    def test_no_rref_vs_rref_differ_by_constant(self):
        s_no_ref = PoissonPolar2D(m=0, rho_m=1.0, r1=1.0, r2=2.0, gamma=1.0)
        s_with_ref = PoissonPolar2D(m=0, rho_m=1.0, r1=1.0, r2=2.0, gamma=1.0, r_ref=3.0)
        r = np.linspace(0.1, 10.0, 50)
        diff = s_no_ref.psi_m(r) - s_with_ref.psi_m(r)
        assert_allclose(diff, diff[0], rtol=1e-12)
