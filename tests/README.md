# Test suite for passess

## Philosophy

The tests are designed to verify mathematical correctness from first principles, independently of the implementation. Rather than testing that the code reproduces its own formulas, the suite attacks the problem from multiple independent angles: hand-computed reference values, the governing PDE itself, and brute-force numerical integration of the full Green's function. If the code passes all three kinds of checks simultaneously, the underlying math must be right.

This approach proved its value early on: the physics-based tests (categories E and H) caught a missing factor of 2pi in the original derivation document, while the hand-computed tests (categories A through D) passed because they were derived from the same flawed formula. The discrepancy between the two groups pinpointed the error.

## Test categories

### A. Direct integral verification (m != 0, generic)

Pick concrete parameter values (m, rho_m, r1, r2, gamma) and evaluate the closed-form integrals by hand at points in each of the three radial regions (r < r1, r1 <= r <= r2, r > r2). Every intermediate arithmetic step is shown in the test comments so the expected values can be checked on paper.

Two parameter sets are tested: m=1 with unit parameters (TestDirectIntegralM1) and m=3 with non-trivial rho_m and gamma (TestDirectIntegralM3). This ensures the prefactor gamma/|m| scales correctly and the power-law exponents are right for different mode numbers.

### B. Special exponent case |m| = 2

When |m| = 2, the outer integral exponent 1 - |m| = -1 produces a logarithm instead of a power law. This is the only value of m that triggers a qualitatively different code path in the outer integral. The tests (TestLogExponentM2) verify all three radial regions with hand-computed values that include ln terms.

A companion test (TestNegativeModeSymmetry) confirms that m = -2 gives identical radial profiles to m = +2, since psi_m depends only on |m|.

### C. m = 0 mode (axisymmetric)

The m = 0 case uses an entirely different code path involving logarithmic integrals (int r' ln(r') dr') and gauge fixing. TestM0Mode verifies:

- The outer region (r > r2) gives the expected -4 pi gamma M ln(r) behavior.
- The inner region (r < r1) is constant (r-independent), consistent with the 2D result that the potential inside a cylindrical shell is uniform.
- The interior region (r1 < r < r2) matches the split integral formula.
- The gauge reference point gives exactly zero.
- Two different gauge choices differ by a constant everywhere.

### D. Delta-function (thin shell) limit

A shell of width dr = 1e-6 should approximate a delta-function source. The expected potential is 8 pi^2 gamma g_m(r, rp) rho_m rp dr, where g_m is the radial Green's function. TestDeltaFunctionLimit verifies this for multiple m values (including m = 0 with gauge) at several evaluation points, using rtol = 1e-4 to account for the finite shell width.

This test connects the finite-shell formulas back to the Green's function that was derived in the theory document.

### E. Laplacian verification (PDE check)

The most important category. These tests do not use the analytical formulas at all. Instead, they compute the modal Laplacian psi_m'' + (1/r) psi_m' - m^2/r^2 psi_m numerically via central finite differences and check it against the Poisson equation:

- **Outside the shell** (TestLaplacianOutsideShell): the Laplacian must vanish (homogeneous equation). This is checked for m = 1, 2, 3 and m = 0.
- **Inside the shell** (TestLaplacianInsideShell): the Laplacian must equal -4 pi gamma rho_m (the source term). This is the test that caught the missing 2pi factor in the derivation document, when the computed Laplacian was -2 gamma rho_m instead of -4 pi gamma rho_m.

The finite-difference step size is chosen to balance truncation error against floating-point cancellation. For m = 0 outside the shell, a larger step (h = 1e-4) is needed because the function values are O(100), causing catastrophic cancellation in the second-derivative stencil at smaller h.

### F. Continuity at shell boundaries

The potential psi_m(r) is defined piecewise across three radial regions. TestContinuityAtBoundaries verifies that both psi_m and its radial derivative are continuous at r = r1 and r = r2. Derivative continuity is expected because the source is an extended shell (not a delta function, which would produce a derivative jump).

Centered finite differences offset slightly from each boundary are used to evaluate the derivative from both sides, avoiding the truncation-error issues that arise from one-sided stencils at the piecewise junction.

### G. Symmetry and far-field behavior

TestFarFieldBehavior checks asymptotic and structural properties:

- **m = 0 far field**: for r >> r2, psi_0(r) grows as -4 pi gamma M ln(r), verified by fitting the slope on a log scale at r = 100, 200, 500.
- **m != 0 far field**: the potential decays as r^{-|m|}, verified by checking the ratio psi(200)/psi(100) = (1/2)^{|m|}.
- **Spatial conversion**: to_spatial(r, phi) = psi_m(r) exp(i m phi).
- **Physical field**: for real rho_m, Re[to_spatial] = psi_m(r) cos(m phi).
- **Linearity**: scaling rho_m or gamma scales the potential proportionally.

### H. Numerical Green's function integration

The strongest independent check. TestGreensFunctionNumericalIntegration computes the potential by brute-force 2D numerical quadrature of the full spatial Green's function:

    psi(r, phi) = 4 pi gamma int int [-(1/2pi) ln|R - R'|] rho(r', phi') r' dphi' dr'

using scipy.integrate.dblquad. This computation is completely independent of the Fourier modal decomposition. It does not know about g_m, mode numbers, or power-law integrals. For m != 0 the complex spatial field is compared directly. For m = 0, potential differences are compared to eliminate the gauge ambiguity.

### Additional tests

- **TestComplexDensity**: complex rho_m is handled correctly via linearity (psi_m is proportional to rho_m, so complex input produces complex output).
- **TestSuperposition**: the potential from two non-overlapping shells equals the sum of individual potentials.
- **TestInputValidation**: invalid shell radii (r1 >= r2 or r1 <= 0) raise ValueError.
- **TestBoundaryExactValues**: psi_m evaluated exactly at r = r1 or r = r2 matches the value from infinitesimally nearby points, confirming the piecewise branches agree at their junctions.
- **TestRhoToSpatial**: the density reconstruction rho_m exp(i m phi) is nonzero inside the shell, zero outside, and correct at the boundaries.
- **TestM0NoGauge**: m = 0 without a reference radius produces finite output and differs from the gauged version by a constant.
