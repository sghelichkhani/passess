## 3D Green's function solution of the gravitational Poisson problem in spherical coordinates

In this document we derive the analytical solution to the three-dimensional gravitational Poisson equation in spherical coordinates $(r, \theta, \phi)$. The density field $\rho(r, \theta, \phi)$ is decomposed into spherical harmonics in the angular directions, reducing the problem to a one-dimensional radial Green's function for each harmonic degree $l$. This is the standard multipole expansion approach (Arfken & Weber, Section 9.7; Jackson, Chapter 3).

The gravitational potential $\psi(\mathbf{r})$ satisfies

$$ \nabla^2\psi(\mathbf{r}) = -4\pi G\,\rho(\mathbf{r}), $$

where $G$ is Newton's gravitational constant and $\rho(\mathbf{r})$ is the mass density distribution. The gravitational acceleration is obtained via $\vec{g} = -\nabla\psi$.


### The Laplacian in spherical coordinates

In spherical coordinates $(r, \theta, \phi)$, the Laplacian takes the form

$$ \nabla^2 = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial}{\partial r}\right) + \frac{1}{r^2\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial}{\partial\theta}\right) + \frac{1}{r^2\sin^2\theta}\frac{\partial^2}{\partial\phi^2}. $$

The angular part is the (negative of the) angular momentum operator, whose eigenfunctions are the spherical harmonics $Y_l^m(\theta,\phi)$:

$$ \left[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial}{\partial\theta}\right) + \frac{1}{\sin^2\theta}\frac{\partial^2}{\partial\phi^2}\right]Y_l^m(\theta,\phi) = -l(l+1)\,Y_l^m(\theta,\phi). $$

This eigenvalue relation is what makes the spherical harmonic expansion so powerful: it converts the angular part of the Laplacian into a simple multiplicative factor.


### 3D Green's function and the Dirac delta in spherical coordinates

We define the Green's function by

$$ \nabla^2 G(\mathbf{r}_1, \mathbf{r}_2) = -\delta(\mathbf{r}_1 - \mathbf{r}_2), $$

so that the solution of the Poisson equation is given by the superposition integral

$$ \psi(\mathbf{r}_1) = 4\pi G\int G(\mathbf{r}_1, \mathbf{r}_2)\,\rho(\mathbf{r}_2)\,d\tau_2, \qquad d\tau_2 = r_2^2\sin\theta_2\,dr_2\,d\theta_2\,d\phi_2. $$

In three-dimensional free space, the Green's function is (Arfken & Weber, Eq. 9.175)

$$ G(\mathbf{r}_1, \mathbf{r}_2) = \frac{1}{4\pi\,|\mathbf{r}_1 - \mathbf{r}_2|}. $$

Our goal is to expand this in spherical harmonics, yielding a mode-by-mode radial Green's function.


### Expansion of the Dirac delta function in spherical harmonics

The three-dimensional Dirac delta function in spherical coordinates factors as (Arfken & Weber, Eq. 9.182)

$$ \delta(\mathbf{r}_1 - \mathbf{r}_2) = \frac{1}{r_1^2}\,\delta(r_1 - r_2)\,\delta(\cos\theta_1 - \cos\theta_2)\,\delta(\phi_1 - \phi_2). $$

The angular delta functions can be expanded using the completeness relation of the spherical harmonics. The spherical harmonics satisfy the completeness relation (Arfken & Weber, Exercises 1.15.11 and 12.6.6):

$$ \delta(\cos\theta_1 - \cos\theta_2)\,\delta(\phi_1 - \phi_2) = \sum_{l=0}^{\infty}\sum_{m=-l}^{l} Y_l^m(\theta_1,\phi_1)\,Y_l^{m*}(\theta_2,\phi_2). $$

Combining these, the full delta function becomes

$$ \delta(\mathbf{r}_1 - \mathbf{r}_2) = \frac{1}{r_1^2}\,\delta(r_1 - r_2)\sum_{l=0}^{\infty}\sum_{m=-l}^{l}Y_l^m(\theta_1,\phi_1)\,Y_l^{m*}(\theta_2,\phi_2). $$


### Spherical harmonic expansion of the Green's function

We expand the Green's function in spherical harmonics (Arfken & Weber, Eq. 9.181):

$$ G(\mathbf{r}_1, \mathbf{r}_2) = \sum_{l=0}^{\infty}\sum_{m=-l}^{l}g_l(r_1, r_2)\,Y_l^m(\theta_1,\phi_1)\,Y_l^{m*}(\theta_2,\phi_2). $$

The key point is that the summation index $l$ is the same for both spherical harmonics. This is a consequence of the symmetry of the Green's function: a mass anomaly of degree $l$ can only generate a potential of degree $l$. The radial functions $g_l(r_1, r_2)$ are independent of $m$ (they depend only on the degree $l$), because the radial equation that determines them does not involve $m$.


### Derivation of the radial ODE for each harmonic degree

Substituting the spherical harmonic expansions of both $G$ and $\delta$ into $\nabla^2 G = -\delta$, we apply the Laplacian. The angular part of $\nabla^2$ acting on $Y_l^m$ produces $-l(l+1)/r_1^2$ times $Y_l^m$ (from the eigenvalue relation above). The radial part of $\nabla^2$ acts only on $g_l(r_1, r_2)$. Using the orthogonality of the spherical harmonics (integrating both sides against $Y_l^{m*}(\theta_1,\phi_1)$ over the sphere), each $(l,m)$ mode decouples and we obtain:

$$ \frac{1}{r_1^2}\frac{d}{dr_1}\left(r_1^2\frac{dg_l}{dr_1}\right) - \frac{l(l+1)}{r_1^2}\,g_l(r_1,r_2) = -\frac{1}{r_1^2}\,\delta(r_1 - r_2). $$

Multiplying both sides by $r_1^2$:

$$ \boxed{ \frac{d}{dr_1}\left(r_1^2\frac{dg_l}{dr_1}\right) - l(l+1)\,g_l(r_1,r_2) = -\delta(r_1 - r_2). } \tag{1} $$

This can also be written as (Arfken & Weber, Eq. 9.183)

$$ r_1\frac{d^2}{dr_1^2}\left[r_1\,g_l(r_1,r_2)\right] - l(l+1)\,g_l(r_1,r_2) = -\delta(r_1 - r_2). $$

To see the equivalence, define $u_l(r_1) = r_1\,g_l(r_1,r_2)$. Then $g_l = u_l/r_1$ and

$$ \frac{d}{dr_1}\left(r_1^2\frac{dg_l}{dr_1}\right) = \frac{d}{dr_1}\left(r_1^2\frac{d}{dr_1}\frac{u_l}{r_1}\right) = \frac{d}{dr_1}\left(r_1\,u_l' - u_l\right) = r_1\,u_l'' + u_l' - u_l' = r_1\,u_l'', $$

confirming that Eq. (1) is equivalent to $r_1\,u_l'' - l(l+1)\,u_l/r_1 = -\delta(r_1 - r_2)/1$, i.e., $r_1\,u_l'' - l(l+1)\,g_l = -\delta$.

This is now a one-dimensional problem in the radial variable $r_1$.


### Construction of the radial Green's function

**Homogeneous solutions.** For $r_1 \neq r_2$, the right-hand side of Eq. (1) vanishes. We seek solutions of

$$ \frac{d}{dr_1}\left(r_1^2\frac{dg_l}{dr_1}\right) = l(l+1)\,g_l. $$

Trying $g_l = r_1^n$, we get $n(n+1) = l(l+1)$, which gives $n = l$ or $n = -(l+1)$. So the two independent homogeneous solutions are $r_1^l$ and $r_1^{-(l+1)}$.

**Boundary conditions.** We impose free-space conditions:
- Regularity at $r_1 = 0$: the solution must remain finite as $r_1 \to 0$, so for the interior branch ($r_1 < r_2$) we choose $r_1^l$.
- Decay at $r_1 \to \infty$: the solution must vanish at infinity, so for the exterior branch ($r_1 > r_2$) we choose $r_1^{-(l+1)}$.

Thus we write

$$ g_l(r_1, r_2) = \begin{cases} A\,r_1^l, & r_1 < r_2, \\ B\,r_1^{-(l+1)}, & r_1 > r_2. \end{cases} $$

**Continuity at $r_1 = r_2$.** The Green's function must be continuous:

$$ A\,r_2^l = B\,r_2^{-(l+1)}, $$

which gives

$$ B = A\,r_2^{2l+1}. \tag{2} $$

**Jump condition from the delta function.** Integrating Eq. (1) across a small interval $(r_2 - \varepsilon, r_2 + \varepsilon)$ and taking $\varepsilon \to 0$:

$$ \int_{r_2-\varepsilon}^{r_2+\varepsilon}\frac{d}{dr_1}\left(r_1^2\frac{dg_l}{dr_1}\right)dr_1 - l(l+1)\int_{r_2-\varepsilon}^{r_2+\varepsilon}g_l\,dr_1 = -1. $$

The second integral vanishes as $\varepsilon \to 0$ because $g_l$ is continuous and bounded near $r_2$. The first integral is a total derivative:

$$ \left[r_1^2\frac{dg_l}{dr_1}\right]_{r_1=r_2^-}^{r_1=r_2^+} = -1. $$

Computing the derivatives from each branch:

$$ r_1^2\frac{dg_l}{dr_1} = \begin{cases} l\,A\,r_1^{l+1}, & r_1 < r_2, \\ -(l+1)\,B\,r_1^{-l}, & r_1 > r_2. \end{cases} $$

Evaluating the jump at $r_1 = r_2$:

$$ -(l+1)\,B\,r_2^{-l} - l\,A\,r_2^{l+1} = -1. $$

Using the continuity relation (2), $B\,r_2^{-l} = A\,r_2^{2l+1}\,r_2^{-l} = A\,r_2^{l+1}$. Substituting:

$$ -(l+1)\,A\,r_2^{l+1} - l\,A\,r_2^{l+1} = -A\,r_2^{l+1}\left[(l+1) + l\right] = -(2l+1)\,A\,r_2^{l+1} = -1. $$

Solving for $A$:

$$ A = \frac{1}{(2l+1)\,r_2^{l+1}}. $$

Substituting back:

- For $r_1 < r_2$: $g_l = \frac{1}{2l+1}\,\frac{r_1^l}{r_2^{l+1}}$.
- For $r_1 > r_2$: From (2), $B = \frac{r_2^{2l+1}}{(2l+1)\,r_2^{l+1}} = \frac{r_2^l}{2l+1}$, so $g_l = \frac{1}{2l+1}\,\frac{r_2^l}{r_1^{l+1}}$.

Both cases combine into:

$$ \boxed{ g_l(r_1, r_2) = \frac{1}{2l+1}\,\frac{r_<^l}{r_>^{l+1}}, } \tag{3} $$

where $r_< = \min(r_1, r_2)$ and $r_> = \max(r_1, r_2)$.

This is Eq. (9.185) in Arfken & Weber.


### Complete Green's function in spherical harmonic form

Assembling the radial Green's function with the angular expansion:

$$ \boxed{ G(\mathbf{r}_1, \mathbf{r}_2) = \sum_{l=0}^{\infty}\sum_{m=-l}^{l}\frac{1}{2l+1}\,\frac{r_<^l}{r_>^{l+1}}\,Y_l^m(\theta_1,\phi_1)\,Y_l^{m*}(\theta_2,\phi_2). } \tag{4} $$

Since we already know from the free-space result that $G = 1/(4\pi|\mathbf{r}_1 - \mathbf{r}_2|)$, Eq. (4) provides the well-known expansion (Arfken & Weber, Eq. 9.187):

$$ \frac{1}{4\pi}\,\frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|} = \sum_{l=0}^{\infty}\sum_{m=-l}^{l}\frac{1}{2l+1}\,\frac{r_<^l}{r_>^{l+1}}\,Y_l^m(\theta_1,\phi_1)\,Y_l^{m*}(\theta_2,\phi_2). $$

This can also be connected to the Legendre polynomial addition theorem (Arfken & Weber, Eq. 9.189):

$$ P_l(\cos\gamma) = \frac{4\pi}{2l+1}\sum_{m=-l}^{l}Y_l^m(\theta_1,\phi_1)\,Y_l^{m*}(\theta_2,\phi_2), $$

where $\gamma$ is the angle between the vectors $\mathbf{r}_1$ and $\mathbf{r}_2$, to recover the standard generating function expansion $1/|\mathbf{r}_1 - \mathbf{r}_2| = \sum_l (r_<^l/r_>^{l+1})\,P_l(\cos\gamma)$.


## Solution for a general mass anomaly and its spherical harmonic coefficients

### Spherical harmonic expansion of the density

We expand the density $\rho(\mathbf{r})$ in spherical harmonics:

$$ \rho(r,\theta,\phi) = \sum_{l=0}^{\infty}\sum_{m=-l}^{l}\rho_{lm}(r)\,Y_l^m(\theta,\phi), $$

where the spherical harmonic coefficients of the density are

$$ \rho_{lm}(r) = \int_0^{2\pi}\int_0^{\pi}\rho(r,\theta,\phi)\,Y_l^{m*}(\theta,\phi)\,\sin\theta\,d\theta\,d\phi. $$

The integers $(l,m)$ are the degree and order of the spherical harmonic, the 3D analog of the azimuthal wavenumber $m$ in the 2D polar case.

### The potential inherits the same spherical harmonic structure

Write

$$ \psi(r,\theta,\phi) = \sum_{l=0}^{\infty}\sum_{m=-l}^{l}\psi_{lm}(r)\,Y_l^m(\theta,\phi). $$

Inserting the Green's function representation into

$$ \psi(\mathbf{r}_1) = 4\pi G\int G(\mathbf{r}_1, \mathbf{r}_2)\,\rho(\mathbf{r}_2)\,d\tau_2, $$

and substituting the spherical harmonic expansions of both $G$ and $\rho$:

$$ \psi(\mathbf{r}_1) = 4\pi G\int r_2^2\,dr_2\int\sin\theta_2\,d\theta_2\,d\phi_2 \left[\sum_{l,m}g_l(r_1,r_2)\,Y_l^m(\theta_1,\phi_1)\,Y_l^{m*}(\theta_2,\phi_2)\right]\left[\sum_{l',m'}\rho_{l'm'}(r_2)\,Y_{l'}^{m'}(\theta_2,\phi_2)\right]. $$

The angular integration uses the orthonormality of the spherical harmonics:

$$ \int Y_l^{m*}(\theta_2,\phi_2)\,Y_{l'}^{m'}(\theta_2,\phi_2)\,\sin\theta_2\,d\theta_2\,d\phi_2 = \delta_{ll'}\delta_{mm'}. $$

This collapses the double sum to a single sum, giving:

$$ \psi(\mathbf{r}_1) = 4\pi G\sum_{l=0}^{\infty}\sum_{m=-l}^{l}Y_l^m(\theta_1,\phi_1)\int_0^{\infty}g_l(r_1,r_2)\,\rho_{lm}(r_2)\,r_2^2\,dr_2. $$

Reading off the spherical harmonic coefficient of $\psi$:

$$ \psi_{lm}(r_1) = 4\pi G\int_0^{\infty}g_l(r_1,r_2)\,\rho_{lm}(r_2)\,r_2^2\,dr_2. $$

Substituting the radial Green's function from Eq. (3) and splitting the integral at $r_2 = r_1$:

$$ \psi_{lm}(r) = 4\pi G\left[\int_0^{r}\frac{1}{2l+1}\,\frac{r_2^l}{r^{l+1}}\,\rho_{lm}(r_2)\,r_2^2\,dr_2 + \int_r^{\infty}\frac{1}{2l+1}\,\frac{r^l}{r_2^{l+1}}\,\rho_{lm}(r_2)\,r_2^2\,dr_2\right]. $$

Factoring out the $r$-dependent terms:

$$ \boxed{ \psi_{lm}(r) = \frac{4\pi G}{2l+1}\left[\frac{1}{r^{l+1}}\int_0^{r}\rho_{lm}(r')\,r'^{l+2}\,dr' + r^l\int_r^{\infty}\rho_{lm}(r')\,r'^{1-l}\,dr'\right]. } \tag{5} $$

This is the **multipole expansion** of the gravitational potential (Arfken & Weber, Eq. 9.188). The first term represents the contribution from sources **interior** to the observation radius ($r' < r$), which falls off as $r^{-(l+1)}$ — the familiar multipole decay. The second term represents contributions from sources **exterior** to the observation radius ($r' > r$), which grows as $r^l$ from the inside.

If the density has compact support on a shell $a \le r' \le b$, simply replace $(0, \infty)$ by $(a, b)$ in the integrals.

### Structural parallel across all formulations

The result has the same structure as the 2D cases:

| Formulation | Spectral basis | Prefactor | Interior kernel | Exterior kernel |
|---|---|---|---|---|
| 2D polar ($m \neq 0$) | $e^{im\phi}$ | $\gamma/\|m\|$ | $r^{-\|m\|}\,r'^{\|m\|+1}$ | $r^{\|m\|}\,r'^{1-\|m\|}$ |
| 2D Cartesian ($k \neq 0$) | $e^{ikx}$ | $2\pi\gamma/\|k\|$ | $e^{-\|k\|z}\,e^{\|k\|z'}$ | $e^{\|k\|z}\,e^{-\|k\|z'}$ |
| 3D spherical | $Y_l^m(\theta,\phi)$ | $4\pi G/(2l+1)$ | $r^{-(l+1)}\,r'^{l+2}$ | $r^l\,r'^{1-l}$ |

In every case: spectral decomposition in the lateral/angular directions, splitting of the radial/vertical integral at the observation point, and the product of "growing" and "decaying" solutions of the corresponding homogeneous ODE.


### Discrete shell approximation

For a density distribution given on discrete radial shells at radii $r_i$ with thicknesses $\Delta h_i$:

$$ \psi_{lm}(r) \approx \frac{4\pi G}{2l+1}\left[\frac{1}{r^{l+1}}\sum_{r_i < r}\rho_{lm}(r_i)\,r_i^{l+2}\,\Delta h_i + r^l\sum_{r_i \ge r}\rho_{lm}(r_i)\,r_i^{1-l}\,\Delta h_i\right]. $$

This can be rearranged as

$$ \psi_{lm}(r) \approx \frac{4\pi G}{2l+1}\left[\sum_{r_i < r}\left(\frac{r_i}{r}\right)^{l+1}r_i\,\rho_{lm}(r_i)\,\Delta h_i + \sum_{r_i \ge r}\left(\frac{r}{r_i}\right)^l r_i\,\rho_{lm}(r_i)\,\Delta h_i\right], $$

which makes the attenuation factors $(r_i/r)^{l+1}$ and $(r/r_i)^l$ explicit. Higher harmonic degrees $l$ are attenuated more strongly with radial separation, just as higher wavenumbers $|k|$ are attenuated more strongly with depth in the Cartesian case.
