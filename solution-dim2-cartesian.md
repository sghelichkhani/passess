## 2D Green's function solution of the gravitational Poisson problem in Cartesian coordinates

In this document we derive the analytical solution to the two-dimensional gravitational Poisson equation in Cartesian coordinates $(x,z)$. The physical setup is that of an infinite cylinder: the density field is invariant in the out-of-plane direction (say $y$), so the three-dimensional problem reduces to two dimensions. The coordinate $x$ plays the role of the "lateral" (periodic/infinite) direction, while $z$ is the "vertical" (depth) direction along which the Green's function structure lives.

The gravitational potential $\psi(x,z)$ satisfies

$$ \nabla^2\psi(x,z) = -4\pi\gamma\,\rho(x,z), $$

where $\rho(x,z)$ is the mass anomaly per unit out-of-plane length and $\gamma$ is the gravitational constant. The 2D Laplacian in Cartesian coordinates is simply

$$ \nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial z^2}. $$

As in the polar-coordinate formulation, the potential is defined up to an arbitrary additive constant, and only $-\nabla\psi = \vec{g}$ (the gravitational acceleration) is physically unique.


### 2D Green's function in Cartesian coordinates

We define the Green's function $G(\mathbf{R},\mathbf{R'})$ by

$$ \nabla^2 G(\mathbf{R},\mathbf{R'}) = -\delta(\mathbf{R} - \mathbf{R'}), $$

where $\mathbf{R} = (x,z)$ and $\mathbf{R'} = (x',z')$. The solution of the Poisson equation is then given by the superposition integral

$$ \psi(\mathbf{R}) = 4\pi\gamma \int_{\mathbb{R}^2} G(\mathbf{R},\mathbf{R'})\,\rho(\mathbf{R'})\,dA', \qquad dA' = dx'\,dz'. $$

In two-dimensional free space, the Green's function for the Laplacian is known (Arfken & Weber, Table 9.5) to be

$$ G(\mathbf{R},\mathbf{R'}) = -\frac{1}{2\pi}\ln|\mathbf{R} - \mathbf{R'}|, $$

where $|\mathbf{R} - \mathbf{R'}| = \sqrt{(x-x')^2 + (z-z')^2}$. Our goal is to decompose this Green's function spectrally in the $x$-direction, arriving at a one-dimensional Green's function problem in $z$.


### Fourier decomposition in the lateral direction

We expand the Green's function in Fourier modes along the $x$-direction:

$$ G(\mathbf{R},\mathbf{R'}) = \frac{1}{2\pi}\int_{-\infty}^{\infty} \hat{g}_k(z,z')\,e^{ik(x-x')}\,dk. $$

Simultaneously, the two-dimensional Dirac delta function in Cartesian coordinates separates as

$$ \delta(\mathbf{R} - \mathbf{R'}) = \delta(x - x')\,\delta(z - z'). $$

The lateral delta function has the standard Fourier representation

$$ \delta(x - x') = \frac{1}{2\pi}\int_{-\infty}^{\infty} e^{ik(x-x')}\,dk. $$

Therefore, the right-hand side of the Green's function equation becomes

$$ -\delta(\mathbf{R} - \mathbf{R'}) = -\frac{1}{2\pi}\int_{-\infty}^{\infty} e^{ik(x-x')}\,dk\;\delta(z-z'). $$


### Derivation of the one-dimensional ODE for each Fourier mode

Substituting the Fourier expansion of $G$ into $\nabla^2 G = -\delta(\mathbf{R}-\mathbf{R'})$, we apply the 2D Laplacian. Acting on $e^{ik(x-x')}$ with $\partial^2/\partial x^2$ produces $-k^2 e^{ik(x-x')}$, while $\partial^2/\partial z^2$ acts only on $\hat{g}_k(z,z')$. Thus:

$$ \frac{1}{2\pi}\int_{-\infty}^{\infty}\left[\frac{d^2\hat{g}_k}{dz^2} - k^2\,\hat{g}_k(z,z')\right]e^{ik(x-x')}\,dk = -\frac{1}{2\pi}\int_{-\infty}^{\infty} e^{ik(x-x')}\,dk\;\delta(z-z'). $$

Since the Fourier modes $e^{ik(x-x')}$ are linearly independent (i.e., equating the integrands for each $k$), we obtain the following one-dimensional equation for each wavenumber $k$:

$$ \boxed{ \frac{d^2\hat{g}_k}{dz^2} - k^2\,\hat{g}_k(z,z') = -\delta(z-z'). } \tag{1} $$

This is the **modified Helmholtz equation** in one dimension (Arfken & Weber, Exercise 10.5.11; Suzuki, Lecture Notes Ch. 17.3). The parameter $k$ here is the lateral Fourier wavenumber; it enters as a "mass" term $k^2$ that controls the exponential decay rate in the $z$-direction.


### Construction of the 1D Green's function for $k \neq 0$

For $z \neq z'$, the right-hand side of Eq. (1) vanishes and we have the homogeneous equation

$$ \frac{d^2\hat{g}_k}{dz^2} - k^2\,\hat{g}_k = 0. $$

The two linearly independent solutions are $e^{|k|z}$ and $e^{-|k|z}$. (We write $|k|$ since the equation depends on $k^2$, ensuring the analysis holds for both positive and negative $k$.)

We impose the free-space boundary conditions: the Green's function must vanish as $z \to +\infty$ and as $z \to -\infty$. This requires:

- For $z < z'$ (below the source): the solution must remain bounded as $z \to -\infty$, so we choose $e^{|k|z}$.
- For $z > z'$ (above the source): the solution must remain bounded as $z \to +\infty$, so we choose $e^{-|k|z}$.

Thus we write

$$ \hat{g}_k(z,z') = \begin{cases} A\,e^{|k|z}, & z < z', \\ B\,e^{-|k|z}, & z > z'. \end{cases} $$

**Continuity at $z = z'$:** The Green's function must be continuous at the source point, so

$$ A\,e^{|k|z'} = B\,e^{-|k|z'}, $$

which gives

$$ B = A\,e^{2|k|z'}. \tag{2} $$

**Jump condition from the delta function:** To determine the remaining constant $A$, we integrate Eq. (1) across a small interval $(z'-\varepsilon, z'+\varepsilon)$ and take $\varepsilon \to 0$. The term involving $k^2\hat{g}_k$ is continuous and contributes nothing in the limit. Only the second-derivative term survives:

$$ \int_{z'-\varepsilon}^{z'+\varepsilon} \frac{d^2\hat{g}_k}{dz^2}\,dz = -\int_{z'-\varepsilon}^{z'+\varepsilon}\delta(z-z')\,dz = -1. $$

The left-hand side is a total derivative:

$$ \left[\frac{d\hat{g}_k}{dz}\right]_{z=z'^-}^{z=z'^+} = -1. $$

Now we compute the derivatives from each branch:

$$ \frac{d\hat{g}_k}{dz} = \begin{cases} |k|\,A\,e^{|k|z}, & z < z', \\ -|k|\,B\,e^{-|k|z}, & z > z'. \end{cases} $$

Evaluating the jump at $z = z'$:

$$ \left(-|k|\,B\,e^{-|k|z'}\right) - \left(|k|\,A\,e^{|k|z'}\right) = -1. $$

Using the continuity condition (2), we have $B\,e^{-|k|z'} = A\,e^{|k|z'}$, so both terms are equal:

$$ -|k|\,A\,e^{|k|z'} - |k|\,A\,e^{|k|z'} = -2|k|\,A\,e^{|k|z'} = -1. $$

Solving for $A$:

$$ A = \frac{1}{2|k|}\,e^{-|k|z'}. $$

Substituting back into the piecewise solution:

- For $z < z'$: $\hat{g}_k = \frac{1}{2|k|}\,e^{-|k|z'}\,e^{|k|z} = \frac{1}{2|k|}\,e^{-|k|(z'-z)}$.
- For $z > z'$: Using (2), $B = \frac{1}{2|k|}\,e^{-|k|z'}\,e^{2|k|z'} = \frac{1}{2|k|}\,e^{|k|z'}$, so $\hat{g}_k = \frac{1}{2|k|}\,e^{|k|z'}\,e^{-|k|z} = \frac{1}{2|k|}\,e^{-|k|(z-z')}$.

Both cases combine into the compact form:

$$ \boxed{ \hat{g}_k(z,z') = \frac{1}{2|k|}\,e^{-|k||z-z'|}, \qquad k \neq 0. } \tag{3} $$

This is the one-dimensional Green's function for the modified Helmholtz operator $(d^2/dz^2 - k^2)$, with free-space (vanishing at infinity) boundary conditions.


### The case $k = 0$

When $k = 0$, Eq. (1) reduces to

$$ \frac{d^2\hat{g}_0}{dz^2} = -\delta(z-z'). $$

This is the 1D Laplace Green's function. The homogeneous solutions are $1$ and $z$. There is no solution that vanishes at both $z \to +\infty$ and $z \to -\infty$ (the 1D Laplace equation has no decaying solutions). The free-space Green's function is

$$ \boxed{ \hat{g}_0(z,z') = -\frac{1}{2}|z - z'| + C, } \tag{4} $$

where $C$ is an arbitrary constant (gauge freedom), analogous to the $m=0$ mode in the polar formulation.

This can be verified by direct differentiation: $d|z-z'|/dz = \text{sgn}(z-z')$, and $d^2|z-z'|/dz^2 = 2\delta(z-z')$, so $\hat{g}_0'' = -\delta(z-z')$ as required.

As in the polar case, the $k=0$ mode corresponds to the mean (laterally uniform) part of the density. It produces a potential that grows linearly with distance from the source, reflecting the fact that a uniform infinite line mass has no natural reference point at which to set the potential to zero.


### Full free-space Green's function in Fourier form

Combining the results for all wavenumbers:

$$ \boxed{ G(\mathbf{R},\mathbf{R'}) = \frac{1}{2\pi}\int_{-\infty}^{\infty} \hat{g}_k(z,z')\,e^{ik(x-x')}\,dk = \frac{1}{2\pi}\left[\hat{g}_0(z,z') + \int_{-\infty}^{\infty}\frac{1}{2|k|}\,e^{-|k||z-z'|}\,e^{ik(x-x')}\,dk\right]. } $$

This Fourier integral representation is equivalent (up to the additive gauge constant from the $k=0$ mode) to the standard closed-form 2D Green's function

$$ G(\mathbf{R},\mathbf{R'}) = -\frac{1}{2\pi}\ln|\mathbf{R} - \mathbf{R'}|, \qquad |\mathbf{R} - \mathbf{R'}| = \sqrt{(x-x')^2 + (z-z')^2}. $$

One can verify this equivalence by evaluating the integral using the known identity (Gradshteyn & Ryzhik, 3.951):

$$ \int_0^{\infty} \frac{e^{-\alpha t}\cos(\beta t)}{t}\,dt = -\frac{1}{2}\ln(\alpha^2 + \beta^2) + C', $$

with $\alpha = |z-z'|$ and $\beta = (x-x')$, which reproduces the logarithmic form.


## Solution for a general mass anomaly and its Fourier wavenumbers

### Fourier expansion of the density anomaly

We expand the density $\rho(x,z)$ in lateral Fourier modes:

$$ \rho(x,z) = \frac{1}{2\pi}\int_{-\infty}^{\infty}\hat{\rho}_k(z)\,e^{ikx}\,dk, \qquad \hat{\rho}_k(z) = \int_{-\infty}^{\infty}\rho(x,z)\,e^{-ikx}\,dx. $$

The integer or continuous index $k$ is the lateral wavenumber, the direct Cartesian analog of the azimuthal mode $m$ in the polar formulation.

### The potential inherits the same Fourier structure

Write

$$ \psi(x,z) = \frac{1}{2\pi}\int_{-\infty}^{\infty}\hat{\psi}_k(z)\,e^{ikx}\,dk. $$

Inserting the Green's function representation into

$$ \psi(\mathbf{R}) = 4\pi\gamma\int G(\mathbf{R},\mathbf{R'})\,\rho(\mathbf{R'})\,dA', \qquad dA' = dx'\,dz', $$

and using the orthogonality of the Fourier modes $e^{ikx}$, each mode decouples. The derivation proceeds as follows. Substituting the Fourier expansions of both $G$ and $\rho$:

$$ \psi(\mathbf{R}) = 4\pi\gamma\int_{-\infty}^{\infty}dz'\int_{-\infty}^{\infty}dx'\left[\frac{1}{2\pi}\int_{-\infty}^{\infty}\hat{g}_k(z,z')\,e^{ik(x-x')}\,dk\right]\left[\frac{1}{2\pi}\int_{-\infty}^{\infty}\hat{\rho}_{k'}(z')\,e^{ik'x'}\,dk'\right]. $$

Performing the $x'$-integration first, using $\int_{-\infty}^{\infty} e^{i(k'-k)x'}\,dx' = 2\pi\,\delta(k'-k)$, collapses the $k'$-integral and yields:

$$ \psi(\mathbf{R}) = 4\pi\gamma\,\frac{1}{2\pi}\int_{-\infty}^{\infty}e^{ikx}\left[\int_{-\infty}^{\infty}\hat{g}_k(z,z')\,\hat{\rho}_k(z')\,dz'\right]dk. $$

Reading off the Fourier coefficient of $\psi$:

$$ \hat{\psi}_k(z) = 4\pi\gamma\int_{-\infty}^{\infty}\hat{g}_k(z,z')\,\hat{\rho}_k(z')\,dz'. $$


#### Modes $k \neq 0$

Substituting the Green's function from Eq. (3):

$$ \hat{\psi}_k(z) = 4\pi\gamma\int_{-\infty}^{\infty}\frac{1}{2|k|}\,e^{-|k||z-z'|}\,\hat{\rho}_k(z')\,dz'. $$

Splitting the integral at $z' = z$ (separating contributions from below and above the observation point):

$$ \hat{\psi}_k(z) = \frac{2\pi\gamma}{|k|}\left[\int_{-\infty}^{z} e^{-|k|(z-z')}\,\hat{\rho}_k(z')\,dz' + \int_{z}^{\infty} e^{-|k|(z'-z)}\,\hat{\rho}_k(z')\,dz'\right]. $$

Rearranging the exponentials to make the structure more explicit:

$$ \boxed{ \hat{\psi}_k(z) = \frac{2\pi\gamma}{|k|}\left[e^{-|k|z}\int_{-\infty}^{z}\hat{\rho}_k(z')\,e^{|k|z'}\,dz' + e^{|k|z}\int_{z}^{\infty}\hat{\rho}_k(z')\,e^{-|k|z'}\,dz'\right], \quad k \neq 0. } $$

This is the direct Cartesian analog of the polar result. The first term collects contributions from sources **below** the observation depth (the "interior" integral), weighted by the decaying exponential $e^{|k|(z'-z)}$ for $z' < z$. The second term collects contributions from sources **above** (the "exterior" integral), weighted by $e^{|k|(z-z')}$ for $z' > z$. The structural parallel with the polar case is:

| Polar ($m \neq 0$) | Cartesian ($k \neq 0$) |
|---|---|
| $\frac{\gamma}{\|m\|}$ prefactor | $\frac{2\pi\gamma}{\|k\|}$ prefactor |
| $r^{-\|m\|}\int_0^r \rho_m(r')\,r'^{\|m\|+1}\,dr'$ | $e^{-\|k\|z}\int_{-\infty}^{z}\hat{\rho}_k(z')\,e^{\|k\|z'}\,dz'$ |
| $r^{\|m\|}\int_r^{\infty}\rho_m(r')\,r'^{1-\|m\|}\,dr'$ | $e^{\|k\|z}\int_{z}^{\infty}\hat{\rho}_k(z')\,e^{-\|k\|z'}\,dz'$ |

The power-law decay $(r_</r_>)^{|m|}$ in polar coordinates is replaced by exponential decay $e^{-|k||z-z'|}$ in Cartesian coordinates. Both express the same physics: the gravitational influence of a spectral mode decays with distance from the source, faster for higher wavenumbers.

If the density has compact support on a layer $z_a \le z' \le z_b$, simply replace $(-\infty, \infty)$ by $(z_a, z_b)$ in the integrals.


#### Mode $k = 0$ (laterally uniform)

$$ \boxed{ \hat{\psi}_0(z) = 4\pi\gamma\int_{-\infty}^{\infty}\left[-\frac{1}{2}|z-z'| + C\right]\hat{\rho}_0(z')\,dz'. } $$

Splitting at $z' = z$:

$$ \hat{\psi}_0(z) = -2\pi\gamma\left[\int_{-\infty}^{z}(z-z')\,\hat{\rho}_0(z')\,dz' + \int_{z}^{\infty}(z'-z)\,\hat{\rho}_0(z')\,dz'\right] + 4\pi\gamma\,C\int_{-\infty}^{\infty}\hat{\rho}_0(z')\,dz'. $$

The last term is a pure constant proportional to the total line mass; it can be fixed by a gauge choice (for example, set $\psi = 0$ at some reference depth $z_0$). This is completely analogous to the $m=0$ gauge freedom in the polar formulation.


### Discrete layer approximation

For practical computation, the density is often given on discrete layers at depths $z_i$ with thicknesses $\Delta h_i$. The radial integrals become sums:

$$ \hat{\psi}_k(z) \approx \frac{2\pi\gamma}{|k|}\left[\sum_{z_i < z} e^{-|k|(z - z_i)}\,\hat{\rho}_k(z_i)\,\Delta h_i + \sum_{z_i \ge z} e^{-|k|(z_i - z)}\,\hat{\rho}_k(z_i)\,\Delta h_i\right], \quad k \neq 0. $$

Each term has a clear physical interpretation: the contribution of a density layer at depth $z_i$ to the potential at depth $z$ decays exponentially with the separation $|z - z_i|$, at a rate controlled by the lateral wavenumber $|k|$. Short-wavelength anomalies (large $|k|$) are attenuated much more rapidly with depth than long-wavelength ones (small $|k|$), which is the well-known spectral attenuation property of potential fields.
