## 3D Green's function solution of the gravitational Poisson problem in Cartesian coordinates

In this document we derive the analytical solution to the three-dimensional gravitational Poisson equation in Cartesian coordinates $(x, y, z)$. The density field $\rho(x, y, z)$ is decomposed into Fourier modes in the two horizontal directions $(x, y)$, leaving a one-dimensional Green's function problem in the vertical direction $z$. This is the direct 3D extension of the 2D Cartesian formulation, and the natural framework for problems posed in rectangular or slab-like geometries.

The gravitational potential $\psi(\mathbf{r})$ satisfies

$$ \nabla^2\psi(\mathbf{r}) = -4\pi G\,\rho(\mathbf{r}), $$

where $G$ is Newton's gravitational constant and $\rho(\mathbf{r})$ is the mass density distribution. The 3D Laplacian in Cartesian coordinates is

$$ \nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}. $$


### 3D Green's function in Cartesian coordinates

We define the Green's function by

$$ \nabla^2 G(\mathbf{r}, \mathbf{r'}) = -\delta(\mathbf{r} - \mathbf{r'}), $$

where $\mathbf{r} = (x,y,z)$ and $\mathbf{r'} = (x',y',z')$. The solution of the Poisson equation is then

$$ \psi(\mathbf{r}) = 4\pi G\int G(\mathbf{r}, \mathbf{r'})\,\rho(\mathbf{r'})\,d^3\mathbf{r'}, \qquad d^3\mathbf{r'} = dx'\,dy'\,dz'. $$

In three-dimensional free space, the Green's function for the Laplacian is (Arfken & Weber, Table 9.5)

$$ G(\mathbf{r}, \mathbf{r'}) = \frac{1}{4\pi\,|\mathbf{r} - \mathbf{r'}|}, $$

where $|\mathbf{r} - \mathbf{r'}| = \sqrt{(x-x')^2 + (y-y')^2 + (z-z')^2}$. Our goal is to decompose this Green's function spectrally in the horizontal directions $(x,y)$, arriving at a one-dimensional Green's function problem in $z$.


### Fourier decomposition in the horizontal directions

We expand the Green's function in two-dimensional Fourier modes along $(x,y)$:

$$ G(\mathbf{r}, \mathbf{r'}) = \frac{1}{(2\pi)^2}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} \hat{g}_{\mathbf{k}}(z,z')\,e^{i\mathbf{k}\cdot(\mathbf{x}-\mathbf{x'})}\,dk_x\,dk_y, $$

where $\mathbf{k} = (k_x, k_y)$ is the horizontal wavevector and $\mathbf{x} = (x,y)$ is the horizontal position vector, so $\mathbf{k}\cdot(\mathbf{x}-\mathbf{x'}) = k_x(x-x') + k_y(y-y')$.

The three-dimensional Dirac delta function in Cartesian coordinates separates as

$$ \delta(\mathbf{r} - \mathbf{r'}) = \delta(x-x')\,\delta(y-y')\,\delta(z-z'). $$

The horizontal delta functions have the standard 2D Fourier representation

$$ \delta(x-x')\,\delta(y-y') = \frac{1}{(2\pi)^2}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}e^{i\mathbf{k}\cdot(\mathbf{x}-\mathbf{x'})}\,dk_x\,dk_y. $$

Therefore the right-hand side of the Green's function equation becomes

$$ -\delta(\mathbf{r} - \mathbf{r'}) = -\frac{1}{(2\pi)^2}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}e^{i\mathbf{k}\cdot(\mathbf{x}-\mathbf{x'})}\,dk_x\,dk_y\;\delta(z-z'). $$


### Derivation of the one-dimensional ODE for each Fourier mode

Substituting the Fourier expansion of $G$ into $\nabla^2 G = -\delta(\mathbf{r}-\mathbf{r'})$, we apply the 3D Laplacian. The horizontal derivatives $\partial^2/\partial x^2 + \partial^2/\partial y^2$ act on $e^{i\mathbf{k}\cdot(\mathbf{x}-\mathbf{x'})}$ and produce $-(k_x^2 + k_y^2)\,e^{i\mathbf{k}\cdot(\mathbf{x}-\mathbf{x'})}$, while $\partial^2/\partial z^2$ acts only on $\hat{g}_{\mathbf{k}}(z,z')$. Thus:

$$ \frac{1}{(2\pi)^2}\int\int\left[\frac{d^2\hat{g}_{\mathbf{k}}}{dz^2} - (k_x^2 + k_y^2)\,\hat{g}_{\mathbf{k}}(z,z')\right]e^{i\mathbf{k}\cdot(\mathbf{x}-\mathbf{x'})}\,dk_x\,dk_y = -\frac{1}{(2\pi)^2}\int\int e^{i\mathbf{k}\cdot(\mathbf{x}-\mathbf{x'})}\,dk_x\,dk_y\;\delta(z-z'). $$

Defining the horizontal wavenumber magnitude

$$ k_h = |\mathbf{k}| = \sqrt{k_x^2 + k_y^2}, $$

and equating the integrands (since the 2D Fourier modes are linearly independent), we obtain the one-dimensional equation for each wavevector $\mathbf{k}$:

$$ \boxed{ \frac{d^2\hat{g}_{\mathbf{k}}}{dz^2} - k_h^2\,\hat{g}_{\mathbf{k}}(z,z') = -\delta(z-z'). } \tag{1} $$

This is the same **modified Helmholtz equation** that appears in the 2D Cartesian formulation, with $k_h = \sqrt{k_x^2 + k_y^2}$ playing the role of the single wavenumber $|k|$. The key difference from the 2D case is that $k_h$ is now the magnitude of a two-dimensional wavevector rather than a single Fourier wavenumber.

Note that $\hat{g}_{\mathbf{k}}$ depends on $\mathbf{k}$ only through $k_h = |\mathbf{k}|$ (the ODE involves only $k_h^2$, not $k_x$ and $k_y$ separately). This is a consequence of the isotropy of the Laplacian in the horizontal plane.


### Construction of the 1D Green's function for $k_h \neq 0$

The construction is identical to the 2D Cartesian case (see that document for a step-by-step derivation). We summarise the key results here for completeness.

For $z \neq z'$, the homogeneous equation $\hat{g}'' - k_h^2\hat{g} = 0$ has solutions $e^{k_h z}$ and $e^{-k_h z}$. Imposing free-space boundary conditions (decay as $z \to \pm\infty$):

$$ \hat{g}_{\mathbf{k}}(z,z') = \begin{cases} A\,e^{k_h z}, & z < z', \\ B\,e^{-k_h z}, & z > z'. \end{cases} $$

**Continuity at $z = z'$:**

$$ A\,e^{k_h z'} = B\,e^{-k_h z'}, \qquad \Longrightarrow \qquad B = A\,e^{2k_h z'}. \tag{2} $$

**Jump condition from the delta function.** Integrating Eq. (1) across $(z'-\varepsilon, z'+\varepsilon)$ and taking $\varepsilon \to 0$, the $k_h^2\hat{g}$ term vanishes (bounded and continuous), leaving

$$ \left[\frac{d\hat{g}_{\mathbf{k}}}{dz}\right]_{z'^-}^{z'^+} = -1. $$

Computing from each branch:

$$ \left(-k_h\,B\,e^{-k_h z'}\right) - \left(k_h\,A\,e^{k_h z'}\right) = -2k_h\,A\,e^{k_h z'} = -1, $$

where we used continuity (2) to replace $B\,e^{-k_h z'} = A\,e^{k_h z'}$. Solving:

$$ A = \frac{1}{2k_h}\,e^{-k_h z'}. $$

Both branches combine to:

$$ \boxed{ \hat{g}_{\mathbf{k}}(z,z') = \frac{1}{2k_h}\,e^{-k_h|z-z'|}, \qquad k_h \neq 0. } \tag{3} $$

This is the same modified Helmholtz Green's function as in the 2D case, with $|k|$ replaced by $k_h = \sqrt{k_x^2 + k_y^2}$.


### The case $k_h = 0$ (horizontally uniform)

When $k_h = 0$ (i.e., $k_x = k_y = 0$), Eq. (1) reduces to

$$ \frac{d^2\hat{g}_{\mathbf{0}}}{dz^2} = -\delta(z-z'). $$

As in the 2D Cartesian case, the free-space Green's function is

$$ \boxed{ \hat{g}_{\mathbf{0}}(z,z') = -\frac{1}{2}|z-z'| + C, } \tag{4} $$

where $C$ is an arbitrary gauge constant. This mode corresponds to the laterally uniform part of the density (a horizontal slab), whose gravitational effect is a linearly growing potential — reflecting the well-known result that a uniform infinite slab of surface density $\sigma$ produces a gravitational field $g = 2\pi G\sigma$ on each side.


### Full free-space Green's function in Fourier form

Combining all wavevectors:

$$ \boxed{ G(\mathbf{r}, \mathbf{r'}) = \frac{1}{(2\pi)^2}\int\int \hat{g}_{\mathbf{k}}(z,z')\,e^{i\mathbf{k}\cdot(\mathbf{x}-\mathbf{x'})}\,dk_x\,dk_y = \frac{1}{(2\pi)^2}\int\int \frac{1}{2k_h}\,e^{-k_h|z-z'|}\,e^{i\mathbf{k}\cdot(\mathbf{x}-\mathbf{x'})}\,dk_x\,dk_y. } $$

This Fourier integral representation is equivalent to the closed-form result $G = 1/(4\pi|\mathbf{r}-\mathbf{r'}|)$. The equivalence can be verified by converting to polar coordinates in $\mathbf{k}$-space. Setting $k_x = k_h\cos\alpha$, $k_y = k_h\sin\alpha$, and denoting $\rho_h = |\mathbf{x}-\mathbf{x'}|$ as the horizontal separation, we can write $\mathbf{k}\cdot(\mathbf{x}-\mathbf{x'}) = k_h\rho_h\cos(\alpha - \alpha_0)$ for some angle $\alpha_0$. The angular integral over $\alpha$ produces $2\pi\,J_0(k_h\rho_h)$ (where $J_0$ is the Bessel function of the first kind), giving

$$ G = \frac{1}{2\pi}\int_0^{\infty}\frac{1}{2k_h}\,e^{-k_h|z-z'|}\,J_0(k_h\rho_h)\,k_h\,dk_h = \frac{1}{4\pi}\int_0^{\infty}e^{-k_h|z-z'|}\,J_0(k_h\rho_h)\,dk_h. $$

This is a standard Hankel transform (Gradshteyn & Ryzhik, 6.611), which evaluates to

$$ G = \frac{1}{4\pi}\,\frac{1}{\sqrt{\rho_h^2 + (z-z')^2}} = \frac{1}{4\pi\,|\mathbf{r}-\mathbf{r'}|}, $$

confirming the equivalence.


## Solution for a general mass anomaly and its Fourier wavenumbers

### Fourier expansion of the density

We expand $\rho(\mathbf{r})$ in horizontal Fourier modes:

$$ \rho(x,y,z) = \frac{1}{(2\pi)^2}\int\int\hat{\rho}_{\mathbf{k}}(z)\,e^{i\mathbf{k}\cdot\mathbf{x}}\,dk_x\,dk_y, $$

where

$$ \hat{\rho}_{\mathbf{k}}(z) = \int\int\rho(x,y,z)\,e^{-i\mathbf{k}\cdot\mathbf{x}}\,dx\,dy. $$

The wavevector $\mathbf{k} = (k_x, k_y)$ labels the horizontal spectral content, the 3D Cartesian analog of $(l,m)$ in the spherical formulation.


### The potential inherits the same Fourier structure

Write

$$ \psi(x,y,z) = \frac{1}{(2\pi)^2}\int\int\hat{\psi}_{\mathbf{k}}(z)\,e^{i\mathbf{k}\cdot\mathbf{x}}\,dk_x\,dk_y. $$

Inserting the Green's function representation into

$$ \psi(\mathbf{r}) = 4\pi G\int G(\mathbf{r}, \mathbf{r'})\,\rho(\mathbf{r'})\,d^3\mathbf{r'}, $$

and substituting the Fourier expansions of both $G$ and $\rho$:

$$ \psi(\mathbf{r}) = 4\pi G\int dz'\int dx'\,dy'\left[\frac{1}{(2\pi)^2}\int\int\hat{g}_{\mathbf{k}}(z,z')\,e^{i\mathbf{k}\cdot(\mathbf{x}-\mathbf{x'})}\,d^2\mathbf{k}\right]\left[\frac{1}{(2\pi)^2}\int\int\hat{\rho}_{\mathbf{k'}}(z')\,e^{i\mathbf{k'}\cdot\mathbf{x'}}\,d^2\mathbf{k'}\right]. $$

Performing the $(x',y')$-integration first:

$$ \int\int e^{i(\mathbf{k'}-\mathbf{k})\cdot\mathbf{x'}}\,dx'\,dy' = (2\pi)^2\,\delta^{(2)}(\mathbf{k'}-\mathbf{k}), $$

where $\delta^{(2)}(\mathbf{k'}-\mathbf{k}) = \delta(k_x'-k_x)\,\delta(k_y'-k_y)$. This collapses the $\mathbf{k'}$-integral (replacing $\mathbf{k'}$ with $\mathbf{k}$), giving:

$$ \psi(\mathbf{r}) = 4\pi G\,\frac{1}{(2\pi)^2}\int\int e^{i\mathbf{k}\cdot\mathbf{x}}\left[\int_{-\infty}^{\infty}\hat{g}_{\mathbf{k}}(z,z')\,\hat{\rho}_{\mathbf{k}}(z')\,dz'\right]d^2\mathbf{k}. $$

Reading off the Fourier coefficient of $\psi$:

$$ \hat{\psi}_{\mathbf{k}}(z) = 4\pi G\int_{-\infty}^{\infty}\hat{g}_{\mathbf{k}}(z,z')\,\hat{\rho}_{\mathbf{k}}(z')\,dz'. $$


#### Modes $k_h \neq 0$

Substituting the Green's function from Eq. (3):

$$ \hat{\psi}_{\mathbf{k}}(z) = 4\pi G\int_{-\infty}^{\infty}\frac{1}{2k_h}\,e^{-k_h|z-z'|}\,\hat{\rho}_{\mathbf{k}}(z')\,dz' = \frac{2\pi G}{k_h}\int_{-\infty}^{\infty}e^{-k_h|z-z'|}\,\hat{\rho}_{\mathbf{k}}(z')\,dz'. $$

Splitting the integral at $z' = z$:

$$ \boxed{ \hat{\psi}_{\mathbf{k}}(z) = \frac{2\pi G}{k_h}\left[e^{-k_h z}\int_{-\infty}^{z}\hat{\rho}_{\mathbf{k}}(z')\,e^{k_h z'}\,dz' + e^{k_h z}\int_{z}^{\infty}\hat{\rho}_{\mathbf{k}}(z')\,e^{-k_h z'}\,dz'\right], \quad k_h \neq 0. } \tag{5} $$

This is structurally identical to the 2D Cartesian result, with $|k|$ replaced by $k_h = \sqrt{k_x^2+k_y^2}$. The first term collects contributions from sources below the observation depth, the second from above, and the exponential decay rate is controlled by $k_h$. The prefactor is $2\pi G/k_h$ (compared to $2\pi\gamma/|k|$ in the 2D case, with $G$ replacing $\gamma$ as the gravitational constant carries different dimensions in 3D vs 2D).

If the density has compact vertical support on a layer $z_a \le z' \le z_b$, simply replace $(-\infty,\infty)$ by $(z_a,z_b)$ in the integrals.


#### Mode $k_h = 0$ (horizontally uniform)

$$ \boxed{ \hat{\psi}_{\mathbf{0}}(z) = 4\pi G\int_{-\infty}^{\infty}\left[-\frac{1}{2}|z-z'| + C\right]\hat{\rho}_{\mathbf{0}}(z')\,dz'. } $$

Splitting at $z'=z$:

$$ \hat{\psi}_{\mathbf{0}}(z) = -2\pi G\left[\int_{-\infty}^{z}(z-z')\,\hat{\rho}_{\mathbf{0}}(z')\,dz' + \int_{z}^{\infty}(z'-z)\,\hat{\rho}_{\mathbf{0}}(z')\,dz'\right] + 4\pi G\,C\int_{-\infty}^{\infty}\hat{\rho}_{\mathbf{0}}(z')\,dz'. $$

The last term is a constant proportional to the total mass per unit horizontal area; it is fixed by a gauge choice.


### Structural parallel across all formulations

The result has the same structure as all other formulations in this package:

| Formulation | Spectral basis | Prefactor | Vertical/radial Green's function |
|---|---|---|---|
| 2D polar ($m \neq 0$) | $e^{im\phi}$ | $\gamma/\|m\|$ | $(r_</r_>)^{\|m\|}$ (power law) |
| 2D Cartesian ($k \neq 0$) | $e^{ikx}$ | $2\pi\gamma/\|k\|$ | $e^{-\|k\|\|z-z'\|}$ (exponential) |
| 3D spherical ($l \geq 0$) | $Y_l^m(\theta,\phi)$ | $4\pi G/(2l+1)$ | $r_<^l/r_>^{l+1}$ (power law) |
| 3D Cartesian ($k_h \neq 0$) | $e^{i\mathbf{k}\cdot\mathbf{x}}$ | $2\pi G/k_h$ | $e^{-k_h\|z-z'\|}$ (exponential) |

In every case: spectral decomposition in the lateral/angular directions, a 1D Green's function integral in the radial/vertical direction, and splitting at the observation point into "below/interior" and "above/exterior" contributions. The key parameter controlling the decay rate is the spectral mode number ($|m|$, $|k|$, $l$, or $k_h$), and higher modes are attenuated more strongly with distance from the source.


### Discrete layer approximation

For a density distribution given on discrete horizontal layers at depths $z_i$ with thicknesses $\Delta h_i$:

$$ \hat{\psi}_{\mathbf{k}}(z) \approx \frac{2\pi G}{k_h}\left[\sum_{z_i < z}e^{-k_h(z-z_i)}\,\hat{\rho}_{\mathbf{k}}(z_i)\,\Delta h_i + \sum_{z_i \ge z}e^{-k_h(z_i-z)}\,\hat{\rho}_{\mathbf{k}}(z_i)\,\Delta h_i\right], \quad k_h \neq 0. $$

Each term has the same clear physical interpretation as in the 2D case: the contribution of a density layer at $z_i$ decays exponentially with the vertical separation $|z-z_i|$, at a rate set by the horizontal wavenumber $k_h$. Short-wavelength horizontal anomalies are attenuated much more rapidly with depth than long-wavelength ones.
