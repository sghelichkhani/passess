## 2D Green's function solution of the gravitational Poisson problem in polar coordinates

In this case, we work on the plane with polar coordinates, that is, $\mathbf{R} = (r,\phi)$. Let the 2D Laplacian be

$$ \nabla^2 = \frac{\partial^2}{\partial r^2}+\frac{1}{r}\frac{\partial}{\partial r}+\frac{1}{r^2}\frac{\partial^2}{\partial \phi^2}. $$


For our problems, which are 2 dimensional, the formulations are obtained by assuming invariance in the out-of-plane direction (an "infinite cylinder" reduction), and hence, the gravitational potential $\psi(r,\phi)$ satisfies the following equation:

$$ \nabla^2\psi(r,\phi) = -4\pi \gamma\,\rho(r,\phi), $$

where $\rho(r,\phi)$ is the **mass anomaly per unit out-of-plane length** (so that $\int \rho\,dA$ is a line mass). The potential is defined up to an arbitrary additive constant, just like the 3d case, and only $-\nabla\psi = \vec{g}$ is unique.

### 2D delta function in polar coordinates
As usual, we start by the following definition of a Green's function:

$$ \nabla^2 G(\mathbf{R}, \mathbf{R'}) = -\delta(\mathbf{R} - \mathbf{R'}). $$

Then the solution of Poisson's equation is

$$ \psi(\mathbf R)=4\pi \gamma\int_{\mathbb R^2} G(\mathbf R,\mathbf R')\,\rho(\mathbf R')\,dA',
\qquad dA'=r'\,dr'\,d\phi'. $$


The 2D Dirac delta in polar coordinates is given by:

$$ \delta(\mathbf R-\mathbf R')=\frac{1}{r}\,\delta(r-r')\,\delta(\phi-\phi'),
\qquad \mathbf R=(r,\phi),\ \mathbf R'=(r',\phi'). $$

The angular delta has the Fourier series

$$ \delta(\phi-\phi')=\frac{1}{2\pi}\sum_{m=-\infty}^{\infty}e^{im(\phi-\phi')}. $$

Therefore, the Dirac delta in polar coordinates is given by:

$$ \delta(\mathbf R-\mathbf R')
=\frac{1}{2\pi r}\,\delta(r-r')\sum_{m=-\infty}^{\infty}e^{im(\phi-\phi')}. $$

### Green's function construction
Let's define the (free-space) Green's function $G(\mathbf R,\mathbf R')$ by the usual form of the Green's function for the Poisson equation:

$$ \nabla^2G(\mathbf R,\mathbf R')=-\delta(\mathbf R-\mathbf R'). $$


### Fourier (azimuthal) expansion of the Green's function
Use the rotational symmetry by expanding $G$ in angular Fourier modes:

$$ G(\mathbf R,\mathbf R')=\sum_{m=-\infty}^{\infty} g_m(r,r')\,e^{im(\phi-\phi')}. $$

Apply $\nabla^2$ with respect to $(r,\phi)$ and since we have:

$$ \frac{\partial^2}{\partial \phi^2}e^{im(\phi-\phi')}=-m^2 e^{im(\phi-\phi')}, $$

we obtain the following one-dimensional PDE, in azimuthal mode-by-mode,

$$ \left[\frac{1}{r}\frac{d}{dr}\left(r\frac{d}{dr}\right)-\frac{m^2}{r^2}\right]g_m(r,r')
= -\frac{1}{2\pi r}\,\delta(r-r').
\tag{1} $$

This means that to have the solution we only require solving for $g_m(r,r')$ for each $m$.

### Radial Green's function for $m\neq 0$
For $r\neq r'$, the right-hand side vanishes and (1) is homogeneous. For $m\neq 0$ the independent homogeneous solutions are $r^{|m|}$ and $r^{-|m|}$.
Impose free-space conditions:
- regularity at $r=0$ for the interior branch,
- decay as $r\to\infty$ for the exterior branch (so we choose $r^{-|m|}$ outside).

Thus, for $m\neq 0$,

$$ g_m(r,r')=
\begin{cases}
A\,r^{|m|}, & r<r',\\
B\,r^{-|m|}, & r>r'.
\end{cases} $$

Continuity at $r=r'$ gives $A r'^{|m|}=B r'^{-|m|}$, so $B=A r'^{2|m|}$.

To determine $A$, integrate (1) across a small interval $(r'-\varepsilon,r'+\varepsilon)$ and use the fact that only the total derivative contributes:

$$ \int_{r'-\varepsilon}^{r'+\varepsilon}\frac{d}{dr}\left(r\frac{dg_m}{dr}\right)\,dr
= -\int_{r'-\varepsilon}^{r'+\varepsilon}\frac{1}{2\pi}\delta(r-r')\,dr
= -\frac{1}{2\pi}. $$

Hence the jump condition

$$ \Big[r\,g_m'(r,r')\Big]_{r=r'^+}-\Big[r\,g_m'(r,r')\Big]_{r=r'^-}=-\frac{1}{2\pi}. $$

Compute the left-hand side:

$$ r g_m' =
\begin{cases}
|m|A r^{|m|}, & r<r',\\
-|m|B r^{-|m|}, & r>r'.
\end{cases} $$

At $r=r'$,

$$ (-|m|B r'^{-|m|})-(|m|A r'^{|m|})
= -|m|A r'^{|m|}-|m|A r'^{|m|}
= -2|m|A r'^{|m|}. $$

Equating to $-1/(2\pi)$ yields $A=\dfrac{1}{4\pi |m|}\,r'^{-|m|}$. Therefore,

$$ \boxed{
g_m(r,r')=\frac{1}{4\pi |m|}\left(\frac{r_<}{r_>}\right)^{|m|},\qquad m\neq 0,
} $$

where $r_< = \min(r,r')$ and $r_> = \max(r,r')$.

### Radial Green's function for $m=0$
For $m=0$, the homogeneous solutions are $\ln r$ and a constant. Free-space choice that matches the delta forcing is

$$ \boxed{
g_0(r,r')=-\frac{1}{2\pi}\ln(r_>) + C,
} $$

with arbitrary constant $C$ (gauge).

### Full free-space Green's function in polar Fourier form
Combine the modes:

$$ \boxed{
G(\mathbf R,\mathbf R')=
g_0(r,r')+\sum_{\substack{m=-\infty\\ m\neq 0}}^{\infty}
\frac{1}{4\pi |m|}\left(\frac{r_<}{r_>}\right)^{|m|}e^{im(\phi-\phi')}.
} $$

This series is equivalent (up to an additive constant) to the standard closed form

$$ \boxed{
G(\mathbf R,\mathbf R')=-\frac{1}{2\pi}\ln|\mathbf R-\mathbf R'| + C,
\qquad |\mathbf R-\mathbf R'|=\sqrt{r^2+r'^2-2rr'\cos(\phi-\phi')}.
} $$

##  Solution for a general mass anomaly and its azimuthal wave-numbers

### Fourier expansion of the anomaly
Expand $\rho(r,\phi)$ in angular Fourier modes:

$$ \rho(r,\phi)=\sum_{m=-\infty}^{\infty}\rho_m(r)\,e^{im\phi},
\qquad
\rho_m(r)=\frac{1}{2\pi}\int_{0}^{2\pi}\rho(r,\phi)\,e^{-im\phi}\,d\phi. $$

The integer $m$ is the azimuthal wavenumber, the direct 2D analog of the $(l,m)$ indices in 3D spherical harmonics (here there is only one angular index).

### The potential inherits the same Fourier structure
Write

$$ \psi(r,\phi)=\sum_{m=-\infty}^{\infty}\psi_m(r)\,e^{im\phi}. $$

Insert the Green representation into

$$ \psi(\mathbf R)=4\pi \gamma\int G(\mathbf R,\mathbf R')\,\rho(\mathbf R')\,dA',
\qquad dA'=r'\,dr'\,d\phi'. $$

Using orthogonality of $e^{im\phi}$, each mode decouples and we obtain

#### Modes $m\neq 0$

$$ \boxed{
\psi_m(r)=4\pi \gamma\int_{0}^{\infty} g_m(r,r')\,\rho_m(r')\,r'\,dr'
= \frac{\gamma}{|m|}\left[
r^{-|m|}\int_{0}^{r}\rho_m(r')\,r'^{|m|+1}\,dr'
+
r^{|m|}\int_{r}^{\infty}\rho_m(r')\,r'^{1-|m|}\,dr'
\right],
\quad m\neq 0.
} $$

If $\rho$ is supported only on an annulus $a\le r'\le b$, simply replace $(0,\infty)$ by $(a,b)$ in the integrals.

#### Mode $m=0$ (axisymmetric)

$$ \boxed{ \psi_0(r)=4\pi \gamma\int_{0}^{\infty}\left[-\frac{1}{2\pi}\ln(r_>)+C\right]\rho_0(r')\,r'\,dr'.
} $$

Equivalently, splitting at $r'=r$,

$$ \boxed{ \psi_0(r)= -2\gamma\left[ \ln r\int_{0}^{r}\rho_0(r')\,r'\,dr' + \int_{r}^{\infty}\rho_0(r')\,r'\ln r'\,dr' \right] + 4\pi \gamma\,C\int_{0}^{\infty}\rho_0(r')\,r'\,dr'. } $$

The last term is a pure constant shift proportional to the total line mass; it can be fixed by a gauge choice (for example set $\psi(r_0)=0$ at some reference radius $r_0$, or enforce zero mean).
