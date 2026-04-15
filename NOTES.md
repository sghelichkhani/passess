## Implementation notes

### Monopole and dipole removal (3D spherical case)

When computing the gravitational potential from density anomalies, it is common practice to zero out the $l=0$ (monopole) and $l=1$ (dipole) spherical harmonic coefficients of the solution. The monopole term corresponds to the total mass anomaly, and the dipole term corresponds to a shift of the centre of mass. Both are typically removed because:

- The monopole produces a uniform radial field that is already accounted for by the reference state.
- The dipole corresponds to a centre-of-mass offset that has no physical meaning for self-gravitating bodies in equilibrium.

In a previous implementation (`lib_poisson.py`), this was done via:

```python
static_geoid.set_coeffs([0., 0., 0., 0.], [0, 1, 1, 1], [0, -1, 0, +1])
```

which sets the $(l,m) = (0,0), (1,-1), (1,0), (1,+1)$ coefficients to zero.

### Data layout convention

In the earlier implementation, the density field was stored as `density[ir, c, l, m]` where:
- `ir` indexes the radial shell
- `c` indexes cosine/sine components (real spherical harmonics)
- `l` is the spherical harmonic degree
- `m` is the spherical harmonic order
