# passess (**P**oissoin Equation analytical solutions (by which we **assess** numerical solutions))

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package providing analytical solutions to the Poisson equation for validating and benchmarking numerical solvers in gravitational physics and computational astrophysics.

## Overview

Passess delivers exact analytical solutions to the gravitational Poisson problem, enabling researchers to rigorously test and validate numerical methods. The package is designed for academic researchers working on N-body simulations, gravitational dynamics, and partial differential equation solvers where accuracy verification against known solutions is critical.

## Features

- **Analytical Solutions**: Exact solutions to the Poisson equation for various configurations
- **Multiple Geometries**: Support for different coordinate systems and boundary conditions
- **Benchmark Suite**: Pre-configured test cases for systematic validation of numerical solvers
- **Research-Grade**: Designed for academic rigor and reproducibility

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/sghelichkhani/passess.git
```

Or clone the repository and install in development mode:

```bash
git clone https://github.com/sghelichkhani/passess.git
cd passess
pip install -e .
```

## Use Cases

- Validation of numerical Poisson solvers
- Benchmarking gravitational field computations
- Testing PDE solver accuracy and convergence
- Educational demonstrations of analytical vs. numerical methods

## Citation

If you use Passess in your research, please cite it in your publications.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
