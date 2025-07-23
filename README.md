# Heat Pipe Solver

A Python-based 2D heat pipe conduction model using the FiPy finite volume library. This solver simulates temperature evolution in heat pipes with temperature-dependent material properties and is designed to replicate the experimental conditions from Faghri et al.

## Overview

This project implements a heat pipe solver using FiPy for finite volume discretization. The solver supports:
- 2D cylindrical coordinate system
- Temperature-dependent material properties for sodium, steel, and wick materials  
- Multiple regions: vapor core, wick, and wall with different thermal properties
- Experimental data comparison with Faghri et al. and STAR-CCM+ simulations
- Parallel computing support (though overhead currently negates benefits)

## Installation

### Prerequisites
- Python 3.11
- conda or mamba package manager

### Setup using Conda

1. Clone the repository:
```bash
git clone https://github.com/kubaniak/heat-pipe-solver.git
cd heat-pipe-solver
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate fipy-env
```

3. Verify the installation by running tests:
```bash
# Test FiPy installation
python tests/test_fipy.py

# Test parallel capabilities (optional)
python tests/test_parallel.py
```

## Project Structure

### Core Directories

- **`src/2d/`** - Main solver implementation
  - `main.py` - Primary heat pipe solver with temperature-dependent properties
  - `verification_examples.py` - 10 FiPy verification examples for learning
  - `mesh.py` - 2D mesh generation functions
  - `material_properties.py` - Temperature-dependent material property functions
  - `params.py` - Parameter loading and management utilities
  - `utils/` - Helper functions for plotting and data processing

- **`tests/`** - Test scripts
  - `test_fipy.py` - Basic FiPy functionality test
  - `test_parallel.py` - Parallel processing capability test

- **`config/`** - Configuration files
  - `faghri.json` - Experimental parameters for Faghri et al. conditions

- **`Faghri_Data/`** - Experimental reference data from Faghri et al.

- **`plots/`** - Generated plots and visualizations

### Additional Files

- `environment.yml` - Conda environment specification
- `LICENSE` - Apache 2.0 license

## Usage

### Getting Started

**Important:** Before running the main solver, familiarize yourself with FiPy by running the verification examples:

```bash
cd src/2d
python verification_examples.py
```

This script contains 10 examples with verdicts that demonstrate FiPy usage patterns and validate the installation.

### Running the Heat Pipe Solver

```bash
cd src/2d
python main.py
```

The solver will:
1. Load parameters from `config/faghri.json`
2. Generate a composite 2D cylindrical mesh
3. Solve the heat conduction equation with temperature-dependent properties
4. Generate plots showing temperature evolution and material property profiles
5. Compare results with experimental data (if available)

### Configuration

To use different experimental conditions, modify `config/faghri.json` or create a new configuration file and specify it as an argument.

Key configuration sections:
- `parameters` - Physical constants and boundary conditions
- `dimensions` - Geometric dimensions of the heat pipe
- `mesh` - Mesh refinement parameters

### Solver Options

The `main.py` script contains regions that can be commented/uncommented:

1. **Simple properties** - Constant material properties (faster)
2. **Temperature-dependent properties** - Realistic material behavior (more accurate)

Set `PLOT_RICCARDO_COMPARISON = False` to disable STAR-CCM+ comparison plots.

## Dependencies

All required dependencies are specified in `environment.yml`:

- **Core libraries:**
  - `python=3.11`
  - `fipy` - Finite volume PDE solver
  - `numpy` - Numerical arrays
  - `scipy` - Scientific computing
  - `matplotlib` - Plotting
  - `pandas` - Data manipulation
  - `tqdm` - Progress bars

- **Parallel computing:**
  - `petsc` - Parallel linear algebra
  - `petsc4py` - Python bindings for PETSc  
  - `mpi4py` - MPI for Python

- **Build tools:**
  - `pip` - Package installer

## Performance Notes

- **Parallelization**: While parallel support is implemented, current testing shows overhead costs exceed benefits for this problem size
- **Mesh refinement**: Balance between accuracy and computational cost can be adjusted in mesh parameters
- **Material properties**: Temperature-dependent properties significantly increase computation time but improve accuracy

## Known Issues

- The solver currently shows incorrect results and requires debugging
- Parallel execution may be slower than serial due to overhead
- Some experimental data comparison features require specific file structures

## Contributing

1. Ensure all tests pass: `python tests/test_fipy.py`
2. Follow existing code style and documentation patterns
3. Add appropriate tests for new features

## License

Licensed under the Apache License 2.0. See `LICENSE` file for details.

## References

- Faghri et al. experimental data for heat pipe validation
- FiPy documentation: https://www.ctcms.nist.gov/fipy/
- STAR-CCM+ comparison data (Riccardo's simulations)

