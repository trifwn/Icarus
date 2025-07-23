# JAX Airfoil Integration Examples

This directory contains comprehensive examples demonstrating how to integrate JAX airfoils into real-world aerodynamic analysis and optimization workflows. These examples showcase the power of automatic differentiation and JIT compilation for advanced engineering applications.

## Examples Overview

### 01_aerodynamic_analysis.py
**Aerodynamic Analysis Workflow Integration**

Demonstrates comprehensive aerodynamic analysis workflows using JAX airfoils:
- Batch processing of multiple airfoils with different Reynolds numbers
- Gradient-based sensitivity analysis of aerodynamic parameters
- Performance comparison between JAX and NumPy implementations
- Integration with existing ICARUS solvers and workflows
- Vectorized analysis for efficient parametric studies

**Key Features:**
- Automatic vectorization over Reynolds numbers and angles of attack
- JIT compilation for performance optimization
- Gradient computation through aerodynamic analysis pipeline
- Seamless integration with existing ICARUS infrastructure

**Run Time:** ~30 seconds
**Output:** Comprehensive aerodynamic analysis plots and performance metrics

### 02_shape_optimization.py
**Advanced Shape Optimization Using Gradient-Based Methods**

Showcases sophisticated optimization workflows enabled by automatic differentiation:
- Single-objective optimization (maximize L/D ratio)
- Multi-objective optimization with Pareto frontier analysis
- Constrained optimization with geometric and performance constraints
- Robust optimization under operating condition uncertainty
- Multi-point optimization across different flight conditions

**Key Features:**
- Gradient-based optimization using JAX automatic differentiation
- Penalty methods for constraint handling
- Monte Carlo sampling for robust optimization
- Pareto frontier exploration for multi-objective problems

**Run Time:** ~45 seconds
**Output:** Optimization convergence plots, Pareto frontiers, and optimized airfoil shapes

### 03_parametric_studies.py
**Comprehensive Parametric Studies with Batch Processing**

Demonstrates efficient design space exploration and statistical analysis:
- Full factorial design space exploration
- Design of Experiments (DOE) with Latin Hypercube Sampling
- Response surface modeling and visualization
- Statistical analysis of design parameter effects
- Correlation analysis and sensitivity studies

**Key Features:**
- Efficient batch processing of large parameter spaces
- Statistical analysis and correlation studies
- Response surface modeling for design insights
- Comparison of full factorial vs. DOE approaches

**Run Time:** ~60 seconds
**Output:** Response surfaces, parameter sensitivity plots, and statistical analysis

## Prerequisites

### Required Packages
```python
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, vmap, jit
from ICARUS.airfoils import Airfoil
from ICARUS.airfoils.naca4 import NACA4
from ICARUS.core.units import calc_reynolds
```

### System Requirements
- JAX installation with CPU or GPU support
- ICARUS aerodynamic analysis framework
- Python 3.8+ with NumPy, Matplotlib, and SciPy

## Usage Instructions

### Running Individual Examples

Each example can be run independently:

```bash
# Aerodynamic analysis workflow
python 01_aerodynamic_analysis.py

# Shape optimization examples
python 02_shape_optimization.py

# Parametric studies
python 03_parametric_studies.py
```

### Integration with Your Workflows

These examples are designed to be modular and can be adapted for your specific needs:

1. **Modify Analysis Conditions**: Update Reynolds numbers, angles of attack, and Mach numbers in the setup functions
2. **Change Airfoil Parameters**: Adjust NACA parameter ranges or add custom airfoil geometries
3. **Customize Objectives**: Modify optimization objectives for your specific performance requirements
4. **Extend Analysis**: Add additional performance metrics or constraints

### Performance Optimization Tips

1. **JIT Compilation**: Functions are decorated with `@jit` for optimal performance
2. **Vectorization**: Use `vmap` for efficient batch processing
3. **Memory Management**: Large parameter studies may require chunking for memory efficiency
4. **GPU Acceleration**: JAX automatically uses GPU when available

## Key Capabilities Demonstrated

### Automatic Differentiation
- Gradient computation through complex aerodynamic analysis chains
- Sensitivity analysis with respect to design parameters
- Efficient optimization using gradient-based methods

### Batch Processing
- Vectorized analysis over multiple operating conditions
- Efficient parameter space exploration
- Parallel processing of design variations

### Integration Features
- Seamless integration with existing ICARUS solvers
- Backward compatibility with NumPy-based workflows
- Coordinate extraction for external solver integration

### Advanced Optimization
- Multi-objective optimization with Pareto frontiers
- Constrained optimization with penalty methods
- Robust optimization under uncertainty
- Statistical design of experiments

## Example Applications

### Aerospace Design
- Wing section optimization for aircraft design
- Propeller blade section analysis
- Turbomachinery airfoil optimization

### Wind Energy
- Wind turbine blade section optimization
- Performance analysis across operating conditions
- Robust design for varying wind conditions

### Research and Development
- Parametric studies for design space understanding
- Sensitivity analysis for design parameter importance
- Validation of aerodynamic models and methods

## Performance Benchmarks

Typical performance improvements with JAX implementation:

- **Batch Analysis**: 10-50x speedup over sequential NumPy analysis
- **Gradient Computation**: Automatic differentiation vs. finite differences
- **JIT Compilation**: 2-5x speedup after compilation overhead
- **Memory Efficiency**: Reduced memory allocation through vectorization

## Extending the Examples

### Adding New Airfoil Types
```python
# Add custom airfoil geometries
from ICARUS.airfoils.naca5 import NACA5
custom_airfoil = NACA5(L=2, P=3, Q=0, XX=12, n_points=200)
```

### Custom Performance Metrics
```python
@jit
def custom_performance_metric(cl, cd, cm):
    """Define your custom performance metric"""
    # Example: weighted combination of L/D and moment
    ld_ratio = cl / (cd + 1e-6)
    moment_penalty = jnp.abs(cm)
    return ld_ratio - 0.1 * moment_penalty
```

### Integration with CFD Solvers
```python
def integrate_with_cfd_solver(airfoil):
    """Template for CFD solver integration"""
    # Extract coordinates
    upper_coords = airfoil.upper_surface
    lower_coords = airfoil.lower_surface

    # Interface with your CFD solver
    # results = your_cfd_solver.analyze(upper_coords, lower_coords)

    return results
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch sizes or use chunking for large parameter studies
2. **Compilation Time**: First run includes JIT compilation overhead
3. **Numerical Stability**: Add small epsilon values to avoid division by zero
4. **Gradient Issues**: Ensure all operations are differentiable

### Performance Tips

1. **Warm-up Runs**: Run small examples first to compile functions
2. **Batch Size Tuning**: Find optimal batch sizes for your hardware
3. **Memory Monitoring**: Monitor memory usage for large studies
4. **GPU Utilization**: Ensure JAX is using GPU when available

## Further Reading

- [JAX Documentation](https://jax.readthedocs.io/)
- [ICARUS Framework Documentation](https://icarus-framework.readthedocs.io/)
- [Aerodynamic Shape Optimization Literature](https://doi.org/10.2514/1.J050071)
- [Design of Experiments Methods](https://doi.org/10.1080/00401706.1989.10488474)

## Contributing

These examples are designed to be educational and extensible. Contributions for additional integration scenarios, performance improvements, or new analysis methods are welcome.

For questions or suggestions, please refer to the main ICARUS documentation or create an issue in the project repository.
