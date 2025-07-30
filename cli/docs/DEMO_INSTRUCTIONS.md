# ICARUS CLI Demo - Quick Start Guide

This guide provides instructions for running the ICARUS CLI demo functionality.

## Prerequisites

Make sure you have the following dependencies installed:

```bash
pip install textual rich matplotlib numpy
```

## Running the Demo

### Interactive TUI Mode

To launch the interactive Terminal User Interface:

```bash
python -m cli
```

This will open a full-screen terminal interface with tabs for:
- Airfoil Analysis
- Airplane Analysis
- Export functionality

### Command-Line Mode

#### Airfoil Analysis

```bash
python -m cli --cli airfoil --target NACA2412 --reynolds 1000000
```

Optional parameters:
- `--min-aoa`: Minimum angle of attack (default: -10)
- `--max-aoa`: Maximum angle of attack (default: 15)
- `--aoa-step`: Angle of attack step size (default: 0.5)
- `--output`: Output file for results in JSON format

#### Airplane Analysis

```bash
python -m cli --cli airplane --target demo_airplane --velocity 50 --altitude 1000
```

Optional parameters:
- `--output`: Output file for results in JSON format

### Running Tests

To verify that the CLI functionality is working correctly:

```bash
python cli/test_cli.py
```

This will run tests for both airfoil and airplane analysis.

## Demo Features

### Airfoil Analysis

- NACA airfoil validation and parameter extraction
- XFoil integration for aerodynamic analysis
- Performance metrics calculation (CL, CD, L/D)
- Detailed performance reports

### Airplane Analysis

- Demo aircraft configuration with wing, fuselage, and tail
- AVL integration for aerodynamic analysis
- Flight performance calculations
- Stability characteristics

### Visualization and Export

- Rich table display of results
- Multiple export formats (JSON, CSV, MATLAB)
- Detailed text reports

## Troubleshooting

### Common Issues

1. **"Textual not available" error**: Install the Textual package with `pip install textual`
2. **"Target file does not exist" error**: Use a valid NACA airfoil code (e.g., NACA2412) or "demo_airplane" for airplane analysis
3. **TUI not displaying**: Make sure your terminal supports full-screen applications

### Debug Mode

Enable verbose logging:

```bash
python -m cli --verbose
```

## Additional Resources

For more detailed information, see:
- `cli/README_DEMO.md`: Comprehensive documentation
- `cli/test_demo.py`: Full test suite
- `.kiro/specs/icarus-cli-revamp/design.md`: Design documentation
