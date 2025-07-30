# ICARUS CLI Visualization System

The ICARUS CLI Visualization System provides comprehensive plotting and charting capabilities for aerodynamic analysis results. Built with matplotlib and designed for integration with the Textual TUI framework, it offers interactive plotting, customization, export, and real-time update capabilities.

## Features

### 🎯 Core Capabilities

- **Interactive Plotting**: Create line plots, scatter plots, bar charts, and specialized aerospace plots
- **Chart Generation**: Generate standardized charts from ICARUS analysis results
- **Plot Customization**: Apply themes, colors, styles, and formatting options
- **Export System**: Export to multiple formats (PNG, PDF, SVG, JSON) with quality presets
- **Real-time Updates**: Live plot updates during long-running analyses
- **TUI Integration**: Seamless integration with Textual-based user interface

### 📊 Supported Plot Types

#### Interactive Plots
- **Line Plots**: Multi-series line charts with customizable styles
- **Scatter Plots**: Point-based data visualization with markers
- **Bar Plots**: Categorical data visualization
- **Polar Plots**: Circular coordinate system plots
- **Contour Plots**: 2D contour visualization
- **Surface Plots**: 3D surface visualization

#### Analysis Charts
- **Airfoil Polar**: CL vs Alpha, CD vs Alpha, CM vs Alpha, CL vs CD
- **Airplane Polar**: Extended polar analysis with L/D ratios
- **Pressure Distribution**: Surface pressure coefficient plots
- **Geometry Visualization**: Airfoil shapes and wing planforms
- **Convergence Analysis**: Residual and force convergence plots
- **Comparison Charts**: Multi-case analysis comparison
- **Sensitivity Analysis**: Parameter sensitivity visualization

### 🎨 Customization Options

#### Color Palettes
- **Aerospace**: Professional aerospace industry colors
- **Publication**: Black and white for academic publications
- **Colorblind**: Colorblind-friendly palette
- **Vibrant**: High-contrast colors for presentations
- **Default**: Standard matplotlib colors

#### Styling Options
- Line widths, styles, and markers
- Grid appearance and transparency
- Font configurations (size, family, weight)
- Legend positioning and styling
- Axis labels and titles
- Background colors and themes

### 📤 Export Formats

#### Image Formats
- **PNG**: High-quality raster images (default 300 DPI)
- **JPEG**: Compressed raster images
- **TIFF**: Uncompressed raster images

#### Vector Formats
- **PDF**: Publication-quality vector graphics
- **SVG**: Scalable vector graphics for web
- **EPS**: Encapsulated PostScript for LaTeX

#### Data Formats
- **JSON**: Plot data and metadata
- **HTML**: Interactive web-based plots (with Plotly)

#### Quality Presets
- **Draft**: 150 DPI, optimized for speed
- **Standard**: 300 DPI, balanced quality/size
- **High**: 600 DPI, maximum quality
- **Publication**: 300 DPI with tight layout
- **Presentation**: Large format for slides

## Architecture

### Component Overview

```
VisualizationManager
├── InteractivePlotter    # Creates interactive plots
├── ChartGenerator       # Generates analysis charts
├── PlotCustomizer      # Applies styling and themes
├── ExportManager       # Handles export operations
└── RealTimeUpdater     # Manages live updates
```

### Key Classes

#### VisualizationManager
Central coordinator for all visualization operations. Provides unified API for creating, customizing, and exporting plots.

#### InteractivePlotter
Creates interactive plots with support for multiple data series, various plot types, and real-time interaction capabilities.

#### ChartGenerator
Generates standardized charts from ICARUS analysis results, with templates for common aerospace analysis types.

#### PlotCustomizer
Applies comprehensive customizations including colors, styles, fonts, and layout options with preset themes.

#### ExportManager
Handles export to multiple formats with quality presets and batch export capabilities.

#### RealTimeUpdater
Manages real-time plot updates during analysis execution with support for different data source types.

## Usage Examples

### Basic Interactive Plotting

```python
from visualization import VisualizationManager
import numpy as np

# Initialize visualization manager
viz_manager = VisualizationManager()

# Create sample data
x = np.linspace(0, 10, 100)
data = {
    'x': x.tolist(),
    'y': {'sin(x)': np.sin(x).tolist(), 'cos(x)': np.cos(x).tolist()},
    'xlabel': 'X Values',
    'ylabel': 'Y Values'
}

# Create interactive plot
plot_id = viz_manager.create_interactive_plot(
    data=data,
    plot_type="line",
    title="Trigonometric Functions"
)
```

### Analysis Chart Generation

```python
# ICARUS airfoil analysis results
airfoil_results = {
    'alpha': [-5, 0, 5, 10, 15],
    'cl': [-0.2, 0.0, 0.5, 1.0, 1.2],
    'cd': [0.008, 0.006, 0.008, 0.015, 0.025],
    'cm': [0.02, 0.0, -0.02, -0.05, -0.08]
}

# Generate airfoil polar chart
chart_id = viz_manager.generate_chart(
    analysis_results=airfoil_results,
    chart_type="airfoil_polar"
)
```

### Plot Customization

```python
# Apply aerospace theme
customizations = {
    'color_palette': 'aerospace',
    'line_widths': 2,
    'grid': {'visible': True, 'alpha': 0.3},
    'legend': {'show': True, 'loc': 'upper right'},
    'font_config': 'publication'
}

viz_manager.customize_plot(plot_id, customizations)
```

### Export Operations

```python
# Export single plot
viz_manager.export_plot(
    plot_id=plot_id,
    output_path="analysis_results.pdf",
    format="pdf",
    quality_preset="publication"
)

# Batch export
viz_manager.batch_export(
    plot_ids=[plot_id1, plot_id2, chart_id],
    output_directory="./exports",
    format="png",
    quality_preset="high"
)
```

### Real-time Updates

```python
# Create data source
def analysis_data_source():
    # Return current analysis state
    return {
        'iteration': current_iteration,
        'residual': current_residual,
        'cl': current_cl
    }

# Start real-time updates
viz_manager.start_real_time_updates(
    plot_id=plot_id,
    data_source=analysis_data_source,
    update_interval=1.0
)
```

## TUI Integration

The visualization system integrates seamlessly with the Textual TUI framework through the `VisualizationScreen` component:

```python
from visualization.tui_visualization_screen import VisualizationScreen

# Add to your Textual app
class IcarusApp(App):
    def action_show_visualization(self):
        self.push_screen(VisualizationScreen())
```

### TUI Features

- **Tabbed Interface**: Organized tabs for different operations
- **Active Plots Management**: View and manage all active plots
- **Interactive Creation**: Form-based plot creation
- **Live Customization**: Real-time plot customization
- **Export Interface**: Guided export with format selection
- **Real-time Controls**: Start/stop/pause real-time updates

## Requirements

### Core Dependencies
```
matplotlib>=3.7.0    # Core plotting library
numpy>=1.24.0       # Numerical operations
rich>=13.0.0        # Terminal output formatting
```

### Optional Dependencies
```
plotly>=5.0.0       # Interactive HTML exports
textual>=0.41.0     # TUI framework integration
```

## Installation

The visualization system is included with the ICARUS CLI. Ensure you have the required dependencies:

```bash
pip install matplotlib numpy rich
```

For full functionality including TUI integration:

```bash
pip install -r cli/requirements.txt
```

## Testing

Run the comprehensive test suite:

```bash
python cli/visualization/test_visualization_system.py
```

Run the demonstration:

```bash
# Command-line demo
python cli/demo_visualization_system.py

# ICARUS integration demo
python cli/demo_visualization_system.py icarus

# TUI demo (requires textual)
python cli/demo_visualization_system.py tui
```

## Performance Considerations

### Memory Management
- Automatic cleanup of closed plots
- Configurable data buffer limits for real-time updates
- Efficient data serialization for exports

### Rendering Performance
- Optimized matplotlib backends
- Batch operations for multiple plots
- Asynchronous export operations

### File Size Optimization
- Quality presets for different use cases
- Vector formats for scalable graphics
- Compression options for raster formats

## Integration with ICARUS

The visualization system is designed to work seamlessly with ICARUS analysis modules:

### Supported Analysis Types
- **Airfoil Analysis**: XFoil, panel methods
- **Airplane Analysis**: AVL, vortex lattice methods
- **CFD Results**: Pressure distributions, flow fields
- **Optimization**: Parameter studies, sensitivity analysis

### Data Format Compatibility
- Direct integration with ICARUS data structures
- Automatic unit handling and conversion
- Metadata preservation for traceability

## Error Handling

The system provides comprehensive error handling:

- **Input Validation**: Data format and parameter validation
- **Graceful Degradation**: Fallback options for missing features
- **User Feedback**: Clear error messages and suggestions
- **Recovery Options**: Automatic cleanup and state restoration

## Future Enhancements

### Planned Features
- **3D Visualization**: Enhanced 3D plotting capabilities
- **Animation Support**: Animated plots for time-series data
- **Interactive Widgets**: Sliders and controls for parameter exploration
- **Cloud Integration**: Direct export to cloud storage services
- **Collaborative Features**: Shared plotting sessions

### Web Migration Readiness
The visualization system is designed with web migration in mind:
- API-based architecture for frontend flexibility
- JSON serialization for web compatibility
- Plotly integration for interactive web plots
- RESTful endpoints for remote plotting

## Contributing

When contributing to the visualization system:

1. **Follow the Architecture**: Use the established component pattern
2. **Add Tests**: Include comprehensive tests for new features
3. **Update Documentation**: Keep README and docstrings current
4. **Consider Performance**: Optimize for large datasets
5. **Maintain Compatibility**: Ensure backward compatibility

## License

This visualization system is part of the ICARUS CLI project and follows the same licensing terms.
