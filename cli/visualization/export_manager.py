"""Export Manager - Handles plot export to various formats

This module provides comprehensive export capabilities for plots,
supporting multiple formats and publication-quality output.
"""

import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from matplotlib.figure import Figure
from rich.console import Console


class ExportManager:
    """Manages plot export operations."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the export manager.

        Args:
            console: Rich console for output (optional)
        """
        self.console = console or Console()

        # Supported export formats
        self.supported_formats = {
            "png": {
                "extension": ".png",
                "description": "Portable Network Graphics",
                "default_dpi": 300,
                "supports_transparency": True,
            },
            "pdf": {
                "extension": ".pdf",
                "description": "Portable Document Format",
                "default_dpi": 300,
                "vector": True,
            },
            "svg": {
                "extension": ".svg",
                "description": "Scalable Vector Graphics",
                "vector": True,
                "supports_transparency": True,
            },
            "eps": {
                "extension": ".eps",
                "description": "Encapsulated PostScript",
                "vector": True,
            },
            "jpg": {
                "extension": ".jpg",
                "description": "JPEG Image",
                "default_dpi": 300,
                "supports_transparency": False,
            },
            "tiff": {
                "extension": ".tiff",
                "description": "Tagged Image File Format",
                "default_dpi": 300,
                "supports_transparency": True,
            },
            "html": {
                "extension": ".html",
                "description": "Interactive HTML with Plotly",
                "interactive": True,
            },
            "json": {
                "extension": ".json",
                "description": "Plot data in JSON format",
                "data_only": True,
            },
        }

        # Quality presets
        self.quality_presets = {
            "draft": {"dpi": 150, "quality": 70},
            "standard": {"dpi": 300, "quality": 85},
            "high": {"dpi": 600, "quality": 95},
            "publication": {"dpi": 300, "quality": 100, "bbox_inches": "tight"},
            "presentation": {"dpi": 150, "figsize": (12, 8)},
        }

    def export_plot(
        self,
        plot: Figure,
        output_path: Union[str, Path],
        format: str = "png",
        **options,
    ) -> bool:
        """Export a plot to file.

        Args:
            plot: Figure to export
            output_path: Path where to save the plot
            format: Export format
            **options: Additional export options

        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)

            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Add extension if not present
            if not output_path.suffix:
                extension = self.supported_formats[format]["extension"]
                output_path = output_path.with_suffix(extension)

            # Apply quality preset if specified
            if "quality_preset" in options:
                preset = self.quality_presets.get(options["quality_preset"], {})
                options.update(preset)

            # Export based on format
            if format in ["png", "pdf", "svg", "eps", "jpg", "tiff"]:
                self._export_matplotlib(plot, output_path, format, **options)
            elif format == "html":
                self._export_html(plot, output_path, **options)
            elif format == "json":
                self._export_json(plot, output_path, **options)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.console.print(f"[green]✓[/green] Exported plot to {output_path}")
            return True

        except Exception as e:
            self.console.print(f"[red]✗[/red] Export failed: {e}")
            return False

    def _export_matplotlib(
        self,
        plot: Figure,
        output_path: Path,
        format: str,
        **options,
    ) -> None:
        """Export using matplotlib's savefig."""
        # Get format info
        format_info = self.supported_formats[format]

        # Set default options
        save_options = {
            "dpi": options.get("dpi", format_info.get("default_dpi", 300)),
            "bbox_inches": options.get("bbox_inches", "tight"),
            "pad_inches": options.get("pad_inches", 0.1),
            "facecolor": options.get("facecolor", "white"),
            "edgecolor": options.get("edgecolor", "none"),
        }

        # Format-specific options
        if format == "png":
            save_options["transparent"] = options.get("transparent", False)
        elif format == "jpg":
            save_options["quality"] = options.get("quality", 95)
        elif format == "pdf":
            save_options["metadata"] = options.get(
                "metadata",
                {"Title": "ICARUS Analysis Plot", "Creator": "ICARUS CLI"},
            )

        # Save the plot
        plot.savefig(str(output_path), format=format, **save_options)

    def _export_html(self, plot: Figure, output_path: Path, **options) -> None:
        """Export as interactive HTML using plotly conversion."""
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot as plotly_plot

            # Convert matplotlib to plotly (simplified)
            fig_plotly = self._convert_to_plotly(plot)

            # Export as HTML
            plotly_plot(
                fig_plotly,
                filename=str(output_path),
                auto_open=False,
                include_plotlyjs=options.get("include_plotlyjs", True),
            )

        except ImportError:
            # Fallback: save as static HTML with embedded image
            self._export_html_fallback(plot, output_path, **options)

    def _export_html_fallback(self, plot: Figure, output_path: Path, **options) -> None:
        """Fallback HTML export without plotly."""
        # Save plot as PNG first
        png_path = output_path.with_suffix(".png")
        self._export_matplotlib(plot, png_path, "png", **options)

        # Create HTML with embedded image
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ICARUS Analysis Plot</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .plot-container {{ text-align: center; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="plot-container">
                <h1>ICARUS Analysis Plot</h1>
                <img src="{png_path.name}" alt="Analysis Plot">
            </div>
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)

    def _export_json(self, plot: Figure, output_path: Path, **options) -> None:
        """Export plot data as JSON."""
        plot_data = self._extract_plot_data(plot)

        with open(output_path, "w") as f:
            json.dump(plot_data, f, indent=2, default=str)

    def _extract_plot_data(self, plot: Figure) -> Dict[str, Any]:
        """Extract data from a matplotlib figure."""
        data = {
            "figure_info": {
                "size": plot.get_size_inches().tolist(),
                "dpi": plot.dpi,
                "title": plot._suptitle.get_text() if plot._suptitle else None,
            },
            "axes": [],
        }

        for i, ax in enumerate(plot.axes):
            ax_data = {
                "index": i,
                "title": ax.get_title(),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "xlim": ax.get_xlim(),
                "ylim": ax.get_ylim(),
                "lines": [],
                "collections": [],
            }

            # Extract line data
            for j, line in enumerate(ax.lines):
                line_data = {
                    "index": j,
                    "label": line.get_label(),
                    "color": line.get_color(),
                    "linestyle": line.get_linestyle(),
                    "linewidth": line.get_linewidth(),
                    "marker": line.get_marker(),
                    "markersize": line.get_markersize(),
                    "xdata": line.get_xdata().tolist(),
                    "ydata": line.get_ydata().tolist(),
                }
                ax_data["lines"].append(line_data)

            # Extract collection data (for scatter plots, etc.)
            for j, collection in enumerate(ax.collections):
                collection_data = {"index": j, "type": type(collection).__name__}
                ax_data["collections"].append(collection_data)

            data["axes"].append(ax_data)

        return data

    def _convert_to_plotly(self, plot: Figure) -> "go.Figure":
        """Convert matplotlib figure to plotly (simplified)."""
        import plotly.graph_objects as go

        fig_plotly = go.Figure()

        # Convert first axis only (simplified)
        if plot.axes:
            ax = plot.axes[0]

            for line in ax.lines:
                fig_plotly.add_trace(
                    go.Scatter(
                        x=line.get_xdata(),
                        y=line.get_ydata(),
                        mode="lines+markers",
                        name=line.get_label(),
                        line=dict(color=line.get_color(), width=line.get_linewidth()),
                        marker=dict(size=line.get_markersize()),
                    ),
                )

            fig_plotly.update_layout(
                title=ax.get_title(),
                xaxis_title=ax.get_xlabel(),
                yaxis_title=ax.get_ylabel(),
            )

        return fig_plotly

    def batch_export(
        self,
        plots: List[Figure],
        output_directory: Union[str, Path],
        format: str = "png",
        naming_pattern: str = "plot_{index}",
        **options,
    ) -> Dict[int, bool]:
        """Export multiple plots in batch.

        Args:
            plots: List of figures to export
            output_directory: Directory to save plots
            format: Export format
            naming_pattern: Pattern for naming files (can include {index})
            **options: Export options

        Returns:
            Dictionary mapping plot indices to success status
        """
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for i, plot in enumerate(plots):
            filename = naming_pattern.format(index=i)
            output_path = output_dir / filename

            results[i] = self.export_plot(
                plot=plot,
                output_path=output_path,
                format=format,
                **options,
            )

        successful = sum(results.values())
        total = len(plots)
        self.console.print(
            f"[green]✓[/green] Batch export: {successful}/{total} successful",
        )

        return results

    def create_export_preset(self, name: str, options: Dict[str, Any]) -> None:
        """Create an export preset for reuse.

        Args:
            name: Name of the preset
            options: Export options dictionary
        """
        if not hasattr(self, "export_presets"):
            self.export_presets = {}

        self.export_presets[name] = options.copy()
        self.console.print(f"[green]✓[/green] Created export preset: {name}")

    def export_with_preset(
        self,
        plot: Figure,
        output_path: Union[str, Path],
        preset_name: str,
    ) -> bool:
        """Export using a preset.

        Args:
            plot: Figure to export
            output_path: Output path
            preset_name: Name of the preset to use

        Returns:
            True if successful, False otherwise
        """
        if (
            not hasattr(self, "export_presets")
            or preset_name not in self.export_presets
        ):
            raise ValueError(f"Export preset '{preset_name}' not found")

        options = self.export_presets[preset_name]
        format = options.pop("format", "png")

        return self.export_plot(plot, output_path, format, **options)

    def get_supported_formats(self) -> Dict[str, Dict[str, Any]]:
        """Get information about supported formats.

        Returns:
            Dictionary of format information
        """
        return self.supported_formats.copy()

    def get_quality_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get available quality presets.

        Returns:
            Dictionary of quality presets
        """
        return self.quality_presets.copy()

    def get_export_presets(self) -> List[str]:
        """Get list of available export presets.

        Returns:
            List of preset names
        """
        if not hasattr(self, "export_presets"):
            return []
        return list(self.export_presets.keys())

    def estimate_file_size(
        self,
        plot: Figure,
        format: str = "png",
        **options,
    ) -> Optional[str]:
        """Estimate the file size for export.

        Args:
            plot: Figure to analyze
            format: Export format
            **options: Export options

        Returns:
            Estimated file size as string or None if cannot estimate
        """
        try:
            # Get figure dimensions
            width, height = plot.get_size_inches()
            dpi = options.get(
                "dpi",
                self.supported_formats[format].get("default_dpi", 300),
            )

            # Calculate pixel dimensions
            pixel_width = int(width * dpi)
            pixel_height = int(height * dpi)
            total_pixels = pixel_width * pixel_height

            # Estimate based on format
            if format == "png":
                # PNG: roughly 3-4 bytes per pixel (with compression)
                estimated_bytes = total_pixels * 3.5
            elif format == "jpg":
                # JPEG: varies with quality, roughly 0.5-2 bytes per pixel
                quality = options.get("quality", 95)
                compression_factor = 0.5 + (quality / 100) * 1.5
                estimated_bytes = total_pixels * compression_factor
            elif format in ["pdf", "svg", "eps"]:
                # Vector formats: harder to estimate, rough approximation
                estimated_bytes = total_pixels * 0.1  # Very rough estimate
            else:
                return None

            # Convert to human-readable format
            if estimated_bytes < 1024:
                return f"{estimated_bytes:.0f} B"
            elif estimated_bytes < 1024**2:
                return f"{estimated_bytes/1024:.1f} KB"
            elif estimated_bytes < 1024**3:
                return f"{estimated_bytes/(1024**2):.1f} MB"
            else:
                return f"{estimated_bytes/(1024**3):.1f} GB"

        except Exception:
            return None
