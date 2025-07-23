"""Plot Customizer - Applies customizations to existing plots

This module provides comprehensive plot customization capabilities,
allowing users to modify colors, styles, labels, and other plot properties.
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.console import Console


class PlotCustomizer:
    """Applies customizations to existing plots."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the plot customizer.

        Args:
            console: Rich console for output (optional)
        """
        self.console = console or Console()

        # Available customization options
        self.color_palettes = {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "aerospace": ["#003f5c", "#2f4b7c", "#665191", "#a05195", "#d45087"],
            "publication": ["#000000", "#333333", "#666666", "#999999", "#cccccc"],
            "colorblind": ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"],
            "vibrant": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"],
        }

        self.line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
        self.markers = ["o", "s", "^", "v", "D", "p", "*", "h", "+", "x"]

        # Font configurations
        self.font_configs = {
            "small": {"size": 8},
            "medium": {"size": 10},
            "large": {"size": 12},
            "xlarge": {"size": 14},
            "publication": {"size": 10, "family": "serif"},
            "presentation": {"size": 14, "weight": "bold"},
        }

    def apply_customizations(
        self,
        plot: Figure,
        customizations: Dict[str, Any],
    ) -> Figure:
        """Apply customizations to a plot.

        Args:
            plot: Figure to customize
            customizations: Dictionary of customization options

        Returns:
            Customized Figure object
        """
        try:
            # Apply figure-level customizations
            self._apply_figure_customizations(plot, customizations)

            # Apply axis-level customizations
            for i, ax in enumerate(plot.axes):
                axis_custom = customizations.get(f"axis_{i}", customizations)
                self._apply_axis_customizations(ax, axis_custom)

            # Apply line/data customizations
            self._apply_data_customizations(plot, customizations)

            # Apply text customizations
            self._apply_text_customizations(plot, customizations)

            # Apply layout customizations
            self._apply_layout_customizations(plot, customizations)

            self.console.print("[green]✓[/green] Applied plot customizations")
            return plot

        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to apply customizations: {e}")
            raise

    def _apply_figure_customizations(self, fig: Figure, custom: Dict[str, Any]) -> None:
        """Apply figure-level customizations."""
        # Figure size
        if "figsize" in custom:
            fig.set_size_inches(custom["figsize"])

        # Background color
        if "facecolor" in custom:
            fig.patch.set_facecolor(custom["facecolor"])

        # DPI
        if "dpi" in custom:
            fig.set_dpi(custom["dpi"])

        # Title
        if "suptitle" in custom:
            title_config = custom["suptitle"]
            if isinstance(title_config, str):
                fig.suptitle(title_config)
            elif isinstance(title_config, dict):
                fig.suptitle(
                    title_config.get("text", ""),
                    fontsize=title_config.get("fontsize", 14),
                    fontweight=title_config.get("fontweight", "bold"),
                    color=title_config.get("color", "black"),
                )

    def _apply_axis_customizations(self, ax: Axes, custom: Dict[str, Any]) -> None:
        """Apply axis-level customizations."""
        # Axis labels
        if "xlabel" in custom:
            label_config = custom["xlabel"]
            if isinstance(label_config, str):
                ax.set_xlabel(label_config)
            elif isinstance(label_config, dict):
                ax.set_xlabel(
                    label_config.get("text", ""),
                    fontsize=label_config.get("fontsize", 10),
                    fontweight=label_config.get("fontweight", "normal"),
                    color=label_config.get("color", "black"),
                )

        if "ylabel" in custom:
            label_config = custom["ylabel"]
            if isinstance(label_config, str):
                ax.set_ylabel(label_config)
            elif isinstance(label_config, dict):
                ax.set_ylabel(
                    label_config.get("text", ""),
                    fontsize=label_config.get("fontsize", 10),
                    fontweight=label_config.get("fontweight", "normal"),
                    color=label_config.get("color", "black"),
                )

        # Axis title
        if "title" in custom:
            title_config = custom["title"]
            if isinstance(title_config, str):
                ax.set_title(title_config)
            elif isinstance(title_config, dict):
                ax.set_title(
                    title_config.get("text", ""),
                    fontsize=title_config.get("fontsize", 12),
                    fontweight=title_config.get("fontweight", "bold"),
                    color=title_config.get("color", "black"),
                )

        # Axis limits
        if "xlim" in custom:
            ax.set_xlim(custom["xlim"])
        if "ylim" in custom:
            ax.set_ylim(custom["ylim"])

        # Grid
        if "grid" in custom:
            grid_config = custom["grid"]
            if isinstance(grid_config, bool):
                ax.grid(grid_config)
            elif isinstance(grid_config, dict):
                ax.grid(
                    grid_config.get("visible", True),
                    alpha=grid_config.get("alpha", 0.3),
                    color=grid_config.get("color", "gray"),
                    linestyle=grid_config.get("linestyle", "-"),
                    linewidth=grid_config.get("linewidth", 0.5),
                )

        # Axis background
        if "facecolor" in custom:
            ax.set_facecolor(custom["facecolor"])

        # Spines
        if "spines" in custom:
            spine_config = custom["spines"]
            for spine_name, spine_props in spine_config.items():
                if spine_name in ax.spines:
                    spine = ax.spines[spine_name]
                    if "visible" in spine_props:
                        spine.set_visible(spine_props["visible"])
                    if "color" in spine_props:
                        spine.set_color(spine_props["color"])
                    if "linewidth" in spine_props:
                        spine.set_linewidth(spine_props["linewidth"])

        # Tick parameters
        if "tick_params" in custom:
            ax.tick_params(**custom["tick_params"])

    def _apply_data_customizations(self, fig: Figure, custom: Dict[str, Any]) -> None:
        """Apply data/line customizations."""
        # Color palette
        if "color_palette" in custom:
            palette_name = custom["color_palette"]
            if palette_name in self.color_palettes:
                colors = self.color_palettes[palette_name]

                for ax in fig.axes:
                    for i, line in enumerate(ax.lines):
                        color = colors[i % len(colors)]
                        line.set_color(color)

        # Line styles
        if "line_styles" in custom:
            styles = custom["line_styles"]
            if isinstance(styles, list):
                for ax in fig.axes:
                    for i, line in enumerate(ax.lines):
                        style = styles[i % len(styles)]
                        line.set_linestyle(style)

        # Line widths
        if "line_widths" in custom:
            widths = custom["line_widths"]
            if isinstance(widths, (int, float)):
                # Apply same width to all lines
                for ax in fig.axes:
                    for line in ax.lines:
                        line.set_linewidth(widths)
            elif isinstance(widths, list):
                # Apply different widths
                for ax in fig.axes:
                    for i, line in enumerate(ax.lines):
                        width = widths[i % len(widths)]
                        line.set_linewidth(width)

        # Markers
        if "markers" in custom:
            markers = custom["markers"]
            if isinstance(markers, list):
                for ax in fig.axes:
                    for i, line in enumerate(ax.lines):
                        marker = markers[i % len(markers)]
                        line.set_marker(marker)

        # Marker sizes
        if "marker_sizes" in custom:
            sizes = custom["marker_sizes"]
            if isinstance(sizes, (int, float)):
                for ax in fig.axes:
                    for line in ax.lines:
                        line.set_markersize(sizes)
            elif isinstance(sizes, list):
                for ax in fig.axes:
                    for i, line in enumerate(ax.lines):
                        size = sizes[i % len(sizes)]
                        line.set_markersize(size)

        # Alpha (transparency)
        if "alpha" in custom:
            alpha = custom["alpha"]
            for ax in fig.axes:
                for line in ax.lines:
                    line.set_alpha(alpha)

    def _apply_text_customizations(self, fig: Figure, custom: Dict[str, Any]) -> None:
        """Apply text customizations."""
        # Font configuration
        if "font_config" in custom:
            font_name = custom["font_config"]
            if font_name in self.font_configs:
                font_config = self.font_configs[font_name]
                plt.rcParams.update(
                    {
                        "font.size": font_config.get("size", 10),
                        "font.family": font_config.get("family", "sans-serif"),
                        "font.weight": font_config.get("weight", "normal"),
                    },
                )

        # Legend customizations
        if "legend" in custom:
            legend_config = custom["legend"]
            for ax in fig.axes:
                if ax.get_legend():
                    # Remove existing legend
                    ax.get_legend().remove()

                # Create new legend with customizations
                if legend_config.get("show", True):
                    ax.legend(
                        loc=legend_config.get("loc", "best"),
                        fontsize=legend_config.get("fontsize", 10),
                        frameon=legend_config.get("frameon", True),
                        fancybox=legend_config.get("fancybox", True),
                        shadow=legend_config.get("shadow", False),
                        ncol=legend_config.get("ncol", 1),
                        bbox_to_anchor=legend_config.get("bbox_to_anchor", None),
                    )

    def _apply_layout_customizations(self, fig: Figure, custom: Dict[str, Any]) -> None:
        """Apply layout customizations."""
        # Tight layout
        if custom.get("tight_layout", True):
            fig.tight_layout()

        # Subplot adjustments
        if "subplots_adjust" in custom:
            fig.subplots_adjust(**custom["subplots_adjust"])

        # Constrained layout
        if "constrained_layout" in custom:
            fig.set_constrained_layout(custom["constrained_layout"])

    def create_customization_preset(
        self,
        name: str,
        customizations: Dict[str, Any],
    ) -> None:
        """Create a customization preset for reuse.

        Args:
            name: Name of the preset
            customizations: Customization dictionary
        """
        if not hasattr(self, "presets"):
            self.presets = {}

        self.presets[name] = customizations.copy()
        self.console.print(f"[green]✓[/green] Created customization preset: {name}")

    def apply_preset(self, plot: Figure, preset_name: str) -> Figure:
        """Apply a customization preset.

        Args:
            plot: Figure to customize
            preset_name: Name of the preset to apply

        Returns:
            Customized Figure object
        """
        if not hasattr(self, "presets") or preset_name not in self.presets:
            raise ValueError(f"Preset '{preset_name}' not found")

        return self.apply_customizations(plot, self.presets[preset_name])

    def get_available_presets(self) -> List[str]:
        """Get list of available customization presets.

        Returns:
            List of preset names
        """
        if not hasattr(self, "presets"):
            return []
        return list(self.presets.keys())

    def get_customization_options(self) -> Dict[str, Any]:
        """Get available customization options.

        Returns:
            Dictionary of available options
        """
        return {
            "color_palettes": list(self.color_palettes.keys()),
            "line_styles": self.line_styles,
            "markers": self.markers,
            "font_configs": list(self.font_configs.keys()),
            "figure_options": ["figsize", "facecolor", "dpi", "suptitle"],
            "axis_options": [
                "xlabel",
                "ylabel",
                "title",
                "xlim",
                "ylim",
                "grid",
                "facecolor",
                "spines",
                "tick_params",
            ],
            "data_options": [
                "color_palette",
                "line_styles",
                "line_widths",
                "markers",
                "marker_sizes",
                "alpha",
            ],
            "text_options": ["font_config", "legend"],
            "layout_options": ["tight_layout", "subplots_adjust", "constrained_layout"],
        }

    def generate_customization_template(
        self,
        plot_type: str = "general",
    ) -> Dict[str, Any]:
        """Generate a customization template.

        Args:
            plot_type: Type of plot to generate template for

        Returns:
            Template dictionary
        """
        if plot_type == "aerospace":
            return {
                "color_palette": "aerospace",
                "font_config": "publication",
                "grid": {
                    "visible": True,
                    "alpha": 0.3,
                    "color": "gray",
                    "linestyle": "-",
                },
                "legend": {
                    "show": True,
                    "loc": "best",
                    "fontsize": 10,
                    "frameon": True,
                },
                "line_widths": 2,
                "tight_layout": True,
            }
        elif plot_type == "publication":
            return {
                "color_palette": "publication",
                "font_config": "publication",
                "figsize": (6, 4),
                "dpi": 300,
                "grid": {
                    "visible": True,
                    "alpha": 0.2,
                    "color": "black",
                    "linestyle": "-",
                    "linewidth": 0.5,
                },
                "legend": {
                    "show": True,
                    "loc": "best",
                    "fontsize": 9,
                    "frameon": False,
                },
                "line_widths": 1.5,
                "tight_layout": True,
            }
        elif plot_type == "presentation":
            return {
                "color_palette": "vibrant",
                "font_config": "presentation",
                "figsize": (12, 8),
                "grid": {
                    "visible": True,
                    "alpha": 0.4,
                    "color": "gray",
                    "linestyle": "--",
                },
                "legend": {
                    "show": True,
                    "loc": "best",
                    "fontsize": 12,
                    "frameon": True,
                    "shadow": True,
                },
                "line_widths": 3,
                "marker_sizes": 8,
                "tight_layout": True,
            }
        else:
            return {
                "color_palette": "default",
                "font_config": "medium",
                "grid": {"visible": True, "alpha": 0.3},
                "legend": {"show": True, "loc": "best"},
                "line_widths": 2,
                "tight_layout": True,
            }
