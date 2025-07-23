"""Layout Helper Utilities for ICARUS CLI

Provides utility functions and classes to assist with responsive layouts,
widget positioning, and dynamic layout management.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from textual.containers import Container
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.geometry import Size
from textual.widget import Widget


class LayoutType(Enum):
    """Types of layout arrangements."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    GRID = "grid"
    STACK = "stack"
    FLEX = "flex"


class Alignment(Enum):
    """Alignment options for layouts."""

    START = "start"
    CENTER = "center"
    END = "end"
    STRETCH = "stretch"
    SPACE_BETWEEN = "space_between"
    SPACE_AROUND = "space_around"


@dataclass
class LayoutConstraints:
    """Constraints for widget layout."""

    min_width: Optional[int] = None
    max_width: Optional[int] = None
    min_height: Optional[int] = None
    max_height: Optional[int] = None

    # Flex properties
    flex_grow: float = 0.0
    flex_shrink: float = 1.0
    flex_basis: Optional[int] = None

    # Alignment
    align_self: Optional[Alignment] = None

    # Margins and padding
    margin_top: int = 0
    margin_right: int = 0
    margin_bottom: int = 0
    margin_left: int = 0

    padding_top: int = 0
    padding_right: int = 0
    padding_bottom: int = 0
    padding_left: int = 0

    @property
    def margin(self) -> Tuple[int, int, int, int]:
        """Get margin as (top, right, bottom, left)."""
        return (
            self.margin_top,
            self.margin_right,
            self.margin_bottom,
            self.margin_left,
        )

    @property
    def padding(self) -> Tuple[int, int, int, int]:
        """Get padding as (top, right, bottom, left)."""
        return (
            self.padding_top,
            self.padding_right,
            self.padding_bottom,
            self.padding_left,
        )


@dataclass
class GridPosition:
    """Position in a grid layout."""

    row: int
    column: int
    row_span: int = 1
    column_span: int = 1


class LayoutHelper:
    """Helper class for managing responsive layouts."""

    @staticmethod
    def calculate_optimal_columns(
        container_width: int,
        item_min_width: int,
        item_max_width: int,
        gap: int = 1,
    ) -> int:
        """Calculate optimal number of columns for a grid layout."""
        if container_width <= 0 or item_min_width <= 0:
            return 1

        # Try different column counts and find the best fit
        best_columns = 1
        best_waste = float("inf")

        max_possible_columns = container_width // (item_min_width + gap)

        for columns in range(1, max_possible_columns + 1):
            available_width = container_width - (gap * (columns - 1))
            item_width = available_width // columns

            if item_width < item_min_width:
                break

            if item_width > item_max_width:
                item_width = item_max_width

            used_width = (item_width * columns) + (gap * (columns - 1))
            waste = container_width - used_width

            if waste < best_waste:
                best_waste = waste
                best_columns = columns

        return best_columns

    @staticmethod
    def distribute_space(
        total_space: int,
        items: List[Dict[str, Any]],
        gap: int = 0,
    ) -> List[int]:
        """Distribute space among items based on their constraints."""
        if not items:
            return []

        # Calculate total gap space
        total_gap = gap * (len(items) - 1) if len(items) > 1 else 0
        available_space = total_space - total_gap

        if available_space <= 0:
            return [0] * len(items)

        # Initialize sizes with minimum widths
        sizes = []
        remaining_space = available_space

        for item in items:
            min_size = item.get("min_width", 0)
            sizes.append(min_size)
            remaining_space -= min_size

        if remaining_space <= 0:
            return sizes

        # Distribute remaining space based on flex_grow
        total_flex_grow = sum(item.get("flex_grow", 0) for item in items)

        if total_flex_grow > 0:
            for i, item in enumerate(items):
                flex_grow = item.get("flex_grow", 0)
                if flex_grow > 0:
                    additional_space = int(
                        (flex_grow / total_flex_grow) * remaining_space,
                    )
                    max_size = item.get("max_width", float("inf"))
                    sizes[i] = min(sizes[i] + additional_space, max_size)
        else:
            # Distribute equally if no flex_grow specified
            additional_per_item = remaining_space // len(items)
            for i, item in enumerate(items):
                max_size = item.get("max_width", float("inf"))
                sizes[i] = min(sizes[i] + additional_per_item, max_size)

        return sizes

    @staticmethod
    def create_responsive_grid(
        widgets: List[Widget],
        container_width: int,
        item_min_width: int = 20,
        item_max_width: int = 40,
        gap: int = 1,
    ) -> Container:
        """Create a responsive grid layout."""
        columns = LayoutHelper.calculate_optimal_columns(
            container_width,
            item_min_width,
            item_max_width,
            gap,
        )

        grid_container = Container()
        current_row = None

        for i, widget in enumerate(widgets):
            if i % columns == 0:
                # Start new row
                current_row = Horizontal()
                grid_container.mount(current_row)

            if current_row:
                current_row.mount(widget)

        return grid_container

    @staticmethod
    def create_flex_layout(
        widgets: List[Tuple[Widget, LayoutConstraints]],
        layout_type: LayoutType = LayoutType.HORIZONTAL,
        alignment: Alignment = Alignment.START,
        gap: int = 1,
    ) -> Container:
        """Create a flex layout with constraints."""
        if layout_type == LayoutType.HORIZONTAL:
            container = Horizontal()
        else:
            container = Vertical()

        for widget, constraints in widgets:
            # Apply constraints to widget styles
            if constraints.min_width:
                widget.styles.min_width = constraints.min_width
            if constraints.max_width:
                widget.styles.max_width = constraints.max_width
            if constraints.min_height:
                widget.styles.min_height = constraints.min_height
            if constraints.max_height:
                widget.styles.max_height = constraints.max_height

            # Apply margins
            widget.styles.margin = constraints.margin

            container.mount(widget)

        return container

    @staticmethod
    def calculate_aspect_ratio_size(
        target_width: int,
        target_height: int,
        aspect_ratio: float,
    ) -> Tuple[int, int]:
        """Calculate size maintaining aspect ratio."""
        # Try fitting by width
        width_based_height = int(target_width / aspect_ratio)
        if width_based_height <= target_height:
            return target_width, width_based_height

        # Fit by height
        height_based_width = int(target_height * aspect_ratio)
        return height_based_width, target_height

    @staticmethod
    def get_optimal_text_columns(
        text_length: int,
        container_width: int,
        min_column_width: int = 20,
    ) -> int:
        """Calculate optimal number of text columns."""
        if container_width < min_column_width:
            return 1

        max_columns = container_width // min_column_width

        # Try different column counts
        for columns in range(1, max_columns + 1):
            chars_per_column = text_length // columns
            column_width = container_width // columns

            # Check if text fits reasonably in columns
            if chars_per_column <= column_width * 3:  # Rough estimate
                return columns

        return max_columns

    @staticmethod
    def create_sidebar_layout(
        sidebar_widget: Widget,
        main_widget: Widget,
        sidebar_width: Union[int, str] = "30%",
        sidebar_position: str = "left",
        collapsible: bool = True,
    ) -> Container:
        """Create a sidebar layout."""
        if sidebar_position == "left":
            container = Horizontal()
            container.mount(sidebar_widget)
            container.mount(main_widget)
        else:  # right
            container = Horizontal()
            container.mount(main_widget)
            container.mount(sidebar_widget)

        # Set sidebar width
        if isinstance(sidebar_width, str) and sidebar_width.endswith("%"):
            percentage = int(sidebar_width[:-1])
            sidebar_widget.styles.width = f"{percentage}%"
            main_widget.styles.width = f"{100 - percentage}%"
        else:
            sidebar_widget.styles.width = sidebar_width

        # Add collapsible behavior if requested
        if collapsible:
            sidebar_widget.add_class("collapsible-sidebar")

        return container

    @staticmethod
    def create_header_footer_layout(
        header_widget: Optional[Widget],
        main_widget: Widget,
        footer_widget: Optional[Widget],
        header_height: int = 3,
        footer_height: int = 1,
    ) -> Container:
        """Create a header-main-footer layout."""
        container = Vertical()

        if header_widget:
            header_widget.styles.height = header_height
            container.mount(header_widget)

        main_widget.styles.height = "1fr"  # Take remaining space
        container.mount(main_widget)

        if footer_widget:
            footer_widget.styles.height = footer_height
            container.mount(footer_widget)

        return container

    @staticmethod
    def create_tabbed_layout(
        tabs: List[Tuple[str, Widget]],
        default_tab: int = 0,
    ) -> Container:
        """Create a tabbed layout."""
        from textual.widgets import TabbedContent

        tabbed_content = TabbedContent()

        for i, (tab_name, widget) in enumerate(tabs):
            with tabbed_content.add_pane(tab_name, id=f"tab_{i}"):
                tabbed_content.mount(widget)

        # Set active tab
        if 0 <= default_tab < len(tabs):
            tabbed_content.active = f"tab_{default_tab}"

        return tabbed_content

    @staticmethod
    def apply_responsive_classes(
        widget: Widget,
        breakpoint_classes: Dict[str, List[str]],
    ) -> None:
        """Apply responsive CSS classes based on breakpoints."""
        for breakpoint, classes in breakpoint_classes.items():
            for css_class in classes:
                widget.add_class(f"{css_class}-{breakpoint}")

    @staticmethod
    def calculate_grid_positions(item_count: int, columns: int) -> List[GridPosition]:
        """Calculate grid positions for items."""
        positions = []

        for i in range(item_count):
            row = i // columns
            column = i % columns
            positions.append(GridPosition(row, column))

        return positions

    @staticmethod
    def create_masonry_layout(
        widgets: List[Widget],
        columns: int = 3,
        gap: int = 1,
    ) -> Container:
        """Create a masonry-style layout (simplified for terminal)."""
        # Create column containers
        column_containers = [Vertical() for _ in range(columns)]
        column_heights = [0] * columns

        # Distribute widgets to columns (simplified algorithm)
        for widget in widgets:
            # Find shortest column
            shortest_column = min(range(columns), key=lambda i: column_heights[i])

            # Add widget to shortest column
            column_containers[shortest_column].mount(widget)

            # Estimate height (simplified)
            estimated_height = getattr(widget, "estimated_height", 5)
            column_heights[shortest_column] += estimated_height + gap

        # Create main container
        main_container = Horizontal()
        for column in column_containers:
            main_container.mount(column)

        return main_container

    @staticmethod
    def get_layout_info(container: Container) -> Dict[str, Any]:
        """Get information about a container's layout."""
        info = {
            "type": type(container).__name__,
            "children_count": len(container.children),
            "size": getattr(container, "size", Size(0, 0)),
            "styles": {},
        }

        # Extract relevant style information
        if hasattr(container, "styles"):
            styles = container.styles
            info["styles"] = {
                "width": getattr(styles, "width", None),
                "height": getattr(styles, "height", None),
                "margin": getattr(styles, "margin", None),
                "padding": getattr(styles, "padding", None),
            }

        return info
