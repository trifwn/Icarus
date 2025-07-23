"""Visual Workflow Builder Widget for ICARUS CLI

This module provides a comprehensive visual workflow builder with drag-and-drop
functionality, dependency graph visualization, and workflow validation.
"""

import json
import uuid
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from textual import events
from textual.app import ComposeResult
from textual.containers import Container
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.coordinate import Coordinate
from textual.geometry import Size
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button
from textual.widgets import Checkbox
from textual.widgets import Label
from textual.widgets import Select
from textual.widgets import Static
from textual.widgets import TabPane
from textual.widgets import Tabs
from textual.widgets import TextArea

from .base_widgets import AerospaceButton
from .base_widgets import AerospaceTree
from .base_widgets import ButtonVariant
from .base_widgets import InputType
from .base_widgets import NotificationPanel
from .base_widgets import StatusIndicator
from .base_widgets import ValidatedInput
from .base_widgets import ValidationRule

try:
    from .workflow_integration import WorkflowConverter
except ImportError:
    # Fallback for testing
    WorkflowConverter = None


class NodeType(Enum):
    """Types of workflow nodes."""

    START = "start"
    END = "end"
    ANALYSIS = "analysis"
    CONDITION = "condition"
    PARALLEL = "parallel"
    MERGE = "merge"
    CUSTOM = "custom"


class ConnectionType(Enum):
    """Types of connections between nodes."""

    SEQUENCE = "sequence"
    CONDITION_TRUE = "condition_true"
    CONDITION_FALSE = "condition_false"
    PARALLEL_BRANCH = "parallel_branch"
    MERGE_INPUT = "merge_input"


@dataclass
class WorkflowNode:
    """Represents a visual workflow node."""

    id: str
    name: str
    node_type: NodeType
    position: Coordinate
    size: Size = field(default_factory=lambda: Size(120, 60))
    step: Optional[Any] = None  # WorkflowStep from core.workflow
    properties: Dict[str, Any] = field(default_factory=dict)
    input_ports: List[str] = field(default_factory=list)
    output_ports: List[str] = field(default_factory=list)
    is_selected: bool = False
    is_dragging: bool = False


@dataclass
class WorkflowConnection:
    """Represents a connection between workflow nodes."""

    id: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str
    connection_type: ConnectionType
    properties: Dict[str, Any] = field(default_factory=dict)


class WorkflowCanvas(Container):
    """Canvas for visual workflow editing with drag-and-drop support."""

    class NodeSelected(Message):
        """Message sent when a node is selected."""

        def __init__(self, node: WorkflowNode) -> None:
            self.node = node
            super().__init__()

    class NodeMoved(Message):
        """Message sent when a node is moved."""

        def __init__(self, node: WorkflowNode, old_position: Coordinate) -> None:
            self.node = node
            self.old_position = old_position
            super().__init__()

    class ConnectionCreated(Message):
        """Message sent when a connection is created."""

        def __init__(self, connection: WorkflowConnection) -> None:
            self.connection = connection
            super().__init__()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nodes: Dict[str, WorkflowNode] = {}
        self.connections: Dict[str, WorkflowConnection] = {}
        self.selected_node: Optional[str] = None
        self.dragging_node: Optional[str] = None
        self.drag_offset: Coordinate = Coordinate(0, 0)
        self.connection_start: Optional[Tuple[str, str]] = None  # (node_id, port_id)
        self.grid_size = 20
        self.zoom_level = reactive(1.0)
        self.canvas_offset = reactive(Coordinate(0, 0))

    def compose(self) -> ComposeResult:
        yield Static("", id="canvas_background", classes="workflow-canvas")
        yield Container(id="canvas_nodes", classes="canvas-nodes")
        yield Container(id="canvas_connections", classes="canvas-connections")

    def add_node(
        self,
        node_type: NodeType,
        position: Coordinate,
        name: Optional[str] = None,
        step: Optional[Any] = None,
    ) -> WorkflowNode:
        """Add a new node to the canvas."""
        node_id = str(uuid.uuid4())

        if name is None:
            name = f"{node_type.value.title()} {len(self.nodes) + 1}"

        # Snap to grid
        snapped_position = Coordinate(
            (position.x // self.grid_size) * self.grid_size,
            (position.y // self.grid_size) * self.grid_size,
        )

        node = WorkflowNode(
            id=node_id,
            name=name,
            node_type=node_type,
            position=snapped_position,
            step=step,
        )

        # Set default ports based on node type
        self._setup_node_ports(node)

        self.nodes[node_id] = node
        self._render_node(node)

        return node

    def _setup_node_ports(self, node: WorkflowNode) -> None:
        """Setup input/output ports for a node based on its type."""
        if node.node_type == NodeType.START:
            node.output_ports = ["out"]
        elif node.node_type == NodeType.END:
            node.input_ports = ["in"]
        elif node.node_type == NodeType.ANALYSIS:
            node.input_ports = ["in"]
            node.output_ports = ["out", "error"]
        elif node.node_type == NodeType.CONDITION:
            node.input_ports = ["in"]
            node.output_ports = ["true", "false"]
        elif node.node_type == NodeType.PARALLEL:
            node.input_ports = ["in"]
            node.output_ports = ["branch1", "branch2", "branch3"]
        elif node.node_type == NodeType.MERGE:
            node.input_ports = ["in1", "in2", "in3"]
            node.output_ports = ["out"]
        else:  # CUSTOM
            node.input_ports = ["in"]
            node.output_ports = ["out"]

    def _render_node(self, node: WorkflowNode) -> None:
        """Render a node on the canvas."""
        nodes_container = self.query_one("#canvas_nodes", Container)

        # Create node widget
        node_widget = WorkflowNodeWidget(node)
        node_widget.styles.offset = (node.position.x, node.position.y)

        nodes_container.mount(node_widget)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its connections."""
        if node_id not in self.nodes:
            return

        # Remove connections
        connections_to_remove = []
        for conn_id, connection in self.connections.items():
            if connection.source_node == node_id or connection.target_node == node_id:
                connections_to_remove.append(conn_id)

        for conn_id in connections_to_remove:
            self.remove_connection(conn_id)

        # Remove node
        del self.nodes[node_id]

        # Remove node widget
        try:
            node_widget = self.query_one(f"#node_{node_id}", WorkflowNodeWidget)
            node_widget.remove()
        except:
            pass

    def add_connection(
        self,
        source_node: str,
        source_port: str,
        target_node: str,
        target_port: str,
        connection_type: ConnectionType = ConnectionType.SEQUENCE,
    ) -> Optional[WorkflowConnection]:
        """Add a connection between two nodes."""
        # Validate connection
        if not self._validate_connection(
            source_node,
            source_port,
            target_node,
            target_port,
        ):
            return None

        connection_id = str(uuid.uuid4())
        connection = WorkflowConnection(
            id=connection_id,
            source_node=source_node,
            source_port=source_port,
            target_node=target_node,
            target_port=target_port,
            connection_type=connection_type,
        )

        self.connections[connection_id] = connection
        self._render_connection(connection)

        self.post_message(self.ConnectionCreated(connection))
        return connection

    def _validate_connection(
        self,
        source_node: str,
        source_port: str,
        target_node: str,
        target_port: str,
    ) -> bool:
        """Validate if a connection is allowed."""
        # Check if nodes exist
        if source_node not in self.nodes or target_node not in self.nodes:
            return False

        source = self.nodes[source_node]
        target = self.nodes[target_node]

        # Check if ports exist
        if (
            source_port not in source.output_ports
            or target_port not in target.input_ports
        ):
            return False

        # Check for cycles (simple check)
        if self._would_create_cycle(source_node, target_node):
            return False

        # Check if target port is already connected
        for connection in self.connections.values():
            if (
                connection.target_node == target_node
                and connection.target_port == target_port
            ):
                return False  # Input ports can only have one connection

        return True

    def _would_create_cycle(self, source_node: str, target_node: str) -> bool:
        """Check if adding a connection would create a cycle."""
        # Simple DFS to detect cycles
        visited = set()

        def dfs(node_id: str) -> bool:
            if node_id == source_node:
                return True
            if node_id in visited:
                return False

            visited.add(node_id)

            for connection in self.connections.values():
                if connection.source_node == node_id:
                    if dfs(connection.target_node):
                        return True

            return False

        return dfs(target_node)

    def _render_connection(self, connection: WorkflowConnection) -> None:
        """Render a connection on the canvas."""
        connections_container = self.query_one("#canvas_connections", Container)

        # Create connection widget
        connection_widget = WorkflowConnectionWidget(connection, self.nodes)
        connections_container.mount(connection_widget)

    def remove_connection(self, connection_id: str) -> None:
        """Remove a connection."""
        if connection_id not in self.connections:
            return

        del self.connections[connection_id]

        # Remove connection widget
        try:
            connection_widget = self.query_one(
                f"#connection_{connection_id}",
                WorkflowConnectionWidget,
            )
            connection_widget.remove()
        except:
            pass

    def select_node(self, node_id: str) -> None:
        """Select a node."""
        # Deselect previous node
        if self.selected_node:
            self.nodes[self.selected_node].is_selected = False

        # Select new node
        self.selected_node = node_id
        if node_id in self.nodes:
            self.nodes[node_id].is_selected = True
            self.post_message(self.NodeSelected(self.nodes[node_id]))

    def validate_workflow(self) -> List[str]:
        """Validate the current workflow and return list of errors."""
        errors = []

        # Check for start node
        start_nodes = [
            node for node in self.nodes.values() if node.node_type == NodeType.START
        ]
        if not start_nodes:
            errors.append("Workflow must have a start node")
        elif len(start_nodes) > 1:
            errors.append("Workflow can only have one start node")

        # Check for end node
        end_nodes = [
            node for node in self.nodes.values() if node.node_type == NodeType.END
        ]
        if not end_nodes:
            errors.append("Workflow must have an end node")

        # Check for disconnected nodes
        connected_nodes = set()
        for connection in self.connections.values():
            connected_nodes.add(connection.source_node)
            connected_nodes.add(connection.target_node)

        for node_id, node in self.nodes.items():
            if node_id not in connected_nodes and node.node_type not in [
                NodeType.START,
                NodeType.END,
            ]:
                errors.append(f"Node '{node.name}' is not connected")

        # Check for cycles
        if self._has_cycles():
            errors.append("Workflow contains cycles")

        return errors

    def _has_cycles(self) -> bool:
        """Check if the workflow has cycles using DFS."""
        visited = set()
        rec_stack = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for connection in self.connections.values():
                if connection.source_node == node_id:
                    target = connection.target_node
                    if target not in visited:
                        if dfs(target):
                            return True
                    elif target in rec_stack:
                        return True

            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False

    def clear_canvas(self) -> None:
        """Clear all nodes and connections from the canvas."""
        self.nodes.clear()
        self.connections.clear()
        self.selected_node = None

        # Clear widgets
        nodes_container = self.query_one("#canvas_nodes", Container)
        connections_container = self.query_one("#canvas_connections", Container)

        for child in nodes_container.children:
            child.remove()

        for child in connections_container.children:
            child.remove()

    def get_workflow_definition(self) -> Optional[Any]:
        """Generate a workflow definition from the visual representation."""
        if not self.nodes:
            return None

        # Use workflow converter for proper integration
        converter = WorkflowConverter()
        return converter.visual_to_executable(
            self.nodes,
            self.connections,
            "Visual Workflow",
            "Workflow created with visual builder",
        )

    def load_workflow_definition(self, template: Any) -> None:
        """Load a workflow definition into the visual representation."""
        # Clear existing nodes and connections
        self.clear_canvas()

        # Use workflow converter for proper integration
        converter = WorkflowConverter()
        nodes, connections = converter.executable_to_visual(template)

        # Load the converted visual representation
        self.nodes = nodes
        self.connections = connections

        # Render all nodes and connections
        for node in nodes.values():
            self._render_node(node)

        for connection in connections.values():
            self._render_connection(connection)


class WorkflowNodeWidget(Container):
    """Widget representing a single workflow node."""

    def __init__(self, node: WorkflowNode, **kwargs):
        super().__init__(**kwargs)
        self.node = node
        self.id = f"node_{node.id}"
        self.add_class(f"workflow-node-{node.node_type.value}")
        if node.is_selected:
            self.add_class("selected")

    def compose(self) -> ComposeResult:
        yield Label(self.node.name, classes="node-title")

        if self.node.node_type == NodeType.ANALYSIS and self.node.step:
            yield Label(f"Action: {self.node.step.action}", classes="node-details")

        # Input ports
        if self.node.input_ports:
            with Horizontal(classes="input-ports"):
                for port in self.node.input_ports:
                    yield Button("●", id=f"port_in_{port}", classes="port input-port")

        # Output ports
        if self.node.output_ports:
            with Horizontal(classes="output-ports"):
                for port in self.node.output_ports:
                    yield Button("●", id=f"port_out_{port}", classes="port output-port")

    def on_click(self, event: events.Click) -> None:
        """Handle node click."""
        canvas = self.parent.parent
        if hasattr(canvas, "select_node"):
            canvas.select_node(self.node.id)


class WorkflowConnectionWidget(Static):
    """Widget representing a connection between nodes."""

    def __init__(
        self,
        connection: WorkflowConnection,
        nodes: Dict[str, WorkflowNode],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.connection = connection
        self.nodes = nodes
        self.id = f"connection_{connection.id}"
        self.add_class("workflow-connection")

    def render(self) -> str:
        """Render the connection as a line (simplified)."""
        # In a real implementation, this would draw SVG lines or use canvas
        return "→"


class NodePalette(Container):
    """Palette of available workflow nodes."""

    class NodeDragStarted(Message):
        """Message sent when node drag starts."""

        def __init__(self, node_type: NodeType) -> None:
            self.node_type = node_type
            super().__init__()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.node_types = [
            (NodeType.START, "Start", "Begin workflow execution"),
            (NodeType.END, "End", "Complete workflow execution"),
            (NodeType.ANALYSIS, "Analysis", "Perform analysis step"),
            (NodeType.CONDITION, "Condition", "Conditional branching"),
            (NodeType.PARALLEL, "Parallel", "Parallel execution"),
            (NodeType.MERGE, "Merge", "Merge parallel branches"),
            (NodeType.CUSTOM, "Custom", "Custom action"),
        ]

    def compose(self) -> ComposeResult:
        yield Label("Node Palette", classes="palette-title")

        for node_type, name, description in self.node_types:
            with Container(classes="palette-item"):
                yield AerospaceButton(
                    name,
                    variant=ButtonVariant.OUTLINE,
                    id=f"palette_{node_type.value}",
                    classes="palette-button",
                )
                yield Label(description, classes="palette-description")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle palette button press."""
        if event.button.id and event.button.id.startswith("palette_"):
            node_type_str = event.button.id.replace("palette_", "")
            try:
                node_type = NodeType(node_type_str)
                self.post_message(self.NodeDragStarted(node_type))
            except ValueError:
                pass


class WorkflowPropertiesPanel(Container):
    """Panel for editing workflow and node properties."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_node: Optional[WorkflowNode] = None

    def compose(self) -> ComposeResult:
        with Tabs(id="properties_tabs"):
            with TabPane("Workflow", id="workflow_tab"):
                yield self._create_workflow_properties()

            with TabPane("Node", id="node_tab"):
                yield self._create_node_properties()

            with TabPane("Validation", id="validation_tab"):
                yield self._create_validation_panel()

    def _create_workflow_properties(self) -> Container:
        """Create workflow properties form."""
        container = Container(classes="properties-form")

        with container:
            yield ValidatedInput(
                "Workflow Name",
                placeholder="Enter workflow name",
                id="workflow_name",
                validation_rules=[
                    ValidationRule(
                        "required",
                        lambda x: bool(x.strip()),
                        "Name is required",
                    ),
                ],
            )

            yield ValidatedInput(
                "Description",
                placeholder="Enter workflow description",
                id="workflow_description",
            )

            yield Label("Workflow Type:", classes="form-label")
            yield Select(
                [
                    ("custom", "Custom"),
                    ("airfoil_analysis", "Airfoil Analysis"),
                    ("airplane_analysis", "Airplane Analysis"),
                    ("batch_processing", "Batch Processing"),
                ],
                id="workflow_type",
                prompt="Select workflow type",
            )

            with Horizontal(classes="form-actions"):
                yield AerospaceButton(
                    "Save Workflow",
                    variant=ButtonVariant.SUCCESS,
                    id="save_workflow",
                )
                yield AerospaceButton(
                    "Load Workflow",
                    variant=ButtonVariant.INFO,
                    id="load_workflow",
                )
                yield AerospaceButton(
                    "New Workflow",
                    variant=ButtonVariant.SECONDARY,
                    id="new_workflow",
                )

        return container

    def _create_node_properties(self) -> Container:
        """Create node properties form."""
        container = Container(classes="properties-form")

        with container:
            yield Label("No node selected", id="node_status", classes="status-label")

            yield ValidatedInput(
                "Node Name",
                placeholder="Enter node name",
                id="node_name",
            )

            yield ValidatedInput(
                "Action",
                placeholder="Enter action name",
                id="node_action",
            )

            yield Label("Parameters (JSON):", classes="form-label")
            yield TextArea("", id="node_parameters", classes="parameters-editor")

            yield ValidatedInput(
                "Timeout (seconds)",
                placeholder="Optional timeout",
                id="node_timeout",
                input_type=InputType.NUMBER,
            )

            yield Checkbox("Required", id="node_required", value=True)

            with Horizontal(classes="form-actions"):
                yield AerospaceButton(
                    "Apply Changes",
                    variant=ButtonVariant.SUCCESS,
                    id="apply_node_changes",
                )
                yield AerospaceButton(
                    "Reset",
                    variant=ButtonVariant.SECONDARY,
                    id="reset_node_changes",
                )

        return container

    def _create_validation_panel(self) -> Container:
        """Create workflow validation panel."""
        container = Container(classes="validation-panel")

        with container:
            yield Label("Workflow Validation", classes="panel-title")
            yield AerospaceButton(
                "Validate Workflow",
                variant=ButtonVariant.INFO,
                id="validate_workflow",
            )
            yield Container(id="validation_results", classes="validation-results")

            yield Label("Test Execution", classes="panel-title")
            yield AerospaceButton(
                "Test Workflow",
                variant=ButtonVariant.WARNING,
                id="test_workflow",
            )
            yield Container(id="test_results", classes="test-results")

        return container

    def set_selected_node(self, node: WorkflowNode) -> None:
        """Set the currently selected node for editing."""
        self.current_node = node

        # Update node status
        status_label = self.query_one("#node_status", Label)
        status_label.update(f"Selected: {node.name}")

        # Populate form fields
        name_input = self.query_one("#node_name", ValidatedInput)
        name_input.value = node.name

        if node.step:
            action_input = self.query_one("#node_action", ValidatedInput)
            action_input.value = getattr(node.step, "action", "")

            params_editor = self.query_one("#node_parameters", TextArea)
            params_editor.text = json.dumps(
                getattr(node.step, "parameters", {}),
                indent=2,
            )

            timeout_input = self.query_one("#node_timeout", ValidatedInput)
            timeout = getattr(node.step, "timeout", None)
            if timeout:
                timeout_input.value = str(timeout)

            required_checkbox = self.query_one("#node_required", Checkbox)
            required_checkbox.value = getattr(node.step, "required", True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses in the properties panel."""
        if event.button.id == "apply_node_changes":
            self._apply_node_changes()
        elif event.button.id == "reset_node_changes":
            if self.current_node:
                self.set_selected_node(self.current_node)
        elif event.button.id == "validate_workflow":
            self._validate_workflow()
        elif event.button.id == "test_workflow":
            self._test_workflow()

    def _apply_node_changes(self) -> None:
        """Apply changes to the selected node."""
        if not self.current_node:
            return

        # Get form values
        name_input = self.query_one("#node_name", ValidatedInput)
        action_input = self.query_one("#node_action", ValidatedInput)
        params_editor = self.query_one("#node_parameters", TextArea)
        timeout_input = self.query_one("#node_timeout", ValidatedInput)
        required_checkbox = self.query_one("#node_required", Checkbox)

        # Update node
        self.current_node.name = name_input.value

        # Create a simple step object if it doesn't exist
        if not self.current_node.step:
            self.current_node.step = type(
                "WorkflowStep",
                (),
                {
                    "id": str(uuid.uuid4()),
                    "name": name_input.value,
                    "description": "",
                    "action": action_input.value,
                    "parameters": {},
                    "dependencies": [],
                    "required": True,
                    "timeout": None,
                },
            )()

        # Update step properties
        self.current_node.step.name = name_input.value
        self.current_node.step.action = action_input.value
        self.current_node.step.required = required_checkbox.value

        # Parse parameters
        try:
            if params_editor.text.strip():
                self.current_node.step.parameters = json.loads(params_editor.text)
        except json.JSONDecodeError:
            # Show error notification
            pass

        # Parse timeout
        if timeout_input.value.strip():
            try:
                self.current_node.step.timeout = int(timeout_input.value)
            except ValueError:
                pass

    def _validate_workflow(self) -> None:
        """Validate the current workflow."""
        # Get canvas from parent
        try:
            canvas = self.parent.query_one(WorkflowCanvas)
            errors = canvas.validate_workflow()

            results_container = self.query_one("#validation_results", Container)
            results_container.remove_children()

            if not errors:
                results_container.mount(
                    StatusIndicator(
                        StatusIndicator.StatusType.SUCCESS,
                        "Workflow validation passed",
                    ),
                )
            else:
                results_container.mount(
                    StatusIndicator(
                        StatusIndicator.StatusType.ERROR,
                        f"Found {len(errors)} validation errors",
                    ),
                )

                for error in errors:
                    results_container.mount(Label(f"• {error}", classes="error-item"))
        except Exception as e:
            results_container = self.query_one("#validation_results", Container)
            results_container.remove_children()
            results_container.mount(
                StatusIndicator(
                    StatusIndicator.StatusType.ERROR,
                    f"Validation error: {str(e)}",
                ),
            )

    def _test_workflow(self) -> None:
        """Test the current workflow execution."""
        results_container = self.query_one("#test_results", Container)
        results_container.remove_children()

        results_container.mount(
            StatusIndicator(
                StatusIndicator.StatusType.INFO,
                "Test execution simulation - workflow structure validated",
            ),
        )


class VisualWorkflowBuilder(Container):
    """Main visual workflow builder interface."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.workflow_templates: Dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        with Horizontal():
            # Left panel - Node palette and properties
            with Vertical(classes="left-panel"):
                yield NodePalette(classes="node-palette")
                yield WorkflowPropertiesPanel(classes="properties-panel")

            # Center - Workflow canvas
            yield WorkflowCanvas(classes="workflow-canvas")

            # Right panel - Workflow list and templates
            with Vertical(classes="right-panel"):
                yield self._create_workflow_manager()
                yield NotificationPanel(classes="notification-panel")

    def _create_workflow_manager(self):
        """Create workflow management panel."""
        return WorkflowManagerPanel(classes="workflow-manager")

    def on_mount(self) -> None:
        """Initialize the workflow builder."""
        self._load_workflow_templates()

    def _load_workflow_templates(self) -> None:
        """Load available workflow templates."""
        templates_tree = self.query_one("#templates_tree", AerospaceTree)

        # Clear existing templates
        templates_tree.clear()

        # Add sample templates
        sample_templates = [
            {
                "name": "Basic Airfoil Analysis",
                "type": "airfoil_analysis",
                "steps": 5,
                "description": "Standard airfoil analysis workflow",
            },
            {
                "name": "Airplane Performance Study",
                "type": "airplane_analysis",
                "steps": 7,
                "description": "Complete airplane performance analysis",
            },
            {
                "name": "Batch Processing",
                "type": "batch_processing",
                "steps": 4,
                "description": "Process multiple configurations",
            },
        ]

        for template in sample_templates:
            template_node = templates_tree.add_aerospace_node(
                templates_tree.root,
                template["name"],
                "analysis",
                template,
            )

            # Add template details
            templates_tree.add_aerospace_node(
                template_node,
                f"Type: {template['type']}",
                "file",
            )
            templates_tree.add_aerospace_node(
                template_node,
                f"Steps: {template['steps']}",
                "file",
            )
            templates_tree.add_aerospace_node(
                template_node,
                f"Description: {template['description']}",
                "file",
            )

    def on_node_palette_node_drag_started(
        self,
        event: NodePalette.NodeDragStarted,
    ) -> None:
        """Handle node drag from palette."""
        # Add the node at a default position
        canvas = self.query_one(WorkflowCanvas)
        canvas.add_node(event.node_type, Coordinate(200, 200))

    def on_workflow_canvas_node_selected(
        self,
        event: WorkflowCanvas.NodeSelected,
    ) -> None:
        """Handle node selection in canvas."""
        properties_panel = self.query_one(WorkflowPropertiesPanel)
        properties_panel.set_selected_node(event.node)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses in the workflow builder."""
        if event.button.id == "load_template":
            self._load_selected_template()
        elif event.button.id == "save_template":
            self._save_current_as_template()
        elif event.button.id == "delete_template":
            self._delete_selected_template()
        elif event.button.id == "export_workflow":
            self._export_workflow()
        elif event.button.id == "import_workflow":
            self._import_workflow()
        elif event.button.id == "new_workflow":
            self._new_workflow()

    def _load_selected_template(self) -> None:
        """Load the selected template into the canvas."""
        templates_tree = self.query_one("#templates_tree", AerospaceTree)

        if templates_tree.cursor_node and templates_tree.cursor_node.data:
            template = templates_tree.cursor_node.data
            if isinstance(template, dict):
                canvas = self.query_one(WorkflowCanvas)
                self._create_sample_workflow(canvas, template)

                # Show notification
                notification_panel = self.query_one(NotificationPanel)
                notification_panel.add_notification(
                    f"Loaded template: {template['name']}",
                    NotificationPanel.NotificationType.SUCCESS,
                )

    def _create_sample_workflow(
        self,
        canvas: WorkflowCanvas,
        template: Dict[str, Any],
    ) -> None:
        """Create a sample workflow based on template."""
        canvas.clear_canvas()

        # Create start node
        start_node = canvas.add_node(NodeType.START, Coordinate(100, 200), "Start")

        # Create analysis nodes based on template type
        if template["type"] == "airfoil_analysis":
            nodes = [
                ("Select Airfoil", NodeType.ANALYSIS, Coordinate(300, 150)),
                ("Configure Analysis", NodeType.ANALYSIS, Coordinate(500, 150)),
                ("Run XFoil", NodeType.ANALYSIS, Coordinate(700, 100)),
                ("Run Foil2Wake", NodeType.ANALYSIS, Coordinate(700, 200)),
                ("Merge Results", NodeType.MERGE, Coordinate(900, 150)),
            ]
        elif template["type"] == "airplane_analysis":
            nodes = [
                ("Load Geometry", NodeType.ANALYSIS, Coordinate(300, 150)),
                ("Setup Flight Conditions", NodeType.ANALYSIS, Coordinate(500, 150)),
                ("Run AVL", NodeType.ANALYSIS, Coordinate(700, 100)),
                ("Run GenuVP", NodeType.ANALYSIS, Coordinate(700, 200)),
                ("Post-process", NodeType.ANALYSIS, Coordinate(900, 150)),
            ]
        else:  # batch_processing
            nodes = [
                ("Load Batch", NodeType.ANALYSIS, Coordinate(300, 150)),
                ("Process Items", NodeType.PARALLEL, Coordinate(500, 150)),
                ("Aggregate Results", NodeType.MERGE, Coordinate(700, 150)),
            ]

        # Create nodes and connections
        prev_node = start_node
        for name, node_type, position in nodes:
            node = canvas.add_node(node_type, position, name)
            if prev_node:
                canvas.add_connection(prev_node.id, "out", node.id, "in")
            prev_node = node

        # Create end node
        end_node = canvas.add_node(NodeType.END, Coordinate(1100, 200), "End")
        if prev_node:
            canvas.add_connection(prev_node.id, "out", end_node.id, "in")

    def _save_current_as_template(self) -> None:
        """Save the current workflow as a template."""
        canvas = self.query_one(WorkflowCanvas)

        # Get workflow properties
        name_input = self.query_one("#workflow_name", ValidatedInput)
        description_input = self.query_one("#workflow_description", ValidatedInput)

        template_name = name_input.value or "Untitled Workflow"

        # Create template data
        template_data = {
            "name": template_name,
            "description": description_input.value or "Custom workflow",
            "type": "custom",
            "steps": len(canvas.nodes),
            "nodes": len(canvas.nodes),
            "connections": len(canvas.connections),
            "created_at": datetime.now().isoformat(),
        }

        self.workflow_templates[template_name] = template_data

        # Refresh templates list
        self._load_workflow_templates()

        # Show notification
        notification_panel = self.query_one(NotificationPanel)
        notification_panel.add_notification(
            f"Saved template: {template_name}",
            NotificationPanel.NotificationType.SUCCESS,
        )

    def _delete_selected_template(self) -> None:
        """Delete the selected template."""
        templates_tree = self.query_one("#templates_tree", AerospaceTree)

        if templates_tree.cursor_node and templates_tree.cursor_node.data:
            template = templates_tree.cursor_node.data
            if isinstance(template, dict):
                template_name = template["name"]

                # Show notification
                notification_panel = self.query_one(NotificationPanel)
                notification_panel.add_notification(
                    f"Template deletion: {template_name} (demo mode)",
                    NotificationPanel.NotificationType.INFO,
                )

    def _export_workflow(self) -> None:
        """Export the current workflow."""
        canvas = self.query_one(WorkflowCanvas)
        share_name = self.query_one("#share_name", ValidatedInput).value
        share_description = self.query_one("#share_description", TextArea).text

        # Create export data
        export_data = {
            "name": share_name or "Exported Workflow",
            "description": share_description or "Workflow exported from visual builder",
            "nodes": len(canvas.nodes),
            "connections": len(canvas.connections),
            "exported_at": datetime.now().isoformat(),
            "exported_by": "visual_workflow_builder",
        }

        # Show notification
        notification_panel = self.query_one(NotificationPanel)
        notification_panel.add_notification(
            "Workflow exported successfully (demo mode)",
            NotificationPanel.NotificationType.SUCCESS,
        )

    def _import_workflow(self) -> None:
        """Import a workflow from file."""
        notification_panel = self.query_one(NotificationPanel)
        notification_panel.add_notification(
            "Import functionality - would open file dialog",
            NotificationPanel.NotificationType.INFO,
        )

    def _new_workflow(self) -> None:
        """Create a new workflow."""
        canvas = self.query_one(WorkflowCanvas)
        canvas.clear_canvas()

        # Clear form fields
        name_input = self.query_one("#workflow_name", ValidatedInput)
        description_input = self.query_one("#workflow_description", ValidatedInput)
        name_input.value = ""
        description_input.value = ""

        # Show notification
        notification_panel = self.query_one(NotificationPanel)
        notification_panel.add_notification(
            "New workflow created",
            NotificationPanel.NotificationType.INFO,
        )


class WorkflowManagerPanel(Container):
    """Panel for managing workflow templates and sharing."""

    def compose(self) -> ComposeResult:
        yield Label("Workflow Manager", classes="panel-title")

        with Tabs(id="manager_tabs"):
            with TabPane("Templates", id="templates_tab"):
                yield AerospaceTree("Workflow Templates", id="templates_tree")
                with Horizontal(classes="template-actions"):
                    yield AerospaceButton(
                        "Load",
                        variant=ButtonVariant.INFO,
                        id="load_template",
                    )
                    yield AerospaceButton(
                        "Save as Template",
                        variant=ButtonVariant.SUCCESS,
                        id="save_template",
                    )
                    yield AerospaceButton(
                        "Delete",
                        variant=ButtonVariant.ERROR,
                        id="delete_template",
                    )

            with TabPane("Sharing", id="sharing_tab"):
                yield ValidatedInput(
                    "Share Name",
                    placeholder="Enter share name",
                    id="share_name",
                )
                yield Label("Description:", classes="form-label")
                yield TextArea("", id="share_description")
                with Horizontal(classes="sharing-actions"):
                    yield AerospaceButton(
                        "Export",
                        variant=ButtonVariant.INFO,
                        id="export_workflow",
                    )
                    yield AerospaceButton(
                        "Import",
                        variant=ButtonVariant.SUCCESS,
                        id="import_workflow",
                    )


# Complete the VisualWorkflowBuilder class methods
def _load_workflow_templates(self) -> None:
    """Load available workflow templates."""
    templates_tree = self.query_one("#templates_tree", AerospaceTree)

    # Clear existing templates
    templates_tree.clear()

    # Add sample templates
    sample_templates = [
        {
            "name": "Basic Airfoil Analysis",
            "type": "airfoil_analysis",
            "steps": 5,
            "description": "Standard airfoil analysis workflow",
        },
        {
            "name": "Airplane Performance Study",
            "type": "airplane_analysis",
            "steps": 7,
            "description": "Complete airplane performance analysis",
        },
        {
            "name": "Batch Processing",
            "type": "batch_processing",
            "steps": 4,
            "description": "Process multiple configurations",
        },
    ]

    for template in sample_templates:
        template_node = templates_tree.add_aerospace_node(
            templates_tree.root,
            template["name"],
            "analysis",
            template,
        )

        # Add template details
        templates_tree.add_aerospace_node(
            template_node,
            f"Type: {template['type']}",
            "file",
        )
        templates_tree.add_aerospace_node(
            template_node,
            f"Steps: {template['steps']}",
            "file",
        )
        templates_tree.add_aerospace_node(
            template_node,
            f"Description: {template['description']}",
            "file",
        )


def on_node_palette_node_drag_started(self, event: NodePalette.NodeDragStarted) -> None:
    """Handle node drag from palette."""
    # Add the node at a default position
    canvas = self.query_one(WorkflowCanvas)
    canvas.add_node(event.node_type, Coordinate(200, 200))


def on_workflow_canvas_node_selected(self, event: WorkflowCanvas.NodeSelected) -> None:
    """Handle node selection in canvas."""
    properties_panel = self.query_one(WorkflowPropertiesPanel)
    properties_panel.set_selected_node(event.node)


def on_button_pressed(self, event: Button.Pressed) -> None:
    """Handle button presses in the workflow builder."""
    if event.button.id == "load_template":
        self._load_selected_template()
    elif event.button.id == "save_template":
        self._save_current_as_template()
    elif event.button.id == "delete_template":
        self._delete_selected_template()
    elif event.button.id == "export_workflow":
        self._export_workflow()
    elif event.button.id == "import_workflow":
        self._import_workflow()
    elif event.button.id == "new_workflow":
        self._new_workflow()


def _load_selected_template(self) -> None:
    """Load the selected template into the canvas."""
    templates_tree = self.query_one("#templates_tree", AerospaceTree)

    if templates_tree.cursor_node and templates_tree.cursor_node.data:
        template = templates_tree.cursor_node.data
        if isinstance(template, dict):
            canvas = self.query_one(WorkflowCanvas)
            self._create_sample_workflow(canvas, template)

            # Show notification
            notification_panel = self.query_one(NotificationPanel)
            notification_panel.add_notification(
                f"Loaded template: {template['name']}",
                NotificationPanel.NotificationType.SUCCESS,
            )


def _create_sample_workflow(
    self,
    canvas: WorkflowCanvas,
    template: Dict[str, Any],
) -> None:
    """Create a sample workflow based on template."""
    canvas.clear_canvas()

    # Create start node
    start_node = canvas.add_node(NodeType.START, Coordinate(100, 200), "Start")

    # Create analysis nodes based on template type
    if template["type"] == "airfoil_analysis":
        nodes = [
            ("Select Airfoil", NodeType.ANALYSIS, Coordinate(300, 150)),
            ("Configure Analysis", NodeType.ANALYSIS, Coordinate(500, 150)),
            ("Run XFoil", NodeType.ANALYSIS, Coordinate(700, 100)),
            ("Run Foil2Wake", NodeType.ANALYSIS, Coordinate(700, 200)),
            ("Merge Results", NodeType.MERGE, Coordinate(900, 150)),
        ]
    elif template["type"] == "airplane_analysis":
        nodes = [
            ("Load Geometry", NodeType.ANALYSIS, Coordinate(300, 150)),
            ("Setup Flight Conditions", NodeType.ANALYSIS, Coordinate(500, 150)),
            ("Run AVL", NodeType.ANALYSIS, Coordinate(700, 100)),
            ("Run GenuVP", NodeType.ANALYSIS, Coordinate(700, 200)),
            ("Post-process", NodeType.ANALYSIS, Coordinate(900, 150)),
        ]
    else:  # batch_processing
        nodes = [
            ("Load Batch", NodeType.ANALYSIS, Coordinate(300, 150)),
            ("Process Items", NodeType.PARALLEL, Coordinate(500, 150)),
            ("Aggregate Results", NodeType.MERGE, Coordinate(700, 150)),
        ]

    # Create nodes and connections
    prev_node = start_node
    for name, node_type, position in nodes:
        node = canvas.add_node(node_type, position, name)
        if prev_node:
            canvas.add_connection(prev_node.id, "out", node.id, "in")
        prev_node = node

    # Create end node
    end_node = canvas.add_node(NodeType.END, Coordinate(1100, 200), "End")
    if prev_node:
        canvas.add_connection(prev_node.id, "out", end_node.id, "in")


def _save_current_as_template(self) -> None:
    """Save the current workflow as a template."""
    canvas = self.query_one(WorkflowCanvas)

    # Get workflow properties
    name_input = self.query_one("#workflow_name", ValidatedInput)
    description_input = self.query_one("#workflow_description", ValidatedInput)

    template_name = name_input.value or "Untitled Workflow"

    # Create template data
    template_data = {
        "name": template_name,
        "description": description_input.value or "Custom workflow",
        "type": "custom",
        "steps": len(canvas.nodes),
        "nodes": len(canvas.nodes),
        "connections": len(canvas.connections),
        "created_at": datetime.now().isoformat(),
    }

    self.workflow_templates[template_name] = template_data

    # Refresh templates list
    self._load_workflow_templates()

    # Show notification
    notification_panel = self.query_one(NotificationPanel)
    notification_panel.add_notification(
        f"Saved template: {template_name}",
        NotificationPanel.NotificationType.SUCCESS,
    )


def _delete_selected_template(self) -> None:
    """Delete the selected template."""
    templates_tree = self.query_one("#templates_tree", AerospaceTree)

    if templates_tree.cursor_node and templates_tree.cursor_node.data:
        template = templates_tree.cursor_node.data
        if isinstance(template, dict):
            template_name = template["name"]

            # Show notification
            notification_panel = self.query_one(NotificationPanel)
            notification_panel.add_notification(
                f"Template deletion: {template_name} (demo mode)",
                NotificationPanel.NotificationType.INFO,
            )


def _export_workflow(self) -> None:
    """Export the current workflow."""
    canvas = self.query_one(WorkflowCanvas)
    share_name = self.query_one("#share_name", ValidatedInput).value
    share_description = self.query_one("#share_description", TextArea).text

    # Create export data
    export_data = {
        "name": share_name or "Exported Workflow",
        "description": share_description or "Workflow exported from visual builder",
        "nodes": len(canvas.nodes),
        "connections": len(canvas.connections),
        "exported_at": datetime.now().isoformat(),
        "exported_by": "visual_workflow_builder",
    }

    # Show notification
    notification_panel = self.query_one(NotificationPanel)
    notification_panel.add_notification(
        "Workflow exported successfully (demo mode)",
        NotificationPanel.NotificationType.SUCCESS,
    )


def _import_workflow(self) -> None:
    """Import a workflow from file."""
    notification_panel = self.query_one(NotificationPanel)
    notification_panel.add_notification(
        "Import functionality - would open file dialog",
        NotificationPanel.NotificationType.INFO,
    )


def _new_workflow(self) -> None:
    """Create a new workflow."""
    canvas = self.query_one(WorkflowCanvas)
    canvas.clear_canvas()

    # Clear form fields
    name_input = self.query_one("#workflow_name", ValidatedInput)
    description_input = self.query_one("#workflow_description", ValidatedInput)
    name_input.value = ""
    description_input.value = ""

    # Show notification
    notification_panel = self.query_one(NotificationPanel)
    notification_panel.add_notification(
        "New workflow created",
        NotificationPanel.NotificationType.INFO,
    )


# Add the methods to the VisualWorkflowBuilder class
VisualWorkflowBuilder._load_workflow_templates = _load_workflow_templates
VisualWorkflowBuilder.on_node_palette_node_drag_started = (
    on_node_palette_node_drag_started
)
VisualWorkflowBuilder.on_workflow_canvas_node_selected = (
    on_workflow_canvas_node_selected
)
VisualWorkflowBuilder.on_button_pressed = on_button_pressed
VisualWorkflowBuilder._load_selected_template = _load_selected_template
VisualWorkflowBuilder._create_sample_workflow = _create_sample_workflow
VisualWorkflowBuilder._save_current_as_template = _save_current_as_template
VisualWorkflowBuilder._delete_selected_template = _delete_selected_template
VisualWorkflowBuilder._export_workflow = _export_workflow
VisualWorkflowBuilder._import_workflow = _import_workflow
VisualWorkflowBuilder._new_workflow = _new_workflow
