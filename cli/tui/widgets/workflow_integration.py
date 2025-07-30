"""Workflow Integration Module

This module provides integration between the visual workflow builder
and the core ICARUS workflow engine, enabling seamless conversion
between visual representations and executable workflows.
"""

import json
import uuid
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from textual.coordinate import Coordinate

from ..core.workflow import WorkflowDefinition
from ..core.workflow import WorkflowEngine
from ..core.workflow import WorkflowStep
from ..core.workflow import WorkflowStorage
from ..core.workflow import WorkflowTemplate
from ..core.workflow import WorkflowType
from .visual_workflow_builder import ConnectionType
from .visual_workflow_builder import NodeType
from .visual_workflow_builder import WorkflowConnection
from .visual_workflow_builder import WorkflowNode


class WorkflowConverter:
    """Converts between visual workflow representations and executable workflows."""

    def __init__(self):
        self.workflow_engine = WorkflowEngine()
        self.workflow_storage = WorkflowStorage()

    def visual_to_executable(
        self,
        nodes: Dict[str, WorkflowNode],
        connections: Dict[str, WorkflowConnection],
        workflow_name: str = "Visual Workflow",
        workflow_description: str = "Workflow created with visual builder",
        workflow_type: WorkflowType = WorkflowType.CUSTOM,
    ) -> Optional[WorkflowTemplate]:
        """Convert visual workflow to executable workflow template."""

        if not nodes:
            return None

        # Find start node
        start_nodes = [
            node for node in nodes.values() if node.node_type == NodeType.START
        ]
        if not start_nodes:
            return None

        # Convert visual nodes to workflow steps
        steps = []
        node_to_step_map = {}

        for node in nodes.values():
            if node.node_type in [NodeType.START, NodeType.END]:
                continue  # Skip start/end nodes in executable workflow

            # Create workflow step from visual node
            step = self._create_step_from_node(node)
            if step:
                steps.append(step)
                node_to_step_map[node.id] = step.id

        # Set up dependencies based on connections
        for step in steps:
            step_node = None
            for node in nodes.values():
                if node.step and hasattr(node.step, "id") and node.step.id == step.id:
                    step_node = node
                    break

            if step_node:
                dependencies = []
                for connection in connections.values():
                    if connection.target_node == step_node.id:
                        source_node = nodes.get(connection.source_node)
                        if source_node and source_node.id in node_to_step_map:
                            dependencies.append(node_to_step_map[source_node.id])

                step.dependencies = dependencies

        # Create workflow template
        template = WorkflowTemplate(
            id=str(uuid.uuid4()),
            name=workflow_name,
            description=workflow_description,
            type=workflow_type,
            steps=steps,
            parameters={},
            metadata={
                "created_with": "visual_workflow_builder",
                "visual_layout": self._serialize_visual_layout(nodes, connections),
            },
        )

        return template

    def executable_to_visual(
        self,
        template: WorkflowTemplate,
    ) -> Tuple[Dict[str, WorkflowNode], Dict[str, WorkflowConnection]]:
        """Convert executable workflow template to visual representation."""

        nodes = {}
        connections = {}

        # Check if template has visual layout metadata
        visual_layout = template.metadata.get("visual_layout")
        if visual_layout:
            return self._deserialize_visual_layout(visual_layout, template)

        # Generate visual layout from workflow structure
        return self._generate_visual_layout(template)

    def _create_step_from_node(self, node: WorkflowNode) -> Optional[WorkflowStep]:
        """Create a workflow step from a visual node."""

        if node.node_type == NodeType.ANALYSIS:
            # Use existing step if available, otherwise create new one
            if node.step:
                return node.step

            return WorkflowStep(
                id=str(uuid.uuid4()),
                name=node.name,
                description=f"Analysis step: {node.name}",
                action=node.properties.get("action", "custom_analysis"),
                parameters=node.properties.get("parameters", {}),
                dependencies=[],
                required=node.properties.get("required", True),
                timeout=node.properties.get("timeout"),
                validation=node.properties.get("validation"),
            )

        elif node.node_type == NodeType.CONDITION:
            return WorkflowStep(
                id=str(uuid.uuid4()),
                name=node.name,
                description=f"Conditional step: {node.name}",
                action="evaluate_condition",
                parameters={
                    "condition": node.properties.get("condition", "true"),
                    "true_action": node.properties.get("true_action", "continue"),
                    "false_action": node.properties.get("false_action", "skip"),
                },
                dependencies=[],
                required=node.properties.get("required", True),
            )

        elif node.node_type == NodeType.PARALLEL:
            return WorkflowStep(
                id=str(uuid.uuid4()),
                name=node.name,
                description=f"Parallel execution: {node.name}",
                action="parallel_execute",
                parameters={
                    "max_workers": node.properties.get("max_workers", 2),
                    "timeout": node.properties.get("timeout", 300),
                },
                dependencies=[],
                required=node.properties.get("required", True),
            )

        elif node.node_type == NodeType.MERGE:
            return WorkflowStep(
                id=str(uuid.uuid4()),
                name=node.name,
                description=f"Merge results: {node.name}",
                action="merge_results",
                parameters={
                    "merge_strategy": node.properties.get("merge_strategy", "combine"),
                    "output_format": node.properties.get("output_format", "json"),
                },
                dependencies=[],
                required=node.properties.get("required", True),
            )

        elif node.node_type == NodeType.CUSTOM:
            return WorkflowStep(
                id=str(uuid.uuid4()),
                name=node.name,
                description=f"Custom step: {node.name}",
                action=node.properties.get("action", "custom_action"),
                parameters=node.properties.get("parameters", {}),
                dependencies=[],
                required=node.properties.get("required", True),
                timeout=node.properties.get("timeout"),
            )

        return None

    def _serialize_visual_layout(
        self,
        nodes: Dict[str, WorkflowNode],
        connections: Dict[str, WorkflowConnection],
    ) -> Dict[str, Any]:
        """Serialize visual layout for storage."""

        serialized_nodes = {}
        for node_id, node in nodes.items():
            serialized_nodes[node_id] = {
                "id": node.id,
                "name": node.name,
                "node_type": node.node_type.value,
                "position": {"x": node.position.x, "y": node.position.y},
                "size": {"width": node.size.width, "height": node.size.height},
                "properties": node.properties,
                "input_ports": node.input_ports,
                "output_ports": node.output_ports,
            }

        serialized_connections = {}
        for conn_id, connection in connections.items():
            serialized_connections[conn_id] = {
                "id": connection.id,
                "source_node": connection.source_node,
                "source_port": connection.source_port,
                "target_node": connection.target_node,
                "target_port": connection.target_port,
                "connection_type": connection.connection_type.value,
                "properties": connection.properties,
            }

        return {
            "nodes": serialized_nodes,
            "connections": serialized_connections,
            "layout_version": "1.0",
            "created_at": datetime.now().isoformat(),
        }

    def _deserialize_visual_layout(
        self,
        layout_data: Dict[str, Any],
        template: WorkflowTemplate,
    ) -> Tuple[Dict[str, WorkflowNode], Dict[str, WorkflowConnection]]:
        """Deserialize visual layout from storage."""

        nodes = {}
        connections = {}

        # Create step lookup
        step_lookup = {step.id: step for step in template.steps}

        # Deserialize nodes
        for node_data in layout_data.get("nodes", {}).values():
            node = WorkflowNode(
                id=node_data["id"],
                name=node_data["name"],
                node_type=NodeType(node_data["node_type"]),
                position=Coordinate(
                    node_data["position"]["x"],
                    node_data["position"]["y"],
                ),
                properties=node_data.get("properties", {}),
                input_ports=node_data.get("input_ports", []),
                output_ports=node_data.get("output_ports", []),
            )

            # Link to workflow step if available
            for step in template.steps:
                if step.name == node.name:
                    node.step = step
                    break

            nodes[node.id] = node

        # Deserialize connections
        for conn_data in layout_data.get("connections", {}).values():
            connection = WorkflowConnection(
                id=conn_data["id"],
                source_node=conn_data["source_node"],
                source_port=conn_data["source_port"],
                target_node=conn_data["target_node"],
                target_port=conn_data["target_port"],
                connection_type=ConnectionType(conn_data["connection_type"]),
                properties=conn_data.get("properties", {}),
            )
            connections[connection.id] = connection

        return nodes, connections

    def _generate_visual_layout(
        self,
        template: WorkflowTemplate,
    ) -> Tuple[Dict[str, WorkflowNode], Dict[str, WorkflowConnection]]:
        """Generate visual layout from workflow structure."""

        nodes = {}
        connections = {}

        # Create start node
        start_node = WorkflowNode(
            id="start_node",
            name="Start",
            node_type=NodeType.START,
            position=Coordinate(100, 200),
        )
        nodes[start_node.id] = start_node

        # Create nodes for workflow steps
        x_offset = 300
        y_base = 150
        step_nodes = {}

        for i, step in enumerate(template.steps):
            # Determine node type based on step action
            node_type = self._infer_node_type(step)

            # Calculate position
            y_offset = y_base + (i % 3) * 100
            if i > 0 and i % 3 == 0:
                x_offset += 200

            node = WorkflowNode(
                id=f"step_{step.id}",
                name=step.name,
                node_type=node_type,
                position=Coordinate(x_offset, y_offset),
                step=step,
            )

            nodes[node.id] = node
            step_nodes[step.id] = node

        # Create end node
        end_node = WorkflowNode(
            id="end_node",
            name="End",
            node_type=NodeType.END,
            position=Coordinate(x_offset + 200, 200),
        )
        nodes[end_node.id] = end_node

        # Create connections based on dependencies
        for step in template.steps:
            step_node = step_nodes[step.id]

            if not step.dependencies:
                # Connect to start node
                connection = WorkflowConnection(
                    id=f"conn_start_{step.id}",
                    source_node=start_node.id,
                    source_port="out",
                    target_node=step_node.id,
                    target_port="in",
                    connection_type=ConnectionType.SEQUENCE,
                )
                connections[connection.id] = connection
            else:
                # Connect to dependency nodes
                for dep_id in step.dependencies:
                    if dep_id in step_nodes:
                        dep_node = step_nodes[dep_id]
                        connection = WorkflowConnection(
                            id=f"conn_{dep_id}_{step.id}",
                            source_node=dep_node.id,
                            source_port="out",
                            target_node=step_node.id,
                            target_port="in",
                            connection_type=ConnectionType.SEQUENCE,
                        )
                        connections[connection.id] = connection

        # Connect final steps to end node
        final_steps = [
            step
            for step in template.steps
            if not any(step.id in other.dependencies for other in template.steps)
        ]

        for step in final_steps:
            if step.id in step_nodes:
                step_node = step_nodes[step.id]
                connection = WorkflowConnection(
                    id=f"conn_{step.id}_end",
                    source_node=step_node.id,
                    source_port="out",
                    target_node=end_node.id,
                    target_port="in",
                    connection_type=ConnectionType.SEQUENCE,
                )
                connections[connection.id] = connection

        return nodes, connections

    def _infer_node_type(self, step: WorkflowStep) -> NodeType:
        """Infer visual node type from workflow step."""

        action = step.action.lower()

        if "condition" in action or "evaluate" in action:
            return NodeType.CONDITION
        elif "parallel" in action or "batch" in action:
            return NodeType.PARALLEL
        elif "merge" in action or "combine" in action or "aggregate" in action:
            return NodeType.MERGE
        elif any(
            keyword in action
            for keyword in ["analysis", "solve", "compute", "calculate"]
        ):
            return NodeType.ANALYSIS
        else:
            return NodeType.CUSTOM

    def save_visual_workflow(
        self,
        nodes: Dict[str, WorkflowNode],
        connections: Dict[str, WorkflowConnection],
        workflow_name: str,
        workflow_description: str = "",
        workflow_type: WorkflowType = WorkflowType.CUSTOM,
    ) -> bool:
        """Save visual workflow as a template."""

        try:
            template = self.visual_to_executable(
                nodes,
                connections,
                workflow_name,
                workflow_description,
                workflow_type,
            )

            if template:
                definition = WorkflowDefinition(template=template)
                self.workflow_storage.save_definition(definition)
                return True

            return False

        except Exception as e:
            print(f"Error saving visual workflow: {e}")
            return False

    def load_visual_workflow(
        self,
        template_id: str,
    ) -> Optional[Tuple[Dict[str, WorkflowNode], Dict[str, WorkflowConnection]]]:
        """Load visual workflow from template."""

        try:
            definition = self.workflow_storage.get_definition(template_id)
            if definition:
                return self.executable_to_visual(definition.template)

            return None

        except Exception as e:
            print(f"Error loading visual workflow: {e}")
            return None

    def validate_visual_workflow(
        self,
        nodes: Dict[str, WorkflowNode],
        connections: Dict[str, WorkflowConnection],
    ) -> List[str]:
        """Validate visual workflow and return list of errors."""

        errors = []

        # Convert to executable workflow for validation
        template = self.visual_to_executable(nodes, connections)
        if not template:
            errors.append("Cannot convert visual workflow to executable format")
            return errors

        # Validate workflow structure
        if not template.steps:
            errors.append("Workflow must contain at least one analysis step")

        # Validate step dependencies
        step_ids = {step.id for step in template.steps}
        for step in template.steps:
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    errors.append(
                        f"Step '{step.name}' has invalid dependency: {dep_id}",
                    )

        # Check for circular dependencies
        if self._has_circular_dependencies(template.steps):
            errors.append("Workflow contains circular dependencies")

        # Validate step actions
        for step in template.steps:
            if not step.action:
                errors.append(f"Step '{step.name}' is missing action")
            elif not self._is_valid_action(step.action):
                errors.append(f"Step '{step.name}' has invalid action: {step.action}")

        return errors

    def _has_circular_dependencies(self, steps: List[WorkflowStep]) -> bool:
        """Check for circular dependencies in workflow steps."""

        visited = set()
        rec_stack = set()

        def dfs(step_id: str) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)

            # Find step by ID
            step = next((s for s in steps if s.id == step_id), None)
            if not step:
                return False

            for dep_id in step.dependencies:
                if dep_id not in visited:
                    if dfs(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True

            rec_stack.remove(step_id)
            return False

        for step in steps:
            if step.id not in visited:
                if dfs(step.id):
                    return True

        return False

    def _is_valid_action(self, action: str) -> bool:
        """Check if an action is valid."""

        # Get available actions from workflow engine
        valid_actions = set(self.workflow_engine.action_handlers.keys())

        # Add common action patterns
        valid_patterns = [
            "select_",
            "configure_",
            "run_",
            "save_",
            "load_",
            "analyze_",
            "compute_",
            "calculate_",
            "process_",
            "validate_",
            "export_",
            "import_",
            "merge_",
            "parallel_",
            "condition_",
            "custom_",
        ]

        # Check exact match
        if action in valid_actions:
            return True

        # Check pattern match
        for pattern in valid_patterns:
            if action.startswith(pattern):
                return True

        return False

    def export_workflow(
        self,
        nodes: Dict[str, WorkflowNode],
        connections: Dict[str, WorkflowConnection],
        export_format: str = "json",
    ) -> Optional[str]:
        """Export visual workflow to specified format."""

        try:
            template = self.visual_to_executable(nodes, connections)
            if not template:
                return None

            export_data = {
                "workflow": {
                    "id": template.id,
                    "name": template.name,
                    "description": template.description,
                    "type": template.type.value,
                    "version": template.version,
                    "created_at": template.created_at.isoformat(),
                    "updated_at": template.updated_at.isoformat(),
                },
                "steps": [
                    {
                        "id": step.id,
                        "name": step.name,
                        "description": step.description,
                        "action": step.action,
                        "parameters": step.parameters,
                        "dependencies": step.dependencies,
                        "required": step.required,
                        "timeout": step.timeout,
                        "max_retries": step.max_retries,
                    }
                    for step in template.steps
                ],
                "visual_layout": self._serialize_visual_layout(nodes, connections),
                "export_info": {
                    "format": export_format,
                    "exported_at": datetime.now().isoformat(),
                    "exported_by": "visual_workflow_builder",
                },
            }

            if export_format.lower() == "json":
                return json.dumps(export_data, indent=2)
            elif export_format.lower() == "yaml":
                import yaml

                return yaml.dump(export_data, default_flow_style=False)
            else:
                return json.dumps(export_data, indent=2)

        except Exception as e:
            print(f"Error exporting workflow: {e}")
            return None
