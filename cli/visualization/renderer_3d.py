"""3D Renderer - Advanced 3D visualization for aerospace models

This module provides 3D rendering capabilities for the ICARUS CLI visualization system.
It supports rendering airfoils, wings, and complete aircraft models with interactive controls.
"""

import math
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from rich.console import Console


class Point3D:
    """3D point representation."""

    def __init__(self, x: float, y: float, z: float):
        """Initialize a 3D point.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
        """
        self.x = x
        self.y = y
        self.z = z

    def rotate_x(self, angle: float) -> "Point3D":
        """Rotate around X axis.

        Args:
            angle: Rotation angle in radians

        Returns:
            Rotated point
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        y = self.y * cos_a - self.z * sin_a
        z = self.y * sin_a + self.z * cos_a
        return Point3D(self.x, y, z)

    def rotate_y(self, angle: float) -> "Point3D":
        """Rotate around Y axis.

        Args:
            angle: Rotation angle in radians

        Returns:
            Rotated point
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x = self.x * cos_a + self.z * sin_a
        z = -self.x * sin_a + self.z * cos_a
        return Point3D(x, self.y, z)

    def rotate_z(self, angle: float) -> "Point3D":
        """Rotate around Z axis.

        Args:
            angle: Rotation angle in radians

        Returns:
            Rotated point
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x = self.x * cos_a - self.y * sin_a
        y = self.x * sin_a + self.y * cos_a
        return Point3D(x, y, self.z)

    def project(
        self,
        width: int,
        height: int,
        fov: float,
        distance: float,
    ) -> Tuple[int, int]:
        """Project 3D point to 2D screen coordinates.

        Args:
            width: Screen width
            height: Screen height
            fov: Field of view
            distance: Viewing distance

        Returns:
            Tuple of (x, y) screen coordinates
        """
        factor = fov / (distance + self.z)
        x = self.x * factor + width / 2
        y = -self.y * factor + height / 2
        return int(x), int(y)


class Mesh:
    """3D mesh representation."""

    def __init__(self, vertices: List[Point3D], edges: List[Tuple[int, int]]):
        """Initialize a 3D mesh.

        Args:
            vertices: List of vertices
            edges: List of edges as pairs of vertex indices
        """
        self.vertices = vertices
        self.edges = edges
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0

    def rotate(self, angle_x: float, angle_y: float, angle_z: float) -> None:
        """Rotate the mesh.

        Args:
            angle_x: Rotation around X axis in radians
            angle_y: Rotation around Y axis in radians
            angle_z: Rotation around Z axis in radians
        """
        self.rotation_x += angle_x
        self.rotation_y += angle_y
        self.rotation_z += angle_z

    def get_transformed_vertices(self) -> List[Point3D]:
        """Get vertices with current rotation applied.

        Returns:
            List of transformed vertices
        """
        transformed = []
        for vertex in self.vertices:
            # Apply rotations
            rotated = vertex
            rotated = rotated.rotate_x(self.rotation_x)
            rotated = rotated.rotate_y(self.rotation_y)
            rotated = rotated.rotate_z(self.rotation_z)
            transformed.append(rotated)
        return transformed


class AirfoilMesh(Mesh):
    """Airfoil mesh representation."""

    @classmethod
    def from_coordinates(
        cls,
        upper: List[Tuple[float, float]],
        lower: List[Tuple[float, float]],
    ) -> "AirfoilMesh":
        """Create an airfoil mesh from coordinates.

        Args:
            upper: Upper surface coordinates as (x, y) pairs
            lower: Lower surface coordinates as (x, y) pairs

        Returns:
            AirfoilMesh instance
        """
        vertices = []

        # Add upper surface vertices
        for x, y in upper:
            vertices.append(Point3D(x, y, 0.0))

        # Add lower surface vertices
        for x, y in lower:
            vertices.append(Point3D(x, y, 0.0))

        # Create edges
        edges = []

        # Upper surface edges
        for i in range(len(upper) - 1):
            edges.append((i, i + 1))

        # Lower surface edges
        lower_start = len(upper)
        for i in range(len(lower) - 1):
            edges.append((lower_start + i, lower_start + i + 1))

        # Connect leading and trailing edges
        edges.append((0, lower_start))
        edges.append((len(upper) - 1, len(upper) + len(lower) - 1))

        return cls(vertices, edges)

    @classmethod
    def create_naca_4digit(cls, code: str, points: int = 50) -> "AirfoilMesh":
        """Create a NACA 4-digit airfoil mesh.

        Args:
            code: NACA 4-digit code (e.g., "0012")
            points: Number of points per surface

        Returns:
            AirfoilMesh instance
        """
        if len(code) != 4:
            raise ValueError("NACA code must be 4 digits")

        try:
            m = int(code[0]) / 100.0  # maximum camber
            p = int(code[1]) / 10.0  # location of maximum camber
            t = int(code[2:]) / 100.0  # thickness
        except ValueError:
            raise ValueError("Invalid NACA code format")

        upper = []
        lower = []

        # Generate points
        for i in range(points):
            x = 1.0 - 0.5 * (1.0 - math.cos(math.pi * i / (points - 1)))

            # Thickness distribution
            yt = (
                5
                * t
                * (
                    0.2969 * math.sqrt(x)
                    - 0.1260 * x
                    - 0.3516 * x**2
                    + 0.2843 * x**3
                    - 0.1015 * x**4
                )
            )

            if m == 0:
                # Symmetric airfoil
                upper.append((x, yt))
                lower.append((x, -yt))
            else:
                # Cambered airfoil
                if x <= p:
                    yc = m * (2 * p * x - x**2) / p**2
                    dyc = 2 * m * (p - x) / p**2
                else:
                    yc = m * (1 - 2 * p + 2 * p * x - x**2) / (1 - p) ** 2
                    dyc = 2 * m * (p - x) / (1 - p) ** 2

                theta = math.atan(dyc)
                upper.append((x - yt * math.sin(theta), yc + yt * math.cos(theta)))
                lower.append((x + yt * math.sin(theta), yc - yt * math.cos(theta)))

        return cls.from_coordinates(upper, lower)


class WingMesh(Mesh):
    """Wing mesh representation."""

    @classmethod
    def from_airfoils(
        cls,
        root_airfoil: AirfoilMesh,
        tip_airfoil: AirfoilMesh,
        span: float,
        sweep: float = 0.0,
        dihedral: float = 0.0,
    ) -> "WingMesh":
        """Create a wing mesh from root and tip airfoils.

        Args:
            root_airfoil: Root airfoil mesh
            tip_airfoil: Tip airfoil mesh
            span: Wing span
            sweep: Sweep angle in degrees
            dihedral: Dihedral angle in degrees

        Returns:
            WingMesh instance
        """
        vertices = []
        edges = []

        # Convert angles to radians
        sweep_rad = math.radians(sweep)
        dihedral_rad = math.radians(dihedral)

        # Add root airfoil vertices
        for vertex in root_airfoil.vertices:
            vertices.append(Point3D(vertex.x, vertex.y, 0.0))

        # Calculate tip position with sweep and dihedral
        tip_x_offset = span * math.tan(sweep_rad)
        tip_y_offset = span * math.sin(dihedral_rad)
        tip_z = span * math.cos(dihedral_rad)

        # Add tip airfoil vertices
        for vertex in tip_airfoil.vertices:
            vertices.append(
                Point3D(vertex.x + tip_x_offset, vertex.y + tip_y_offset, tip_z),
            )

        # Add root airfoil edges
        num_root_vertices = len(root_airfoil.vertices)
        for edge in root_airfoil.edges:
            edges.append(edge)

        # Add tip airfoil edges
        for edge in tip_airfoil.edges:
            edges.append((edge[0] + num_root_vertices, edge[1] + num_root_vertices))

        # Connect root and tip
        for i in range(num_root_vertices):
            edges.append((i, i + num_root_vertices))

        return cls(vertices, edges)


class AirplaneMesh(Mesh):
    """Airplane mesh representation."""

    @classmethod
    def create_simple_airplane(
        cls,
        wingspan: float = 10.0,
        fuselage_length: float = 12.0,
        tail_span: float = 3.0,
    ) -> "AirplaneMesh":
        """Create a simple airplane mesh.

        Args:
            wingspan: Wing span
            fuselage_length: Fuselage length
            tail_span: Horizontal tail span

        Returns:
            AirplaneMesh instance
        """
        vertices = []
        edges = []

        # Fuselage
        fuselage_width = fuselage_length / 10
        fuselage_height = fuselage_width * 1.2

        # Nose
        vertices.append(Point3D(0, 0, 0))  # 0: nose

        # Cockpit
        vertices.append(Point3D(fuselage_length * 0.2, 0, 0))  # 1: cockpit top

        # Wing root
        wing_pos = fuselage_length * 0.4
        vertices.append(Point3D(wing_pos, 0, 0))  # 2: wing root top
        vertices.append(Point3D(wing_pos, 0, -fuselage_width / 2))  # 3: wing root left
        vertices.append(Point3D(wing_pos, 0, fuselage_width / 2))  # 4: wing root right

        # Wing tips
        vertices.append(Point3D(wing_pos, 0, -wingspan / 2))  # 5: wing tip left
        vertices.append(Point3D(wing_pos, 0, wingspan / 2))  # 6: wing tip right

        # Tail root
        tail_pos = fuselage_length * 0.8
        vertices.append(Point3D(tail_pos, 0, 0))  # 7: tail root
        vertices.append(Point3D(tail_pos, fuselage_height, 0))  # 8: vertical tail top
        vertices.append(Point3D(tail_pos, 0, -tail_span / 2))  # 9: horizontal tail left
        vertices.append(
            Point3D(tail_pos, 0, tail_span / 2),
        )  # 10: horizontal tail right

        # Rear
        vertices.append(Point3D(fuselage_length, 0, 0))  # 11: rear

        # Fuselage edges
        edges.append((0, 1))  # Nose to cockpit
        edges.append((1, 2))  # Cockpit to wing root
        edges.append((2, 7))  # Wing root to tail root
        edges.append((7, 11))  # Tail root to rear

        # Wing edges
        edges.append((2, 3))  # Wing root top to left
        edges.append((2, 4))  # Wing root top to right
        edges.append((3, 5))  # Wing root left to wing tip left
        edges.append((4, 6))  # Wing root right to wing tip right

        # Tail edges
        edges.append((7, 8))  # Tail root to vertical tail top
        edges.append((8, 11))  # Vertical tail top to rear
        edges.append((7, 9))  # Tail root to horizontal tail left
        edges.append((7, 10))  # Tail root to horizontal tail right
        edges.append((9, 11))  # Horizontal tail left to rear
        edges.append((10, 11))  # Horizontal tail right to rear

        return cls(vertices, edges)


class Renderer3D:
    """3D renderer for aerospace models."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the 3D renderer.

        Args:
            console: Rich console for output (optional)
        """
        self.console = console or Console()
        self.width = 80
        self.height = 24
        self.fov = 100
        self.distance = 5
        self.meshes: Dict[str, Mesh] = {}
        self.active_mesh: Optional[str] = None

    def add_mesh(self, name: str, mesh: Mesh) -> None:
        """Add a mesh to the renderer.

        Args:
            name: Mesh name
            mesh: Mesh object
        """
        self.meshes[name] = mesh
        if self.active_mesh is None:
            self.active_mesh = name

    def set_active_mesh(self, name: str) -> bool:
        """Set the active mesh.

        Args:
            name: Mesh name

        Returns:
            True if successful, False if mesh not found
        """
        if name in self.meshes:
            self.active_mesh = name
            return True
        return False

    def rotate_mesh(
        self,
        name: str,
        angle_x: float,
        angle_y: float,
        angle_z: float,
    ) -> bool:
        """Rotate a mesh.

        Args:
            name: Mesh name
            angle_x: Rotation around X axis in degrees
            angle_y: Rotation around Y axis in degrees
            angle_z: Rotation around Z axis in degrees

        Returns:
            True if successful, False if mesh not found
        """
        if name in self.meshes:
            self.meshes[name].rotate(
                math.radians(angle_x),
                math.radians(angle_y),
                math.radians(angle_z),
            )
            return True
        return False

    def render_to_text(self, width: int = 80, height: int = 24) -> str:
        """Render the active mesh to text.

        Args:
            width: Render width
            height: Render height

        Returns:
            Text representation of the rendered mesh
        """
        if not self.active_mesh or self.active_mesh not in self.meshes:
            return "[No active mesh]"

        self.width = width
        self.height = height

        # Create empty canvas
        canvas = [[" " for _ in range(width)] for _ in range(height)]

        mesh = self.meshes[self.active_mesh]
        transformed_vertices = mesh.get_transformed_vertices()

        # Project vertices to 2D
        points_2d = []
        for vertex in transformed_vertices:
            if vertex.z < -self.distance:
                # Skip vertices behind the camera
                points_2d.append(None)
            else:
                points_2d.append(vertex.project(width, height, self.fov, self.distance))

        # Draw edges
        for edge in mesh.edges:
            p1 = points_2d[edge[0]]
            p2 = points_2d[edge[1]]

            if p1 is None or p2 is None:
                continue

            x1, y1 = p1
            x2, y2 = p2

            # Draw line using Bresenham's algorithm
            self._draw_line(canvas, x1, y1, x2, y2)

        # Convert canvas to string
        result = ""
        for row in canvas:
            result += "".join(row) + "\n"

        return result

    def _draw_line(
        self,
        canvas: List[List[str]],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> None:
        """Draw a line on the canvas using Bresenham's algorithm.

        Args:
            canvas: Canvas to draw on
            x1, y1: Start point
            x2, y2: End point
        """
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if 0 <= x1 < self.width and 0 <= y1 < self.height:
                canvas[y1][x1] = "█"

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def create_airfoil(self, name: str, naca_code: str) -> bool:
        """Create an airfoil mesh from a NACA code.

        Args:
            name: Mesh name
            naca_code: NACA airfoil code

        Returns:
            True if successful
        """
        try:
            mesh = AirfoilMesh.create_naca_4digit(naca_code)
            self.add_mesh(name, mesh)
            return True
        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to create airfoil: {e}")
            return False

    def create_wing(
        self,
        name: str,
        root_airfoil: str,
        tip_airfoil: str,
        span: float,
        sweep: float = 0.0,
        dihedral: float = 0.0,
    ) -> bool:
        """Create a wing mesh from root and tip airfoils.

        Args:
            name: Mesh name
            root_airfoil: Root airfoil mesh name
            tip_airfoil: Tip airfoil mesh name
            span: Wing span
            sweep: Sweep angle in degrees
            dihedral: Dihedral angle in degrees

        Returns:
            True if successful
        """
        if root_airfoil not in self.meshes or tip_airfoil not in self.meshes:
            self.console.print("[red]✗[/red] Airfoil not found")
            return False

        try:
            mesh = WingMesh.from_airfoils(
                self.meshes[root_airfoil],
                self.meshes[tip_airfoil],
                span,
                sweep,
                dihedral,
            )
            self.add_mesh(name, mesh)
            return True
        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to create wing: {e}")
            return False

    def create_airplane(
        self,
        name: str,
        wingspan: float = 10.0,
        length: float = 12.0,
    ) -> bool:
        """Create a simple airplane mesh.

        Args:
            name: Mesh name
            wingspan: Wing span
            length: Fuselage length

        Returns:
            True if successful
        """
        try:
            mesh = AirplaneMesh.create_simple_airplane(wingspan, length)
            self.add_mesh(name, mesh)
            return True
        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to create airplane: {e}")
            return False

    def export_to_file(self, filename: Union[str, Path], format: str = "obj") -> bool:
        """Export the active mesh to a file.

        Args:
            filename: Output filename
            format: Export format (obj, stl)

        Returns:
            True if successful
        """
        if not self.active_mesh or self.active_mesh not in self.meshes:
            self.console.print("[red]✗[/red] No active mesh")
            return False

        try:
            mesh = self.meshes[self.active_mesh]

            with open(filename, "w") as f:
                if format.lower() == "obj":
                    # Write OBJ format
                    f.write(f"# {self.active_mesh}\n")

                    # Write vertices
                    for vertex in mesh.vertices:
                        f.write(f"v {vertex.x} {vertex.y} {vertex.z}\n")

                    # Write edges as lines
                    for edge in mesh.edges:
                        # OBJ indices are 1-based
                        f.write(f"l {edge[0] + 1} {edge[1] + 1}\n")

                elif format.lower() == "stl":
                    # Write STL format (ASCII)
                    f.write(f"solid {self.active_mesh}\n")

                    # This is a simplified STL export that doesn't properly create faces
                    # A real implementation would need to triangulate the mesh
                    for edge in mesh.edges:
                        v1 = mesh.vertices[edge[0]]
                        v2 = mesh.vertices[edge[1]]

                        # Create a simple triangle with a third point
                        v3 = Point3D(v1.x, v1.y, v1.z + 0.01)

                        # Calculate normal (simplified)
                        nx, ny, nz = 0, 0, 1

                        f.write("facet normal {nx} {ny} {nz}\n")
                        f.write("  outer loop\n")
                        f.write(f"    vertex {v1.x} {v1.y} {v1.z}\n")
                        f.write(f"    vertex {v2.x} {v2.y} {v2.z}\n")
                        f.write(f"    vertex {v3.x} {v3.y} {v3.z}\n")
                        f.write("  endloop\n")
                        f.write("endfacet\n")

                    f.write(f"endsolid {self.active_mesh}\n")

                else:
                    self.console.print(
                        f"[red]✗[/red] Unsupported export format: {format}",
                    )
                    return False

            self.console.print(f"[green]✓[/green] Exported mesh to {filename}")
            return True

        except Exception as e:
            self.console.print(f"[red]✗[/red] Export failed: {e}")
            return False
