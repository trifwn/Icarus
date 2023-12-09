"""
==================
Merged Wing Class
==================
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from ICARUS.Core.types import FloatArray
from ICARUS.Vehicle.lifting_surface import Lifting_Surface


class Merged_Wing:
    "Class to represent a Wing of an airplane"

    def __init__(
        self,
        name: str,
        wing_segments: list[Lifting_Surface],
        symmetries: list[bool] | None = None,
    ) -> None:
        """
        Initializes the Wing Object

        Args:
            name (str): Name of the wing e.g. Main Wing
            wing_segments (list[Lifting_Surface]): List of Wing_Segments
            symmetries (_type_): Symmetries that the wing has
        """

        # Create a grid of points to store all Wing Segments
        self.name: str = name
        NM = 0
        self._wing_segments = wing_segments
        for segment in wing_segments:
            NM += segment.N * segment.M

        self.grid: FloatArray = np.empty((NM, 3))
        self.grid_lower: FloatArray = np.empty((NM, 3))
        self.grid_upper: FloatArray = np.empty((NM, 3))

        NM = 0
        for segment in wing_segments:
            print(segment.M, segment.N, NM, segment.M * segment.N)
            self.grid[NM : NM + segment.M * segment.N, :] = np.reshape(segment.grid, (segment.M * segment.N, 3))
            self.grid_lower[NM : NM + segment.M * segment.N, :] = np.reshape(
                segment.grid_upper,
                (segment.M * segment.N, 3),
            )
            self.grid_upper[NM : NM + segment.M * segment.N, :] = np.reshape(
                segment.grid_lower,
                (segment.M * segment.N, 3),
            )
            NM += segment.M * segment.N

    def plot_wing(self) -> None:
        """
        Plots the wing
        """

        fig: Figure = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection="3d")
        ax.set_title("Wing")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.axis("equal")
        ax.view_init(30, 150)

        # Plot the wing grid
        for grid, c in zip([self.grid_lower, self.grid_upper], ["b", "r"]):
            X = grid[:, 0]
            Y = grid[:, 1]
            Z = grid[:, 2]

            ax.plot_trisurf(X, Y, Z, color=c)
        fig.show()

    def export_grid(self, filename: str) -> None:
        "Writes the grid to a file"

        with open(filename, "w") as file:
            file.write("X,Y,Z\n")
            for grid in [self.grid]:
                for point in grid:
                    file.write(f"{point[0]},{point[1]},{point[2]}\n")
