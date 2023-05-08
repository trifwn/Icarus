import io

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from ICARUS.Core.struct import Struct


class State:
    def __init__(self, plane, name, trip):
        self.name = name
        self.trip = trip

        self.longitudal = Struct()
        self.lateral = Struct()

        self.longitudal.stateSpace = Struct()
        self.longitudal.stateSpace.A = plane.Along
        self.longitudal.stateSpace.B = 0

        eigvalLong, eigvecLong = np.linalg.eig(plane.Along)
        self.longitudal.eigenValues = eigvalLong
        self.longitudal.eigenVectors = eigvecLong

        self.lateral.stateSpace = Struct()
        self.lateral.stateSpace.A = plane.Alat
        self.lateral.stateSpace.B = 0

        eigvalLat, eigvecLat = np.linalg.eig(plane.Alat)
        self.lateral.eigenValues = eigvalLat
        self.lateral.eigenVectors = eigvecLat

    def plotEig(self):
        # extract real part
        x = [ele.real for ele in self.longitudal.eigenValues]
        # extract imaginary part
        y = [ele.imag for ele in self.longitudal.eigenValues]
        plt.scatter(x, y, color="r")

        # extract real part
        x = [ele.real for ele in self.lateral.eigenValues]
        # extract imaginary part
        y = [ele.imag for ele in self.lateral.eigenValues]
        plt.scatter(x, y, color="b", marker="x")

        plt.ylabel("Imaginary")
        plt.xlabel("Real")
        plt.grid()
        plt.show()

    def __str__(self):
        ss = io.StringIO()
        ss.write(f"State: {self.name}\n")
        ss.write(f"Trim: {self.trip}\n")
        ss.write(f"\n{45*'--'}\n")

        ss.write(f"\nLongitudal State:\n")
        ss.write(
            f"Eigen Values: {[round(item,3) for item in self.longitudal.eigenValues]}\n",
        )
        ss.write("Eigen Vectors:\n")
        for item in self.longitudal.eigenVectors:
            ss.write(f"\t{[round(i,3) for i in item]}\n")
        ss.write("\nThe State Space Matrix:\n")
        ss.write(
            tabulate(self.longitudal.stateSpace.A, tablefmt="github", floatfmt=".3f"),
        )

        ss.write(f"\n\n{45*'--'}\n")

        ss.write("\nLateral State:\n")
        ss.write(
            f"Eigen Values: {[round(item,3) for item in self.lateral.eigenValues]}\n",
        )
        ss.write("Eigen Vectors:\n")
        for item in self.lateral.eigenVectors:
            ss.write(f"\t{[round(i,3) for i in item]}\n")
        ss.write("\nThe State Space Matrix:\n")
        ss.write(tabulate(self.lateral.stateSpace.A, tablefmt="github", floatfmt=".3f"))
        return ss.getvalue()
