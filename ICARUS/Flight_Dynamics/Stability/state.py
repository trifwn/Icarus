from ICARUS.Core.struct import Struct
import numpy as np
import matplotlib.pyplot as plt


class State():
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
        plt.scatter(x, y, color="b", marker='x')

        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.grid()
        plt.show()
