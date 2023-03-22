import numpy as np
import matplotlib as plt


class Surface:
    def __init__(self, Spoint, Epoint, airfoil, Schord, Echord):
        self.x0 = Spoint[0]
        self.y0 = Spoint[1]
        self.z0 = Spoint[2]

        self.x1 = Epoint[0]
        self.y1 = Epoint[1]
        self.z1 = Epoint[2]

        self.airfoil = airfoil
        self.chord = [Schord, Echord]

    def returnSymmetric(self):
        Spoint = [self.x1, -self.y1, self.z1]
        if self.y0 == 0:
            Epoint = [self.x0, 0.01 * self.y1, self.z0]
        else:
            Epoint = [self.x0, -self.y0, self.z0]
        airf = self.airfoil
        return Spoint, Epoint, airf, self.chord[1], self.chord[0]

    def set_airfoil(self, airfoil):
        self.airfoil = airfoil

    def startStrip(self):
        strip = [
            self.x0 + self.chord[0] * np.hstack((self.airfoil._x_upper,
                                                 self.airfoil._x_lower)),
            self.y0 + np.repeat(0, 2*self.airfoil.n_points),
            self.z0 + self.chord[0] * np.hstack((self.airfoil._y_upper,
                                                 self.airfoil._y_lower)),
        ]
        return np.array(strip)

    def endStrip(self):
        strip = [
            self.x1 + self.chord[1] * np.hstack((self.airfoil._x_upper,
                                                 self.airfoil._x_lower)),
            self.y1 + np.repeat(0, 2*self.airfoil.n_points),
            self.z1 + self.chord[1] * np.hstack((self.airfoil._y_upper,
                                                 self.airfoil._y_lower)),
        ]
        return np.array(strip)
