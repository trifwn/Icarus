import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


class Strip:
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
            self.x0
            + self.chord[0] * np.hstack((self.airfoil._x_upper, self.airfoil._x_lower)),
            self.y0 + np.repeat(0, 2 * self.airfoil.n_points),
            self.z0
            + self.chord[0] * np.hstack((self.airfoil._y_upper, self.airfoil._y_lower)),
        ]
        return np.array(strip)

    def endStrip(self):
        strip = [
            self.x1
            + self.chord[1] * np.hstack((self.airfoil._x_upper, self.airfoil._x_lower)),
            self.y1 + np.repeat(0, 2 * self.airfoil.n_points),
            self.z1
            + self.chord[1] * np.hstack((self.airfoil._y_upper, self.airfoil._y_lower)),
        ]
        return np.array(strip)

    def getInterStrip(self, idx, n_points_span=10):
        """Interpolate between start and end strips
        Args:
            idx: index of interpolation
            n_points: number of points to interpolate in the span direction
        Returns:
            strip: 3xn_points array of points
        """
        x = np.linspace(self.x0, self.x1, n_points_span)
        y = np.linspace(self.y0, self.y1, n_points_span)
        z = np.linspace(self.z0, self.z1, n_points_span)
        c = np.linspace(self.chord[0], self.chord[1], n_points_span)

        strip = [
            x[idx] + c[idx] * self.airfoil._x_lower,
            y[idx] + np.repeat(0, self.airfoil.n_points),
            z[idx] + c[idx] * self.airfoil.camber_line_NACA4(self.airfoil._x_lower),
        ]
        return np.array(strip)

    def plotStrip(self, fig=None, ax=None, movement=None, color=None):
        pltshow = False
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.set_title("Strip")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.axis("scaled")
            ax.view_init(30, 150)
            pltshow = True

        if movement is None:
            movement = np.zeros(3)

        xs, ys, zs = [], [], []
        N = 2
        for i in range(N):
            x, y, z = self.getInterStrip(i, N)
            xs.append(x + movement[0])
            ys.append(y + movement[1])
            zs.append(z + movement[2])
        X = np.array(xs)
        Y = np.array(ys)
        Z = np.array(zs)

        if color is None:
            my_color = "red"
        else:
            my_color = np.tile(color, (Z.shape[0], Z.shape[1])).reshape(
                Z.shape[0], Z.shape[1], 4,
            )

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=my_color)

        if pltshow:
            plt.show()
