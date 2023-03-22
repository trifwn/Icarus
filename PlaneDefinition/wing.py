import numpy as np
import matplotlib.pyplot as plt
from .surface import Surface


class Wing:
    def __init__(self, name, airfoil, Origin, Orientation, isSymmetric, span, sweepOffset, dihAngle, chordFun, chord, spanFun):

        self.N = 10
        self.M = 10
        self.name = name
        self.Gamma = dihAngle * np.pi/180
        self.isSymmetric = isSymmetric
        self.span = span
        self.airfoil = airfoil
        self.Origin = Origin

        # ORIENTATION
        self.pitch, self.yaw, self.roll = Orientation * np.pi/180
        R_pitch = np.array([
            [np.cos(self.pitch), 0, np.sin(self.pitch)],
            [0, 1, 0],
            [-np.sin(self.pitch), 0, np.cos(self.pitch)]
        ])
        R_yaw = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw), 0],
            [np.sin(self.yaw),  np.cos(self.yaw), 0],
            [0, 0, 1]
        ])
        R_roll = np.array([
            [1, 0, 0],
            [0,  np.cos(self.roll), -np.sin(self.roll)],
            [0,  np.sin(self.roll),  np.cos(self.roll)]
        ])
        self.Rmat = R_yaw.dot(R_pitch).dot(R_roll)

        # Make Dihedral Angle Distribution
        if isSymmetric == True:
            self.Dchord = chordFun(self.N, *chord)
            self.Dspan = spanFun(span/2, self.N)
            self.xoff = (self.Dspan - span/2) * (sweepOffset / (span/2))
            self.Ddihedr = (self.Dspan - span/2)*np.sin(self.Gamma)
        else:
            self.Dchord = chordFun(self.N, *chord)
            self.Dspan = spanFun(span, self.N)
            self.xoff = self.Dspan * sweepOffset / span
            self.Ddihedr = self.Dspan*np.sin(self.Gamma)

        # Create Grid
        self.createGrid()

        # Create Surfaces
        self.createSurfaces()

    def createSurfaces(self):
        surfaces = []
        symSurfaces = []
        startPoint = [
            self.Origin[0] + self.xoff[0],
            self.Origin[1] + self.Dspan[0],
            self.Origin[2] + self.Ddihedr[0],
        ]
        endPoint = [
            self.Origin[0] + self.xoff[-1],
            self.Origin[1] + self.Dspan[-1],
            self.Origin[2] + self.Ddihedr[-1],
        ]

        if self.isSymmetric == True:
            surf = Surface(startPoint, endPoint, self.airfoil,
                           self.Dchord[0], self.Dchord[-1])
            surfaces.append(surf)
            symSurfaces.append(Surface(*surf.returnSymmetric()))
        else:
            surf = Surface(startPoint, endPoint, self.airfoil,
                           self.Dchord[0], self.Dchord[-1])
            surfaces.append(surf)
        self.surfaces = surfaces
        self.allSurfaces = [*surfaces, *symSurfaces]

    def plotWing(self, fig=None, ax = None):
        if fig == None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_title(self.name)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.axis('scaled')
            ax.view_init(30, 150)

        for surf in self.allSurfaces:
            s1 = np.matmul(self.Rmat, surf.startStrip())
            s2 = np.matmul(self.Rmat, surf.endStrip())
            ax.plot(*s1, '-', color='red')
            ax.plot(*s2, '-', color='blue')

        for i in np.arange(0, self.N-1):
            for j in np.arange(0, self.M-1):
                for item in [self.panels_lower, self.panels_upper]:
                    p1, p3, p4, p2 = item[i, j, :, :]
                    xs = np.reshape([p1[0], p2[0], p3[0], p4[0]], (2, 2))
                    ys = np.reshape([p1[1], p2[1], p3[1], p4[1]], (2, 2))
                    zs = np.reshape([p1[2], p2[2], p3[2], p4[2]], (2, 2))
                    ax.plot_wireframe(xs, ys, zs, linewidth=0.5)
                    if self.isSymmetric == True:
                        ax.plot_wireframe(xs, -ys, zs, linewidth=0.5)


    def grid2panels(self, grid):
        panels = np.empty((self.N-1, self.M-1, 4, 3))
        for i in np.arange(0, self.N-1):
            for j in np.arange(0, self.M-1):
                panels[i, j, 0, :] = grid[i+1, j]
                panels[i, j, 1, :] = grid[i, j]
                panels[i, j, 2, :] = grid[i, j+1]
                panels[i, j, 3, :] = grid[i+1, j+1]
        return panels

    def createGrid(self):
        xs = np.empty((self.M, self.N))
        xs_upper = np.empty((self.M, self.N))
        xs_lower = np.empty((self.M, self.N))

        ys = np.empty((self.M, self.N))
        ys_upper = np.empty((self.M, self.N))
        ys_lower = np.empty((self.M, self.N))

        zs = np.empty((self.M, self.N))
        zs_upper = np.empty((self.M, self.N))
        zs_lower = np.empty((self.M, self.N))

        for i in np.arange(0, self.M):
            xpos = (self.Dchord)*(i/(self.M-1))
            xs[i, :] = self.Origin[0] + self.xoff + xpos
            xs_lower[i, :] = xs[i, :]
            xs_upper[i, :] = xs[i, :]

            ys[i, :] = self.Origin[1] + self.Dspan
            ys_lower[i, :] = ys[i, :]
            ys_upper[i, :] = ys[i, :]

            for j in np.arange(0, self.N):
                zs_upper[i, j] = self.Origin[2] + self.Ddihedr[j] + \
                    self.Dchord[j] * self.airfoil.y_upper(i/(self.M-1))
                zs_lower[i, j] = self.Origin[2] + self.Ddihedr[j] + \
                    self.Dchord[j] * self.airfoil.y_lower(i/(self.M-1))
                zs[i, j] = self.Origin[2] + self.Ddihedr[j] + self.Dchord[j] * \
                    self.airfoil.camber_line(
                        i/(self.M-1))  # camber_line y_upper
            xs[i, :], ys[i, :], zs[i, :] = np.matmul(
                self.Rmat, [xs[i, :], ys[i, :], zs[i, :]])

            xs_lower[i, :], ys_lower[i, :], zs_lower[i, :] = np.matmul(
                self.Rmat, [xs_lower[i, :], ys_lower[i, :], zs_lower[i, :]])

            xs_upper[i, :], ys_upper[i, :], zs_upper[i, :] = np.matmul(
                self.Rmat, [xs_upper[i, :], ys_upper[i, :], zs_upper[i, :]])

        self.grid = np.array((xs, ys, zs)).T
        self.grid_upper = np.array((xs_upper, ys_upper, zs_upper)).T
        self.grid_lower = np.array((xs_lower, ys_lower, zs_lower)).T

        self.panels = self.grid2panels(self.grid)
        self.panels_lower = self.grid2panels(self.grid_lower)
        self.panels_upper = self.grid2panels(self.grid_upper)


def linSpan(sp, Ni):
    return np.linspace(0, sp, Ni)


def uniformChord(Ni, ch=1):
    return ch * np.ones(Ni)


def linearChord(Ni, ch1, ch2):
    return np.linspace(ch1, ch2, Ni)
