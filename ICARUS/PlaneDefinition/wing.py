import numpy as np
import matplotlib.pyplot as plt
from .surface import Surface


class Wing:
    def __init__(self, name, airfoil, Origin, Orientation, isSymmetric, span, sweepOffset, dihAngle, chordFun, chord, spanFun, N, M, mass=1):
        """Wing Class"""

        self.N = N
        self.M = M

        self.name = name
        self.airfoil = airfoil
        self.Origin = Origin
        self.Orientation = Orientation
        self.isSymmetric = isSymmetric
        self.span = span
        self.sweepOffset = sweepOffset
        self.dihAngle = dihAngle
        self.chordFun = chordFun
        self.chord = chord
        self.spanFun = spanFun
        self.mass = mass

        self.Gamma = dihAngle * np.pi/180

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

        # Find Chords MAC-SMC
        self.meanChords()

        # Calculate Areas
        self.findArea()

        # Calculate Volumes
        self.findVolume()

        # Find Center of Mass
        self.centerMass()

        # Calculate Moments
        self.inertia(self.mass, self.CG)

    def splitSymmetric(self):
        """Split Symmetric Wing into two Wings"""
        if self.isSymmetric == True:
            left = Wing(name=f"L{self.name}",
                        airfoil=self.airfoil,
                        Origin=[self.Origin[0], self.Origin[1] -
                                self.span/2, self.Origin[2]],
                        Orientation=self.Orientation,
                        isSymmetric=False,
                        span=self.span/2,
                        sweepOffset=self.sweepOffset,
                        dihAngle=self.dihAngle,
                        chordFun=self.chordFun,
                        chord=self.chord[::-1],
                        spanFun=self.spanFun,
                        N=self.N,
                        M=self.M,
                        mass=self.mass/2)

            right = Wing(name=f"R{self.name}",
                         airfoil=self.airfoil,
                         Origin=self.Origin,
                         Orientation=self.Orientation,
                         isSymmetric=False,
                         span=self.span/2,
                         sweepOffset=self.sweepOffset,
                         dihAngle=self.dihAngle,
                         chordFun=self.chordFun,
                         chord=self.chord,
                         spanFun=self.spanFun,
                         N=self.N,
                         M=self.M,
                         mass=self.mass/2)
            return left, right
        else:
            print("Cannot Split Body it is not symmetric")

    def createSurfaces(self):
        """Create Surfaces given the Grid and Airfoil"""
        surfaces = []
        symSurfaces = []
        startPoint = [
            self.xoff[0],
            self.Dspan[0],
            self.Ddihedr[0],
        ]
        endPoint = [
            self.xoff[-1],
            self.Dspan[-1],
            self.Ddihedr[-1],
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

    def plotWing(self, fig=None, ax=None, movement=None):
        """Plot Wing in 3D"""
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_title(self.name)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        if movement is None:
            movement = np.zeros(3)

        for surf in self.allSurfaces:
            s1 = np.matmul(self.Rmat, surf.startStrip())
            s2 = np.matmul(self.Rmat, surf.endStrip())
            for item in [s1, s2]:
                item[0, :] += self.Origin[0] + movement[0]
                item[1, :] += self.Origin[1] + movement[1]
                item[2, :] += self.Origin[2] + movement[2]

            ax.plot(*s1, '-', color='red')
            ax.plot(*s2, '-', color='blue')

        for i in np.arange(0, self.N-1):
            for j in np.arange(0, self.M-1):
                for item in [self.panels_lower, self.panels_upper]:
                    p1, p3, p4, p2 = item[i, j, :, :]
                    xs = np.reshape(
                        [p1[0], p2[0], p3[0], p4[0]], (2, 2)) + movement[0]
                    ys = np.reshape(
                        [p1[1], p2[1], p3[1], p4[1]], (2, 2)) + movement[1]
                    zs = np.reshape(
                        [p1[2], p2[2], p3[2], p4[2]], (2, 2)) + movement[2]
                    ax.plot_wireframe(xs, ys, zs, linewidth=0.5)
                    if self.isSymmetric == True:
                        ax.plot_wireframe(xs, -ys, zs, linewidth=0.5)
        ax.axis('scaled')
        ax.view_init(30, 150)

    def grid2panels(self, grid):
        """Convert Grid to Panels"""
        panels = np.empty((self.N-1, self.M-1, 4, 3))
        for i in np.arange(0, self.N-1):
            for j in np.arange(0, self.M-1):
                panels[i, j, 0, :] = grid[i+1, j]
                panels[i, j, 1, :] = grid[i, j]
                panels[i, j, 2, :] = grid[i, j+1]
                panels[i, j, 3, :] = grid[i+1, j+1]
        return panels

    def createGrid(self):
        """Create Grid for Wing"""
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
            xs[i, :] = self.xoff + xpos
            xs_lower[i, :] = xs[i, :]
            xs_upper[i, :] = xs[i, :]

            ys[i, :] = self.Dspan
            ys_lower[i, :] = ys[i, :]
            ys_upper[i, :] = ys[i, :]

            for j in np.arange(0, self.N):
                zs_upper[i, j] = self.Ddihedr[j] + \
                    self.Dchord[j] * self.airfoil.y_upper(i/(self.M-1))
                zs_lower[i, j] = self.Ddihedr[j] + \
                    self.Dchord[j] * self.airfoil.y_lower(i/(self.M-1))
                zs[i, j] = self.Ddihedr[j] + \
                    self.Dchord[j] * self.airfoil.camber_line(i/(self.M-1))

            # ROTATE ACCORDING TO RMAT
            xs[i, :], ys[i, :], zs[i, :] = np.matmul(
                self.Rmat, [xs[i, :], ys[i, :], zs[i, :]])

            xs_lower[i, :], ys_lower[i, :], zs_lower[i, :] = np.matmul(
                self.Rmat, [xs_lower[i, :], ys_lower[i, :], zs_lower[i, :]])

            xs_upper[i, :], ys_upper[i, :], zs_upper[i, :] = np.matmul(
                self.Rmat, [xs_upper[i, :], ys_upper[i, :], zs_upper[i, :]])

        for item in [xs, xs_upper, xs_lower]:
            item += self.Origin[0]

        for item in [ys, ys_upper, ys_lower]:
            item += self.Origin[1]

        for item in [zs, zs_upper, zs_lower]:
            item += self.Origin[2]

        self.grid = np.array((xs, ys, zs)).T
        self.grid_upper = np.array((xs_upper, ys_upper, zs_upper)).T
        self.grid_lower = np.array((xs_lower, ys_lower, zs_lower)).T

        self.panels = self.grid2panels(self.grid)
        self.panels_lower = self.grid2panels(self.grid_lower)
        self.panels_upper = self.grid2panels(self.grid_upper)

    def meanChords(self):
        "Finds the Mean Aerodynamic Chord (MAC) of the wing."
        num = 0
        denum = 0
        for i in np.arange(0, self.N-1):
            num += ((self.Dchord[i] + self.Dchord[i+1])/2)**2 *\
                (self.Dspan[i+1] - self.Dspan[i])
            denum += (self.Dchord[i] + self.Dchord[i+1])/2 *\
                (self.Dspan[i+1] - self.Dspan[i])
        self.MAC = num/denum

        # Finds Standard Mean Chord
        num = 0
        denum = 0
        for i in np.arange(0, self.N-1):
            num += (self.Dchord[i] + self.Dchord[i+1])/2 *\
                (self.Dspan[i+1] - self.Dspan[i])
            denum += (self.Dspan[i+1] - self.Dspan[i])
        self.SMC = num/denum

    def findAspectRatio(self):
        """Finds the Aspect Ratio of the wing."""
        self.AR = (self.span ** 2)/self.Area

    def findArea(self):
        "Finds the area of the wing."

        self.Area = 0
        self.S = 0
        for i in np.arange(0, self.N-1):
            self.S += 2*(self.grid_upper[i+1, 0, 1] -
                         self.grid_upper[i, 0, 1]) * \
                (self.Dchord[i] + self.Dchord[i+1])/2

        g_up = self.grid_upper
        g_low = self.grid_lower
        for i in np.arange(0, self.N-1):
            for j in np.arange(0, self.M-1):
                AB1 = (g_up[i+1, j, :] - g_up[i, j, :])
                AB2 = (g_up[i+1, j+1, :] - g_up[i, j+1, :])

                AD1 = (g_up[i, j+1, :] - g_up[i, j, :])
                AD2 = (g_up[i+1, j+1, :] - g_up[i+1, j, :])

                Area_up = np.linalg.norm(np.cross((AB1+AB2)/2, (AD1+AD2)/2))

                AB1 = (g_low[i+1, j, :] - g_low[i, j, :])
                AB2 = (g_low[i+1, j+1, :] - g_low[i, j+1, :])

                AD1 = (g_low[i, j+1, :] - g_low[i, j, :])
                AD2 = (g_low[i+1, j+1, :] - g_low[i+1, j, :])

                Area_low = np.linalg.norm(np.cross((AB1+AB2)/2, (AD1+AD2)/2))

                self.Area += Area_up + Area_low

        # Find Aspect Ratio
        self.findAspectRatio()

    def findVolume(self):
        """Finds the volume of the wing. This is done by finding the volume of the wing
        as the sum of a series of panels."""

        self.VolumeDist = np.empty((self.N-1, self.M-1))
        self.VolumeDist2 = np.empty((self.N-1, self.M-1))

        g_up = self.grid_upper
        g_low = self.grid_lower
        self.AreasB = np.zeros((self.N-1))
        self.AreasF = np.zeros((self.N-1))

        # We divide the wing into a set of lower and upper panels that form
        # a tetrahedron. We then find the volume of each tetrahedron and sum.
        # This is equivalent to finding the area of the front and back faces of
        # the tetrahedron taking the average and multiplying by the height.
        # We then have to subtract the volume of the trianglular prism that is
        # formed by the slanted edges of the tetrahedron.
        for i in np.arange(0, self.N-1):
            for j in np.arange(0, self.M-1):

                # Area of the front face
                AB1 = (g_up[i+1, j, :] - g_up[i, j, :])
                AB2 = (g_low[i+1, j, :] - g_low[i, j, :])

                AD1 = (g_up[i, j, :] - g_low[i, j, :])
                AD2 = (g_up[i+1, j, :] - g_low[i+1, j, :])
                Area_front_v = (np.cross((AB1+AB2)/2, (AD1+AD2)/2))
                Area_front = np.linalg.norm(Area_front_v)

                # Area of the back face
                AB1 = (g_up[i+1, j+1, :] - g_up[i, j+1, :])
                AB2 = (g_low[i+1, j+1, :] - g_low[i, j+1, :])

                AD1 = (g_up[i, j+1, :] - g_low[i, j+1, :])
                AD2 = (g_up[i+1, j+1, :] - g_low[i+1, j+1, :])
                Area_back_v = (np.cross((AB1+AB2)/2, (AD1+AD2)/2))
                Area_back = np.linalg.norm(Area_back_v)

                # Height of the tetrahedron
                dx1 = g_up[i, j+1, 0] - g_up[i, j, 0]
                dx2 = g_up[i+1, j+1, 0] - g_up[i+1, j, 0]
                dx3 = g_low[i, j+1, 0] - g_low[i, j, 0]
                dx4 = g_low[i+1, j+1, 0] - g_low[i+1, j, 0]
                dx = (dx1 + dx2 + dx3 + dx4)/4

                # Volume of the tetrahedron
                self.VolumeDist[i, j] = 0.5 * (Area_front + Area_back) * dx

        self.Volume = np.sum(self.VolumeDist)
        if self.isSymmetric == True:
            self.Volume = self.Volume * 2

    def centerMass(self):
        """Finds the center of mass of the wing.
        This is done by summing the volume of each panel 
        and dividing by the total volume."""
        x_cm = 0
        y_cm = 0
        z_cm = 0

        g_up = self.grid_upper
        g_low = self.grid_lower
        for i in np.arange(0, self.N-1):
            for j in np.arange(0, self.M-1):
                x_upp1 = (g_up[i, j+1, 0] + g_up[i, j, 0])/2
                x_upp2 = (g_up[i+1, j+1, 0] + g_up[i+1, j, 0])/2

                x_low1 = (g_low[i, j+1, 0] + g_low[i, j, 0])/2
                x_low2 = (g_low[i+1, j+1, 0] + g_low[i+1, j, 0])/2
                x = ((x_upp1 + x_upp2)/2 + (x_low1 + x_low2)/2)/2

                y_upp1 = (g_up[i+1, j, 1] + g_up[i, j, 1])/2
                y_upp2 = (g_up[i+1, j+1, 1] + g_up[i, j+1, 1])/2

                y_low1 = (g_low[i+1, j, 1] + g_low[i, j, 1])/2
                y_low2 = (g_low[i+1, j+1, 1] + g_low[i, j+1, 1])/2
                y = ((y_upp1 + y_upp2)/2 + (y_low1 + y_low2)/2)/2

                z_upp1 = (g_up[i+1, j, 2] + g_up[i+1, j, 2])/2
                z_upp2 = (g_up[i+1, j, 2] + g_up[i+1, j, 2])/2

                z_low1 = (g_low[i+1, j, 2] + g_low[i+1, j, 2])/2
                z_low2 = (g_low[i+1, j, 2] + g_low[i+1, j, 2])/2
                z = ((z_upp1 + z_upp2)/2 + (z_low1 + z_low2)/2)/2

                if self.isSymmetric == True:
                    x_cm += self.VolumeDist[i, j] * 2 * x
                    y_cm += 0
                    z_cm += self.VolumeDist[i, j] * 2 * z
                else:
                    x_cm += self.VolumeDist[i, j] * x
                    y_cm += self.VolumeDist[i, j] * y
                    z_cm += self.VolumeDist[i, j] * z

        self.CG = np.array((x_cm, y_cm, z_cm)) / self.Volume

    def inertia(self, mass, cog):
        """Finds the inertia of the wing."""
        I_xx = 0
        I_yy = 0
        I_zz = 0
        I_xz = 0
        I_xy = 0
        I_yz = 0

        for i in np.arange(0, self.N-1):
            for j in np.arange(0, self.M-1):
                x_upp = (self.grid_upper[i, j+1, 0] +
                         self.grid_upper[i, j, 0])/2
                x_low = (self.grid_lower[i, j+1, 0] +
                         self.grid_lower[i, j, 0])/2

                y_upp = (self.grid_upper[i+1, j, 1] +
                         self.grid_upper[i, j, 1])/2
                y_low = (self.grid_lower[i+1, j, 1] +
                         self.grid_lower[i, j, 1])/2

                z_upp1 = (self.grid_upper[i+1, j, 2] +
                          self.grid_upper[i+1, j, 2])/2
                z_upp2 = (self.grid_upper[i+1, j, 2] +
                          self.grid_upper[i+1, j, 2])/2
                z_upp = (z_upp1 + z_upp2)/2

                z_low1 = (self.grid_lower[i+1, j, 2] +
                          self.grid_lower[i+1, j, 2])/2
                z_low2 = (self.grid_lower[i+1, j, 2] +
                          self.grid_lower[i+1, j, 2])/2
                z_low = (z_low1 + z_low2)/2

                xd = ((x_upp + x_low)/2 - cog[0])**2
                zd = ((z_upp + z_low)/2 - cog[2])**2
                if self.isSymmetric == True:
                    yd = (-(y_upp + y_low)/2 - cog[1])**2
                    yd += ((y_upp + y_low)/2 - cog[1])**2
                else:
                    yd = ((y_upp + y_low)/2 - cog[1])**2

                I_xx += self.VolumeDist[i, j] * (yd + zd)
                I_yy += self.VolumeDist[i, j] * (xd + zd)
                I_zz += self.VolumeDist[i, j] * (xd + yd)

                xd = np.sqrt(xd)
                yd = np.sqrt(yd)
                zd = np.sqrt(zd)

                I_xz += self.VolumeDist[i, j] * (xd * zd)
                I_xy += self.VolumeDist[i, j] * (xd * yd)
                I_yz += self.VolumeDist[i, j] * (yd * zd)

        self.I = np.array((I_xx, I_yy, I_zz, I_xz, I_xy, I_yz)
                          ) * (mass / self.Volume)


def linSpan(sp, Ni):
    """Returns a linearly spaced span array."""
    return np.linspace(0, sp, Ni).round(12)


def uniformChord(Ni, ch=1):
    """Returns a uniform chord array."""
    return ch * np.ones(Ni)


def linearChord(Ni, ch1, ch2):
    """Returns a linearly spaced chord array."""
    return np.linspace(ch1, ch2, Ni).round(12)
