import jax
import numpy as np

from ICARUS.core.types import FloatArray


@jax.jit
def vortexL(
    xp: float,
    yp: float,
    zp: float,
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
    gamma: float,
) -> tuple[float, float, float]:
    """Computes the velocities induced at a point xp,yp,zp
    by a vortex line given its two end points and circulation

    Args:
        xp: x coordinate of point
        yp: y coordinate of point
        zp: z coordinate of point
        x1: x coordinate of line startpoint
        y1: y coordinate of line startpoint
        z1: z coordinate of line startpoint
        x2: x coordinate of line endpoint
        y2: y coordinate of line endpoint
        z2: z coordinate of line endpoint
        gamma: vorticity

    Returns:
        u,v,w: induced velocities
    """
    crossx: float = (yp - y1) * (zp - z2) - (zp - z1) * (yp - y2)
    crossy: float = -(xp - x1) * (zp - z2) + (zp - z1) * (xp - x2)
    crossz: float = (xp - x1) * (yp - y2) - (yp - y1) * (xp - x2)

    cross_mag: float = crossx**2 + crossy**2 + crossz**2
    r1: float = np.sqrt((xp - x1) ** 2 + (yp - y1) ** 2 + (zp - z1) ** 2)
    r2: float = np.sqrt((xp - x2) ** 2 + (yp - y2) ** 2 + (zp - z2) ** 2)
    r0: float = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    e: float = 1e-9
    if r1 < e or r2 < e or cross_mag < e:
        return 0, 0, 0
    r0dr1 = (x2 - x1) * (xp - x1) + (y2 - y1) * (yp - y1) + (z2 - z1) * (zp - z1)
    r0dr2 = (x2 - x1) * (xp - x2) + (y2 - y1) * (yp - y2) + (z2 - z1) * (zp - z2)

    epsilon = 0.0001
    d = cross_mag / r0
    filt = 1  # - np.exp(-(d/epsilon)**2)
    K: float = filt * (gamma / (4 * np.pi * cross_mag)) * (r0dr1 / r1 - r0dr2 / r2)
    u: float = K * crossx
    v: float = K * crossy
    w: float = K * crossz

    return u, v, w


@jax.jit
def voring(
    x: float,
    y: float,
    z: float,
    j: int,
    k: int,
    grid: FloatArray,
    gamma: float = 1,
) -> tuple[FloatArray, FloatArray]:
    """Vorticity Ring Element. Computes the velocities induced at a point x,y,z
    by a vortex ring given its grid lower corner coordinates

    Args:
        x: x coordinate of point
        y: y coordinate of point
        z: z coordinate of point
        j: specifies i index of grid (gd[j,k])
        k: specifies j index of grid (gd[j,k])
        grid: grid of geometry
        gamma: Circulation. Defaults to 1 (When we use nondimensional solve).

    Returns:
        U: induced velocities vector
    """
    u1, v1, w1 = vortexL(
        x,
        y,
        z,
        grid[j, k, 0],
        grid[j, k, 1],
        grid[j, k, 2],
        grid[j + 1, k, 0],
        grid[j + 1, k, 1],
        grid[j + 1, k, 2],
        gamma,
    )

    u2, v2, w2 = vortexL(
        x,
        y,
        z,
        grid[j + 1, k, 0],
        grid[j + 1, k, 1],
        grid[j + 1, k, 2],
        grid[j + 1, k + 1, 0],
        grid[j + 1, k + 1, 1],
        grid[j + 1, k + 1, 2],
        gamma,
    )
    u3, v3, w3 = vortexL(
        x,
        y,
        z,
        grid[j + 1, k + 1, 0],
        grid[j + 1, k + 1, 1],
        grid[j + 1, k + 1, 2],
        grid[j, k + 1, 0],
        grid[j, k + 1, 1],
        grid[j, k + 1, 2],
        gamma,
    )
    u4, v4, w4 = vortexL(
        x,
        y,
        z,
        grid[j, k + 1, 0],
        grid[j, k + 1, 1],
        grid[j, k + 1, 2],
        grid[j, k, 0],
        grid[j, k, 1],
        grid[j, k, 2],
        gamma,
    )

    u: float = u1 + u2 + u3 + u4
    v: float = v1 + v2 + v3 + v4
    w: float = w1 + w2 + w3 + w4

    ustar: float = u2 + u4
    vstar: float = v2 + v4
    wstar: float = w2 + w4

    U: FloatArray = np.array((u, v, w))
    Ustar: FloatArray = np.array((ustar, vstar, wstar))
    return U, Ustar


@jax.jit
def hshoe2(
    x: float,
    y: float,
    z: float,
    k: int,
    j: int,
    grid: FloatArray,
    gamma: float = 1,
) -> tuple[FloatArray, FloatArray]:
    """Vorticity Horseshow Element. Computes the velocities induced at a point x,y,z
    by a horseshow Vortex given its grid lower corner coordinates

    Args:
        x: x coordinate of point
        y: y coordinate of point
        z: z coordinate of point
        k: specifies k index of grid (gd[k,j])
        j: specifies j index of grid (gd[k,j])
        grid: grid of geometry
        gamma: Circulation. Defaults to 1 (When we use nondimensional solve).

    Returns:
        U: induced velocities vector
    """
    u1, v1, w1 = vortexL(
        x,
        y,
        z,
        grid[j, k + 1, 0],
        grid[j, k + 1, 1],
        grid[j, k + 1, 2],
        grid[j, k, 0],
        grid[j, k, 1],
        grid[j, k, 2],
        gamma,
    )
    u2, v2, w2 = vortexL(
        x,
        y,
        z,
        grid[j, k, 0],
        grid[j, k, 1],
        grid[j, k, 2],
        grid[j + 1, k, 0],
        grid[j + 1, k, 1],
        grid[j + 1, k, 2],
        gamma,
    )
    u3, v3, w3 = vortexL(
        x,
        y,
        z,
        grid[j + 1, k, 0],
        grid[j + 1, k, 1],
        grid[j + 1, k, 2],
        grid[j + 1, k + 1, 0],
        grid[j + 1, k + 1, 1],
        grid[j + 1, k + 1, 2],
        gamma,
    )

    u: float = u1 + u2 + u3
    v: float = v1 + v2 + v3
    w: float = w1 + w2 + w3

    ust: float = u1 + u3
    vst: float = v1 + v3
    wst: float = w1 + w3

    U: FloatArray = np.array((u, v, w))
    Ustar: FloatArray = np.array((ust, vst, wst))

    return U, Ustar


@jax.jit
def hshoeSL2(
    x: float,
    y: float,
    z: float,
    i: int,
    j: int,
    grid: FloatArray,
    gamma: float = 1,
) -> tuple[FloatArray, FloatArray]:
    """Slanted Horseshoe Element To Work with Panels

    Args:
        x: x coordinate of point
        y: y coordinate of point
        z: z coordinate of point
        i: specifies i index of grid (gd[i,j])
        j: specifies j index of grid (gd[i,j])
        grid: grid of geometry
        gamma: Circulation. Defaults to 1 (When we use nondimensional solve).

    Returns:
        U, Ustar: induced velocities vector
    """
    u1, v1, w1 = vortexL(
        x,
        y,
        z,
        grid[j, i + 2, 0],
        grid[j, i + 2, 1],
        grid[j, i + 2, 2],
        grid[j, i + 1, 0],
        grid[j, i + 1, 1],
        grid[j, i + 1, 2],
        gamma,
    )
    u2, v2, w2 = vortexL(
        x,
        y,
        z,
        grid[j, i + 1, 0],
        grid[j, i + 1, 1],
        grid[j, i + 1, 2],
        grid[j, i, 0],
        grid[j, i, 1],
        grid[j, i, 2],
        gamma,
    )
    u3, v3, w3 = vortexL(
        x,
        y,
        z,
        grid[j, i, 0],
        grid[j, i, 1],
        grid[j, i, 2],
        grid[j + 1, i, 0],
        grid[j + 1, i, 1],
        grid[j + 1, i, 2],
        gamma,
    )
    u4, v4, w4 = vortexL(
        x,
        y,
        z,
        grid[j + 1, i, 0],
        grid[j + 1, i, 1],
        grid[j + 1, i, 2],
        grid[j + 1, i + 1, 0],
        grid[j + 1, i + 1, 1],
        grid[j + 1, i + 1, 2],
        gamma,
    )
    u5, v5, w5 = vortexL(
        x,
        y,
        z,
        grid[j + 1, i + 1, 0],
        grid[j + 1, i + 1, 1],
        grid[j + 1, i + 1, 2],
        grid[j + 1, i + 2, 0],
        grid[j + 1, i + 2, 1],
        grid[j + 1, i + 2, 2],
        gamma,
    )

    u: float = u1 + u2 + u3 + u4 + u5
    v: float = v1 + v2 + v3 + v4 + v5
    w: float = w1 + w2 + w3 + w4 + w5

    ust: float = u1 + u2 - u3 - u4
    vst: float = v1 + v2 - v3 - v4
    wst: float = w1 + w2 - w3 - w4

    U: FloatArray = np.array((u, v, w))
    Ustar: FloatArray = np.array((ust, vst, wst))

    return U, Ustar


@jax.jit
def symm_wing_panels(
    x: float,
    y: float,
    z: float,
    i: int,
    j: int,
    grid: FloatArray,
    gamma: float = 1,
) -> tuple[FloatArray, FloatArray]:
    """
    Computes the induced velocities at a point (x,y,z) by panel[i,j] the velocities induce only by the chordwise vortices,
    using the slanted horseshow model and accounting for a symmetric wing around the x axis.

    Args:
        x (float): X coordinate of the point
        y (float): Y coordinate of the point
        z (float): Z coordinate of the point
        i (int): Index of the panel
        j (int): Index of the panel
        grid (FloatArray): Array of the panels
        gamma (float, optional): Circulation of the panel. Defaults to 1.

    Returns:
        tuple[FloatArray, FloatArray]: _description_
    """

    U1, U1st = voring(x, y, z, i, j, grid, gamma)
    U2, U2st = voring(x, -y, z, i, j, grid, gamma)

    U_ind = np.array([U1[0] + U2[0], U1[1] - U2[1], U1[2] + U2[2]])
    U_ind_st = np.array([U1st[0] + U2st[0], U1st[1] - U2st[1], U1st[2] + U2st[2]])
    return U_ind, U_ind_st


def ground_effect(x: float, y: float, z: float, i: int, j: int, panel: FloatArray) -> tuple[FloatArray, FloatArray]:
    """
    Computes the induced velocities at a point (x,y,z) by panel[i,j] the velocities induce only by the chordwise vortices,
    using the slanted horseshow model and accounting for the ground effect by reflecting the panels along the z axis.

    Args:
        x (float): X coordinate of the point
        y (float): Y coordinate of the point
        z (float): Z coordinate of the point
        i (int): Index of the panel
        j (int): Index of the panel
        panel (FloatArray): Array of the panels

    Returns:
        tuple[FloatArray, FloatArray]: The induced velocities at the point (x,y,z) by panel[i,j] the velocities induce only by the chordwise vortices
    """
    U1, U1st = hshoeSL2(x, y, z, i, j, panel)
    U2, U2st = hshoeSL2(x, y, -z, i, j, panel)

    U_ind = np.array([U1[0] + U2[0], U1[1] + U2[1], U1[2] - U2[2]])
    U_ind_st = np.array([U1st[0] + U2st[0], U1st[1] + U2st[1], U1st[2] - U2st[2]])
    return U_ind, U_ind_st


# try:
#     from jax import jit
#     # Jit all functions


# except ImportError:
#     pass
