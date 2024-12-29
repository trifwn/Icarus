import numpy as np

from ICARUS.core.types import FloatArray


def interpolate(
    xa: FloatArray | list[float],
    ya: FloatArray | list[float],
    queryPoints: FloatArray | list[float],
) -> FloatArray:
    """A cubic spline interpolation on a given set of points (x,y)
    Recalculates everything on every call which is far from efficient but does the job for now
    should eventually be replaced by an external helper class

    Args:
        xa (FloatArray | list[float]): X coordinates of the points
        ya (FloatArray | list[float]): Y coordinates of the points
        queryPoints (FloatArray | list[float]): X coordinates of the points to interpolate

    Returns:
        FloatArray: coordinates of the points to interpolate

    """
    # PreCompute() from Paint Mono which in turn adapted:
    # NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING
    # ISBN 0-521-43108-5, page 113, section 3.3.
    # http://paint-mono.googlecode.com/svn/trunk/src/PdnLib/SplineInterpolator.cs

    # number of points
    n: int = len(xa)
    u: FloatArray = np.zeros(n)
    y2: FloatArray = np.zeros(n)

    for i in range(1, n - 1):
        # This is the decomposition loop of the tridiagonal algorithm.
        # y2 and u are used for temporary storage of the decomposed factors.

        wx = xa[i + 1] - xa[i - 1]
        sig = (xa[i] - xa[i - 1]) / wx
        p = sig * y2[i - 1] + 2.0

        y2[i] = (sig - 1.0) / p

        ddydx = (ya[i + 1] - ya[i]) / (xa[i + 1] - xa[i]) - (ya[i] - ya[i - 1]) / (xa[i] - xa[i - 1])

        u[i] = (6.0 * ddydx / wx - sig * u[i - 1]) / p

    y2[n - 1] = 0

    # This is the backsubstitution loop of the tridiagonal algorithm
    # ((int i = n - 2; i >= 0; --i):
    for i in range(n - 2, -1, -1):
        y2[i] = y2[i] * y2[i + 1] + u[i]

    # interpolate() adapted from Paint Mono which in turn adapted:
    # NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING
    # ISBN 0-521-43108-5, page 113, section 3.3.
    # http://paint-mono.googlecode.com/svn/trunk/src/PdnLib/SplineInterpolator.cs

    results = np.zeros(n)

    # loop over all query points
    for i in range(len(queryPoints)):
        # bisection. This is optimal if sequential calls to this
        # routine are at random values of x. If sequential calls
        # are in order, and closely spaced, one would do better
        # to store previous values of klo and khi and test if

        klo = 0
        khi = n - 1

        while khi - klo > 1:
            k = (khi + klo) >> 1
            if xa[k] > queryPoints[i]:
                khi = k
            else:
                klo = k

        h = xa[khi] - xa[klo]
        a = (xa[khi] - queryPoints[i]) / h
        b = (queryPoints[i] - xa[klo]) / h

        # Cubic spline polynomial is now evaluated.
        results[i] = a * ya[klo] + b * ya[khi] + ((a * a * a - a) * y2[klo] + (b * b * b - b) * y2[khi]) * (h * h) / 6.0

    return results
