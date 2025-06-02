import numpy as np
from scipy.interpolate import interp1d

from ICARUS.core.types import FloatArray


class Interpolator(interp1d):
    """
    Interpolator class that allows for extrapolation

    Note:
        * This class is a wrapper around 'scipy.interpolate.interp1d'

    Args:
        :x: x-coordinates
        :y: y-coordinates
        :kind: Interpolation type
        :bounds_error: (bool) If True, a ValueError is raised when interpolated values are requested outside of the domain of the input data
        :fill_value: (str or float) If a string, it must be one of 'extrapolate', 'constant', 'nearest', 'zero', 'slinear', 'quadratic', or 'cubic'
    """

    def __getstate__(self):
        """
        This method defines the state to be pickled for interp1d objects.

        We exclude the private member '_InterpolatorBase__spline' as it's
        recreated during object initialization.

        Returns:
            dict: A dictionary containing the picklable state of the object.
        """
        return {
            "kind": self._kind,
            "bounds_error": self.bounds_error,
            "fill_value": self.fill_value,
            # "assume_sorted": self.assume_sorted,
            "copy": self.copy,
            "x": self.x.copy(),
            "y": self.y.copy(),
            # "args": self.args,
            # "kwargs": self.kwargs,
        }

    def __setstate__(self, state):
        """
        This method reconstructs the interp1d object from the pickled state.

        Args:
            state (dict): The pickled state of the object.
        """
        self.kind = state["kind"]
        self.bounds_error = state["bounds_error"]
        self.fill_value = state["fill_value"]
        # self.assume_sorted = state["assume_sorted"]
        self.copy = state["copy"]
        self.xi = state["x"]
        self.yi = state["y"]
        # self.args = state["args"]
        # self.kwargs = state["kwargs"]
        # Recreate the spline object during initialization
        self.__init__(
            self.xi,
            self.yi,
            kind=self.kind,
            bounds_error=self.bounds_error,
            fill_value=self.fill_value,
            # assume_sorted=self.assume_sorted,
            copy=self.copy,
            # args=self.args,
            # kwargs=self.kwargs,
        )


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
