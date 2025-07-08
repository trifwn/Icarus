from typing import Any

from scipy.interpolate import interp1d


class ScipyInterpolator1D(interp1d):
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

    def __getstate__(self) -> dict[str, Any]:
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

    def __setstate__(self, state: dict[str, Any]) -> None:
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
        ScipyInterpolator1D.__init__(
            self,
            x=self.xi,
            y=self.yi,
            kind=self.kind,
            bounds_error=self.bounds_error,
            fill_value=self.fill_value,
            copy=self.copy,
        )
