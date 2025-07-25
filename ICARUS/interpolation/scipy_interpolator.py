import pickle
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
from scipy import interpolate
from typing_extensions import Self
from typing_extensions import TypeAlias

# --- Type Aliases for Enhanced Readability and Precision ---

# Represents array-like objects that can be converted to a NumPy array
ArrayLike: TypeAlias = Union[np.ndarray, List[Any], Tuple[Any, ...]]

# Represents the coordinates for 1D interpolation
Points1D: TypeAlias = np.ndarray

# Represents coordinates for N-D interpolation:
# - A single (n_points, n_dims) array
# - A tuple of 1D arrays, one for each dimension
PointsND: TypeAlias = Union[np.ndarray, Tuple[np.ndarray, ...]]

# Union of all possible underlying SciPy interpolator objects
InterpolatorFunc: TypeAlias = Union[
    interpolate.interp1d,
    interpolate.PchipInterpolator,
    interpolate.Akima1DInterpolator,
    interpolate.CubicSpline,
    interpolate.UnivariateSpline,
    interpolate.BarycentricInterpolator,
    interpolate.KroghInterpolator,
    interpolate.RegularGridInterpolator,
    interpolate.Rbf,
    interpolate.RBFInterpolator,
    interpolate.NearestNDInterpolator,
    interpolate.LinearNDInterpolator,
    interpolate.CloughTocher2DInterpolator,
]


class ScipyInterpolator:
    """
    A serializable, robust wrapper for SciPy's interpolation functions.

    This class provides a unified, scikit-learn-like interface (`fit`, `predict`)
    for various SciPy interpolators and ensures that the fitted interpolator can
    be serialized and deserialized using libraries like pickle.

    Attributes:
    -----------
    method : str
        The interpolation method being used.
    kwargs : dict
        Additional keyword arguments for the SciPy interpolator.
    is_fitted : bool
        True if the interpolator has been fitted to data.
    interpolator : Optional[InterpolatorFunc]
        The underlying SciPy interpolator object after fitting.
    x_data : Optional[PointsND]
        The stored input data points (features).
    y_data : Optional[np.ndarray]
        The stored output data values (targets).
    grid_data : Optional[Tuple[np.ndarray, ...]]
        The stored grid coordinates for 'regular_grid' interpolation.
    """

    # --- Supported Methods ---
    _SUPPORTED_1D_METHODS: Dict[str, Type[Any]] = {
        "linear": interpolate.interp1d,
        "cubic": interpolate.interp1d,
        "nearest": interpolate.interp1d,
        "previous": interpolate.interp1d,
        "next": interpolate.interp1d,
        "quadratic": interpolate.interp1d,
        "pchip": interpolate.PchipInterpolator,
        "akima": interpolate.Akima1DInterpolator,
        "cubic_spline": interpolate.CubicSpline,
        "univariate_spline": interpolate.UnivariateSpline,
        "barycentric": interpolate.BarycentricInterpolator,
        "krogh": interpolate.KroghInterpolator,
    }

    _SUPPORTED_ND_METHODS: Dict[str, Union[Type[Any], Callable[..., np.ndarray]]] = {
        "griddata": interpolate.griddata,
        "regular_grid": interpolate.RegularGridInterpolator,
        "rbf": interpolate.Rbf,
        "rbf_interpolator": interpolate.RBFInterpolator,
        "nearest_nd": interpolate.NearestNDInterpolator,
        "linear_nd": interpolate.LinearNDInterpolator,
        "clough_tocher": interpolate.CloughTocher2DInterpolator,
    }

    _SUPPORTED_METHODS = {**_SUPPORTED_1D_METHODS, **_SUPPORTED_ND_METHODS}

    def __init__(self, method: str = "linear", **kwargs: Any) -> None:
        """
        Initializes the ScipyInterpolator.

        Parameters:
        -----------
        method : str, default='linear'
            The interpolation method to use. See class documentation for supported
            methods.
        **kwargs : Any
            Additional keyword arguments to be passed to the underlying
            SciPy interpolation function.

        Raises:
        -------
        ValueError
            If the specified method is not supported.
        """
        if method not in self._SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: '{method}'. Supported methods are: "
                f"{list(self._SUPPORTED_METHODS.keys())}",
            )

        self.method: str = method
        self.kwargs: Dict[str, Any] = kwargs
        self.is_fitted: bool = False
        self.interpolator: Optional[InterpolatorFunc] = None
        self.x_data: Optional[PointsND] = None
        self.y_data: Optional[np.ndarray] = None
        self.grid_data: Optional[Tuple[np.ndarray, ...]] = None

    def fit(
        self,
        x: PointsND,
        y: ArrayLike,
        grid: Optional[Tuple[ArrayLike, ...]] = None,
    ) -> Self:
        """
        Fits the interpolator to the provided data.

        Parameters:
        -----------
        x : PointsND
            The input data points.
            - For 1D: A 1D array of x-coordinates.
            - For N-D: A (n_samples, n_dims) array or a tuple of 1D coordinate arrays.
        y : ArrayLike
            The output data values corresponding to the input points.
        grid : Optional[Tuple[ArrayLike, ...]], default=None
            For 'regular_grid' interpolation, a tuple of 1D arrays representing
            the grid coordinates for each dimension.

        Returns:
        --------
        Self
            The fitted interpolator instance for method chaining.
        """
        # --- Data Sanitization and Storage ---
        y_arr = np.asarray(y)
        if isinstance(x, (list, tuple)):
            x_sanitized: PointsND = tuple(np.asarray(arr) for arr in x)
        else:
            x_sanitized = np.asarray(x)

        self.y_data = y_arr
        self.x_data = x_sanitized
        if grid:
            self.grid_data = tuple(np.asarray(g) for g in grid)

        # --- Create and fit the interpolator ---
        if self.method in self._SUPPORTED_1D_METHODS:
            # Ensure x is a 1D array for 1D methods
            if not isinstance(self.x_data, np.ndarray) or self.x_data.ndim != 1:
                raise ValueError("For 1D methods, x must be a 1D array.")
            self._fit_1d(self.x_data, self.y_data)
        elif self.method in self._SUPPORTED_ND_METHODS:
            self._fit_nd(self.x_data, self.y_data, self.grid_data)

        self.is_fitted = True
        return self

    def _fit_1d(self, x: Points1D, y: np.ndarray) -> None:
        """Helper to fit 1D interpolators."""
        interpolator_class = self._SUPPORTED_1D_METHODS[self.method]
        if self.method in [
            "linear",
            "cubic",
            "nearest",
            "previous",
            "next",
            "quadratic",
        ]:
            self.interpolator = interpolator_class(
                x,
                y,
                kind=self.method,
                **self.kwargs,
            )
        else:
            self.interpolator = interpolator_class(x, y, **self.kwargs)

    def _fit_nd(
        self,
        x: PointsND,
        y: np.ndarray,
        grid: Optional[Tuple[np.ndarray, ...]] = None,
    ) -> None:
        """Helper to fit N-D interpolators."""
        if self.method == "griddata":
            # No persistent interpolator object; handled in predict()
            pass
        elif self.method == "regular_grid":
            if grid is None:
                raise ValueError(
                    "Grid coordinates must be provided for 'regular_grid' method.",
                )
            self.interpolator = interpolate.RegularGridInterpolator(
                grid,
                y,
                **self.kwargs,
            )
        elif self.method == "rbf":
            if not isinstance(x, tuple):
                # Rbf expects separate coordinate arrays, not a single points array
                x = tuple(x[:, i] for i in range(x.shape[1]))
            self.interpolator = interpolate.Rbf(*x, y, **self.kwargs)
        else:  # Methods that expect a single (n_points, n_dims) array
            points = x
            if isinstance(x, tuple):
                # Convert tuple of coordinates to a single points array
                points = np.column_stack(x)

            interpolator_class = self._SUPPORTED_ND_METHODS[self.method]
            self.interpolator = interpolator_class(points, y, **self.kwargs)

    def predict(self, x_new: ArrayLike) -> np.ndarray:
        """
        Evaluates the interpolator at new points.

        Parameters:
        -----------
        x_new : ArrayLike
            The new points at which to evaluate the interpolation. The required
            format depends on the method (e.g., array of points, tuple of
            coordinates).

        Returns:
        --------
        np.ndarray
            The interpolated values at the new points.

        Raises:
        -------
        ValueError
            If the interpolator has not been fitted yet.
        """
        if not self.is_fitted or (
            self.interpolator is None and self.method != "griddata"
        ):
            raise ValueError("Interpolator must be fitted before calling 'predict'.")

        # Special handling for griddata, which is a function, not an object
        if self.method == "griddata":
            assert self.x_data is not None and self.y_data is not None
            points = (
                np.column_stack(self.x_data)
                if isinstance(self.x_data, tuple)
                else self.x_data
            )
            # Use method from kwargs for griddata, e.g. 'cubic', 'nearest'
            griddata_method = self.kwargs.get("method", "linear")
            res = interpolate.griddata(
                points,
                self.y_data,
                x_new,
                method=griddata_method,
            )
            return np.asarray(res)

        assert self.interpolator is not None
        res = self.interpolator(x_new)
        return np.asarray(res)

    def __call__(self, x_new: ArrayLike) -> np.ndarray:
        """Makes the instance callable, equivalent to `predict`."""
        return self.predict(x_new)

    def __getstate__(self) -> Dict[str, Any]:
        """
        Serializes the interpolator's state to a dictionary for pickling.
        """
        state = self.__dict__.copy()
        del state["interpolator"]  # Remove non-serializable object

        if self.is_fitted and self.x_data is not None:
            # Explicitly store the type of x_data to ensure correct reconstruction
            if isinstance(self.x_data, tuple):
                state["x_data"] = [arr.tolist() for arr in self.x_data]
                state["x_data_type"] = "tuple"
            else:
                state["x_data"] = self.x_data.tolist()
                state["x_data_type"] = "array"

            if self.y_data is not None:
                state["y_data"] = self.y_data.tolist()
            if self.grid_data is not None:
                state["grid_data"] = [arr.tolist() for arr in self.grid_data]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Deserializes the state from a dictionary to reconstruct the object.
        """
        self.__dict__.update(state)

        if self.is_fitted and self.x_data is not None and self.y_data is not None:
            # Reconstruct data from lists back to numpy arrays using the stored type info
            x_data_type = state.get("x_data_type")
            if x_data_type == "tuple":
                self.x_data = tuple(np.array(arr) for arr in self.x_data)
            elif x_data_type == "array":
                self.x_data = np.array(self.x_data)

            if self.y_data is not None:
                self.y_data = np.array(self.y_data)
            if self.grid_data is not None:
                self.grid_data = tuple(np.array(arr) for arr in self.grid_data)

            self._recreate_interpolator()
        else:
            self.interpolator = None

    def _recreate_interpolator(self) -> None:
        """Recreates the interpolator from stored data after deserialization."""
        if not self.is_fitted or self.x_data is None or self.y_data is None:
            raise ValueError("Cannot recreate interpolator from incomplete state.")

        if self.method in self._SUPPORTED_1D_METHODS:
            assert isinstance(self.x_data, np.ndarray) and self.x_data.ndim == 1
            self._fit_1d(self.x_data, self.y_data)
        elif self.method in self._SUPPORTED_ND_METHODS:
            self._fit_nd(self.x_data, self.y_data, self.grid_data)

    def __repr__(self) -> str:
        """Provides a concise string representation of the object."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"ScipyInterpolator(method='{self.method}', {status})"


# --- Example Usage (Now works correctly) ---
if __name__ == "__main__":
    print("--- 1D Interpolation Example (Cubic Spline) ---")
    x1d = np.linspace(0, 10, 10)
    y1d = np.sin(x1d)
    x1d_new = np.linspace(0, 10, 51)

    interp_1d = ScipyInterpolator(method="cubic_spline")
    interp_1d.fit(x1d, y1d)
    y1d_pred = interp_1d.predict(x1d_new)

    serialized_interp = pickle.dumps(interp_1d)
    deserialized_interp_1d = pickle.loads(serialized_interp)
    y1d_pred_deserialized = deserialized_interp_1d.predict(x1d_new)

    assert np.allclose(y1d_pred, y1d_pred_deserialized)
    print("✅ 1D serialization and prediction verified successfully!")

    print("\n" + "=" * 50 + "\n")

    print("--- 2D Interpolation Example (LinearND) ---")
    np.random.seed(42)  # for reproducibility
    x2d = np.random.rand(50, 2) * 4.0 - 2.0
    y2d = x2d[:, 0] * np.exp(-(x2d[:, 0] ** 2) - x2d[:, 1] ** 2)
    x2d_new = np.random.rand(10, 2) * 4.0 - 2.0

    interp_2d = ScipyInterpolator(method="linear_nd")
    interp_2d.fit(x2d, y2d)
    y2d_pred = interp_2d(x2d_new)
    print(
        f"Original interpolator ({interp_2d.method}) is fitted: {interp_2d.is_fitted}",
    )

    # This sequence now works without errors
    deserialized_interp_2d = pickle.loads(pickle.dumps(interp_2d))
    print(f"Deserialized interpolator is fitted: {deserialized_interp_2d.is_fitted}")
    y2d_pred_deserialized = deserialized_interp_2d.predict(x2d_new)
    assert np.allclose(y2d_pred, y2d_pred_deserialized, equal_nan=True)
    print("✅ 2D serialization and prediction verified successfully!")
