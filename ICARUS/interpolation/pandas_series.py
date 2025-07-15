from __future__ import annotations

import numpy as np
from pandas import Series

from ICARUS.core.types import FloatArray


def interpolate_series_index(xval: float, series: Series[float]) -> float:
    """Compute xval as the linear interpolation of xval where df is a dataframe and
    df.x are the x coordinates, and df.y are the y coordinates. df.x is expected to be sorted.

    Args:
        xval (float): Value to interpolate
        series (pd.Series): Series to interpolate from

    Returns:
        float: Interpolated Index

    """
    return float(np.interp(xval, series.to_numpy(), series.index.to_numpy()))


def interpolate_series_value(xval: float, series: Series[float]) -> float:
    """Interpolate Pandas Series Value

    Args:
        xval (float): Value to interpolate
        series (pd.Series): Series to interpolate from

    Returns:
        float: Interpolated Value

    """
    # compute xval as the linear interpolation of xval where df is a dataframe and
    #  df.x are the x coordinates, and df.y are the y coordinates. df.x is expected to be sorted.
    return float(np.interp(xval, series.index.to_numpy(), series.to_numpy()))


def interpolate_from_series(
    xs: list[float] | FloatArray,
    series: Series[float],
) -> Series[float]:
    """Interpolate Pandas Series

    Args:
        xs (list[float]): Values to interpolate
        series (pd.Series): Series to interpolate from

    Returns:
        pd.Series: Interpolated Series

    """
    # compute xs as the linear interpolation of xs where df is a dataframe and
    #  df.x are the x coordinates, and df.y are the y coordinates. df.x is expected to be sorted.
    xs_sorted = np.sort(xs)
    return Series(
        np.interp(xs_sorted, series.index.to_numpy(), series.to_numpy()),
        index=xs_sorted,
    )


def get_linear_series(series: Series[float]) -> Series[float]:
    """Get the Linear Part of a Series. We assume that the series is a curve with one linear
    part and some non-linear part. We find the linear part by finding the second derivative
    of the series and then applying a threshold to it. The threshold is set to 0.1. The
    threshold is applied to the absolute value of the second derivative. The threshold is
    applied to the second derivative of the series and the result is a boolean series. The
    boolean series is then used to filter the original series and the result is the linear
    part of the series.

    Args:
        series (pd.Series): Series to filter

    Returns:
        pd.Series: Filtered Series

    """
    # Get Second Derivative
    second_derivative: Series[float] = series.diff().diff()
    # Apply Threshold
    threshold: float = 0.01
    second_derivative_idx: Series[bool] = second_derivative.abs() < threshold
    # Filter Series
    return series[second_derivative_idx]
