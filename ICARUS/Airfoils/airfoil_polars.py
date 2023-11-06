"""
Contains the Airfoil Polars Class and related functions.
The Airfoil Polars Class is used to store the aerodynamic
coefficients of an airfoil at different Reynolds numbers
and angles of attack. The class also contains methods to
interpolate the aerodynamic coefficients at different
Reynolds numbers and angles of attack.

To initialize the Airfoil Polars Class, just pass a the
Struct data from the DataBase to the constructor.

>>> from ICARUS.Airfoils.airfoil_polars import AirfoilPolars
>>> from ICARUS.Database import DB
>>> data = DB.get_airfoil_data("NACA0012")
>>> polars = AirfoilPolars(data)

Then we can get a specific Reynolds Subtable by calling:

>>> reynolds = 1000000
>>> polars.get_reynolds_subtable(reynolds)

We can also interpolate missing values in the table by calling:

>>> polars.fill_polar_table(df)

For any series of data (aka a specific Reynolds) we can get additional
information such as the zero lift angle, the zero lift coefficient of
moment, and the slope of the Cl vs Alpha curve by calling:

>>> df = polars.get_reynolds_subtable(reynolds)
>>> cm_curve = df["Cm"]
>>> cl_curve = df["Cl"]
>>> polars.get_zero_lift(cl_curve)
>>> polars.get_zero_lift_cm(cm_curve, zero_lift_angle)
>>> polars.get_cl_slope(cl_curve)

"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Index

from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray

# from ICARUS.Airfoils.airfoil import Airfoil


def interpolate_series_index(xval: float, series: pd.Series) -> float:
    """
    Compute xval as the linear interpolation of xval where df is a dataframe and
    df.x are the x coordinates, and df.y are the y coordinates. df.x is expected to be sorted.

    Args:
        xval (float): Value to interpolate
        series (pd.Series): Series to interpolate from

    Returns:
        float: Interpolated Value
    """

    return float(np.interp([xval], series.to_numpy(), series.index.to_numpy())[0])


def interpolate_series_value(xval: float, series: pd.Series) -> float:
    """
    Interpolate Pandas Series Value

    Args:
        xval (float): Value to interpolate
        series (pd.Series): Series to interpolate from

    Returns:
        float: Interpolated Value
    """
    # compute xval as the linear interpolation of xval where df is a dataframe and
    #  df.x are the x coordinates, and df.y are the y coordinates. df.x is expected to be sorted.
    return float(np.interp([xval], series.index.to_numpy(), series.to_numpy())[0])


def get_linear_series(series: pd.Series) -> pd.Series:
    """
    Get the Linear Part of a Series. We assume that the series is a curve with one linear
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
    second_derivative: pd.Series = series.diff().diff()
    # Apply Threshold
    threshold: float = 0.01
    second_derivative = second_derivative.abs() < threshold
    # Filter Series
    return series[second_derivative]


class Polars:
    """
    Airfoil Polars Class

    """

    def __init__(
        self,
        data: Struct | dict[str, DataFrame],
    ) -> None:
        self.data: Struct | dict[str, DataFrame] = data

        self.reynolds_keys: list[str] = list(data.keys())
        self.reynolds_nums: list[float] = [float(reyn) for reyn in self.reynolds_keys]

        # MERGE ALL POLARS INTO ONE DATAFRAME
        df: DataFrame = data[self.reynolds_keys[0]].astype("float32").dropna(axis=0, how="all")
        df.rename(
            {
                "CL": f"CL_{self.reynolds_keys[0]}",
                "CD": f"CD_{self.reynolds_keys[0]}",
                "Cm": f"Cm_{self.reynolds_keys[0]}",
                "CM": f"Cm_{self.reynolds_keys[0]}",
            },
            inplace=True,
            axis="columns",
        )
        for reyn in self.reynolds_keys[1:]:
            df2: DataFrame = data[reyn].astype("float32").dropna(axis=0, how="all")
            df2.rename(
                {"CL": f"CL_{reyn}", "CD": f"CD_{reyn}", "Cm": f"Cm_{reyn}", "CM": f"Cm_{reyn}"},
                inplace=True,
                axis="columns",
            )
            df = pd.merge(df, df2, on="AoA", how="outer")

        # SORT BY AoA
        df = df.sort_values("AoA")

        # print(df)
        # FILL NaN Values By neighbors
        df = self.fill_polar_table(df)
        self.df: DataFrame = df
        self.angles: FloatArray = df["AoA"].to_numpy()

        # Flap Angle
        self.flap_angle: float = 0.0  # airfoil.flap_angle

        # Potential Zero Lift Angle
        if "CL_Potential" in df.keys():
            potential_cl: pd.Series = df["CL_Potential"]
            potential_cm: pd.Series = df["Cm_Potential"]
        else:
            least_idx: int = self.reynolds_nums.index(min(self.reynolds_nums))
            potential_cl = df[f"CL_{self.reynolds_keys[least_idx]}"]
            potential_cl.index = Index(df["AoA"].astype("float32"))
            potential_cm = df[f"Cm_{self.reynolds_keys[least_idx]}"]
            potential_cm.index = Index(df["AoA"].astype("float32"))
        self.a_zero_pot: float = self.get_zero_lift(potential_cl)

        # Potential Cm at Zero Lift Angle
        self.cm_pot: float = self.get_zero_lift_cm(potential_cm, self.a_zero_pot)

        # Viscous Zero Lift Angle
        max_idx: int = self.reynolds_nums.index(max(self.reynolds_nums))
        viscous: pd.Series = df[f"CL_{self.reynolds_keys[max_idx]}"]
        viscous.index = Index(df["AoA"].astype("float32"))
        self.a_zero_visc: float = self.get_zero_lift(viscous)

        # Slope of Cl vs Alpha (viscous)
        self.cl_slope_visc: float = self.get_cl_slope(viscous)

    def get_reynolds_subtable(self, reynolds: float | str) -> DataFrame:
        """Get Reynolds Subtable"""
        if isinstance(reynolds, float):
            reynolds = np.format_float_scientific(reynolds, sign=False, precision=3, min_digits=3).replace("+", "")

        if reynolds not in self.reynolds_keys:
            print(self.reynolds_keys)
            print(reynolds)
            # raise ValueError(f"Reynolds Number {reynolds} is not in the list of Reynolds Numbers")

        # Get Reynolds Subtable
        df: DataFrame = self.df[
            [
                "AoA",
                f"CL_{reynolds}",
                f"CD_{reynolds}",
                f"Cm_{reynolds}",
            ]
        ]

        # Rename Columns
        df = df.rename(
            columns={
                f"CL_{reynolds}": "CL",
                f"CD_{reynolds}": "CD",
                f"Cm_{reynolds}": "Cm",
            },
        )

        return df

    @staticmethod
    def get_zero_lift(cl_curve: pd.Series) -> float:
        """Get Zero Lift Angle from Cl Curve"""
        return interpolate_series_index(0.0, cl_curve)

    @staticmethod
    def get_zero_lift_cm(cm_curve: pd.Series, zero_lift_angle: float) -> float:
        """Get Zero Lift Angle from Cl Curve"""
        return interpolate_series_value(zero_lift_angle, cm_curve)

    @staticmethod
    def get_cl_slope(cl_curve: pd.Series) -> float:
        """Get Slope of Cl Curve"""
        cl_linear: pd.Series = get_linear_series(cl_curve)
        # cl_linear.plot()
        return float(cl_linear.diff().mean())

    @staticmethod
    def fill_polar_table(df: DataFrame) -> DataFrame:
        """Fill Nan Values of Panda Dataframe Row by Row
        substituting first backward and then forward

        Args:
            df (pandas.DataFrame): Dataframe with NaN values
        """
        CLs: list[str] = []
        CDs: list[str] = []
        CMs: list[str] = []
        for item in list(df.keys()):
            if item.startswith("CL"):
                CLs.append(item)
            if item.startswith("CD"):
                CDs.append(item)
            if item.startswith("Cm") or item.startswith("CM"):
                CMs.append(item)
        for colums in [CLs, CDs, CMs]:
            df[colums] = df[colums].interpolate(
                method="linear",
                limit_direction="backward",
                axis=1,
            )
            df[colums] = df[colums].interpolate(
                method="linear",
                limit_direction="forward",
                axis=1,
            )
        df.dropna(axis=0, subset=df.columns[1:], how="all", inplace=True)
        return df
