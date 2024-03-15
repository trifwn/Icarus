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
from typing import Any

import matplotlib.pyplot as plt
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
        float: Interpolated Index
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
        name: str,
        data: Struct | dict[str, DataFrame],
    ) -> None:
        self.name = name
        self.data: Struct | dict[str, DataFrame] = data

        self.reynolds_keys: list[str] = list(data.keys())
        self.reynolds_nums: list[float] = sorted([float(reyn) for reyn in self.reynolds_keys])

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
                {
                    "CL": f"CL_{reyn}",
                    "CD": f"CD_{reyn}",
                    "Cm": f"Cm_{reyn}",
                    "CM": f"Cm_{reyn}",
                },
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
        self.a_zero_pot: float = self.get_zero_lift_angle(potential_cl)

        # Potential Cm at Zero Lift Angle
        self.cm_pot: float = self.get_zero_lift_cm(potential_cm, self.a_zero_pot)

        # Viscous Zero Lift Angle
        max_idx: int = self.reynolds_nums.index(max(self.reynolds_nums))
        viscous: pd.Series = df[f"CL_{self.reynolds_keys[max_idx]}"]
        viscous.index = Index(df["AoA"].astype("float32"))
        self.a_zero_visc: float = self.get_zero_lift_angle(viscous)

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

    def get_reynolds_zero_lift_angle(self, reynolds: float | str) -> float:
        """Get Reynolds Zero Lift Angle"""
        df: DataFrame = self.get_reynolds_subtable(reynolds)
        cl_curve: pd.Series = df["CL"]
        cl_curve.index = Index(df["AoA"].astype("float32"))
        return self.get_zero_lift_angle(cl_curve)

    def get_reynolds_cl_slope(self, reynolds: float | str) -> float:
        """Get Reynolds Cl Slope"""
        df: DataFrame = self.get_reynolds_subtable(reynolds)
        cl_curve: pd.Series = df["CL"]
        cl_curve.index = Index(df["AoA"].astype("float32"))
        return self.get_cl_slope(cl_curve)

    def plot_reynolds_cl_curve(
        self,
        reynolds: float | str,
        fig=None,
        ax=None,
    ) -> None: 
        """Plot Reynolds Cl Curve"""
        df: DataFrame = self.get_reynolds_subtable(reynolds)
        cl = df["CL"]
        cl.index = Index(df["AoA"].astype("float32"))
        aoa = df["AoA"]

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        ax.plot(aoa, cl, label=f"Reynolds {reynolds}")
        ax.set_xlabel("Angle of Attack [deg]")
        ax.set_ylabel("Lift Coefficient")
        ax.set_title(f"Reynolds {reynolds} Cl Curve")
        ax.legend()
        ax.grid(True)
        ax.axvline(self.get_zero_lift_angle(cl), color="r", linestyle="--", label="Zero Lift Angle")
        ax.axhline(0.0, color="k", linestyle="--")
    
    def plot_reynolds_cd_curve(
        self,
        reynolds: float | str,
        fig=None,
        ax=None,
    ) -> None:
        """Plot Reynolds Cd Curve"""
        df: DataFrame = self.get_reynolds_subtable(reynolds)
        cd = df["CD"]
        cl = df["CL"]
        cl.index = Index(df["AoA"].astype("float32"))
        aoa = df["AoA"]

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        ax.plot(aoa, cd, label=f"Reynolds {reynolds}")
        ax.set_xlabel("Angle of Attack [deg]")
        ax.set_ylabel("Drag Coefficient")
        ax.set_title(f"Reynolds {reynolds} Cd Curve")
        ax.legend()
        ax.grid(True)
        ax.axvline(self.get_zero_lift_angle(cl), color="r", linestyle="--", label="Zero Lift Angle")
        ax.axhline(0.0, color="k", linestyle="--")
    
    def plot_reynolds_cl_over_cd_curve(
        self,
        reynolds: float | str,
        fig=None,
        ax=None,
    ) -> None:
        """Plot Reynolds Cl Over Cd Curve"""
        df: DataFrame = self.get_reynolds_subtable(reynolds)
        cl = df["CL"]
        cd = df["CD"]
        cl.index = Index(df["AoA"].astype("float32"))
        cd.index = Index(df["AoA"].astype("float32"))
        aoa = df["AoA"]

        if fig is None or ax is None:
            fig, ax = plt.subplots()
        ax.plot(aoa, cl / cd, label=f"Reynolds {reynolds}")
        ax.set_xlabel("Angle of Attack [deg]")
        ax.set_ylabel("Lift to Drag Ratio")
        ax.set_title(f"Reynolds {reynolds} Cl/Cd Curve")
        ax.legend()
        ax.grid(True)
        ax.axvline(self.get_zero_lift_angle(cl), color="r", linestyle="--", label="Zero Lift Angle")
        ax.axhline(0.0, color="k", linestyle="--")

    @staticmethod
    def get_zero_lift_angle(cl_curve: pd.Series) -> float:
        """Get Zero Lift Angle from Cl Curve"""
        return interpolate_series_index(0.0, cl_curve)

    @staticmethod
    def get_zero_lift_cm(cm_curve: pd.Series, zero_lift_angle: float) -> float:
        """Get Zero Lift Cm from Cl Curve"""
        return interpolate_series_index(zero_lift_angle, cm_curve)

    @staticmethod
    def get_cl_slope(cl_curve: pd.Series) -> float:
        """Get Slope of Cl Curve"""
        cl_linear: pd.Series = get_linear_series(cl_curve)
        # cl_linear.plot()
        return float(cl_linear.diff().mean())

    @staticmethod
    def get_positive_stall_idx(cl_curve: pd.Series) -> int:
        """Get Positive Stall Angle"""
        return int(cl_curve.idxmax())

    @staticmethod
    def get_negative_stall_idx(cl_curve: pd.Series) -> int:
        """Get Negative Stall Angle"""
        return int(cl_curve.idxmin())

    @staticmethod
    def get_cl_cd_minimum_idx(cl_curve: pd.Series, cd_curve: pd.Series) -> int:
        """Get Minimum CD/CL Ratio"""
        cd_cl: pd.Series = cd_curve / cl_curve
        slope_cd_cl = cd_cl.diff()

        # Find the minimum slope
        return int(slope_cd_cl.idxmin())

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

    def get_cl_cd_parabolic(self, reynolds: float) -> tuple[FloatArray, FloatArray]:
        """
        A simple profile-drag CD(CL) function for this section.
        The function is parabolic between CL1..CL2 and CL2..CL3,
        with rapid increases in CD below CL1 and above CL3.

        The CD-CL polar is based on a simple interpolation with four CL regions:
        1) negative stall region
        2) parabolic CD(CL) region between negative stall and the drag minimum
        3) parabolic CD(CL) region between the drag minimum and positive stall
        4) positive stall region

                CLpos,CDpos       <-  Region 4 (quadratic above CLpos)
        CL |   pt3--------
        |    /
        |   |                   <-  Region 3 (quadratic above CLcdmin)
        | pt2 CLcdmin,CDmin
        |   |
        |    /                  <-  Region 2 (quadratic below CLcdmin)
        |   pt1_________
        |     CLneg,CDneg       <-  Region 1 (quadratic below CLneg)
        |
        -------------------------
                        CD

        The CD(CL) function is interpolated for stations in between
        defining sections.  Hence, the CDCL declaration on any surface
        must be used either for all sections or for none (unless the SURFACE
        CDCL is specified).
        """

        # Interpolate Reynolds From Values Stored in the Class
        if reynolds not in self.reynolds_nums:
            reynolds_max = min(self.reynolds_nums)
            reynolds_min = max(self.reynolds_nums)
            for reyn in self.reynolds_nums:
                if reyn > reynolds:
                    reynolds_max = reyn
                    break

            for reyn in self.reynolds_nums[::-1]:
                if reyn < reynolds:
                    reynolds_min = reyn
                    break

            # Get CL and CD for the two Reynolds Numbers
            curve_1 = self.get_reynolds_subtable(reynolds_min)
            curve_2 = self.get_reynolds_subtable(reynolds_max)

            # Interpolate curve based on relative distance between Reynolds Numbers
            # (Linear Interpolation)
            curve = curve_1 + (curve_2 - curve_1) * (reynolds - reynolds_min) / (reynolds_max - reynolds_min)

        else:
            curve = self.get_reynolds_subtable(reynolds)

        pos_stall_idx = self.get_positive_stall_idx(curve["CL"])
        neg_stall_idx = self.get_negative_stall_idx(curve["CL"])
        min_cdcl_idx = self.get_cl_cd_minimum_idx(curve["CL"], curve["CD"])

        # pos_stall_angle = curve["AoA"].loc[pos_stall_idx]
        # neg_stall_angle = curve["AoA"].loc[neg_stall_idx]
        # min_cdcl_angle = curve["AoA"].loc[min_cdcl_idx]

        # From Indexes get CL and CD
        cl1 = curve["CL"].loc[neg_stall_idx]
        cd1 = curve["CD"].loc[neg_stall_idx]
        cl2 = curve["CL"].loc[min_cdcl_idx]
        cd2 = curve["CD"].loc[min_cdcl_idx]
        cl3 = curve["CL"].loc[pos_stall_idx]
        cd3 = curve["CD"].loc[pos_stall_idx]

        cl = np.array([cl1, cl2, cl3])
        cd = np.array([cd1, cd2, cd3])

        return cl, cd
