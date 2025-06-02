from __future__ import annotations

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame
from pandas import Index

# from ICARUS.vehicle import Airplane
from ICARUS.core.base_types import Struct
from ICARUS.core.types import FloatArray

from .utils import (
    interpolate_series_index,
    interpolate_series_value,
    get_linear_series,
)

class PolarNotAccurate(Exception):
    """Exception Raised when the Polar is not accurate"""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class AirplaneData:
    """Solver Data Class"""

    def __init__(
        self,
        name: str,
        data: Struct | dict[str, dict[str, DataFrame]],
    ) -> None:
        self.name = name

        self.polars: dict[str, AirplanePolars] = {}
        for solver, subdata in data.items():
            self.add_solver(solver, subdata)

    @property
    def solvers(self) -> list[str]:
        return list(self.polars.keys())

    def get_solver_reynolds(self, solver: str) -> list[float]:
        return self.polars[solver].reynolds_nums

    def get_polars(self, solver: str | None = None) -> AirplanePolars:
        if solver is None:
            return self.polars[list(self.polars.keys())[0]]
        return self.polars[solver]

    def get_polar_data(self, solver: str) -> DataFrame:
        return self.polars[solver].df

    def add_solver(self, solver: str, data: dict[str, DataFrame]) -> None:
        self.polars[solver] = AirplanePolars(self.name, data)

    def add_data(self, data: dict[str, dict[str, DataFrame]]) -> None:
        for solver, subdata in data.items():
            self.add_solver(solver, subdata)


class AirplanePolars:
    """Airfoil Polars Class"""

    def __init__(
        self,
        name: str,
        data: Struct | dict[str, DataFrame],
    ) -> None:
        self.name = name
        self.data: Struct | dict[str, DataFrame] = data

        self.reynolds_keys: list[str] = list(data.keys())
        self.reynolds_nums: list[float] = sorted(
            [float(reyn) for reyn in self.reynolds_keys],
        )

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
            potential_cl: pd.Series[float] = df["CL_Potential"]
            potential_cm: pd.Series[float] = df["Cm_Potential"]
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
        viscous: pd.Series[float] = df[f"CL_{self.reynolds_keys[max_idx]}"]
        viscous.index = Index(df["AoA"].astype("float32"))
        self.a_zero_visc: float = self.get_zero_lift_angle(viscous)

        # Slope of Cl vs Alpha (viscous)
        self.cl_slope_visc: float = self.get_cl_slope(viscous)

    def get_reynolds_subtable(self, reynolds: float | str) -> DataFrame:
        """Get Reynolds Subtable"""
        if isinstance(reynolds, float):
            reynolds = np.format_float_scientific(
                reynolds,
                sign=False,
                precision=3,
                min_digits=3,
            ).replace("+", "")

        if reynolds not in self.reynolds_keys:
            print(self.reynolds_keys)
            print(reynolds)
            # raise ValueError(f"Reynolds Number {float(reynolds):,} is not in the list of Reynolds Numbers")

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
        cl_curve: pd.Series[float] = df["CL"]
        cl_curve.index = Index(df["AoA"].astype("float32"))
        return self.get_zero_lift_angle(cl_curve)

    def get_reynolds_cl_slope(self, reynolds: float | str) -> float:
        """Get Reynolds Cl Slope"""
        df: DataFrame = self.get_reynolds_subtable(reynolds)
        cl_curve: pd.Series[float] = df["CL"]
        cl_curve.index = Index(df["AoA"].astype("float32"))
        cl_vector = cl_curve.to_numpy()
        aoa_vector = np.deg2rad(df["AoA"].to_numpy())
        zero_lift_angle = self.get_zero_lift_angle(cl_curve)
        max_angle = zero_lift_angle + 3
        min_angle = zero_lift_angle - 3
        min_index = int(np.argmin(np.abs(aoa_vector - np.deg2rad(min_angle))))
        max_index = int(np.argmin(np.abs(aoa_vector - np.deg2rad(max_angle))))

        cl_slope = float(
            np.poly1d(
                np.polyfit(
                    aoa_vector[min_index:max_index],
                    cl_vector[min_index:max_index],
                    1,
                ),
            )[1],
        )
        return cl_slope
        # return self.get_cl_slope(cl_curve)

    def get_aero_coefficients(
        self,
        reynolds: float | str,
        aoa: float,
    ) -> tuple[float, float, float]:
        """Get Aero Coefficients"""
        df: DataFrame = self.get_reynolds_subtable(reynolds)
        cl_curve: pd.Series[float] = df["CL"]
        cd_curve: pd.Series[float] = df["CD"]
        cm_curve: pd.Series[float] = df["Cm"]
        cl_curve.index = Index(df["AoA"].astype("float32"))
        cd_curve.index = Index(df["AoA"].astype("float32"))
        cm_curve.index = Index(df["AoA"].astype("float32"))

        cl = interpolate_series_value(aoa, cl_curve)
        cd = interpolate_series_value(aoa, cd_curve)
        cm = interpolate_series_value(aoa, cm_curve)
        return cl, cd, cm

    def reynolds_examine_run(self, reynolds: float | str) -> tuple[bool, str]:
        df: DataFrame = self.get_reynolds_subtable(reynolds)
        cl_curve: pd.Series[float] = df["CL"]
        cd_curve: pd.Series[float] = df["CD"]
        cl_curve.index = Index(df["AoA"].astype("float32"))
        zero_lift = self.get_zero_lift_angle(cl_curve)
        return self.examine_run(cl_curve, cd_curve, zero_lift)

    def plot_reynolds_cl_curve(
        self,
        reynolds: float | str,
        ax: Axes | None = None,
        add_zero_lift: bool = True,
    ) -> None:
        """Plot Reynolds Cl Curve"""
        df: DataFrame = self.get_reynolds_subtable(reynolds)
        cl = df["CL"]
        cl.index = Index(df["AoA"].astype("float32"))
        aoa = df["AoA"]

        if not isinstance(ax, Axes):
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(aoa, cl, label=f"Reynolds {float(reynolds):,}")
        ax.set_xlabel("Angle of Attack [deg]")
        ax.set_ylabel("Lift Coefficient")
        ax.set_title(f"Reynolds {float(reynolds):,} Cl Curve")
        ax.legend()
        ax.grid(True)
        ax.axhline(0.0, color="k", linestyle="--")
        if add_zero_lift:
            ax.axvline(self.get_zero_lift_angle(cl), color="r", linestyle="--")

    def plot_reynolds_cd_curve(
        self,
        reynolds: float | str,
        ax: Axes | None = None,
    ) -> None:
        """Plot Reynolds Cd Curve"""
        df: DataFrame = self.get_reynolds_subtable(reynolds)
        cd = df["CD"]
        cl = df["CL"]
        cl.index = Index(df["AoA"].astype("float32"))
        aoa = df["AoA"]

        if not isinstance(ax, Axes):
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(aoa, cd, label=f"Reynolds {float(reynolds):,}")
        ax.set_xlabel("Angle of Attack [deg]")
        ax.set_ylabel("Drag Coefficient")
        ax.set_title(f"Reynolds {float(reynolds):,} Cd Curve")
        ax.legend()
        ax.grid(True)
        ax.axvline(self.get_zero_lift_angle(cl), color="r", linestyle="--")
        ax.axhline(0.0, color="k", linestyle="--")

    def plot_reynolds_cl_over_cd_curve(
        self,
        reynolds: float | str,
        ax: Axes | None = None,
    ) -> None:
        """Plot Reynolds Cl Over Cd Curve"""
        df: DataFrame = self.get_reynolds_subtable(reynolds)

        cl = df["CL"]
        cd = df["CD"]
        cl.index = Index(df["AoA"].astype("float32"))
        cd.index = Index(df["AoA"].astype("float32"))
        aoa = df["AoA"]
        cl_over_cd = df["CL"] / df["CD"]
        cl_over_cd.index = Index(df["AoA"].astype("float32"))

        # idx = df[df["CL/CD"] > 200].index
        # idx2 = df[df["CL/CD"] < -200].index

        # df.loc[idx, "CL/CD"] = 0
        # df.loc[idx2, "CL/CD"] = 0

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(aoa, cl_over_cd, label=f"Reynolds {float(reynolds):,}")
        ax.set_xlabel("Angle of Attack [deg]")
        ax.set_ylabel("Lift to Drag Ratio")
        ax.set_title(f"Reynolds {float(reynolds):,} Cl/Cd Curve")
        ax.legend()
        ax.grid(True)
        ax.axvline(self.get_zero_lift_angle(cl), color="r", linestyle="--")
        ax.axhline(0.0, color="k", linestyle="--")

    def plot(self) -> Figure:
        # Create 2 subplots and unpack the output array immediately
        fig, axs = plt.subplots(2, 2, figsize=(10.2, 10.0))

        fig.suptitle(f"{self.name} Polars")
        for reyn in self.reynolds_nums:
            self.plot_reynolds_cl_curve(reyn, axs[0, 0])
            self.plot_reynolds_cd_curve(reyn, axs[0, 1])
            self.plot_reynolds_cl_over_cd_curve(reyn, axs[1, 0])

        from ICARUS.database import Database

        DB = Database.get_instance()

        # airplane: Airplane = DB.get_vehicle(self.name.upper())
        # airplane.plot(ax=axs[1, 1], camber=True, max_thickness=True, scatter=False)

        # Clear all axes legends
        for ax in axs.flatten():
            ax.legend().remove()

        ax = axs[0, 0]
        ax.legend(title="Reynolds Numbers", loc="upper left")

        # Add text for the slopes
        # reyn_min = self.reynolds_nums[0]
        # reyn_max = self.reynolds_nums[-1]
        # axs[0, 0].text(
        #     0.7,
        #     0.2,
        #     f"Cl slope at {float(reyn_min):.1e}: {self.get_reynolds_cl_slope(reyn_min)* 180/np.pi:.3f} ",
        #     fontsize=12,
        #     transform=axs[0, 0].transAxes,
        # )
        # axs[0, 0].text(
        #     0.7,
        #     0.1,
        #     f"Cl slope at {float(reyn_max):.1e}: {self.get_reynolds_cl_slope(reyn_max)* 180/np.pi:.3f} ",
        #     fontsize=12,
        #     transform=axs[0, 0].transAxes,
        # )
        fig.tight_layout()
        return fig

    def save_polar_plot_img(self, folder: str, desc: str | None = None) -> None:
        fig = self.plot()
        desc = f"_{desc}" if desc else ""
        filename = os.path.join(folder, f"{self.name}_polars{desc}.png")
        fig.savefig(filename)
        plt.close(fig)
        print(f"Saved {filename}")

    @staticmethod
    def get_zero_lift_angle(cl_curve: pd.Series[float]) -> float:
        """Get Zero Lift Angle from Cl Curve"""
        return interpolate_series_index(0.0, cl_curve)

    @staticmethod
    def get_zero_lift_cm(cm_curve: pd.Series[float], zero_lift_angle: float) -> float:
        """Get Zero Lift Cm from Cl Curve"""
        return interpolate_series_index(zero_lift_angle, cm_curve)

    @staticmethod
    def get_cl_slope(cl_curve: pd.Series[float]) -> float:
        """Get Slope of Cl Curve"""
        cl_linear: pd.Series[float] = get_linear_series(cl_curve)
        # cl_linear.plot()

        return float(cl_linear.diff().mean())

    @staticmethod
    def get_positive_stall_idx(cl_curve: pd.Series[float]) -> int:
        """Get Positive Stall Angle"""
        # Remove NaN Values
        cl_curve = cl_curve.dropna()
        return int(cl_curve.idxmax())

    @staticmethod
    def get_negative_stall_idx(cl_curve: pd.Series[float]) -> int:
        """Get Negative Stall Angle"""
        # Remove NaN Values
        cl_curve = cl_curve.dropna()
        return int(cl_curve.idxmin())

    @staticmethod
    def get_cl_cd_minimum_idx(
        cl_curve: pd.Series[float],
        cd_curve: pd.Series[float],
    ) -> int:
        """Get Minimum CD/CL Ratio"""
        cd_cl: pd.Series[float] = cd_curve / cl_curve
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

    @staticmethod
    def examine_run(
        cl_curve: pd.Series[float],
        cd_curve: pd.Series[float],
        zero_lift_angle: float,
    ) -> tuple[bool, str]:
        is_bad: bool = False
        cl_array = cl_curve.to_numpy()
        cd_array = cd_curve.to_numpy()
        aoa_array = np.deg2rad(cl_curve.index.to_numpy())
        cl_slopes = np.gradient(cl_array, aoa_array)
        cd_slopes = np.gradient(cd_array, aoa_array)
        max_angle = zero_lift_angle + 4
        min_angle = zero_lift_angle
        min_index = np.argmin(np.abs(aoa_array - np.deg2rad(min_angle)))
        max_index = np.argmin(np.abs(aoa_array - np.deg2rad(max_angle)))
        max_positive_CL_slope = 3 * (2 * np.pi)
        min_negative_CL_slope = -0.2 * (2 * np.pi)
        max_positive_CD_slope = 0.9
        min_negative_CD_slope = -0.9

        CL_cond = np.any(
            np.greater(cl_slopes[min_index:max_index], max_positive_CL_slope),
        ) or np.any(
            np.less(cl_slopes[min_index:max_index], min_negative_CL_slope),
        )
        CD_cond = np.any(
            np.greater(cd_slopes[min_index:max_index], max_positive_CD_slope),
        ) or np.any(
            np.less(cd_slopes[min_index:max_index], min_negative_CD_slope),
        )
        is_bad = False
        problem: str = "None"
        if CL_cond and not CD_cond:
            is_bad = True
            problem = "CL problem"
        elif CD_cond and not CL_cond:
            is_bad = True
            problem = "CD problem"
        elif CD_cond and CL_cond:
            problem = "both"
            is_bad = True
        return is_bad, problem

    def get_cl_cd_parabolic(self, reynolds: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        """A simple profile-drag CD(CL) function for this section.
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
            reynolds_max = max(self.reynolds_nums)
            reynolds_min = min(self.reynolds_nums)
            if reynolds_min == reynolds_max:
                reynolds = reynolds_min
                curve = self.get_reynolds_subtable(reynolds)
            else:
                for reyn in self.reynolds_nums:
                    if reyn > reynolds:
                        reynolds_max = reyn
                        break

                for reyn in self.reynolds_nums[::-1]:
                    if reyn < reynolds:
                        reynolds_min = reyn
                        break

                diff_reynolds = reynolds_max - reyn
                diff_reynolds_max = reyn - reynolds_min
                if diff_reynolds < diff_reynolds_max:
                    curve = self.get_reynolds_subtable(reynolds_max)
                else:
                    curve = self.get_reynolds_subtable(reynolds_min)

                # Get CL and CD for the two Reynolds Numbers
                # curve_1 = self.get_reynolds_subtable(reynolds_min)
                # curve_2 = self.get_reynolds_subtable(reynolds_max)
                # Interpolate curve based on relative distance between Reynolds Numbers
                # (Linear Interpolation)

                # curve = curve_1 + (curve_2 - curve_1) * (reynolds - reynolds_min) / (
                #     reynolds_max - reynolds_min
                # )

        else:
            curve = self.get_reynolds_subtable(reynolds)
        try:
            pos_stall_idx = self.get_positive_stall_idx(curve["CL"] / curve["CD"])
            neg_stall_idx = self.get_negative_stall_idx(curve["CL"] / curve["CD"])
            min_cdcl_idx = self.get_cl_cd_minimum_idx(curve["CL"], curve["CD"])
        except ValueError:
            print(curve["CL"])
            print(self.name)
            raise ValueError("Error in getting the indexes")

        # pos_stall_angle = curve["AoA"].loc[pos_stall_idx]
        # neg_stall_angle = curve["AoA"].loc[neg_stall_idx]
        # min_cdcl_angle = curve["AoA"].loc[min_cdcl_idx]

        # From Indexes get CL and CD
        aoa1 = curve["AoA"].loc[neg_stall_idx]
        cl1 = curve["CL"].loc[neg_stall_idx]
        cd1 = curve["CD"].loc[neg_stall_idx]

        aoa2 = curve["AoA"].loc[min_cdcl_idx]
        cl2 = curve["CL"].loc[min_cdcl_idx]
        cd2 = curve["CD"].loc[min_cdcl_idx]

        aoa3 = curve["AoA"].loc[pos_stall_idx]
        cl3 = curve["CL"].loc[pos_stall_idx]
        cd3 = curve["CD"].loc[pos_stall_idx]

        cl = np.array([cl1, cl2, cl3])
        cd = np.array([cd1, cd2, cd3])
        aoas = np.array([aoa1, aoa2, aoa3])
        return cl, cd, aoas

    def plot_avl_drag(self, reynolds: float) -> None:
        """Plot AVL Drag Polar"""
        curve: DataFrame = self.get_reynolds_subtable(reynolds)
        cl_points, cd_points, aoa_points = self.get_cl_cd_parabolic(reynolds)

        def cdcl(CL: float) -> tuple[float, float]:
            """Compute the CD and CD_CL for a given CL using a parabolic model."""
            # Constants
            CLINC = 0.2
            CDINC = 0.0500
            CLMIN, CL0, CLMAX = cl_points
            CDMIN, CD0, CDMAX = cd_points
            # Some matching parameters to make slopes smooth at stall joins
            CDX1 = 2.0 * (CDMIN - CD0) * (CLMIN - CL0) / (CLMIN - CL0) ** 2
            CDX2 = 2.0 * (CDMAX - CD0) * (CLMAX - CL0) / (CLMAX - CL0) ** 2
            CLFAC = 1.0 / CLINC
            # Four formulas are used for CD, depending on the CL
            if CL < CLMIN:
                # Negative stall drag model (slope matches lower side, quadratic drag rise)
                CD = CDMIN + CDINC * (CLFAC * (CL - CLMIN)) ** 2 + CDX1 * (1.0 - (CL - CL0) / (CLMIN - CL0))
                CD_CL = CDINC * CLFAC * 2.0 * (CL - CLMIN)
            elif CL < CL0:
                # Quadratic matching negative stall and minimum drag point with zero slope
                CD = CD0 + (CDMIN - CD0) * (CL - CL0) ** 2 / (CLMIN - CL0) ** 2
                CD_CL = (CDMIN - CD0) * 2.0 * (CL - CL0) / (CLMIN - CL0) ** 2
            elif CL < CLMAX:
                # Quadratic matching positive stall and minimum drag point with zero slope
                CD = CD0 + (CDMAX - CD0) * (CL - CL0) ** 2 / (CLMAX - CL0) ** 2
                CD_CL = (CDMAX - CD0) * 2.0 * (CL - CL0) / (CLMAX - CL0) ** 2
            else:
                # Positive stall drag model (slope matches upper side, quadratic drag rise)
                CD = CDMAX + CDINC * (CLFAC * (CL - CLMAX)) ** 2 - CDX2 * (1.0 - (CL - CL0) / (CLMAX - CL0))
                CD_CL = CDINC * CLFAC * 2.0 * (CL - CLMAX)
            return CD, CD_CL

        cl_range = np.linspace(curve["CL"].min(), curve["CL"].max(), 100)
        aoa_range = np.interp(cl_range, curve["CL"], curve["AoA"])
        cd_avl = [cdcl(c)[0] for c in cl_range]

        # Create the plot
        fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        axs = axs.flatten()

        fig.suptitle(f"Polars vs AVL interpolation \n{self.name} CD-CL Polar at Re = {reynolds:.2e}.")
        # Plot the CD-CL curve generated by the Fortran-translated cdcl function
        axs[0].plot(curve["CD"], curve["CL"], label="Xfoil Polar")
        axs[0].scatter(cd_points, cl_points, color="red", label="Selected Points")
        axs[0].plot(cd_avl, cl_range, label="Drela interpolation", linestyle=":")
        axs[0].set_xlabel("CD")
        axs[0].set_ylabel("CL")
        axs[0].set_title("CD vs CL")
        # Plot the CL/CD vs AoA curve
        axs[1].plot(curve["AoA"], curve["CL"] / curve["CD"])
        axs[1].scatter(aoa_points, cl_points / cd_points, color="red")
        axs[1].plot(aoa_range, cl_range / cd_avl, linestyle=":")
        axs[1].set_xlabel("AoA")
        axs[1].set_ylabel("CL/CD")
        axs[1].set_title("CL/CD vs AoA")
        # Plot the CL vs AoA curve
        axs[2].plot(curve["AoA"], curve["CL"])
        axs[2].scatter(aoa_points, cl_points, color="red")
        axs[2].plot(aoa_range, cl_range, linestyle=":")
        axs[2].set_xlabel("AoA")
        axs[2].set_ylabel("CL")
        axs[2].set_title("CL vs AoA")
        # Plot the CD vs AoA curve
        axs[3].plot(curve["AoA"], curve["CD"])
        axs[3].scatter(aoa_points, cd_points, color="red")
        axs[3].plot(aoa_range, cd_avl, linestyle=":")
        axs[3].set_xlabel("AoA")
        axs[3].set_ylabel("CD")
        axs[3].set_title("CD vs AoA")
        # Add the legend to the plot
        fig.legend(loc="lower right")
        for ax in axs:
            ax.grid()
        fig.tight_layout()
