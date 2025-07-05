from __future__ import annotations

import os

import distinctipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame

from ICARUS.airfoils.airfoil import Airfoil
from ICARUS.airfoils.metrics.polars import AirfoilPolar
from ICARUS.core.types import FloatArray


class AirfoilPolarMap:
    """Airfoil Polars Class"""

    def __init__(
        self,
        airfoil_name: str,
        solver: str,
        data: dict[str, DataFrame] = {},
    ) -> None:
        self.airfoil_name = airfoil_name
        self.solver_name = solver

        reynolds_keys: list[str] = list(data.keys())
        self.reynolds_numbers: list[float] = sorted(
            [float(reyn) for reyn in reynolds_keys],
        )

        # Create an Empty DataFrame with AoA on the index
        df = DataFrame(columns=["AoA"])
        self.polars: dict[float, AirfoilPolar] = {}

        for reyn in reynolds_keys:
            df2: DataFrame = data[reyn].astype("float32").dropna(axis=0, how="all")

            self.polars[float(reyn)] = AirfoilPolar(
                reynolds=float(reyn),
                df=df2,
            )

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
        df = self.fill_polar_table(df)
        self.df: DataFrame = df

    @property
    def reynolds_strings(self) -> list[str]:
        """Get the Reynolds keys of the airfoil polar map."""
        return [polar.reynolds_string for polar in self.polars.values()]

    def is_empty(self) -> bool:
        """Check if the polar map is empty."""
        return len(self.polars) == 0

    @property
    def angles_of_attack(self) -> FloatArray:
        """Get the angles of attack for the airfoil polar map."""
        return self.df["AoA"].values.astype("float64")

    def get_polar(self, reynolds: float) -> AirfoilPolar:
        """Get the airfoil polar for a given Reynolds number."""
        if reynolds in self.polars:
            return self.polars[reynolds]
        else:
            # Simple interpolation/extrapolation
            reynolds_max = max(self.reynolds_numbers)
            reynolds_min = min(self.reynolds_numbers)
            if reynolds > reynolds_max:
                return self.polars[reynolds_max]
            elif reynolds < reynolds_min:
                return self.polars[reynolds_min]
            else:
                # Do linear interpolation between the two closest polars
                # Find the closest Reynolds numbers
                mask = np.array(self.reynolds_numbers) >= reynolds
                prev_idx = int(np.sum(~mask)) - 1
                next_idx = int(np.sum(mask))

                prev_polar = self.polars[self.reynolds_numbers[prev_idx]]
                next_polar = self.polars[self.reynolds_numbers[next_idx]]

                polar = AirfoilPolar.interpolate_polar(
                    polar1=prev_polar,
                    polar2=next_polar,
                    reynolds=reynolds,
                )
                return polar

    def add_polar(self, polar: AirfoilPolar) -> None:
        """Add a polar to the airfoil polar map."""
        reynolds = polar.reynolds
        if reynolds not in self.polars:
            self.polars[reynolds] = polar
            self.reynolds_numbers.append(reynolds)
            self.reynolds_numbers.sort()
            # Update the DataFrame with the new polar data
            self.df = pd.merge(
                self.df,
                polar.df.rename(
                    columns={
                        "CL": f"CL_{polar.reynolds_string}",
                        "CD": f"CD_{polar.reynolds_string}",
                        "Cm": f"Cm_{polar.reynolds_string}",
                    },
                ),
                on="AoA",
                how="outer",
            )
            self.df = self.fill_polar_table(self.df)

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
        df.dropna(axis=0, how="all", subset=df.columns[1:], inplace=True)
        return df

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
        if reynolds not in self.reynolds_numbers:
            reynolds_max = max(self.reynolds_numbers)
            reynolds_min = min(self.reynolds_numbers)
            if reynolds_min == reynolds_max:
                reynolds = reynolds_min
                curve = self.get_polar(reynolds)
            else:
                for reyn in self.reynolds_numbers:
                    if reyn > reynolds:
                        reynolds_max = reyn
                        break

                for reyn in self.reynolds_numbers[::-1]:
                    if reyn < reynolds:
                        reynolds_min = reyn
                        break

                diff_reynolds = reynolds_max - reyn
                diff_reynolds_max = reyn - reynolds_min
                if diff_reynolds < diff_reynolds_max:
                    curve = self.get_polar(reynolds_max)
                else:
                    curve = self.get_polar(reynolds_min)

                # Get CL and CD for the two Reynolds Numbers
                # curve_1 = self.get_reynolds_subtable(reynolds_min)
                # curve_2 = self.get_reynolds_subtable(reynolds_max)
                # Interpolate curve based on relative distance between Reynolds Numbers
                # (Linear Interpolation)

                # curve = curve_1 + (curve_2 - curve_1) * (reynolds - reynolds_min) / (
                #     reynolds_max - reynolds_min
                # )

        else:
            curve = self.get_polar(reynolds)

        try:
            # pos_stall_idx = self.get_positive_stall_idx(curve["CL"] / curve["CD"])
            # neg_stall_idx = self.get_negative_stall_idx(curve["CL"] / curve["CD"])
            # min_cdcl_idx = self.get_cl_cd_minimum_idx(curve["CL"], curve["CD"])
            aoa1, cl1, cd1 = curve.get_positive_stall()
            aoa3, cl3, cd3 = curve.get_negative_stall()
            aoa2, cl2, cd2 = curve.get_minimum_cl_cd()
        except ValueError:
            raise ValueError("Error in getting the indexes")

        cl = np.array([cl1, cl2, cl3])
        cd = np.array([cd1, cd2, cd3])
        aoas = np.array([aoa1, aoa2, aoa3])
        return cl, cd, aoas

    def plot_parabolic_drag_profiles(self, reynolds: float) -> None:
        """Plot AVL Drag Polar"""
        curve: DataFrame = self.get_polar(reynolds).df
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

        fig.suptitle(f"Polars vs AVL interpolation \n{self.airfoil_name} CD-CL Polar at Re = {reynolds:.2e}.")
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

    def plot(self) -> Figure:
        # Create 2 subplots and unpack the output array immediately
        fig, axs = plt.subplots(2, 2, figsize=(10.2, 10.0))

        fig.suptitle(f"{self.airfoil_name} Polars ({self.solver_name})")
        colors = distinctipy.get_colors(len(self.reynolds_numbers))
        for i, reyn in enumerate(self.reynolds_numbers):
            polar = self.get_polar(reyn)
            polar.plot_cl(ax=axs[0, 0], color=colors[i])
            polar.plot_cd(ax=axs[0, 1], color=colors[i])
            polar.plot_cl_over_cd(ax=axs[1, 0], color=colors[i])

        from ICARUS.database import Database

        DB = Database.get_instance()

        airfoil: Airfoil = DB.get_airfoil(self.airfoil_name.upper())
        airfoil.plot(ax=axs[1, 1], camber=True, max_thickness=True, scatter=False)

        # Clear all axes legends
        for ax in axs.flatten():
            if ax.get_legend() is not None:
                ax.legend().remove()

        ax = axs[0, 0]
        ax.legend(title="Reynolds Numbers", loc="upper left")
        fig.tight_layout()
        return fig

    def save_polar_plot_img(self, folder: str, filename: str | None = None) -> None:
        fig = self.plot()

        if filename is None:
            filename = os.path.join(folder, f"{self.airfoil_name}_{self.solver_name}_polars.png")
        else:
            filename = os.path.join(folder, filename)
        fig.savefig(filename)
        plt.close(fig)
        print(f"Saved {filename}")
