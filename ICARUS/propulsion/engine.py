import os
from typing import TypeVar

import jax
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.interpolate import griddata
from scipy.interpolate import RBFInterpolator

from ICARUS.core.types import FloatArray


class Engine:
    def __init__(self) -> None:
        pass

    def load_data_from_df(self, engine_dir: str) -> None:
        prop_dfs: dict[str, DataFrame] = {}
        for prop_data in os.listdir(engine_dir):
            if prop_data.endswith(".csv"):
                prop_name = prop_data[:-4]
                prop_dfs[prop_name] = pd.read_csv(engine_dir + prop_data)

        df: DataFrame = prop_dfs["13x8E"].sort_values(by=["Airspeed [m/s]"])
        df["Thrust [N]"] = df["Thrust [g]"] * 9.81 / 1000

        self.motor: DataFrame = df

        # Extract data from DataFrame
        x = df["Airspeed [m/s]"].values
        y = df["Current [A]"].values
        z = df["Thrust [N]"].values

        # Create function to evaluate thrust from airspeed and current
        thrust_interpolator = RBFInterpolator(np.c_[x, y], z, kernel="linear")

        def thrust(airspeed, current):
            # return griddata((x, y), z, (airspeed, current), method="linear")
            try:
                shape = airspeed.shape
            except AttributeError:
                shape = 1
            airspeed = np.ravel(airspeed)
            current = np.ravel(current)

            xy = np.c_[airspeed, current]
            return thrust_interpolator(xy).reshape(shape)

        current_interpolator = RBFInterpolator(np.c_[x, z], y, kernel="linear")

        def current(airspeed, thrust):
            # return griddata((x, z), y, (airspeed, thrust), method="linear")
            try:
                shape = airspeed.shape
            except AttributeError:
                shape = 1
            airspeed = np.ravel(airspeed)
            thrust = np.ravel(thrust)

            xy = np.c_[airspeed, thrust]
            return current_interpolator(xy).reshape(shape)

        self.thrust_model = thrust
        self.current_model = current

    def plot_engine_map_throttle(self) -> None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(211)
        cm = plt.get_cmap("gist_rainbow")
        for count, throtle_lvl in enumerate(np.sort(self.motor["Throttle [%]"].unique())):
            throtle_lvl_data = self.motor[self.motor["Throttle [%]"] == throtle_lvl]
            ax.plot(
                throtle_lvl_data["Airspeed [m/s]"],
                throtle_lvl_data["Thrust [N]"],
                label=f"{throtle_lvl} %",
                color=cm(count * 20),
            )

        ax.set_xlabel("Airspeed [m/s]")
        ax.set_ylabel("Thrust [N]")
        ax.legend()
        ax.grid()
        fig.show()

        # Add current plot
        ax = fig.add_subplot(212)
        for count, throtle_lvl in enumerate(np.sort(self.motor["Throttle [%]"].unique())):
            throtle_lvl_data = self.motor[self.motor["Throttle [%]"] == throtle_lvl]
            ax.plot(
                throtle_lvl_data["Airspeed [m/s]"],
                throtle_lvl_data["Current [A]"],
                label=f"{throtle_lvl} %",
                color=cm(count * 20),
            )
        ax.set_xlabel("Airspeed [m/s]")
        ax.set_ylabel("Current [A]")
        ax.legend()

    def plot_engine_map(self) -> None:
        # Plot the 3D engine map of current, thrust and velocity
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(121)
        x = self.motor["Airspeed [m/s]"]
        y = self.motor["Current [A]"]
        z = self.motor["Thrust [N]"]

        # Plot original data points
        c = ax.scatter(x, y, c=z, cmap="viridis")
        fig.colorbar(c, label="Thrust [N]")
        # Plot interpolated surface
        x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
        z_interp = self.thrust_model(x_grid, y_grid).reshape(x_grid.shape)

        CS = ax.contourf(x_grid, y_grid, z_interp, cmap="viridis", alpha=0.5, levels=30)
        ax.clabel(CS, inline=True, fontsize=10, fmt="%2.2f")

        ax.set_xlabel("Airspeed [m/s]")
        ax.set_ylabel("Current [A]")
        ax.set_title("Engine Map: Thrust [N] vs Airspeed [m/s] and Current [A]")

        # Add current plot
        ax = fig.add_subplot(122)
        x = self.motor["Airspeed [m/s]"]
        y = self.motor["Thrust [N]"]
        z = self.motor["Current [A]"]

        # Plot original data points
        c = ax.scatter(x, y, c=z, cmap="viridis")
        fig.colorbar(c, label="Current [A]")
        # Plot interpolated surface
        x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
        z_interp = self.current_model(x_grid, y_grid).reshape(x_grid.shape)
        CS = ax.contourf(x_grid, y_grid, z_interp, cmap="viridis", alpha=0.5, levels=30)
        ax.clabel(CS, inline=True, fontsize=10, fmt="%2.2f")

        ax.set_xlabel("Airspeed [m/s]")
        ax.set_ylabel("Thrust [N]")
        ax.set_title("Engine Map: Current [A] vs Airspeed [m/s] and Thrust [N]")
        fig.show()

    K = TypeVar("K", float, FloatArray)

    def thrust(self, velocity: K, current: K) -> K:
        # Interpolating the thrust curve
        thrust: FloatArray = self.thrust_model(velocity, current)
        return thrust

    # @jax.jit
    def current(self, velocity: float | FloatArray, thrust: float | FloatArray) -> FloatArray:
        # Interpolating the current curve
        current: FloatArray = self.current_model(velocity, thrust)
        return current


#
# def get_current(velocity, thrust):
#     # Interpolating the current curve
#     # x = np.array([[Airspeed [m/s]], [thrust]]).T
#     p00: float = 1.245
#     p10: float = -0.2109
#     p01: float = 0.4448
#     p20: float = 0.02435
#     p11: float = 0.08183
#     p02: float = 0.05932

#     x = velocity
#     y = thrust

#     val = p00 + p10 * x + p01 * y + p20 * x**2 + p11 * x * y + p02 * y**2
#     # return self.current_model.predict(x)
#     return val

# # Compare the current model against the engine.current_model for all the data points
# # Data points are in the engine.motor DataFrame
# currents_old = []
# currents_new = []
# currents_true = []
# for index, row in engine.motor.iterrows():
#     currents_old.append(get_current(row["Airspeed [m/s]"], row["Thrust [N]"]))
#     currents_new.append(engine.current_model(row["Airspeed [m/s]"], row["Thrust [N]"]))
#     currents_true.append(row["Current [A]"])

#     print(f"Old: {currents_old[-1]}, New: {currents_new[-1]}, True: {currents_true[-1]}")
#
