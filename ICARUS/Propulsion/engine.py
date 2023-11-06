import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame


class Engine:
    def __init__(self) -> None:
        pass

    def load_data_from_df(self, engine_dir: str) -> None:
        prop_dfs: dict[str, DataFrame] = {}
        for prop_data in os.listdir(engine_dir):
            if prop_data.endswith(".csv"):
                prop_name = prop_data[:-4]
                prop_dfs[prop_name] = pd.read_csv(engine_dir + prop_data)

        prop_dfs.keys()
        motor: DataFrame = prop_dfs["12x6E"].sort_values(by=["Airspeed [m/s]"])
        motor["Thrust [N]"] = motor["Thrust [g]"] * 9.81 / 1000

        self.motor: DataFrame = motor

    def plot_thrust_curve(self) -> None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        for throtle_lvl in self.motor["Throttle [%]"].unique():
            throtle_lvl_data = self.motor[self.motor["Throttle [%]"] == throtle_lvl]
            ax.plot(throtle_lvl_data["Airspeed [m/s]"], throtle_lvl_data["Thrust [N]"], label=f"{throtle_lvl} %")
        ax.legend()
        ax.grid()
        fig.show()

    def get_thrust(self, alpha: float, velocity: float, throttle: DataFrame) -> tuple[float, float]:
        thrust = self.get_available_thrust(velocity)
        thrust_x = thrust * np.cos(alpha)
        thrust_y = thrust * np.sin(alpha)
        return thrust_x, thrust_y

    def get_available_thrust(self, velocity: float) -> tuple[float, float]:
        throtle_levels = self.motor["Throttle [%]"].unique()
        min_throtle = self.motor[self.motor["Throttle [%]"] == min(throtle_levels)]
        max_throtle = self.motor[self.motor["Throttle [%]"] == max(throtle_levels)]

        min_thrust = float(np.interp(velocity, min_throtle["Airspeed [m/s]"], min_throtle["Thrust [N]"]))
        max_thrust = float(np.interp(velocity, max_throtle["Airspeed [m/s]"], max_throtle["Thrust [N]"]))

        return (min_thrust, max_thrust)
