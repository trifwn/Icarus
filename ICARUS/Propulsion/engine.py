import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
import scipy

from ICARUS.Core.types import FloatArray


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
        motor: DataFrame = prop_dfs["13x8E"].sort_values(by=["Airspeed [m/s]"])
        motor["Thrust [N]"] = motor["Thrust [g]"] * 9.81 / 1000

        self.motor: DataFrame = motor
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(degree=2, include_bias=False)

        # define the target and input data for training
        target1 = motor["Thrust [N]"]
        input = motor[["Airspeed [m/s]", "Current [A]"]]
        # Fit a model of the form y = ax^2 + bx + a2 * x2 + b2 * x2 + c
        # where x is the airspeed and x2 is the current
        regression_model_thrust = LinearRegression()
        regression_model_thrust.fit(poly.fit_transform(input), target1)
        self.thrust_model = regression_model_thrust

        # Regression Models for Current and Voltage
        target2 = motor["Current [A]"]
        input = motor[["Airspeed [m/s]", "Thrust [N]"]]
        regression_model_current = LinearRegression()
        regression_model_current.fit(poly.fit_transform(input), target2)
        self.current_model = regression_model_current

    def plot_thrust_curve(self) -> None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        for throtle_lvl in self.motor["Throttle [%]"].unique():
            throtle_lvl_data = self.motor[self.motor["Throttle [%]"] == throtle_lvl]
            ax.plot(
                throtle_lvl_data["Airspeed [m/s]"],
                throtle_lvl_data["Thrust [N]"],
                label=f"{throtle_lvl} %",
            )
        ax.legend()
        ax.grid()
        fig.show()

    def get_thrust(
        self, alpha: float, velocity: float, current: float
    ) -> tuple[float, float]:
        thrust = self.thrust(velocity, current)
        thrust_x = thrust * np.cos(alpha)
        thrust_y = thrust * np.sin(alpha)
        return thrust_x, thrust_y

    def thrust(self, velocity: float, current: float) -> float | FloatArray:
        # Interpolating the thrust curve
        x = np.array([[velocity], [current]]).T
        print(x.shape)
        return self.thrust_model.predict(x)

    def get_current(self, velocity: float, thrust: float) -> float | FloatArray:
        # Interpolating the current curve
        # x = np.array([[velocity], [thrust]]).T
        p00 = 1.245
        p10 = -0.2109
        p01 = 0.4448
        p20 = 0.02435
        p11 = 0.08183
        p02 = 0.05932

        x = velocity
        y = thrust

        val = p00 + p10 * x + p01 * y + p20 * x**2 + p11 * x * y + p02 * y**2
        # return self.current_model.predict(x)
        return val
