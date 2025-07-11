from __future__ import annotations

from pandas import DataFrame

from ICARUS.airfoils.metrics.polar_map import AirfoilPolarMap
from ICARUS.airfoils.metrics.polars import AirfoilPolar


class AirfoilData:
    """Solver Data Class"""

    def __init__(
        self,
        airfoil_name: str,
        polar_maps: dict[str, AirfoilPolarMap] = {},
    ) -> None:
        self.airfoil_name = airfoil_name

        self.polars: dict[str, AirfoilPolarMap] = {}
        for solver, polar_map in polar_maps.items():
            self.add_polar_map(solver, polar_map)

    @property
    def solvers(self) -> list[str]:
        return list(self.polars.keys())

    def get_solver_reynolds(self, solver: str) -> list[float]:
        return self.polars[solver].reynolds_numbers

    def get_polars(self, solver: str | None = None) -> AirfoilPolarMap:
        if solver is None:
            return self.polars[list(self.polars.keys())[0]]
        return self.polars[solver]

    def get_polar_data(self, solver: str) -> DataFrame:
        return self.polars[solver].df

    def add_polar_map(self, solver_name: str, polar_map: AirfoilPolarMap) -> None:
        self.polars[solver_name] = polar_map

    def add_polar(self, solver_name: str, polar: AirfoilPolar) -> None:
        """Add a polar to the airfoil data."""
        if solver_name not in self.polars:
            self.polars[solver_name] = AirfoilPolarMap(self.airfoil_name, solver_name)
        self.polars[solver_name].add_polar(polar)

    # def add_data(self, data: dict[str, dict[str, DataFrame]]) -> None:
    #     for solver, subdata in data.items():
    #         self.add_solver(solver, subdata)
