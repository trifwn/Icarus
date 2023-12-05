import os
from abc import abstractmethod
from typing import Any

from pandas import DataFrame

from .analysis import Analysis
from ICARUS.Core.struct import Struct
from ICARUS.Database import DB


class SolverStrategy:
    """
    Interface for the Solver strategy.
    """

    def run_analysis(self, analysis: Analysis) -> Any:
        """
        Run the provided analysis.

        Args:
            analysis (Analysis): Analysis instance.

        Returns:
            Any: Analysis Results or Error Code.
        """
        pass

    def get_results(self, analysis: Analysis) -> DataFrame | int:
        """
        Get the results of the provided analysis.

        Args:
            analysis (Analysis): Analysis instance.

        Returns:
            DataFrame | int: DataFrame with simulation Results or Error Code.
        """
        return 0


class DefaultSolverStrategy(SolverStrategy):
    """
    Default implementation of the Solver strategy.
    """

    def run_analysis(self, analysis: Analysis) -> Any:
        """
        Run the provided analysis using the default strategy.

        Args:
            analysis (Analysis): Analysis instance.

        Returns:
            Any: Analysis Results or Error Code.
        """
        print(f"Running Analysis {analysis.name}...")
        folder = DB.analyses_db.DATADIR
        if "plane" in analysis.options.keys():
            folder = os.path.join(folder, analysis.options["plane"].value.name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        fname = os.path.join(folder, f"{analysis.name}.json")
        with open(fname, "w") as f:
            f.write(analysis.toJSON())
        return analysis()

    def get_results(self, analysis: Analysis) -> DataFrame | int:
        """
        Get the results of the provided analysis using the default strategy.

        Args:
            analysis (Analysis): Analysis instance.

        Returns:
            DataFrame | int: DataFrame with simulation Results or Error Code.
        """
        return analysis.get_results()


class Solver:
    """
    Abstract class to represent a solver. It is used to run analyses.
    """

    def __init__(self, name: str, solver_type: str, fidelity: int, strategy: SolverStrategy) -> None:
        """
        Initialize the Solver class.

        Args:
            name (str): Solver Name.
            solver_type (str): Solver Type.
            fidelity (int): Fidelity of the solver.
            strategy (SolverStrategy): Solver strategy.
        """
        self.name: str = name
        self.type: str = solver_type
        try:
            assert type(fidelity) == int, "Fidelity must be an integer"
        except AssertionError:
            print("Fidelity must be an integer")
        self.fidelity: int = fidelity
        self.availableAnalyses: dict[str, Analysis] = {}
        self.mode: str | None = None
        self.strategy: SolverStrategy = strategy

    def run(self, analysis_name: str | None = None) -> Any:
        """
        Run the selected analysis.

        Args:
            analysis_name (str | None, optional): Analysis Name. If false it runs
                                            the selected one. Defaults to None.

        Returns:
            Any: Analysis Results or Error Code.
        """
        if analysis_name is None:
            if self.mode is None:
                print("Analysis not selected or provided")
                return -1
            analysis: Analysis = self.availableAnalyses[self.mode]
        else:
            if analysis_name in self.availableAnalyses.keys():
                analysis = self.availableAnalyses[analysis_name]
            else:
                print("Analysis not available")
                return -1
        print(f"Running Solver {self.name}:\n\tAnalysis {analysis.name}...")
        return self.strategy.run_analysis(analysis)

    def get_results(self, analysis_name: str | None = None) -> DataFrame | int:
        """
        Get the results of the selected analysis.

        Args:
            analysis_name (str | None, optional): Analysis Name. If false it runs
                                                    the selected one. Defaults to None.

        Returns:
            DataFrame | int: DataFrame with simulation Results or Error Code.
        """
        if analysis_name is None:
            if self.mode is None:
                print("Analysis not selected or provided")
                return -1
            analysis: Analysis = self.availableAnalyses[self.mode]
        else:
            if analysis_name in self.availableAnalyses.keys():
                analysis = self.availableAnalyses[analysis_name]
            else:
                print("Analysis not available")
                return -1
        return self.strategy.get_results(analysis)
