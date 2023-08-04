import os
from typing import Any

from pandas import DataFrame

from .analysis import Analysis
from ICARUS.Core.struct import Struct
from ICARUS.Database.db import DB


class Solver:
    """
    Abstract class to represent a solver. It is used to run analyses.
    """

    def __init__(self, name: str, solver_type: str, fidelity: int, db: DB) -> None:
        """
        Initialize the Solver class.

        Args:
            name (str): Solver Name.
            solver_type (str): Solver Type.
            fidelity (int): Fidelity of the solver.
            db (DB): Database.
        """
        self.name: str = name
        self.type: str = solver_type
        self.db: DB = db
        try:
            assert type(fidelity) == int, "Fidelity must be an integer"
        except AssertionError:
            print("Fidelity must be an integer")
        self.fidelity: int = fidelity
        self.availableAnalyses: dict[str, Analysis] = {}
        self.mode: str | None = None

    def add_analyses(self, analyses: list[Analysis]) -> None:
        """
        Add analyses to the solver.

        Args:
            analyses (list[Analysis]): List of analyses to add.
        """
        for analysis in analyses:
            if analysis.name in self.availableAnalyses.keys():
                print(f"Analysis {analysis.name} already exists")
                continue
            if analysis.solver_name != self.name:
                print(
                    f"Analysis {analysis.name} is not compatible with solver {self.name} but with {analysis.solver_name}",
                )
                continue
            self.availableAnalyses[analysis.name] = analysis

    def set_analyses(self, analysis: str) -> None:
        """
        Set the analysis to be used.

        Args:
            analysis (str): Analysis Name.
        """
        self.mode = analysis

    def get_analysis(self, analysis: str | None = None) -> Analysis:
        """
        Get the analysis to be used.

        Args:
            analysis (str | None, optional): String with the analysis Name. If None
                                            it return the selected Analysis. Defaults to None.

        Returns:
            Analysis: Analysis Object.
        """
        if analysis is not None:
            try:
                return self.availableAnalyses[analysis]
            except KeyError:
                print(f"Analysis {analysis} not available")
                raise Exception(f"Analysis {analysis} not available")
        else:
            if self.mode is not None:
                try:
                    return self.availableAnalyses[self.mode]
                except KeyError:
                    print(f"Analysis {self.mode} not available")
                    raise Exception(f"Analysis {self.mode} not available")
            else:
                print("Analysis not selected")
                raise Exception("Analysis not selected")

    def print_analysis_options(self) -> None:
        """
        Print the options of the selected analysis.
        """
        if self.mode is not None:
            print(self.availableAnalyses[self.mode])
        else:
            print("Analysis hase not been Selected")

    def get_analysis_options(self, verbose: bool = False) -> Struct:
        """
        Get the options of the selected analysis.

        Args:
            verbose (bool, optional): Displays the option if True. Defaults to False.

        Raises:
            Exception: If the analysis has not been selected.

        Returns:
            Struct: Struct Object containing the analysis options.
        """
        if self.mode is not None:
            return self.availableAnalyses[self.mode].get_options(verbose)
        else:
            print("Analysis hase not been Selected")
            _ = self.available_analyses_names(verbose)
            raise Exception("Analysis hase not been Selected")

    def get_solver_parameters(self, verbose: bool = False) -> Struct:
        """
        Get the solver parameters of the selected analysis.

        Args:
            verbose (bool, optional): Displays the options if True. Defaults to False.

        Raises:
            Exception: If the analysis has not been selected.

        Returns:
            Struct: Struct Object containing the solver parameters.
        """
        if self.mode is not None:
            return self.availableAnalyses[self.mode].get_solver_options(verbose=verbose)
        else:
            print("Analysis hase not been Selected")
            _ = self.available_analyses_names(verbose=verbose)
            raise Exception("Analysis hase not been Selected")

    def set_analysis_options(self, options: Struct | dict[str, Any]) -> None:
        """
        Set the options of the selected analysis.

        Args:
            options (Struct): Struct Object containing the analysis options.
        """
        if self.mode is not None:
            self.availableAnalyses[self.mode].set_all_options(options)
        else:
            print("Analysis hase not been Selected")
            _ = self.available_analyses_names(verbose=True)

    def set_solver_parameters(self, params: Struct | dict[str, Any]) -> None:
        """
        Set the solver parameters of the selected analysis.

        Args:
            params (Struct | dict[str,Any]): Struct Object containing the solver parameters.
        """
        if self.mode is not None:
            self.availableAnalyses[self.mode].set_all_solver_params(params)
        else:
            print("Analysis hase not been Selected")
            _ = self.available_analyses_names(verbose=True)

    def available_analyses_names(self, verbose: bool = False) -> list[str]:
        """
        Return the available analyses names.

        Args:
            verbose (bool, optional): If true it also prints them. Defaults to False.

        Returns:
            list[str]: List of the analyses names
        """
        if verbose:
            print(self)
        return list(self.availableAnalyses.keys())

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

        def saveAnalysis(analysis: Analysis) -> None:
            folder: str = self.db.analysesDB.DATADIR
            if "plane" in analysis.options.keys():
                folder = os.path.join(folder, analysis.options["plane"].value.name)
            if not os.path.exists(folder):
                os.makedirs(folder)
            fname: str = os.path.join(folder, f"{analysis.name}.json")
            with open(fname, "w") as f:
                f.write(analysis.toJSON())

        saveAnalysis(analysis)
        res = analysis()
        return res

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
        res: DataFrame | int = analysis.get_results()
        return res

    def __str__(self) -> str:
        """
        String representation of the Solver.

        Returns:
            str: String representation of the Solver.
        """
        string: str = f"{self.type} Solver {self.name}:\n"
        string += "Available Analyses Are: \n"
        string += "------------------- \n"
        for i, key in enumerate(self.availableAnalyses.keys()):
            string += f"{i}) {key} \n"
        return string

    # TODO: Implement this
    # def worker(self):
    #     tasks = collections.deque()
    #     value = None
    #     while True:
    #         batch = yield value
    #         value = None
    #         if batch is not None:
    #             tasks.extend(batch)
    #         else:
    #             if tasks:
    #                 args = tasks.popleft()
    #                 value = self.run(*args)
