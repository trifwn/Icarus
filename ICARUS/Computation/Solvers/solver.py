from typing import Any

from pandas import DataFrame
from tabulate import tabulate

from ICARUS.Computation.Analyses.analysis import Analysis
from ICARUS.Computation.Solvers.solver_parameters import Parameter
from ICARUS.Core.struct import Struct


class Solver:
    """
    Abstract class to represent a solver. It is used to run analyses.
    """

    def __init__(
        self,
        name: str,
        solver_type: str,
        fidelity: int,
        available_analyses: list[Analysis],
        solver_parameters: list[Parameter] = [],
    ) -> None:
        """
        Initialize the Solver class.

        Args:
            name (str): Solver Name.
            solver_type (str): Solver Type.
            fidelity (int): Fidelity of the solver.
            solver_parameters (list[SolverParameter]): List of Solver Parameters.
            available_analyses (list[Analysis]): List of available Analyses.
        """
        self.name: str = name
        self.type: str = solver_type
        try:
            assert type(fidelity) == int, "Fidelity must be an integer"
        except AssertionError:
            print("Fidelity must be an integer")
        self.fidelity: int = fidelity
        self.analyses: dict[str, Analysis] = {}
        for analysis in available_analyses:
            self.analyses[analysis.name] = analysis
        self.mode: str = "None"
        self.solver_parameters: dict[str, Parameter] = {
            solver_parameter.name: solver_parameter for solver_parameter in solver_parameters
        }

    def print_analysis_options(self) -> None:
        """
        Print the options of the selected analysis.
        """
        if self.mode is not None:
            print(self.analyses[self.mode])
        else:
            print("Analysis hase not been Selected")

    def get_analyses_names(self, verbose: bool = False) -> list[str]:
        if verbose:
            print(self)
        return list(self.analyses.keys())

    def print_solver_parameters(self) -> None:
        """
        Get Solver Options for the current analysis. Solver Options refer to internal solver settings not
        the analysis itself.

        """
        if self.solver_parameters is None:
            print("No Solver Parameters Needed")
            return

        string: str = f"Available Solver Parameters of {self.name} for {self.name}: \n\n"
        table: list[list[str]] = [["VarName", "Value", "Description"]]
        for param in self.solver_parameters.values():
            if param.value is None:
                table.append([param.name, "None", param.description])
            elif hasattr(param.value, "__len__"):
                if len(param.value) > 2:
                    table.append([param.name, "Multiple Values", param.description])
            else:
                table.append([param.name, param.value, param.description])
        string += tabulate(table[1:], headers=table[0], tablefmt="github")
        string += "\n\nIf there are multiple values you should inspect them sepretly by calling the option name\n"
        print(string)

    def select_analysis(self, identifier: str | int) -> None:
        """
        Set the analysis to be used.

        Args:
            analysis (str): Analysis Name.
        """
        if isinstance(identifier, str):
            self.mode = identifier
        elif isinstance(identifier, int):
            self.mode = list(self.analyses.keys())[identifier]
        else:
            raise ValueError("Invalid Analysis Identifier")

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
        # Convert Option Object to struct
        ret = Struct()
        print(self.mode)
        for option in self.analyses[self.mode].options.values():
            ret[option.name] = option.value

        if verbose:
            print(self.analyses[self.mode])
        return ret

    def set_analysis_options(self, options: Struct | dict[str, Any]) -> None:
        """
        Set
        """
        for key in options.keys():
            self.analyses[self.mode].options[key].value = options[key]

    def get_solver_parameters(self, verbose: bool = False) -> Struct:
        """
        Get the solver parameters of the selected analysis.
        """
        # Convert Option Object to struct
        ret = Struct()
        for option in self.solver_parameters.values():
            ret[option.name] = option.value
        if verbose:
            self.print_solver_parameters()
        return ret

    def set_solver_parameters(self, parameters: Struct | dict[str, Any]) -> None:
        """
        Set the solver parameters of the selected analysis.
        """
        for key in parameters.keys():
            if key in self.solver_parameters:
                self.solver_parameters[key].value = parameters[key]

    def define_analysis(self, options: Struct | dict[str, Any], solver_parameters: Struct | dict[str, Any]) -> None:
        """
        Set the options of the selected analysis.

        Args:
            options (Struct): Struct Object containing the analysis options.
        """
        if self.mode is not None:
            self.set_analysis_options(options)
            self.set_solver_parameters(solver_parameters)

    def execute(self, parallel: bool = False) -> Any:
        """
        Run the selected analysis.

        Args:
            parallel (bool, optional): Run in parallel. Defaults to False.

        Returns:
            Any: Analysis Results or Error Code.
        """

        analysis: Analysis = self.analyses[self.mode]
        print(f"Running Solver {self.name}:\n\tAnalysis {analysis.name}...")

        # def saveAnalysis(analysis: Analysis) -> None:
        #     folder: str = DB.analyses_db.DATADIR
        #     if "plane" in analysis.options.keys():
        #         folder = os.path.join(folder, analysis.options["plane"].name)
        #     if not os.path.exists(folder):
        #         os.makedirs(folder)
        #     fname: str = os.path.join(folder, f"{analysis.name}.json")
        #     with open(fname, "w") as f:
        #         f.write(analysis.encode_json())

        # saveAnalysis(analysis)
        solver_parameters = [solver_parameter for solver_parameter in self.solver_parameters.values()]
        res = analysis(solver_parameters, parallel=parallel)
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
            analysis: Analysis = self.analyses[self.mode]
        else:
            if analysis_name in self.analyses.keys():
                analysis = self.analyses[analysis_name]
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
        for i, key in enumerate(self.analyses.keys()):
            string += f"{i}) {key} \n"
        return string

    def __repr__(self) -> str:
        return f"Solver: {self.name}"
