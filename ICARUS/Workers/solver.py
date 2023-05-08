import collections
import os
from typing import Optional
from typing import Union

import numpy as np

from .analysis import Analysis
from ICARUS.Database.db import DB


class Solver:
    def __init__(self, name: str, solverType: str, fidelity: int, db: DB) -> None:
        self.name = name
        self.type = solverType
        self.db = db
        try:
            assert type(fidelity) == int, "Fidelity must be an integer"
        except AssertionError:
            print("Fidelity must be an integer")
        self.fidelity = fidelity
        self.availableAnalyses: dict[str, Analysis] = {}
        self.mode: Union[str, None] = None

    def addAnalyses(self, analyses):
        for analysis in analyses:
            if analysis.name in self.availableAnalyses.keys():
                print(f"Analysis {analysis.name} already exists")
                continue
            if analysis.solverName != self.name:
                print(
                    f"Analysis {analysis.name} is not compatible with solver {self.name} but with {analysis.solverName}",
                )
                continue
            self.availableAnalyses[analysis.name] = analysis

    def setAnalysis(self, analysis: str):
        self.mode = analysis

    def getAnalysis(self, analysis: str | None = None) -> Analysis | None:
        if analysis is not None:
            try:
                return self.availableAnalyses[analysis]
            except KeyError:
                print(f"Analysis {analysis} not available")
        else:
            if self.mode is not None:
                try:
                    return self.availableAnalyses[self.mode]
                except KeyError:
                    print(f"Analysis {self.mode} not available")

    def printOptions(self) -> None:
        if self.mode is not None:
            print(self.availableAnalyses[self.mode])
        else:
            print("Analysis hase not been Selected")

    def getOptions(self, verbose: bool = False):
        if self.mode is not None:
            return self.availableAnalyses[self.mode].getOptions(verbose)
        else:
            print("Analysis hase not been Selected")
            _ = self.getAvailableAnalyses(verbose=verbose)

    def getSolverParameters(self, verbose: bool = False):
        if self.mode is not None:
            return self.availableAnalyses[self.mode].getSolverOptions(verbose)
        else:
            print("Analysis hase not been Selected")
            _ = self.getAvailableAnalyses(verbose=verbose)

    def setOptions(self, options):
        if self.mode is not None:
            return self.availableAnalyses[self.mode].setOptions(options)
        else:
            print("Analysis hase not been Selected")
            _ = self.getAvailableAnalyses(verbose=True)

    def getAvailableAnalyses(self, verbose: bool = False):
        if verbose:
            print(self)
        return list(self.availableAnalyses.keys())

    def run(self, analysis_name: str | None = None):
        if analysis_name is None:
            if self.mode is None:
                print("Analysis not selected or provided")
                return -1
            analysis = self.availableAnalyses[self.mode]
        else:
            if analysis_name in self.availableAnalyses.keys():
                analysis = self.availableAnalyses[analysis_name]
            else:
                print("Analysis not available")
                return -1
        print(f"Running Solver {self.name}:\n\tAnalysis {analysis.name}...")

        def saveAnalysis(analysis: Analysis):
            folder = self.db.analysesDB.DATADIR
            if "plane" in analysis.options.keys():
                folder = os.path.join(folder, analysis.options["plane"].value.name)
            if not os.path.exists(folder):
                os.makedirs(folder)
            fname = os.path.join(folder, f"{analysis.name}.json")
            with open(fname, "w") as f:
                f.write(analysis.toJSON())

        saveAnalysis(analysis)
        res = analysis()
        return res

    def getResults(self, analysis_name: Optional[str] = None) -> Union[np.ndarray, int]:
        if analysis_name is None:
            if self.mode is None:
                print("Analysis not selected or provided")
                return -1
            analysis = self.availableAnalyses[self.mode]
        else:
            if analysis_name in self.availableAnalyses.keys():
                analysis = self.availableAnalyses[analysis_name]
            else:
                print("Analysis not available")
                return -1
        res = analysis.getResults()
        return res

    def __str__(self) -> str:
        string = f"{self.type} Solver {self.name}:\n"
        string += "Available Analyses Are: \n"
        string += "------------------- \n"
        for i, key in enumerate(self.availableAnalyses.keys()):
            string += f"{i}) {key} \n"
        return string

    def worker(self):
        tasks = collections.deque()
        value = None
        while True:
            batch = yield value
            value = None
            if batch is not None:
                tasks.extend(batch)
            else:
                if tasks:
                    args = tasks.popleft()
                    value = self.run(*args)
