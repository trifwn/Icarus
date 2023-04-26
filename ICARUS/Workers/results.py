from ICARUS.Core.struct import Struct
from ICARUS.Workers.analysis import Analysis
from .options import Option
from tabulate import tabulate

class Results():
    def __init__(self, solverName : str, name : str, funcDict : dict, optionsDict : dict):
        self.name = name
        self.solverName = solverName
        self.resFuncs = Struct()
        self.resFuncArgs = Struct()
        for funcName in funcDict.keys():
            self.resFuncs[funcName] = funcDict[funcName]
            self.resFuncArgs[funcName] = optionsDict[funcName]
    
    def getAvailableResFuncs(self):
        return self.resFuncs.keys()
    
    
    def processOptions(self, AnalysisOptions : Analysis, funcName : str):
        pass
    
    def getResults(self, analysis : Analysis, proccess : str = None):
        if proccess is None:
            # process = self.resFuncs.keys()[0]
            pass
        args = self.processOptions(analysis)
        return self.resFunc(*args)
    