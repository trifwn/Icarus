from ICARUS.Core.struct import Struct

class Solver():
    def __init__(self,name,fidelity,analysisDict,optionsDict) -> None:
        self.name = name
        try:
            assert type(fidelity) == int, "Fidelity must be an integer"
        except AssertionError:
            print("Fidelity must be an integer")
        self.fidelity = fidelity
        
        self.availableAnalyses = analysisDict
        options = Struct()
        options.convertFromDict(optionsDict)
        self.options = options
        self.mode = None
        
    def setAnalysis(self,analysis):
        self.mode = analysis
        
    def getAnalysis(self):
        return self.mode
    
    def getOptions(self,analysis,verbose = True):
        if verbose:
            string = f'Available Options for {analysis}: \n'
            string+= '\t---------\t \n'
            string+= "VarName | DefaultValue\t| Description\n"
            string+= '--------------------------------------\n'
            for key,value in self.options[analysis].items():
                string+= f"{value[0]}\t| {value[1]} \t\t| {value[2]} \n"
            print(string)

        return self.options[analysis]
    
    def setOptions(self,analysis, options):
        self.options[analysis] = options
        
    
    def getAvailableAnalyses(self, verbose = True):
        if verbose:
            string = 'Available Analyses: \n'
            string+= '------------------- \n'
            for key in self.availableAnalyses.keys():
                string+= f"{key} \n"
            print(string)

        return list(self.availableAnalyses.keys())
            
    def run(self,analysis,options):
        print(options)
        return 0
        if analysis in self.availableAnalyses.keys():
            res = self.availableAnalyses[analysis](options)
        return res
    
    def getResults(self,analysis,options):
        if analysis in self.availableAnalyses.keys():
            if self.availableAnalyses[analysis].checkRun():
                res = self.availableAnalyses[analysis](options)
                return res
        else:
            print("Analysis not available")
            return None
        
class SolverOptions():
    def __init__(self,analysis, options) -> None:
        self.analysis = analysis
        self.options = options