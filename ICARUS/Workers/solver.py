import collections

class Solver():
    def __init__(self,name,solverType,fidelity) -> None:
        self.name = name
        self.type = solverType
        try:
            assert type(fidelity) == int, "Fidelity must be an integer"
        except AssertionError:
            print("Fidelity must be an integer")
        self.fidelity = fidelity
        self.availableAnalyses = {}
        self.mode = None
        
    def addAnalyses(self,analyses):
        for analysis in analyses:
            if analysis.name in self.availableAnalyses.keys():
                print(f"Analysis {analysis.name} already exists")
                continue
            if analysis.solverName != self.name:
                print(f"Analysis {analysis.name} is not compatible with solver {self.name} but with {analysis.solverName}")
                continue
            self.availableAnalyses[analysis.name] = analysis
        
    def setAnalysis(self,analysis):
        self.mode = analysis
        
    def getAnalysis(self):
        return self.mode
    
    def printOptions(self):
        if self.mode is not None:
            print(self.availableAnalyses[self.mode])
        else:
            print('Analysis hase not been Selected')
    
    def getOptions(self,verbose = False):
        if self.mode is not None:
            return self.availableAnalyses[self.mode].getOptions(verbose)
        else:
            print('Analysis hase not been Selected')
            _ = self.getAvailableAnalyses(verbose=True)
    
    def setOptions(self, options):
        if self.mode is not None:
            return self.availableAnalyses[self.mode].setOptions(options)
        else:
            print('Analysis hase not been Selected')
            _ = self.getAvailableAnalyses(verbose=True)
            
    def __str__(self):
        string = f'Solver {self.name}:\n' 
        string += 'Available Analyses Are: \n'
        string+= '------------------- \n'
        for i,key in enumerate(self.availableAnalyses.keys()):
            string+= f"{i}) {key} \n"
        return string
    
    def getAvailableAnalyses(self, verbose = False):
        if verbose:
            print(self)
        return list(self.availableAnalyses.keys())
            
    def run(self,analysis = None):
        if analysis is None:
            if self.mode is None:
                print("Analysis not selected or provided")
                return -1
            analysis = self.availableAnalyses[self.mode]
        else:
            if analysis in self.availableAnalyses.keys():
                analysis = self.availableAnalyses[analysis]
            else:
                print("Analysis not available")
                return -1
        print(f'Running Solver {self.name}:\n\tAnalysis {analysis.name}...')
        res = analysis()
        return res
    
    def getResults(self,analysis):
        if analysis in self.availableAnalyses.keys():
            if self.availableAnalyses[analysis].checkRun():
                res = None
                return res
        else:
            print("Analysis not available")
            return None

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