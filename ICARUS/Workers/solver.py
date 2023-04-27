import collections
from .analysis import Analysis
from ICARUS.Database.db import DB
import os


class Solver():
    def __init__(self,name : str, solverType : str, fidelity : int, db : DB) -> None:
        self.name = name
        self.type = solverType
        self.db = db
        try:
            assert type(fidelity) == int, "Fidelity must be an integer"
        except AssertionError:
            print("Fidelity must be an integer")
        self.fidelity = fidelity
        self.availableAnalyses = {}
        self.resRetrivalMethods = {}
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
           
    def setAnalysis(self, analysis : str):
        self.mode = analysis
        
    def getAnalysis(self) -> Analysis:
        try:
            return  self.availableAnalyses[self.mode]
        except:
            print(f'Analysis {self.mode} not available')
       
    def printOptions(self):
        if self.mode is not None:
            print(self.availableAnalyses[self.mode])
        else:
            print('Analysis hase not been Selected')
    
    def getOptions(self,verbose : bool = False):
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
            
    def getAvailableAnalyses(self, verbose: bool = False):
        if verbose:
            print(self)
        return list(self.availableAnalyses.keys())
            
    def addResRetrival(self,methods):
        for method in methods:
            if method.name in self.resRetrivalMethods.keys():
                print(f"Method {method.name} already exists")
                continue
            if method.solverName != self.name:
                print(f"Method {method.name} is not compatible with solver {self.name} but with {method.solverName}")
                continue
            self.resRetrivalMethods[method.name] = method
            
    def setRetrivalMethod(self, method : str):
        self.Resmode = method
    
    def getRetrivalMethod(self):
        try:
            return  self.resRetrivalMethods[self.Resmode]
        except:
            print(f'Results Retrival Method {self.Resmode} not available')
            
    def printRetrievalOptions(self):
        if self.mode is not None:
            print(self.resRetrivalMethods[self.Resmode])
        else:
            print('Analysis hase not been Selected')
    
    def getRetrievalOptions(self,verbose : bool = False):
        if self.Resmode is not None:
            return self.resRetrivalMethods[self.Resmode].getOptions(verbose)
        else:
            print('Analysis hase not been Selected')
            _ = self.getAvailableRetrieval(verbose=True)
    
    def setRetrievalOptions(self, options):
        if self.Resmode is not None:
            return self.resRetrivalMethods[self.Resmode].setOptions(options)
        else:
            print('Analysis hase not been Selected')
            _ = self.getAvailableRetrieval(verbose=True)
                   
    def getAvailableRetrieval(self, verbose: bool = False):
        if verbose:
            print(self)
        return list(self.resRetrivalMethods.keys())
            
    def run(self,analysis : str = None):
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
        
        def saveAnalysis(analysis : Analysis):
            folder = self.db.analysesDB.DATADIR
            fname = os.path.join(folder, f'{analysis.name}.json')
            with open(fname, 'w') as f:
                f.write(analysis.toJSON())
            
        saveAnalysis(analysis)
        res = analysis()
        return res
    
    def getResults(self, method: str = None):
        if method is None:
            if self.Resmode is None:
                print("Analysis not selected or provided")
                return -1
            method = self.resRetrivalMethods[self.Resmode]
        else:
            if method in self.resRetrivalMethods.keys():
                method = self.resRetrivalMethods[method]
            else:
                print("method not available")
                return -1
        print(f'Getting Results')
        
        res = method()
        return res
        
    def __str__(self) -> str:
        string = f'{self.type} Solver {self.name}:\n' 
        string += 'Available Analyses Are: \n'
        string+= '------------------- \n'
        for i,key in enumerate(self.availableAnalyses.keys()):
            string+= f"{i}) {key} \n"
        string += '\nAvailable Results: \n'
        string+= '------------------- \n'
        for i,key in enumerate(self.resRetrivalMethods.keys()):
            string+= f"{i}) {key} \n"       
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