from ICARUS.Core.struct import Struct
from .options import Option
from tabulate import tabulate

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
jsonpickle_pd.register_handlers()

class Analysis():
    def __init__(self,solverName, name, runFunc ,options):
        self.solverName = solverName
        self.name = name
        self.options = Struct()
        self.execute = runFunc
        for option in options.keys():
            self.options[option] = Option(option,None,options[option]) 
    
    def __str__(self):
        string = f'Available Options of {self.solverName} for {self.name}: \n\n'
        table = [['VarName','Value','Description']]
        for _,opt in self.options.items():
            if opt.value is None:
                table.append([opt.name,'None',opt.description])
            elif hasattr(opt.value, '__len__'):
                if len(opt.value) > 2:
                    table.append([opt.name,'Multiple Values',opt.description])                
            else:
                table.append([opt.name,opt.value,opt.description])
        string+= tabulate(table[1:],headers=table[0] ,tablefmt="github")
        string+= '\n\nIf there are multiple values you should inspect them sepretly by calling the option name\n'
        return string
    
    def getOptions(self,verbose = True):
        if verbose:
            print(self)
        return self.options
    
    def setOption(self,optionName,optionValue):
        try:
            self.options[optionName].value = optionValue
        except KeyError:
            print(f'Option {optionName} not available')
        
    def setOptions(self, options):
        for option in options:
            self.setOption(option,options[option])
    
    def checkOptions(self):
        flag = True
        for option in self.options:
            if self.options[option].value is None:
                print(f'Option {option} not set')
                flag = False
        return flag
    
    def checkRun(self):
        print('Checking Run')
        return True
    
    def __call__(self):
        if self.checkOptions():
            kwargs = {option:self.options[option].value for option in self.options.keys()}
            res = self.execute(**kwargs)
            print('Analysis Completed')
            return res
        else:
            print(f"Options not set for {self.name} of {self.solverName}. Here is what was passed:")
            print(self)
            return -1
            
    def copy(self):
        optiondict = {k:v.description for k,v in self.options.items()}
        return self.__class__(self.solverName, self.name, self.execute, optiondict)
    
    def __copy__(self):
        optiondict = {k:v.description for k,v in self.options.items()}
        return self.__class__(self.solverName,self.name, self.execute, optiondict)
    
    def toJSON(self):
        return jsonpickle.encode(self)
    
    def __lshift__(self, other):
        """ overloading operator << """
        if not isinstance(other, dict):
            raise TypeError("Can only << a dict")
           
        s = self.__copy__()
        s.__dict__.update(other)
        return s