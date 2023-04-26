import io

class Option():
    __slots__ = ['name','value','description']
    def __init__(self,name,value,description):
        self.name = name
        self.value = value
        self.description = description
    
    def __getstate__(self):
        return (self.name,self.value,self.description)
    
    def __setstate__(self,state):
        self.name,self.value,self.description = state
        
    def __str__(self):
        ss = io.StringIO()
        
        ss.write(f'{self.name} : {self.value}\n')
        ss.write(f'{self.description}\n')
        return ss.getvalue()