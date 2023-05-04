import io


class Option():
    __slots__ = ['name','value','description']
    def __init__(self,name:str,value,description:str) -> None:
        self.name = name
        self.value = value
        self.description :str= description
    
    def __getstate__(self) -> tuple:
        return (self.name,self.value,self.description)
    
    def __setstate__(self,state: tuple)-> None:
        self.name,self.value,self.description = state
        
    def __str__(self):
        ss = io.StringIO()
        
        ss.write(f'{self.name} : {self.value}\n')
        ss.write(f'{self.description}\n')
        return ss.getvalue()
    
    def __repr__(self):
        ss = io.StringIO()
        
        ss.write(f'{self.name} : {self.value}\n')
        ss.write(f'{self.description}\n')
        return ss.getvalue()