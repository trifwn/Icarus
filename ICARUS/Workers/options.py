from ICARUS.Core.struct import Struct

class Option(Struct):
    def __init__(self,name,value,description):
        self.name = name
        self.value = value
        self.description = description
            
    def __copy__(self):
        return self.__class__(**self.__dict__)