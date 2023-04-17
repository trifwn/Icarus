# Class for nested dictionary where you can access data both ways.
class Struct(dict):
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __contains__(self, item):
        return item in self.__dict__

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def pop(self, key, default=None):
        return self.__dict__.pop(key, default)

    def popitem(self, key, default=None):
        return self.__dict__.popitem(key, default)

    def clear(self):
        self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def update(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)

    def setdefault(self, key, default=None):
        return self.__dict__.setdefault(key, default)

    def fromkeys(self, *args, **kwargs):
        return self.__dict__.fromkeys(*args, **kwargs)
    
    def __getstate__(self):
        out = self.__dict__.copy()
        return out

    def __setstate__(self, state):
        self.__dict__.update(state)
