class Struct:
    __slots__ = ['_data', '_depth']
    
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        object.__setattr__(instance, '_data', {})
        object.__setattr__(instance, '_depth', 0)
        return instance
    
    def __init__(self, initial_dict=None):
        object.__setattr__(self, '_data', {})
        object.__setattr__(self, '_depth', 0)
        if initial_dict is not None:
            self.update(initial_dict)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = Struct(value)
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]
            
    def __getattr__(self, key):
        # if key == '_data' or key == '_depth':
        #     return object.__getattribute__(self, key)
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Struct' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = Struct(value)
        self[key] = value

    def __delattr__(self, name):
        if name in self._data:
            del self._data[name]
        else:
            raise AttributeError("Attribute {} not found".format(name))

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        items = []
        for key, value in self._data.items():
            items.append(f"{key}={repr(value)}")
        return f"Struct({', '.join(items)})"

    def __str__(self):
        return str(self._data)

    def __iter__(self):
        return iter(self._data)
    
    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()
    
    def __getstate__(self):
        return self._data
    
    def __setstate__(self, state):
        for key, value in state.items():
            if isinstance(value, dict):
                value = Struct(value)
            self._data[key] = value
        # self._data = state
        
    def update(self, other):
        for key, value in other.items():
            if isinstance(value, dict):
                value = Struct(value)
            self._data[key] = value

    def __invert__(self):
        # This allows us to invert the dictionary using the ~ operator
        return self.invert_nested_dict()
    
    def invert_nested_dict(self):
        def _invert_nested_dict(dd, depth):
            new_dict = {}
            for k, v in dd.items():
                if isinstance(v, dict):
                    new_dict[k] = _invert_nested_dict(v, depth + 1)
                else:
                    new_dict[k] = v

            if depth == 0:
                new_dict = {k: _invert_nested_dict(v, depth + 1) for k, v in new_dict.items()}
            return new_dict

        inverted = _invert_nested_dict(self._data, self._depth)
        return Struct(inverted)

    def tree(self, indent=0):
        for key, value in self.items():
            print('--|' * indent + '- {}:'.format(key), end='')
            if isinstance(value, Struct):
                print('')
                value.tree(indent + 1)
            else:
                print(' {}'.format(value))