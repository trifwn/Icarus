from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Tuple


class Struct:
    __slots__: List[str] = ["_data", "_depth"]

    def __new__(cls, *args, **kwargs) -> "Struct":
        """Create a new Struct instance."""
        instance: "Struct" = super().__new__(cls)
        object.__setattr__(instance, "_data", {})
        object.__setattr__(instance, "_depth", 0)
        return instance

    def __init__(self, initial_dict: Dict[str, Any] | None = None) -> None:
        """Initialize a Struct instance with an optional initial dictionary."""
        object.__setattr__(self, "_data", {})
        object.__setattr__(self, "_depth", 0)
        if initial_dict is not None:
            self.update(initial_dict)

    def __getitem__(self, key: str) -> Any:
        """Get an item from the Struct instance by key."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an item in the Struct instance by key and value."""
        if isinstance(value, dict):
            value = Struct(value)
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete an item from the Struct instance by key."""
        del self._data[key]

    def __getattr__(self, key: str) -> Any:
        """
        Get an attribute from the Struct instance by key.
        Raises AttributeError if the key is not found.
        """
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Struct' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        """Set an attribute in the Struct instance by key and value."""
        if isinstance(value, dict):
            value = Struct(value)
        self[key] = value

    def __delattr__(self, name: str) -> None:
        """
        Delete an attribute from the Struct instance by name.
        Raises AttributeError if the attribute is not found.
        """
        if name in self._data:
            del self._data[name]
        else:
            raise AttributeError(f"Attribute {name} not found")

    def __len__(self) -> int:
        """Get the number of items in the Struct instance."""
        return len(self._data)

    def __repr__(self) -> str:
        """Return a string representation of the Struct instance."""
        items: List[str] = [f"{key}={repr(value)}" for key, value in self._data.items()]
        return f"Struct({', '.join(items)})"

    def __str__(self) -> str:
        """Return a string representation of the data in the Struct instance."""
        return str(self._data)

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the keys in the Struct instance."""
        return iter(self._data)

    def items(self) -> Iterator[Tuple[str, Any]]:
        """Return an iterator over the items in the Struct instance."""
        return self._data.items()

    def keys(self) -> Iterator[str]:
        """Return an iterator over the keys in the Struct instance."""
        return self._data.keys()

    def values(self) -> Iterator[Any]:
        """Return an iterator over the values in the Struct instance."""
        return self._data.values()

    def __getstate__(self) -> Dict[str, Any]:
        """Get the state of the Struct instance."""
        return self._data

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """This method is called when unpickling an instance of Struct. It updates the
        state of the instance by iterating over the items in state, creating a new Struct
        object for any dictionary values it encounters, and assigning the updated
        key-value pairs to self._data.

        Args:
            state (dict): _description_
        """
        for key, value in state.items():
            if isinstance(value, dict):
                value = Struct(value)
            self._data[key] = value
        # self._data = state

    def update(self, other: Any) -> None:
        """This method updates the current Struct object with the key-value pairs from
        another dictionary-like object other. If any of the values in other are
        themselves dictionaries, they are converted to Struct objects before being added
        to the current Struct.

        Args:
            other (Any): Dictionary-like object
        """
        for key, value in other.items():
            if isinstance(value, dict):
                value = Struct(value)
            self._data[key] = value

    def __invert__(self) -> "Struct":
        """This method is called when the ~ operator is used on a Struct object. It
        returns a new Struct object that has all the key-value pairs in the original
        Struct object, but with the keys and values swapped.

        Returns:
            Struct: Inverted Object
        """
        # This allows us to invert the dictionary using the ~ operator
        return self.invert_nested_dict()

    def invert_nested_dict(self) -> "Struct":
        """This method recursively inverts a nested dictionary by creating a new
        dictionary with the same keys as the original, but with inverted values. If any
        of the values are themselves dictionaries, the method calls itself recursively to
        invert those dictionaries as well. The final result is returned as a new Struct
        object.

        Returns:
            Struct: Inverted Dict
        """

        def _invert_nested_dict(dd: Struct, depth: int) -> "Struct":
            new_dict = {}
            for k, v in dd.items():
                if isinstance(v, dict):
                    new_dict[k] = _invert_nested_dict(v, depth + 1)
                else:
                    new_dict[k] = v

            if depth == 0:
                new_dict = {
                    k: _invert_nested_dict(v, depth + 1) for k, v in new_dict.items()
                }
            return new_dict

        inverted = _invert_nested_dict(self._data, self._depth)
        return Struct(inverted)

    def tree(self, indent: int = 0) -> None:
        """This method prints a hierarchical representation of the Struct object, with
        each key-value pair indented by a certain amount based on its depth in the
        structure. If a value is itself a Struct, the method is called recursively to
        print its contents as well.

        Args:
            indent (int, optional): Indentation Level. Defaults to 0.
        """
        for key, value in self.items():
            print("--|" * indent + f"- {key}:", end="")
            if isinstance(value, Struct):
                print("")
                value.tree(indent + 1)
            else:
                print(f" {value}")
