from __future__ import annotations

import itertools
from abc import ABC
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import replace
from typing import Any
from typing import Union
from typing import get_args
from typing import get_origin
from typing import get_type_hints

import numpy as np
import numpy.typing as npt

from ICARUS.core.format import short_format


def iter_field(*, order: int, **kwargs):
    metadata = dict(kwargs.pop("metadata", {}))
    metadata["iter_order"] = order
    return field(metadata=metadata, **kwargs)


@dataclass
class BaseAnalysisInput(ABC):
    """
    Abstract base class for analysis inputs.
    Serves as a data container for a single analysis run.
    """

    def validate(self) -> None:
        """
        Validate the input parameters.
        Check all fields are set and of appropriate types.
        Cast fields to expected types if possible.
        """
        hints = get_type_hints(self.__class__)
        for f in fields(self):
            name = f.name
            value = getattr(self, name)
            expected_type = hints.get(name, None)

            if value is None:
                raise ValueError(f"Field '{name}' is not set (None)")

            if expected_type and not self._is_instance_of_type(value, expected_type):
                try:
                    casted_value = self._cast_to_type(value, expected_type)
                    setattr(self, name, casted_value)
                except Exception as e:
                    raise TypeError(
                        f"Field '{name}' is expected to be of type {expected_type}, "
                        f"got {type(value)} and casting failed: {e}",
                    )

    def _is_instance_of_type(self, value: Any, expected_type: Any) -> bool:
        """
        Helper to handle typing constructs like Optional, Union, etc.
        """
        origin = get_origin(expected_type)

        if origin is Union:
            # Handle Optional[...] or Union[T1, T2, ...]
            return any(
                self._is_instance_of_type(value, arg) for arg in get_args(expected_type) if arg is not type(None)
            )

        if origin is not None:
            # Generic like list[float], dict[str, int], etc.
            return isinstance(value, origin)

        return isinstance(value, expected_type)

    def _cast_to_type(self, value: Any, expected_type: Any) -> Any:
        """
        Attempt to cast value to the expected type.
        Supports some built-in types and common generics like list, tuple.
        """
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        # Handle Union (cast to first non-None type)
        if origin is Union:
            for arg in args:
                if arg is not type(None):
                    try:
                        return self._cast_to_type(value, arg)
                    except Exception:
                        pass
            raise ValueError(f"Cannot cast {value} to any type in {expected_type}")

        # Handle built-in generics like list, tuple
        if origin in (list, tuple):
            # For these, cast the outer container, and attempt to cast each element
            container_type = origin
            item_type = args[0] if args else Any

            if not isinstance(value, container_type):
                # Try casting the container itself (e.g., tuple(value) or list(value))
                value = container_type(value)

            # Cast each element recursively
            casted_items = [self._cast_to_type(v, item_type) for v in value]

            # For tuple, convert back to tuple
            if container_type is tuple:
                return tuple(casted_items)
            else:
                return casted_items

        # For other built-in types or classes, try direct construction
        try:
            return expected_type(value)
        except Exception as e:
            # If casting fails, just raise up
            raise ValueError(f"Failed to cast {value} to {expected_type}: {e}")

    @classmethod
    def get_input(cls, data: dict[str, Any]) -> BaseAnalysisInput:
        """
        Set data from a dictionary to the fields of the dataclass.
        """
        return cls(**data)

    def expand_dataclass(self) -> dict[str, BaseAnalysisInput]:
        iter_field_names = self.get_iter_fields()

        # Extract iterables from iter fields, check they are iterable
        iterables = []
        for name in iter_field_names:
            val = getattr(self, name)
            if val is None:
                continue
                raise ValueError(f"Iter field '{name}' is None, cannot expand")
            if not hasattr(val, "__iter__") or isinstance(val, (str, bytes)):
                continue
                raise ValueError(f"Iter field '{name}' is not iterable")
            iterables.append(val)

        # Outer product of all iterables
        product_iter = itertools.product(*iterables)

        expanded_dict = {}
        for combo in product_iter:
            replace_kwargs = dict(zip(iter_field_names, combo))
            new_instance = replace(self, **replace_kwargs)

            # Build descriptive key string: "field1: value1 | field2: value2 | ..."
            key_parts = [f"{name}= {short_format(val)}" for name, val in replace_kwargs.items()]
            key = " | ".join(key_parts)

            expanded_dict[key] = new_instance

        return expanded_dict

    @classmethod
    def get_iter_fields(cls) -> list:
        """Return list of (field_name, order) sorted by order."""
        iter_fields = []
        for f in fields(cls):
            order = f.metadata.get("iter_order")
            if order is not None:
                iter_fields.append((f.name, order))
        return [name for name, _ in sorted(iter_fields, key=lambda x: x[1])]

    def get_iter_field_shapes(self) -> list[int]:
        """Get lengths of all iter fields in order."""
        iter_field_names = self.get_iter_fields()
        shapes = []
        for name in iter_field_names:
            val = getattr(self, name)
            if val is None or not hasattr(val, "__len__"):
                shapes.append(1)
                continue
                raise ValueError(f"Iter field {name} must be a non-empty iterable")
            shapes.append(len(val))
        return shapes

    def fold_results(self, flat_results: list[Any]) -> npt.NDArray[Any]:
        """
        Fold flat_results into an n-D nested list matching iter field sizes.
        """
        shapes = self.get_iter_field_shapes()
        total_size = np.prod(shapes) if shapes else 1

        if len(flat_results) != total_size:
            raise ValueError(f"Number of results {len(flat_results)} does not match expansion size {total_size}")

        # Create 1D object array from results
        obj_array = np.empty(total_size, dtype=object)
        obj_array[:] = flat_results  # fill array with objects

        # Reshape the *outer* dimension only
        nested = obj_array.reshape(shapes)
        return nested
