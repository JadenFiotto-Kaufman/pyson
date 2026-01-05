"""
pyson - JSON-serializable Python object serialization.

This library serializes Python objects to JSON-compatible Pydantic models,
preserving object identity, circular references, and complex types including:

- Primitives (int, float, bool, str, None)
- Collections (list, tuple, dict)
- Custom objects with __getstate__/__setstate__ support
- Functions with source code, closures, and globals
- Classes (by reference or by value)
- Bound methods and property descriptors

Unlike pickle, the serialized output is human-readable JSON and can be
inspected, modified, or transmitted over JSON-based protocols.

Usage:
    >>> from pyson import serialize, deserialize
    >>> import json
    >>>
    >>> # Serialize any Python object
    >>> payload = serialize({"key": [1, 2, 3]})
    >>> json_str = payload.model_dump_json()
    >>>
    >>> # Deserialize back to Python
    >>> result = deserialize(json.loads(json_str))

For functions and classes from your own modules to be serialized by value
(with source code), register them with cloudpickle:
    >>> from cloudpickle import register_pickle_by_value
    >>> import mymodule
    >>> register_pickle_by_value(mymodule)

To add custom serializers for new types:
    >>> from pyson import register_serializer, SerializedType
    >>> from typing import Literal
    >>>
    >>> class MySerializer(SerializedType):
    ...     type: Literal["my_type"] = "my_type"
    ...     # ... fields and methods
    >>>
    >>> register_serializer(MyClass, MySerializer)
"""

from pyson.serialize import (
    SerializationContext,
    register_serializer,
    register_persistent,
)
from pyson.types import Memo, ReferenceId, SerializedType
from pydantic import BaseModel
from cloudpickle import register_pickle_by_value as register_by_value


class Payload(BaseModel):
    """
    Container for serialized data.

    A Payload contains all the information needed to reconstruct
    the original Python object:

    Attributes:
        memo: Dictionary mapping reference IDs to serialized objects.
              This is the "object graph" containing all serialized data.
        obj: The reference ID of the root object to deserialize.

    The Payload can be converted to JSON using model_dump_json() and
    reconstructed using Payload.model_validate().
    """

    memo: Memo
    obj: ReferenceId


def serialize(obj) -> Payload:
    """
    Serialize a Python object to a Payload.

    The returned Payload can be converted to JSON using model_dump_json()
    and later deserialized using deserialize().

    Args:
        obj: Any Python object to serialize.

    Returns:
        A Payload containing the serialized representation.

    Example:
        >>> payload = serialize([1, 2, {"nested": True}])
        >>> json_str = payload.model_dump_json()
        >>> print(json_str)  # Human-readable JSON
    """
    context = SerializationContext()
    return Payload(memo=context.memo, obj=context.serialize(obj))


def deserialize(payload: dict, persistent_objects: dict[str, object] | None = None):
    """
    Deserialize a payload dictionary back to a Python object.

    The payload should be a dictionary as returned by json.loads()
    on a serialized Payload's JSON representation.

    Args:
        payload: The payload dictionary (from json.loads()).
        persistent_objects: Optional dict mapping persistent IDs to objects.
            Required when deserializing payloads that contain PersistentType
            references (objects serialized via register_persistent).

    Returns:
        The reconstructed Python object.

    Example:
        >>> import json
        >>> payload = serialize([1, 2, 3])
        >>> json_str = payload.model_dump_json()
        >>> result = deserialize(json.loads(json_str))
        >>> assert result == [1, 2, 3]

    Example with persistent objects:
        >>> register_persistent(LargeModel, lambda m: m.path)
        >>> model = LargeModel("/models/gpt.pt")
        >>> payload = serialize(model)
        >>> result = deserialize(payload, persistent_objects={"/models/gpt.pt": model})
    """
    validated = Payload.model_validate(payload)
    context = SerializationContext(validated.memo, persistent_objects)
    return context.deserialize(validated.obj)

