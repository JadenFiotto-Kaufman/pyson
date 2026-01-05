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

Basic Usage:
    >>> from pyson import serialize, deserialize
    >>> import json
    >>>
    >>> # Serialize any Python object
    >>> payload = serialize({"key": [1, 2, 3]})
    >>> json_str = payload.model_dump_json()
    >>>
    >>> # Deserialize back to Python
    >>> result = deserialize(json.loads(json_str))

Linting (for remote execution safety):
    >>> from pyson.lint import NNSIGHT_CONFIG  # or create custom LintConfig
    >>> # Enable validation for remote execution
    >>> payload = serialize(data, lint=NNSIGHT_CONFIG)
    >>> # Will raise ValueError on:
    >>> #   - Forbidden types (pandas DataFrames, file handles, etc.)
    >>> #   - Functions with nonlocal closures
    >>> #   - Code with forbidden imports/calls

Server-Provided Attributes (for remote execution):
    >>> # Mark attributes that should be provided by the server
    >>> class MyModel:
    ...     _server_provided = frozenset({'_cache', '_connection'})
    >>>
    >>> # These attributes are skipped during serialization
    >>> payload = serialize(model)
    >>>
    >>> # Inject server's values during deserialization
    >>> result = deserialize(payload_dict, server_provided={
    ...     '_cache': server_cache,
    ...     '_connection': server_conn
    ... })

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
from pyson.stypes import Memo, ReferenceId, SerializedType
from pydantic import BaseModel
from cloudpickle import register_pickle_by_value as register_by_value

# Import lint module for strict mode validation
from pyson import lint


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


def serialize(obj, *, lint: 'lint.LintConfig | None' = None) -> Payload:
    """
    Serialize a Python object to a Payload.

    The returned Payload can be converted to JSON using model_dump_json()
    and later deserialized using deserialize().

    Args:
        obj: Any Python object to serialize.
        lint: Optional LintConfig for validation. When provided, enables
            checking for forbidden types, nonlocal closures, and dangerous
            imports/calls. Use lint.NNSIGHT_CONFIG for nnsight-compatible
            validation, or create a custom LintConfig for other use cases.

    Returns:
        A Payload containing the serialized representation.

    Raises:
        ValueError: If linting is enabled and the object or its contents
            violate the configured serialization safety rules.

    Example:
        >>> payload = serialize([1, 2, {"nested": True}])
        >>> json_str = payload.model_dump_json()
        >>> print(json_str)  # Human-readable JSON

    Example with linting (nnsight config):
        >>> from pyson.lint import NNSIGHT_CONFIG
        >>> # Raises helpful error instead of serializing DataFrame
        >>> payload = serialize(pandas_df, lint=NNSIGHT_CONFIG)
        ValueError: Cannot serialize DataFrame:
        pandas.DataFrame cannot be serialized.
        Convert to a tensor before serialization:
          tensor_data = torch.tensor(df.values)

    Example with custom lint config:
        >>> from pyson.lint import LintConfig
        >>> config = LintConfig(
        ...     forbidden_modules=frozenset({'os', 'subprocess'}),
        ...     reject_nonlocal=True,
        ... )
        >>> payload = serialize(my_func, lint=config)
    """
    context = SerializationContext(lint=lint)
    return Payload(memo=context.memo, obj=context.serialize(obj))


def deserialize(
    payload: dict,
    persistent_objects: dict[str, object] | None = None,
    server_provided: dict[str, object] | None = None,
):
    """
    Deserialize a payload dictionary back to a Python object.

    The payload should be a dictionary as returned by json.loads()
    on a serialized Payload's JSON representation.

    Args:
        payload: The payload dictionary (from json.loads()).
        persistent_objects: Optional dict mapping persistent IDs to objects.
            Required when deserializing payloads that contain PersistentType
            references (objects serialized via register_persistent).
        server_provided: Optional dict mapping attribute names to values.
            Used to inject values for attributes marked as _server_provided
            on classes. These attributes are skipped during serialization
            and must be provided during deserialization.

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

    Example with server-provided attributes:
        >>> # Class that marks certain attributes as server-provided
        >>> class Model:
        ...     _server_provided = frozenset({'_tokenizer', '_module'})
        >>>
        >>> # These are skipped during serialization
        >>> payload = serialize(model)
        >>>
        >>> # Inject server's values during deserialization
        >>> result = deserialize(
        ...     payload_dict,
        ...     server_provided={
        ...         '_tokenizer': server_tokenizer,
        ...         '_module': server_module,
        ...     }
        ... )
    """
    validated = Payload.model_validate(payload)
    context = SerializationContext(
        memo=validated.memo,
        persistent_objects=persistent_objects,
        server_provided=server_provided,
    )
    return context.deserialize(validated.obj)


__all__ = [
    # Core API
    "serialize",
    "deserialize",
    "Payload",
    # Registration
    "register_serializer",
    "register_persistent",
    "register_by_value",
    # Types
    "SerializationContext",
    "SerializedType",
    "Memo",
    "ReferenceId",
    # Validation
    "lint",
]
