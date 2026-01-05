"""
Serialization context and dispatch logic for the pyson library.

This module contains the SerializationContext class which manages the
serialization/deserialization process, including:
- Memoization for handling circular references and shared objects
- Type dispatch to select the appropriate serializer
- By-value vs by-reference decisions for functions and classes

The dispatch table maps Python types to their serializer classes.
"""

from __future__ import annotations

import types

from cloudpickle.cloudpickle import _PICKLE_BY_VALUE_MODULES

from typing import Callable

from pyson.types import (
    ClassType,
    FunctionType,
    Memo,
    ModuleType,
    ObjectType,
    PersistentType,
    Placeholder,
    ReferenceId,
    SerializedType,
    PrimitiveType,
    ListType,
    TupleType,
    DictType,
    PropertyType,
    MethodType,
    _TYPE_REGISTRY,
)


def register_module_by_value(module: str) -> None:
    """
    Register a module to be serialized by value.
    """
    _PICKLE_BY_VALUE_MODULES.add(module)


# Dispatch table mapping Python types to their serializer classes.
# Used for serialization.

dispatch_table: dict[type, type[SerializedType]] = {}


def register_serializer(
    python_type: type | tuple[type, ...],
    serializer_class: type[SerializedType],
) -> None:
    """
    Register a serializer for one or more Python types.

    This adds the serializer to both:
    - The type registry (for deserialization/JSON parsing)
    - The dispatch table (for serialization)

    The serializer_class must:
    - Inherit from SerializedType
    - Have a unique 'type' literal field
    - Implement serialize() and deserialize()

    Args:
        python_type: The Python type(s) to handle. Can be a single type
            or a tuple of types that all use the same serializer.
        serializer_class: The SerializedType subclass to use.

    Example:
        >>> class MyType(SerializedType):
        ...     type: Literal["my_type"] = "my_type"
        ...     data: str
        ...
        ...     @classmethod
        ...     def serialize(cls, obj, context):
        ...         return cls(data=str(obj))
        ...
        ...     def deserialize(self, referenceID, context):
        ...         result = MyClass(self.data)
        ...         context.memo[referenceID] = result
        ...         return result
        ...
        >>> register_serializer(MyClass, MyType)
    """
    # Register in type registry (for deserialization)
    type_field = serializer_class.model_fields.get("type")
    if type_field and type_field.default:
        _TYPE_REGISTRY[type_field.default] = serializer_class

    # Register in dispatch table (for serialization)
    if isinstance(python_type, tuple):
        for t in python_type:
            dispatch_table[t] = serializer_class
    else:
        dispatch_table[python_type] = serializer_class


# Register all built-in serializers
register_serializer((int, float, bool, str, type(None)), PrimitiveType)
register_serializer(list, ListType)
register_serializer(tuple, TupleType)
register_serializer(dict, DictType)
register_serializer(property, PropertyType)
register_serializer(types.MethodType, MethodType)
# Dispatcher types that handle by-value vs by-ref decisions
register_serializer(types.FunctionType, FunctionType)
register_serializer(types.BuiltinFunctionType, FunctionType)
register_serializer(classmethod, FunctionType)
register_serializer(staticmethod, FunctionType)
register_serializer(type, ClassType)
register_serializer(types.ModuleType, ModuleType)


# =============================================================================
# Persistent Serialization
# =============================================================================

# Maps types to dump functions: dump(obj) -> str (persistent ID)
# Checked BEFORE dispatch_table during serialization
persistent_table: dict[type, Callable] = {}


def register_persistent(
    python_type: type,
    dump: Callable[[object], str],
) -> None:
    """
    Register a persistent serialization handler for a type.

    Persistent serialization stores only an ID string, not the full object.
    This is useful for:
    - Large objects stored externally (files, databases, caches)
    - Objects that should maintain identity across serialization boundaries
    - Singleton-like objects that should be looked up, not recreated

    The dump function is called during serialization to get the persistent ID.
    During deserialization, pass a `persistent_objects` dict to the deserialize
    function that maps persistent IDs to objects.

    Note: persistent_table is checked BEFORE dispatch_table, so registering
    a type here takes priority over any serializer in dispatch_table.

    Args:
        python_type: The Python type to handle.
        dump: Function that takes an object and returns its persistent ID string.

    Example:
        >>> # Register a type for persistent serialization
        >>> register_persistent(LargeModel, lambda m: m.path)
        >>>
        >>> # Serialize
        >>> model = LargeModel("/models/gpt.pt")
        >>> payload = serialize(model)
        >>>
        >>> # Deserialize with persistent_objects mapping
        >>> result = deserialize(payload, persistent_objects={"/models/gpt.pt": model})
    """
    persistent_table[python_type] = dump


# =============================================================================
# Serialization Context
# =============================================================================


class SerializationContext:
    """
    Manages the serialization and deserialization process.

    The context maintains a memo table that:
    - Tracks serialized objects by their id() to handle circular references
    - Preserves object identity (same object = same reference ID)
    - Stores serialized representations for later deserialization

    Attributes:
        memo: Dictionary mapping reference IDs to serialized objects.
        persistent_objects: Dictionary mapping persistent IDs to objects
            (used during deserialization for PersistentType).

    Example:
        >>> context = SerializationContext()
        >>> ref_id = context.serialize([1, 2, 3])
        >>> result = context.deserialize(ref_id)
    """

    def __init__(
        self,
        memo: Memo | None = None,
        persistent_objects: dict[str, object] | None = None,
    ):
        """
        Initialize the serialization context.

        Args:
            memo: Optional existing memo table (used for deserialization).
            persistent_objects: Optional dict mapping persistent IDs to objects.
                Used during deserialization to resolve PersistentType references.
        """
        self.memo = memo or {}
        self.persistent_objects = persistent_objects or {}
        # Keep references to all serialized objects to prevent id() reuse.
        # Python can reuse memory addresses for garbage-collected objects,
        # which would cause incorrect memoization if a new object gets the
        # same id() as a previously serialized (and freed) object.
        self._refs: list = []

    def serialize(self, obj) -> ReferenceId:
        """
        Serialize an object and return its reference ID.

        This method:
        1. Checks if the object is already memoized (returns existing ref)
        2. Adds a placeholder to detect circular references
        3. Dispatches to the appropriate serializer based on type
        4. Stores the serialized result in the memo

        Args:
            obj: The Python object to serialize.

        Returns:
            The reference ID for this object in the memo table.
        """
        referenceID = id(obj)

        # Already serialized - return existing reference (handles circular refs)
        if referenceID in self.memo:
            return referenceID

        # Keep object alive to prevent id() reuse during serialization
        self._refs.append(obj)

        # Add placeholder to detect cycles during serialization
        self.memo[referenceID] = Placeholder

        # Check for persistent serialization FIRST (takes priority over dispatch_table)
        # 1. If object has _persistent_id attribute, use it directly
        # 2. If type is in persistent_table, call the dump function
        persistent_id = None
        if hasattr(obj, "_persistent_id"):
            persistent_id = obj._persistent_id
        elif type(obj) in persistent_table:
            dump_fn = persistent_table[type(obj)]
            persistent_id = dump_fn(obj)

        if persistent_id is not None:
            serialized_obj = PersistentType.serialize(obj, self, persistent_id)
            self.memo[referenceID] = serialized_obj
            return referenceID

        # Dispatch to appropriate serializer
        if type(obj) in dispatch_table:
            serialized_type = dispatch_table[type(obj)]
        else:
            # Default: serialize as a generic object with __dict__
            serialized_type = ObjectType

        # Perform serialization and store result
        serialized_obj = serialized_type.serialize(obj, self)
        self.memo[referenceID] = serialized_obj

        return referenceID

    def deserialize(self, referenceID: ReferenceId):
        """
        Deserialize an object from its reference ID.

        If the object has already been deserialized (memo contains the
        actual object, not a SerializedType), returns it directly.
        This preserves object identity across multiple references.

        Args:
            referenceID: The reference ID to look up.

        Returns:
            The deserialized Python object.
        """
        obj = self.memo[referenceID]

        # Already deserialized - return directly (preserves identity)
        if not isinstance(obj, SerializedType):
            return obj

        # Deserialize and the type's deserialize() will update memo
        return obj.deserialize(referenceID, self)

