"""
Serialization context and dispatch logic for the pyson library.

This module contains the SerializationContext class which manages the
serialization/deserialization process, including:
- Memoization for handling circular references and shared objects
- Type dispatch to select the appropriate serializer
- By-value vs by-reference decisions for functions and classes
- Optional strict mode with linting for remote execution safety

The dispatch table maps Python types to their serializer classes.

Strict Mode:
    When `strict=True` is passed to SerializationContext, additional
    validation is performed:
    - Forbidden types are rejected with helpful error messages
    - Functions with `nonlocal` closures are rejected
    - Source code is validated for forbidden imports/calls

Server-Provided Attributes:
    When deserializing, pass `server_provided` dict to inject attributes
    that were marked as `_server_provided` on classes and skipped during
    serialization.
"""

from __future__ import annotations

import types

from cloudpickle.cloudpickle import _PICKLE_BY_VALUE_MODULES

from typing import Callable

from pyson.stypes import (
    ClassRefType,
    DynamicClassType,
    DynamicModuleType,
    FunctionRefType,
    FunctionType,
    Memo,
    ModuleRefType,
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
# By-Value Serialization Helpers
# =============================================================================


def _should_serialize_by_value(module: str | None) -> bool:
    """
    Check if an object from this module should be serialized by value.

    Objects are serialized by value (with full source/state) when:
    - module is None (dynamically created)
    - module is "__main__" (interactive/script context)
    - module is registered via cloudpickle.register_pickle_by_value()
    - any parent package of module is registered

    Args:
        module: The __module__ attribute of the object, or None.

    Returns:
        True if the object should be serialized by value.
    """
    if module is None:
        return True
    if module == "__main__":
        return True
    if module in _PICKLE_BY_VALUE_MODULES:
        return True
    # Check parent packages (e.g., "mypackage" for "mypackage.submodule")
    parts = module.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent in _PICKLE_BY_VALUE_MODULES:
            return True
    return False


def _should_serialize_function_by_value(func: types.FunctionType) -> bool:
    """
    Check if a function should be serialized by value (with source code).

    Args:
        func: The function to check.

    Returns:
        True if the function should be serialized with its source code.
    """
    return _should_serialize_by_value(getattr(func, "__module__", None))


def _should_serialize_class_by_value(cls: type) -> bool:
    """
    Check if a class should be serialized by value.

    Built-in types (from 'builtins' module) are never serialized by value.

    Args:
        cls: The class to check.

    Returns:
        True if the class should be serialized with its full definition.
    """
    if cls.__module__ == "builtins":
        return False
    return _should_serialize_by_value(getattr(cls, "__module__", None))


def _should_serialize_module_by_value(mod: types.ModuleType) -> bool:
    """
    Check if a module should be serialized by value.

    Args:
        mod: The module to check.

    Returns:
        True if the module should be serialized with its contents.
    """
    return _should_serialize_by_value(mod.__name__)


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
        lint: Optional LintConfig for validation. If provided, enables
            checking for forbidden types, nonlocal closures, etc.
        server_provided: Dictionary of attribute values to inject during
            deserialization for objects with _server_provided class attribute.

    Example:
        >>> context = SerializationContext()
        >>> ref_id = context.serialize([1, 2, 3])
        >>> result = context.deserialize(ref_id)

    Lint Config Example:
        >>> from pyson.lint import NNSIGHT_CONFIG
        >>> context = SerializationContext(lint=NNSIGHT_CONFIG)
        >>> # Will raise error on forbidden types like pandas DataFrames
        >>> ref_id = context.serialize(pandas_df)  # Raises with helpful message

    Server-Provided Example (for large model instances):
        >>> # Class with server-provided attributes (e.g., large AI models)
        >>> class ModelWrapper:
        ...     _server_provided = frozenset({'_model', '_tokenizer'})
        >>> # These are skipped during serialization (they're huge)
        >>> # During deserialization, inject the server's model instances
        >>> context = SerializationContext(
        ...     memo=serialized_memo,
        ...     server_provided={'_model': server_model, '_tokenizer': server_tokenizer}
        ... )
    """

    def __init__(
        self,
        memo: Memo | None = None,
        persistent_objects: dict[str, object] | None = None,
        lint: 'LintConfig | None' = None,
        server_provided: dict[str, object] | None = None,
    ):
        """
        Initialize the serialization context.

        Args:
            memo: Optional existing memo table (used for deserialization).
            persistent_objects: Optional dict mapping persistent IDs to objects.
                Used during deserialization to resolve PersistentType references.
            lint: Optional LintConfig for validation. If provided, checks for
                forbidden types, nonlocal closures, and dangerous imports/calls.
                Use pyson.lint.NNSIGHT_CONFIG for nnsight-compatible validation.
            server_provided: Optional dict mapping attribute names to values.
                Used during deserialization to inject server-provided attributes
                (like large AI model instances) that were skipped during serialization.
        """
        self.memo = memo or {}
        self.persistent_objects = persistent_objects or {}
        self.lint = lint
        self.server_provided = server_provided or {}
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
        2. In strict mode, checks for forbidden types
        3. Adds a placeholder to detect circular references
        4. Dispatches to the appropriate serializer based on type
        5. Stores the serialized result in the memo

        Args:
            obj: The Python object to serialize.

        Returns:
            The reference ID for this object in the memo table.

        Raises:
            ValueError: In strict mode, if the object is a forbidden type
                or contains forbidden code patterns.
        """
        referenceID = id(obj)

        # Already serialized - return existing reference (handles circular refs)
        if referenceID in self.memo:
            return referenceID

        # If linting is enabled, check for forbidden types early
        if self.lint is not None:
            from pyson.lint import check_forbidden_type
            forbidden_msg = check_forbidden_type(obj, self.lint)
            if forbidden_msg:
                raise ValueError(f"Cannot serialize {type(obj).__name__}:\n{forbidden_msg}")

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
            # Direct type match in dispatch table
            serialized_type = dispatch_table[type(obj)]
        elif isinstance(obj, types.FunctionType):
            # Functions - by value or by reference depending on module
            if _should_serialize_function_by_value(obj):
                serialized_type = FunctionType
            else:
                serialized_type = FunctionRefType
        elif isinstance(obj, types.BuiltinFunctionType):
            # Built-in (C) functions are always by reference
            serialized_type = FunctionRefType
        elif isinstance(obj, (classmethod, staticmethod)):
            # classmethod/staticmethod - by value or by reference
            if _should_serialize_function_by_value(obj.__func__):
                serialized_type = FunctionType
            else:
                serialized_type = FunctionRefType
        elif isinstance(obj, type):
            # Class objects - by value or by reference depending on module
            if _should_serialize_class_by_value(obj):
                serialized_type = DynamicClassType
            else:
                serialized_type = ClassRefType
        elif isinstance(obj, types.ModuleType):
            # Module objects - by value or by reference depending on registration
            if _should_serialize_module_by_value(obj):
                serialized_type = DynamicModuleType
            else:
                serialized_type = ModuleRefType
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
