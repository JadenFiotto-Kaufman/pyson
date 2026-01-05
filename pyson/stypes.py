"""
Serialized type definitions for the pyson library.

This module defines the Pydantic models that represent serialized Python objects.
Each SerializedType subclass handles a specific Python type:

- PrimitiveType: int, float, bool, str, None
- ListType, TupleType, DictType: Built-in collections
- ObjectType: Generic objects with __dict__
- FunctionType: Functions, classmethods, staticmethods (by value)
- ClassRefType: Classes serialized by reference (import path)
- DynamicClassType: Classes serialized by value (full definition)
- MethodType: Bound instance methods
- PropertyType: Property descriptors

Each type provides:
- serialize(): Class method to convert Python object to serialized form
- deserialize(): Instance method to reconstruct the Python object

Server-Provided Attributes:
    Classes can define a `_server_provided` attribute (frozenset of strings)
    to specify attributes that should be skipped during serialization and
    injected by the deserializer. This is useful for remote execution where
    certain resources (models, tokenizers) are provided by the server.

    Example:
        class StandardizedTransformer(LanguageModel):
            _server_provided = frozenset({'_module', '_tokenizer', '_model'})
"""

from __future__ import annotations

from typing import Annotated, Literal, Union, Any, TYPE_CHECKING
from pydantic import BaseModel, Field
from cloudpickle.cloudpickle import (
    _function_getstate,
    _get_cell_contents,
)
import importlib
import inspect
import textwrap
import types

if TYPE_CHECKING:
    from pyson.serialize import SerializationContext
else:
    SerializationContext = Any


# =============================================================================
# Type Aliases
# =============================================================================

# Union of Python primitive types that can be directly JSON-serialized
Primitive = Union[int, float, bool, str, type(None)]

# Reference ID type - uses Python object id() for identity tracking
ReferenceId = int

# Placeholder value used to detect circular references during serialization
Placeholder: ReferenceId = -1


# =============================================================================
# Base Class
# =============================================================================


class SerializedType(BaseModel):
    """
    Abstract base class for all serialized type representations.

    Each subclass must implement:
    - type: A literal string discriminator for Pydantic union discrimination
    - serialize(): Class method to convert a Python object to this type
    - deserialize(): Instance method to reconstruct the Python object

    The 'type' field allows Pydantic to automatically determine which
    subclass to use when parsing JSON.
    """

    type: Literal["type"]

    @classmethod
    def serialize(cls, obj: Any, context: SerializationContext):
        """Serialize a Python object to this type."""
        raise NotImplementedError

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        """Deserialize this type back to a Python object."""
        raise NotImplementedError


# =============================================================================
# Primitive Types
# =============================================================================


class PrimitiveType(SerializedType):
    """
    Serializer for primitive types: int, float, bool, str, None.

    These types are directly JSON-serializable and don't require
    special handling.
    """

    type: Literal["primitive"] = "primitive"
    value: Primitive

    @classmethod
    def serialize(cls, obj: Primitive, context: SerializationContext):
        return cls(value=obj)

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        return self.value


# =============================================================================
# Collection Types
# =============================================================================


class ListType(SerializedType):
    """
    Serializer for Python lists.

    Lists are serialized as a list of reference IDs pointing to
    their elements in the memo table.
    """

    type: Literal["list"] = "list"
    items: list[ReferenceId]

    @classmethod
    def serialize(cls, obj: list, context: SerializationContext):
        return cls(items=[context.serialize(item) for item in obj])

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        # Create empty list first and memoize to handle circular refs
        obj = list()
        context.memo[referenceID] = obj

        # Then populate it
        for item in self.items:
            obj.append(context.deserialize(item))

        return obj


class TupleType(SerializedType):
    """
    Serializer for Python tuples.

    Tuples are immutable, so they're handled specially:
    - Elements are deserialized first
    - Then the tuple is created and memoized
    """

    type: Literal["tuple"] = "tuple"
    items: list[ReferenceId]

    @classmethod
    def serialize(cls, obj: tuple, context: SerializationContext):
        return cls(items=[context.serialize(item) for item in obj])

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        # Deserialize elements first
        obj = [context.deserialize(item) for item in self.items]

        # Check if already memoized (can happen with circular refs)
        if referenceID in context.memo and type(context.memo[referenceID]) == tuple:
            return context.memo[referenceID]

        # Create tuple and memoize
        obj = tuple(obj)
        context.memo[referenceID] = obj
        return obj


class DictType(SerializedType):
    """
    Serializer for Python dictionaries.

    Keys and values are stored as separate lists of reference IDs.
    """

    type: Literal["dict"] = "dict"
    keys: list[ReferenceId]
    values: list[ReferenceId]

    @classmethod
    def serialize(cls, obj: dict, context: SerializationContext):
        return cls(
            keys=[context.serialize(key) for key in obj.keys()],
            values=[context.serialize(value) for value in obj.values()],
        )

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        # Create empty dict and memoize first for circular refs
        obj = dict()
        context.memo[referenceID] = obj

        # Then populate it
        for key, value in zip(self.keys, self.values):
            obj[context.deserialize(key)] = context.deserialize(value)

        return obj


# =============================================================================
# Module Type
# =============================================================================


class ModuleRefType(SerializedType):
    """
    Serializer for module objects by reference.

    Stores only the module name, which is used to re-import the module
    during deserialization. Used for stdlib and installed packages.
    """

    type: Literal["module_ref"] = "module_ref"
    name: str

    @classmethod
    def serialize(cls, obj, context: SerializationContext):
        return cls(name=obj.__name__)

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        mod = importlib.import_module(self.name)
        context.memo[referenceID] = mod
        return mod


class DynamicModuleType(SerializedType):
    """
    Serializer for module objects by value.

    Used for modules from __main__ or registered via register_pickle_by_value.
    Stores the module's attributes so they can be reconstructed.
    """

    type: Literal["dynamic_module"] = "dynamic_module"
    name: str
    doc: str | None
    attrs: dict[str, ReferenceId]

    @classmethod
    def serialize(cls, obj, context: SerializationContext):
        # Get module attributes (skip private/dunder and unserializable)
        attrs = {}
        for key in dir(obj):
            if key.startswith("_"):
                continue
            try:
                value = getattr(obj, key)
                # Skip built-in functions (not serializable)
                if isinstance(value, types.BuiltinFunctionType):
                    continue
                attrs[key] = context.serialize(value)
            except Exception:
                # Skip unserializable attributes
                pass

        return cls(
            name=obj.__name__,
            doc=obj.__doc__,
            attrs=attrs,
        )

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        # Create a new module
        mod = types.ModuleType(self.name, self.doc)

        # Memoize before populating (for circular refs)
        context.memo[referenceID] = mod

        # Deserialize and set attributes
        for key, ref in self.attrs.items():
            value = context.deserialize(ref)
            setattr(mod, key, value)

        return mod


# =============================================================================
# Class Types
# =============================================================================


class ClassRefType(SerializedType):
    """
    Serializer for classes by reference.

    Used for classes that can be imported (stdlib, installed packages).
    Stores only the module path and class name.
    """

    type: Literal["class_ref"] = "class_ref"
    name: str
    module: str

    @classmethod
    def serialize(cls, obj: type, context: SerializationContext):
        return cls(name=obj.__name__, module=obj.__module__)

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        # Import the module and get the class
        mod = importlib.import_module(self.module)
        typ = getattr(mod, self.name)
        context.memo[referenceID] = typ
        return typ


def _extract_class_dict(cls: type) -> dict:
    """
    Extract a class's __dict__ excluding inherited attributes.

    Filters out:
    - Attributes inherited from base classes (same object identity)
    - Auto-generated attributes (__dict__, __weakref__)

    Args:
        cls: The class to extract attributes from.

    Returns:
        Dictionary of the class's own (non-inherited) attributes.
    """
    clsdict = {k: cls.__dict__[k] for k in cls.__dict__}

    # Build inherited dict from bases
    if len(cls.__bases__) == 1:
        inherited_dict = cls.__bases__[0].__dict__
    else:
        inherited_dict = {}
        for base in reversed(cls.__bases__):
            inherited_dict.update(base.__dict__)

    # Remove inherited attributes (same object identity)
    to_remove = []
    for name, value in clsdict.items():
        try:
            base_value = inherited_dict[name]
            if value is base_value:
                to_remove.append(name)
        except KeyError:
            pass
    for name in to_remove:
        clsdict.pop(name)

    # Remove unpicklable/auto-generated attributes
    clsdict.pop("__dict__", None)
    clsdict.pop("__weakref__", None)

    return clsdict


class DynamicClassType(SerializedType):
    """
    Serializer for classes by value (dynamic classes).

    Used for classes from __main__ or registered modules.
    Stores the full class definition including:
    - Base classes
    - Class attributes and methods
    - Metadata (name, qualname, module, doc)
    """

    type: Literal["dynamic_class"] = "dynamic_class"
    name: str
    qualname: str
    module: str
    doc: str | None
    bases: list[ReferenceId]
    class_dict: dict[str, ReferenceId]

    @classmethod
    def serialize(cls, obj: type, context: SerializationContext):
        # Serialize base classes
        serialized_bases = [context.serialize(base) for base in obj.__bases__]

        # Extract and serialize class dict (excluding inherited attrs)
        clsdict = _extract_class_dict(obj)

        serialized_dict = {}
        for k, v in clsdict.items():
            # Skip special attributes that will be set automatically
            if k in ("__module__", "__qualname__", "__doc__"):
                continue
            try:
                serialized_dict[k] = context.serialize(v)
            except Exception:
                # Skip unserializable attributes (like some descriptors)
                pass

        return cls(
            name=obj.__name__,
            qualname=obj.__qualname__,
            module=obj.__module__,
            doc=obj.__doc__,
            bases=serialized_bases,
            class_dict=serialized_dict,
        )

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        # Deserialize base classes first
        bases = tuple(context.deserialize(base_ref) for base_ref in self.bases)

        # Create skeleton class and memoize before populating (for circular refs)
        new_class = type(self.name, bases, {"__module__": self.module})
        context.memo[referenceID] = new_class

        # Deserialize and set class dict attributes
        for k, v_ref in self.class_dict.items():
            v = context.deserialize(v_ref)
            try:
                setattr(new_class, k, v)
            except (TypeError, AttributeError):
                # Some attributes can't be set after class creation
                pass

        # Set metadata
        new_class.__qualname__ = self.qualname
        new_class.__doc__ = self.doc

        return new_class


# =============================================================================
# Object Type
# =============================================================================


def _get_server_provided_attrs(cls: type) -> frozenset:
    """
    Get server-provided attribute names from a class hierarchy.

    Walks the MRO to collect all _server_provided attributes.

    Args:
        cls: The class to check.

    Returns:
        Frozenset of attribute names that should be skipped during serialization.
    """
    result = set()
    for klass in cls.__mro__:
        server_provided = getattr(klass, '_server_provided', None)
        if server_provided:
            result.update(server_provided)
    return frozenset(result)


class ObjectType(SerializedType):
    """
    Serializer for generic Python objects.

    Handles objects by serializing their class and state (__dict__ or
    __getstate__). Used as the fallback for objects not matching other types.

    Server-Provided Attributes:
        If the object's class defines `_server_provided` as a frozenset of
        attribute names, those attributes will be skipped during serialization.
        They should be injected during deserialization via `server_provided`
        parameter to the deserialize function.

        Example:
            class MyModel(BaseModel):
                _server_provided = frozenset({'_internal_cache', '_connection'})
    """

    type: Literal["object"] = "object"
    cls: ReferenceId
    state: dict[str, ReferenceId]
    server_provided_attrs: list[str] | None = None  # Track which attrs were skipped

    @classmethod
    def serialize(cls, obj: object, context: SerializationContext):
        # Serialize the object's class
        obj_cls = context.serialize(obj.__class__)

        # Get server-provided attribute names from class hierarchy
        server_provided = _get_server_provided_attrs(type(obj))

        # Get state using __getstate__ or __dict__
        if hasattr(obj, "__getstate__"):
            try:
                state_dict = obj.__getstate__()
            except TypeError:
                state_dict = getattr(obj, "__dict__", {})
        else:
            state_dict = getattr(obj, "__dict__", {})

        # Handle None or non-dict state
        if state_dict is None:
            state_dict = {}
        elif not isinstance(state_dict, dict):
            state_dict = {"__state__": state_dict}

        # Filter out server-provided attributes
        if server_provided and isinstance(state_dict, dict):
            skipped = [k for k in state_dict if k in server_provided]
            state_dict = {k: v for k, v in state_dict.items() if k not in server_provided}
        else:
            skipped = []

        # Recursively serialize all values in state_dict
        serialized_state = {
            key: context.serialize(value) for key, value in state_dict.items()
        }

        return cls(
            cls=obj_cls,
            state=serialized_state,
            server_provided_attrs=skipped if skipped else None,
        )

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        # Deserialize the class
        cls = context.deserialize(self.cls)

        # Create blank instance without calling __init__
        obj = cls.__new__(cls)

        # Memoize before setting state (for circular refs)
        context.memo[referenceID] = obj

        # Deserialize state
        state = {key: context.deserialize(value) for key, value in self.state.items()}

        # Inject server-provided attributes if available
        if self.server_provided_attrs and hasattr(context, 'server_provided'):
            for attr_name in self.server_provided_attrs:
                if attr_name in context.server_provided:
                    state[attr_name] = context.server_provided[attr_name]

        # Apply state using __setstate__ or __dict__
        if hasattr(obj, "__setstate__"):
            obj.__setstate__(state)
        else:
            if hasattr(obj, "__dict__"):
                obj.__dict__.update(state)
            else:
                for k, v in state.items():
                    setattr(obj, k, v)

        return obj


# =============================================================================
# Function Types
# =============================================================================


class FunctionRefType(SerializedType):
    """
    Serializer for functions by reference.

    Used for functions from installed packages that can be imported.
    Stores only the module path and function name.
    """

    type: Literal["function_ref"] = "function_ref"
    name: str
    module: str
    qualname: str

    @classmethod
    def serialize(cls, obj, context: SerializationContext):
        # Handle classmethod/staticmethod wrappers
        if isinstance(obj, classmethod):
            func = obj.__func__
        elif isinstance(obj, staticmethod):
            func = obj.__func__
        else:
            func = obj

        return cls(
            name=func.__name__,
            module=func.__module__,
            qualname=func.__qualname__,
        )

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        # Import the module and traverse to get the function
        mod = importlib.import_module(self.module)

        # Handle qualified names like "ClassName.method_name"
        obj = mod
        for attr in self.qualname.split("."):
            obj = getattr(obj, attr)

        context.memo[referenceID] = obj
        return obj


class FunctionType(SerializedType):
    """
    Serializer for functions, classmethods, and staticmethods.

    Stores the function's source code and reconstructs it by compiling
    the source. Also captures:
    - Globals (only those referenced by the function)
    - Closures (free variables from enclosing scopes)
    - Defaults, kwdefaults, annotations
    - Custom attributes set on the function

    The 'kind' field distinguishes regular functions from classmethod
    and staticmethod wrappers.
    """

    type: Literal["function"] = "function"
    kind: Literal["function", "classmethod", "staticmethod"] = "function"
    source: str
    name: str
    qualname: str
    module: str
    doc: str | None
    annotations: dict[str, ReferenceId]
    defaults: list[ReferenceId] | None
    kwdefaults: dict[str, ReferenceId] | None
    globals: dict[str, ReferenceId]
    closure: list[ReferenceId] | None
    closure_names: list[str] | None  # Variable names from co_freevars
    func_dict: dict[str, ReferenceId]

    @classmethod
    def serialize(
        cls,
        obj: types.FunctionType | classmethod | staticmethod,
        context: SerializationContext,
    ):
        # Extract the underlying function for classmethod/staticmethod
        if isinstance(obj, (classmethod, staticmethod)):
            func = obj.__func__
            kind = "classmethod" if isinstance(obj, classmethod) else "staticmethod"
        else:
            kind = "function"
            func = obj

        # Get source code
        source = inspect.getsource(func)

        # Check for nonlocal closures if linting is enabled
        if context.lint is not None and context.lint.reject_nonlocal:
            from pyson.lint import check_nonlocal
            nonlocal_names = check_nonlocal(func)
            if nonlocal_names:
                names_str = ', '.join(sorted(nonlocal_names))
                raise ValueError(
                    f"Function '{func.__name__}' uses 'nonlocal {names_str}'.\n"
                    f"Functions with nonlocal cannot be serialized because multiple closures\n"
                    f"sharing the same cell would lose shared state.\n"
                    f"\n"
                    f"Refactor to use a class:\n"
                    f"    class Counter:\n"
                    f"        def __init__(self): self.count = 0\n"
                    f"        def increment(self): self.count += 1; return self.count"
                )

        # Use cloudpickle's helper to get function state
        state, slotstate = _function_getstate(func)

        # Serialize only the globals actually used by the function
        serialized_globals = {
            k: context.serialize(v) for k, v in slotstate["__globals__"].items()
        }

        # Serialize defaults
        defaults = slotstate["__defaults__"]
        serialized_defaults = (
            [context.serialize(d) for d in defaults] if defaults else None
        )

        # Serialize keyword-only defaults
        kwdefaults = slotstate["__kwdefaults__"]
        serialized_kwdefaults = (
            {k: context.serialize(v) for k, v in kwdefaults.items()}
            if kwdefaults
            else None
        )

        # Serialize type annotations
        annotations = slotstate["__annotations__"]
        serialized_annotations = {
            k: context.serialize(v) for k, v in annotations.items()
        }

        # Serialize closure (free variables)
        closure = slotstate["__closure__"]
        closure_names = (
            list(func.__code__.co_freevars) if func.__code__.co_freevars else None
        )
        if closure:
            serialized_closure = [
                context.serialize(_get_cell_contents(cell)) for cell in closure
            ]
        else:
            serialized_closure = None

        # Serialize custom attributes on the function
        serialized_func_dict = {k: context.serialize(v) for k, v in state.items()}

        return cls(
            kind=kind,
            source=source,
            name=slotstate["__name__"],
            qualname=slotstate["__qualname__"],
            module=slotstate["__module__"],
            doc=slotstate["__doc__"],
            annotations=serialized_annotations,
            defaults=serialized_defaults,
            kwdefaults=serialized_kwdefaults,
            globals=serialized_globals,
            closure=serialized_closure,
            closure_names=closure_names,
            func_dict=serialized_func_dict,
        )

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        # Dedent the source (in case it was defined inside a class/function)
        source = textwrap.dedent(self.source)

        # Deserialize globals
        func_globals = {"__builtins__": __builtins__}
        for k, v in self.globals.items():
            func_globals[k] = context.deserialize(v)

        # Deserialize closure values
        closure_values = None
        if self.closure and self.closure_names:
            closure_values = [context.deserialize(v) for v in self.closure]

        # If there's a closure, wrap the function in a factory to recreate
        # the closure context
        if closure_values and self.closure_names:
            # Build factory: def _seri_factory_(x, y): <source>; return name
            closure_params = ", ".join(self.closure_names)
            factory_source = f"def _seri_factory_({closure_params}):\n"
            indented_source = textwrap.indent(source, "    ")
            factory_source += indented_source + "\n"
            factory_source += f"    return {self.name}\n"

            # Compile, execute, and call factory
            factory_code = compile(factory_source, f"<seri:{self.name}>", "exec")
            exec(factory_code, func_globals)
            factory = func_globals["_seri_factory_"]
            func = factory(*closure_values)
        else:
            # No closure - compile and extract the function's code object
            module_code = compile(source, f"<seri:{self.name}>", "exec")

            # Find the function's code object in module constants
            func_code = None
            for const in module_code.co_consts:
                if isinstance(const, types.CodeType) and const.co_name == self.name:
                    func_code = const
                    break

            if func_code is None:
                raise ValueError(
                    f"Could not find function '{self.name}' in compiled source"
                )

            # Deserialize defaults
            defaults = (
                tuple(context.deserialize(d) for d in self.defaults)
                if self.defaults
                else None
            )

            # Create the function
            func = types.FunctionType(
                func_code,
                func_globals,
                self.name,
                defaults,
                None,  # No closure
            )

        # Set defaults if we used the factory path
        if closure_values and self.closure_names and self.defaults:
            func.__defaults__ = tuple(context.deserialize(d) for d in self.defaults)

        # Set keyword-only defaults
        if self.kwdefaults:
            func.__kwdefaults__ = {
                k: context.deserialize(v) for k, v in self.kwdefaults.items()
            }

        # Set annotations
        if self.annotations:
            func.__annotations__ = {
                k: context.deserialize(v) for k, v in self.annotations.items()
            }

        # Set metadata
        func.__module__ = self.module
        func.__doc__ = self.doc
        func.__qualname__ = self.qualname

        # Set custom attributes
        for k, v in self.func_dict.items():
            setattr(func, k, context.deserialize(v))

        # Wrap in classmethod/staticmethod if needed
        if self.kind == "classmethod":
            result = classmethod(func)
        elif self.kind == "staticmethod":
            result = staticmethod(func)
        else:
            result = func

        # Memoize the final result
        context.memo[referenceID] = result
        return result


# =============================================================================
# Method and Property Types
# =============================================================================


class MethodType(SerializedType):
    """
    Serializer for bound instance methods.

    A bound method has:
    - __func__: The underlying function
    - __self__: The instance it's bound to

    Both are serialized and recombined on deserialization.
    """

    type: Literal["method"] = "method"
    func: ReferenceId
    self_: ReferenceId

    @classmethod
    def serialize(cls, obj: types.MethodType, context: SerializationContext):
        return cls(
            func=context.serialize(obj.__func__),
            self_=context.serialize(obj.__self__),
        )

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        func = context.deserialize(self.func)
        self_obj = context.deserialize(self.self_)
        result = types.MethodType(func, self_obj)
        context.memo[referenceID] = result
        return result


class PropertyType(SerializedType):
    """
    Serializer for property descriptors.

    Stores the getter, setter, and deleter functions (if present)
    along with the docstring.
    """

    type: Literal["property"] = "property"
    fget: ReferenceId | None
    fset: ReferenceId | None
    fdel: ReferenceId | None
    doc: str | None

    @classmethod
    def serialize(cls, obj: property, context: SerializationContext):
        return cls(
            fget=context.serialize(obj.fget) if obj.fget else None,
            fset=context.serialize(obj.fset) if obj.fset else None,
            fdel=context.serialize(obj.fdel) if obj.fdel else None,
            doc=obj.__doc__,
        )

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        fget = context.deserialize(self.fget) if self.fget else None
        fset = context.deserialize(self.fset) if self.fset else None
        fdel = context.deserialize(self.fdel) if self.fdel else None
        result = property(fget, fset, fdel, self.doc)
        context.memo[referenceID] = result
        return result


# =============================================================================
# Persistent Type
# =============================================================================


class PersistentType(SerializedType):
    """
    Serializer for objects with persistent external storage.

    Used when an object should be serialized as just an ID that can be
    used to retrieve it later (e.g., from a database, file system, or cache).

    During serialization, the persistent_table's dump function is called
    to get the persistent ID. During deserialization, the ID is looked up
    in context.persistent_objects.
    """

    type: Literal["persistent"] = "persistent"
    persistent_id: str

    @classmethod
    def serialize(cls, obj: Any, context: SerializationContext, persistent_id: str):
        """Serialize with a pre-computed persistent ID."""
        return cls(persistent_id=persistent_id)

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        # Look up the object by persistent ID
        if self.persistent_id not in context.persistent_objects:
            raise ValueError(
                f"Persistent ID '{self.persistent_id}' not found in persistent_objects. "
                f"Available IDs: {list(context.persistent_objects.keys())}"
            )

        result = context.persistent_objects[self.persistent_id]
        context.memo[referenceID] = result
        return result


# =============================================================================
# Type Registry
# =============================================================================

# Registry mapping type discriminator strings to SerializedType classes.
# Used for deserialization (JSON parsing).
_TYPE_REGISTRY: dict[str, type[SerializedType]] = {}


# Note: ObjectType, FunctionType, ClassRefType, DynamicClassType, PersistentType, ModuleRefType
# are not in dispatch_table - they're selected by logic in SerializationContext.serialize()
# But we still register them in _TYPE_REGISTRY for deserialization:
for _cls in [
    ObjectType,
    FunctionType,
    FunctionRefType,
    ClassRefType,
    DynamicClassType,
    PersistentType,
    ModuleRefType,
    DynamicModuleType,
]:
    type_field = _cls.model_fields.get("type")
    if type_field and type_field.default:
        _TYPE_REGISTRY[type_field.default] = _cls


def _validate_serialized_value(value: Any) -> SerializedType:
    """
    Validate and parse a serialized value using the type registry.

    This is the custom validator that replaces Pydantic's static
    discriminated union, enabling dynamic type registration.

    Args:
        value: The raw value (dict from JSON or already a SerializedType).

    Returns:
        The parsed SerializedType instance.

    Raises:
        ValueError: If the type discriminator is unknown.
    """
    # Already parsed
    if isinstance(value, SerializedType):
        return value

    # Must be a dict with a 'type' field
    if not isinstance(value, dict) or "type" not in value:
        raise ValueError(f"Expected dict with 'type' field, got {type(value)}")

    type_name = value["type"]
    if type_name not in _TYPE_REGISTRY:
        raise ValueError(
            f"Unknown serialized type '{type_name}'. "
            f"Available types: {list(_TYPE_REGISTRY.keys())}"
        )

    # Parse using the registered class
    return _TYPE_REGISTRY[type_name].model_validate(value)


# =============================================================================
# Discriminated Union (with dynamic support)
# =============================================================================

# Use Annotated with BeforeValidator for dynamic dispatch
from pydantic import BeforeValidator, PlainSerializer


def _serialize_value(value: SerializedType) -> dict:
    """
    Serialize a SerializedType to a dict for JSON output.

    This ensures that subclass-specific fields are included in the output,
    not just the base class fields.
    """
    return value.model_dump()


# SerializedValue uses custom validator (parsing) and serializer (dumping)
SerializedValue = Annotated[
    SerializedType,
    BeforeValidator(_validate_serialized_value),
    PlainSerializer(_serialize_value, return_type=dict),
]

# Type alias for the memo table
Memo = dict[int, SerializedValue]
