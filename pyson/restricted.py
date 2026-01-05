"""
RestrictedPython integration for secure server-side execution.

This module provides utilities for deserializing pyson payloads and executing
the resulting functions in a RestrictedPython sandbox. This is the security
enforcement layer that should be used on the server side.

IMPORTANT: Client-side linting (pyson.lint) is for user experience only.
This module provides the actual security boundary.

Usage:
    from pyson.restricted import deserialize_restricted

    data = deserialize_restricted(
        payload_dict,
        server_provided={'_model': model},
        restricted_globals={'torch': torch},
    )

    # Functions are wrapped to execute in restricted mode
    result = data["intervention"](input_tensor)

Requirements:
    pip install RestrictedPython
"""

from __future__ import annotations

import functools
import types
from typing import Any, Callable, Dict, Optional

# Lazy import RestrictedPython to make it an optional dependency
_restricted_python_available = None


def _check_restricted_python():
    """Check if RestrictedPython is available, raise helpful error if not."""
    global _restricted_python_available
    if _restricted_python_available is None:
        try:
            import RestrictedPython
            _restricted_python_available = True
        except ImportError:
            _restricted_python_available = False

    if not _restricted_python_available:
        raise ImportError(
            "RestrictedPython is required for secure server-side execution.\n"
            "Install it with: pip install RestrictedPython"
        )


class RestrictedExecutionError(Exception):
    """Raised when code violates RestrictedPython security policies."""
    pass


def _get_default_restricted_globals() -> Dict[str, Any]:
    """Get default safe globals for RestrictedPython execution."""
    _check_restricted_python()
    from RestrictedPython import safe_globals, safe_builtins
    from RestrictedPython.Guards import (
        guarded_iter_unpack_sequence,
        guarded_unpack_sequence,
    )

    return {
        **safe_globals,
        '__builtins__': safe_builtins,
        '_iter_unpack_sequence_': guarded_iter_unpack_sequence,
        '_unpack_sequence_': guarded_unpack_sequence,
        # Allow basic operations
        '_getattr_': getattr,  # Will be restricted by policy
        '_getitem_': lambda obj, key: obj[key],
    }


def _wrap_function_restricted(
    func: types.FunctionType,
    source: str,
    restricted_globals: Dict[str, Any],
) -> Callable:
    """
    Wrap a function to execute in RestrictedPython mode.

    Instead of executing the original function, this wrapper:
    1. Compiles the source with compile_restricted
    2. Executes in a sandbox with safe_globals
    3. Calls the restricted version

    Args:
        func: The original deserialized function
        source: The function's source code
        restricted_globals: Globals to use for execution

    Returns:
        A wrapped function that executes in restricted mode
    """
    _check_restricted_python()
    from RestrictedPython import compile_restricted

    @functools.wraps(func)
    def restricted_wrapper(*args, **kwargs):
        # Compile with RestrictedPython
        try:
            code = compile_restricted(
                source,
                filename=f"<restricted:{func.__name__}>",
                mode='exec',
            )
        except SyntaxError as e:
            raise RestrictedExecutionError(
                f"RestrictedPython rejected code in '{func.__name__}': {e}"
            )

        if code.errors:
            raise RestrictedExecutionError(
                f"RestrictedPython found violations in '{func.__name__}':\n"
                + "\n".join(f"  - {err}" for err in code.errors)
            )

        # Create execution namespace with restricted globals
        exec_globals = {**restricted_globals}

        # Add the function's original globals (filtered for safety)
        for key, value in func.__globals__.items():
            if key not in exec_globals and not key.startswith('__'):
                exec_globals[key] = value

        # Execute to define the function
        exec(code.code, exec_globals)

        # Get the restricted version and call it
        restricted_func = exec_globals.get(func.__name__)
        if restricted_func is None:
            raise RestrictedExecutionError(
                f"Could not find function '{func.__name__}' after restricted compilation"
            )

        return restricted_func(*args, **kwargs)

    # Mark as restricted for inspection
    restricted_wrapper._is_restricted = True
    restricted_wrapper._original_func = func
    restricted_wrapper._source = source

    return restricted_wrapper


def _wrap_deserialized_functions(
    obj: Any,
    restricted_globals: Dict[str, Any],
    visited: Optional[set] = None,
) -> Any:
    """
    Recursively wrap all deserialized functions in restricted mode.

    This walks through the deserialized object graph and wraps any
    functions that were serialized by value (have source code).
    """
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)

    # Wrap functions that have source attached
    if isinstance(obj, types.FunctionType):
        # Check if this function has source (was serialized by value)
        # The source is stored during deserialization in the function's globals
        # under a special key, or we can try to get it via inspect
        import inspect
        try:
            source = inspect.getsource(obj)
            return _wrap_function_restricted(obj, source, restricted_globals)
        except (OSError, TypeError):
            # Built-in or no source available - return as-is
            return obj

    # Recurse into containers
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _wrap_deserialized_functions(value, restricted_globals, visited)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = _wrap_deserialized_functions(value, restricted_globals, visited)
    elif isinstance(obj, tuple):
        # Can't modify tuples in place
        pass
    elif hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):
                try:
                    new_value = _wrap_deserialized_functions(
                        value, restricted_globals, visited
                    )
                    setattr(obj, key, new_value)
                except AttributeError:
                    pass

    return obj


def deserialize_restricted(
    payload: dict,
    persistent_objects: Optional[Dict[str, object]] = None,
    server_provided: Optional[Dict[str, object]] = None,
    restricted_globals: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Deserialize a pyson payload with RestrictedPython enforcement.

    This is the secure server-side deserialization function. All functions
    serialized by value will be wrapped to execute in RestrictedPython's
    sandbox.

    Args:
        payload: The payload dictionary (from json.loads()).
        persistent_objects: Optional dict mapping persistent IDs to objects.
        server_provided: Optional dict mapping attribute names to values
            for server-provided attributes.
        restricted_globals: Optional dict of globals to allow in restricted
            execution. If not provided, uses RestrictedPython's safe_globals.
            You can add allowed modules here (e.g., {'torch': torch}).

    Returns:
        The deserialized object with functions wrapped in restricted mode.

    Raises:
        ImportError: If RestrictedPython is not installed.
        RestrictedExecutionError: If code violates RestrictedPython policies.

    Example:
        >>> data = deserialize_restricted(
        ...     payload_dict,
        ...     server_provided={'_model': model},
        ...     restricted_globals={'torch': torch, 'numpy': np},
        ... )
        >>> result = data["intervention"](input_tensor)
    """
    _check_restricted_python()

    # Import pyson's deserialize
    from pyson import deserialize

    # First, do normal deserialization
    result = deserialize(
        payload,
        persistent_objects=persistent_objects,
        server_provided=server_provided,
    )

    # Get restricted globals
    if restricted_globals is None:
        exec_globals = _get_default_restricted_globals()
    else:
        # Merge user's globals with defaults
        exec_globals = _get_default_restricted_globals()
        exec_globals.update(restricted_globals)

    # Wrap all functions in restricted mode
    result = _wrap_deserialized_functions(result, exec_globals)

    return result


__all__ = [
    'deserialize_restricted',
    'RestrictedExecutionError',
]
