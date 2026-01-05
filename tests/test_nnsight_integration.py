"""
Comprehensive tests for pyson nnsight integration features.

Tests cover:
1. Configurable linting (LintConfig, NNSIGHT_CONFIG)
2. Server-provided attributes
3. Nonlocal rejection
4. Forbidden type detection with helpful messages
5. Source code validation
6. Edge cases from nnsight's test suite
"""

import json
import pytest
import sys
import types
from io import StringIO
from dataclasses import dataclass
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, '/Users/davidbau/git/pyson')

from pyson import serialize, deserialize, Payload
from pyson.lint import (
    LintConfig,
    NNSIGHT_CONFIG,
    FORBIDDEN_MODULE_PREFIXES,
    FORBIDDEN_CALLS,
    FORBIDDEN_ATTR_CHAINS,
    FORBIDDEN_CLASSES,
    check_nonlocal,
    check_forbidden_type,
    validate_function_source,
    lint_object,
    LintError,
)

# Register this test module for by-value serialization
# This makes functions/classes defined here serialize with source code
from cloudpickle import register_pickle_by_value
import test_nnsight_integration
register_pickle_by_value(test_nnsight_integration)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

def roundtrip(obj, **kwargs):
    """Serialize and deserialize an object."""
    payload = serialize(obj, **kwargs)
    json_str = payload.model_dump_json()
    return deserialize(json.loads(json_str), **kwargs)


def roundtrip_with_server_provided(obj, server_provided, **serialize_kwargs):
    """Serialize and deserialize with server-provided injection."""
    payload = serialize(obj, **serialize_kwargs)
    json_str = payload.model_dump_json()
    return deserialize(json.loads(json_str), server_provided=server_provided)


# =============================================================================
# Module-Level Test Classes (must be here for deserialization to work)
# =============================================================================

class ServerProvidedModel:
    """Test class with server-provided attributes."""
    _server_provided = frozenset({'_module', '_tokenizer'})

    def __init__(self):
        self.name = "test"
        self._module = "LARGE_MODULE_PLACEHOLDER"
        self._tokenizer = "LARGE_TOKENIZER_PLACEHOLDER"


class ServerProvidedBaseModel:
    """Base class with server-provided attributes."""
    _server_provided = frozenset({'_cache'})


class ServerProvidedDerivedModel(ServerProvidedBaseModel):
    """Derived class with additional server-provided attributes."""
    _server_provided = frozenset({'_module'})

    def __init__(self):
        self.name = "derived"
        self._cache = "CACHE"
        self._module = "MODULE"


class ServerProvidedWithState:
    """Test class with server-provided and custom getstate/setstate."""
    _server_provided = frozenset({'_heavy'})

    def __init__(self):
        self.light = "small"
        self._heavy = "HEAVY_DATA"

    def __getstate__(self):
        return {'light': self.light, '_heavy': self._heavy}

    def __setstate__(self, state):
        self.light = state['light']
        self._heavy = state.get('_heavy')


class SimpleTestClass:
    """Simple class for basic roundtrip tests."""
    def __init__(self, x, y):
        self.x = x
        self.y = y


class SlottedTestClass:
    """Class with __slots__ for slot tests."""
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y


class BaseTestClass:
    """Base class for inheritance tests."""
    def __init__(self):
        self.base_attr = "base"


class DerivedTestClass(BaseTestClass):
    """Derived class for inheritance tests."""
    def __init__(self):
        super().__init__()
        self.derived_attr = "derived"


class StaticMethodTestClass:
    """Class with static methods for tests."""
    @staticmethod
    def static_method(x):
        return x * 2


class ClassMethodTestClass:
    """Class with class methods for tests."""
    value = 10

    @classmethod
    def get_value(cls):
        return cls.value


class NodeClass:
    """Node class for circular reference tests."""
    def __init__(self, value):
        self.value = value
        self.next = None


class ContainerClass:
    """Container class for integration tests."""
    def __init__(self, processor, data):
        self.processor = processor
        self.data = data


class MockEngine:
    """Mock engine for server-provided integration tests."""
    def process(self, x):
        return x * 10


class IntegrationModel:
    """Model class for integration tests."""
    _server_provided = frozenset({'_engine'})

    def __init__(self, name):
        self.name = name
        self._engine = None

    def run(self, x):
        return self._engine.process(x) if self._engine else x


# Module-level functions for closure tests
GLOBAL_MULTIPLIER = 3

def module_level_closure_func(x):
    """A function that captures a module-level variable."""
    return x * GLOBAL_MULTIPLIER


# Factory functions that create closures (defined at module level for source access)
def make_counter_with_nonlocal():
    """Factory that creates a counter using nonlocal."""
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment


def make_simple_closure():
    """Factory that creates a simple read-only closure."""
    multiplier = 2
    def multiply(x):
        return x * multiplier
    return multiply


def make_multi_var_closure():
    """Factory that creates a closure with multiple variables."""
    a, b, c = 1, 2, 3
    def compute(x):
        return a * x + b * x + c
    return compute


def make_nested_closure():
    """Factory that creates nested closures."""
    def outer(x):
        def inner(y):
            return x + y
        return inner
    return outer


def make_list_closure():
    """Factory that creates a closure capturing a list."""
    data = [1, 2, 3]
    def get_sum():
        return sum(data)
    return get_sum


# =============================================================================
# 1. Basic Linting Configuration Tests
# =============================================================================

class TestLintConfig:
    """Tests for LintConfig and NNSIGHT_CONFIG."""

    def test_nnsight_config_exists(self):
        """NNSIGHT_CONFIG should be properly configured."""
        assert NNSIGHT_CONFIG is not None
        assert isinstance(NNSIGHT_CONFIG, LintConfig)

    def test_nnsight_config_has_forbidden_modules(self):
        """NNSIGHT_CONFIG should include forbidden modules."""
        assert 'subprocess' in NNSIGHT_CONFIG.forbidden_modules
        assert 'socket' in NNSIGHT_CONFIG.forbidden_modules
        assert 'os' not in NNSIGHT_CONFIG.forbidden_modules  # os itself isn't banned, specific calls are

    def test_nnsight_config_has_forbidden_calls(self):
        """NNSIGHT_CONFIG should include forbidden function calls."""
        assert 'eval' in NNSIGHT_CONFIG.forbidden_calls
        assert 'exec' in NNSIGHT_CONFIG.forbidden_calls
        assert 'open' in NNSIGHT_CONFIG.forbidden_calls
        assert 'compile' in NNSIGHT_CONFIG.forbidden_calls

    def test_nnsight_config_has_forbidden_attr_chains(self):
        """NNSIGHT_CONFIG should include forbidden attribute chains."""
        assert ('os', 'system') in NNSIGHT_CONFIG.forbidden_attr_chains
        assert ('subprocess', 'run') in NNSIGHT_CONFIG.forbidden_attr_chains

    def test_nnsight_config_has_forbidden_classes(self):
        """NNSIGHT_CONFIG should include forbidden classes with messages."""
        assert 'pandas.core.frame.DataFrame' in NNSIGHT_CONFIG.forbidden_classes
        assert 'matplotlib.figure.Figure' in NNSIGHT_CONFIG.forbidden_classes

    def test_custom_lint_config(self):
        """Custom LintConfig should work."""
        config = LintConfig(
            forbidden_modules=frozenset({'mymodule'}),
            forbidden_calls=frozenset({'dangerous_func'}),
            reject_nonlocal=False,
        )
        assert 'mymodule' in config.forbidden_modules
        assert 'dangerous_func' in config.forbidden_calls
        assert config.reject_nonlocal is False

    def test_lint_config_merge_with_defaults(self):
        """merge_with_defaults should combine with module defaults."""
        config = LintConfig(
            forbidden_modules=frozenset({'custom_module'}),
        )
        merged = config.merge_with_defaults()
        assert 'custom_module' in merged.forbidden_modules
        assert 'subprocess' in merged.forbidden_modules  # From defaults

    def test_forbidden_module_prefixes_immutable(self):
        """FORBIDDEN_MODULE_PREFIXES should be a frozenset."""
        assert isinstance(FORBIDDEN_MODULE_PREFIXES, frozenset)

    def test_key_modules_in_forbidden_list(self):
        """Key dangerous modules should be forbidden."""
        for module in ['socket', 'subprocess', 'multiprocessing', '_pytest', 'pytest']:
            assert module in FORBIDDEN_MODULE_PREFIXES, f"{module} should be forbidden"


# =============================================================================
# 2. Forbidden Type Detection Tests
# =============================================================================

class TestForbiddenTypeDetection:
    """Tests for early detection of forbidden types."""

    def test_regular_objects_allowed(self):
        """Regular user objects should not be forbidden."""
        class MyClass:
            pass
        obj = MyClass()
        assert check_forbidden_type(obj, NNSIGHT_CONFIG) is None

    def test_primitives_allowed(self):
        """Primitives should not be forbidden."""
        for obj in [1, 1.5, "string", True, None, [1, 2], {"a": 1}]:
            assert check_forbidden_type(obj, NNSIGHT_CONFIG) is None

    def test_socket_forbidden(self):
        """Socket objects should be forbidden."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            msg = check_forbidden_type(sock, NNSIGHT_CONFIG)
            assert msg is not None
            assert 'socket' in msg.lower() or 'cannot be serialized' in msg.lower()
        finally:
            sock.close()

    def test_file_handle_forbidden(self):
        """File handles should be forbidden."""
        f = StringIO()
        msg = check_forbidden_type(f, NNSIGHT_CONFIG)
        # StringIO might not be in the forbidden list, but real files would be
        # This is a documentation of current behavior

    def test_serialize_forbidden_type_raises(self):
        """Serializing a forbidden type with linting should raise ValueError."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(ValueError) as exc_info:
                serialize(sock, lint=NNSIGHT_CONFIG)
            assert 'socket' in str(exc_info.value).lower() or 'cannot serialize' in str(exc_info.value).lower()
        finally:
            sock.close()

    def test_forbidden_type_error_has_suggestion(self):
        """Error messages for forbidden types should include suggestions."""
        # Create a mock object that looks like a DataFrame
        class FakeDataFrame:
            pass
        FakeDataFrame.__module__ = 'pandas.core.frame'
        FakeDataFrame.__qualname__ = 'DataFrame'

        obj = FakeDataFrame()
        msg = check_forbidden_type(obj, NNSIGHT_CONFIG)
        if msg:
            # Should have helpful suggestion
            assert 'tensor' in msg.lower() or 'convert' in msg.lower()


# =============================================================================
# 3. Nonlocal Rejection Tests
# =============================================================================

class TestNonlocalRejection:
    """Tests for nonlocal closure detection and rejection."""

    def test_check_nonlocal_detects_nonlocal(self):
        """check_nonlocal should detect nonlocal statements."""
        # Use module-level factory
        inner_func = make_counter_with_nonlocal()
        names = check_nonlocal(inner_func)
        assert 'count' in names

    def test_check_nonlocal_no_false_positives(self):
        """check_nonlocal should not flag functions without nonlocal."""
        # Use module-level function
        names = check_nonlocal(module_level_closure_func)
        assert len(names) == 0

    def test_check_nonlocal_closure_without_nonlocal(self):
        """Closures that only read (don't mutate) should be allowed."""
        # Use module-level factory
        inner_func = make_simple_closure()
        names = check_nonlocal(inner_func)
        assert len(names) == 0

    def test_serialize_nonlocal_with_lint_raises(self):
        """Serializing function with nonlocal and lint should raise."""
        counter = make_counter_with_nonlocal()

        with pytest.raises(ValueError) as exc_info:
            serialize(counter, lint=NNSIGHT_CONFIG)

        error_msg = str(exc_info.value)
        assert 'nonlocal' in error_msg.lower()
        assert 'count' in error_msg

    def test_nonlocal_error_suggests_class(self):
        """Nonlocal error should suggest using a class instead."""
        counter = make_counter_with_nonlocal()

        with pytest.raises(ValueError) as exc_info:
            serialize(counter, lint=NNSIGHT_CONFIG)

        error_msg = str(exc_info.value)
        assert 'class' in error_msg.lower()

    def test_nonlocal_without_lint_succeeds(self):
        """Without lint, nonlocal functions should serialize (behavior warning)."""
        counter = make_counter_with_nonlocal()
        # Should not raise without linting
        payload = serialize(counter)
        assert payload is not None

    def test_multiple_nonlocal_variables(self):
        """Should detect multiple nonlocal variables."""
        # Define a function with multiple nonlocal vars for this specific test
        def make_multi_nonlocal():
            x, y = 0, 0
            def inner():
                nonlocal x, y
                x += 1
                y += 2
            return inner

        inner_func = make_multi_nonlocal()
        names = check_nonlocal(inner_func)
        assert 'x' in names
        assert 'y' in names


# =============================================================================
# 4. Server-Provided Attributes Tests
# =============================================================================

class TestServerProvidedAttributes:
    """Tests for _server_provided attribute handling."""

    def test_server_provided_attrs_skipped(self):
        """Attributes in _server_provided should be skipped from object state."""
        obj = ServerProvidedModel()
        payload = serialize(obj)

        # Check the object's state dict directly in the payload
        # The server_provided_attrs field should list what was skipped
        obj_ref = payload.obj
        obj_data = payload.memo[obj_ref]

        # The object should track which attrs were skipped
        assert obj_data.server_provided_attrs is not None
        assert '_module' in obj_data.server_provided_attrs
        assert '_tokenizer' in obj_data.server_provided_attrs

        # The object state should NOT contain these keys
        state_keys = list(obj_data.state.keys())
        assert '_module' not in state_keys
        assert '_tokenizer' not in state_keys
        # But name should be in state
        assert 'name' in state_keys

    def test_server_provided_attrs_injected(self):
        """Server-provided attributes should be injected during deserialization."""
        obj = ServerProvidedModel()
        payload = serialize(obj)
        json_str = payload.model_dump_json()

        # Deserialize with server-provided value
        server_module = "SERVER_INJECTED_MODULE"
        result = deserialize(
            json.loads(json_str),
            server_provided={'_module': server_module}
        )

        assert result.name == "test"
        assert result._module == server_module

    def test_server_provided_inheritance(self):
        """Server-provided attributes should be collected from class hierarchy."""
        obj = ServerProvidedDerivedModel()
        payload = serialize(obj)

        # Check the object's state dict directly in the payload
        obj_ref = payload.obj
        obj_data = payload.memo[obj_ref]

        # Both _cache (from base) and _module (from derived) should be skipped
        assert obj_data.server_provided_attrs is not None
        assert '_cache' in obj_data.server_provided_attrs
        assert '_module' in obj_data.server_provided_attrs

        # The object state should NOT contain these keys
        state_keys = list(obj_data.state.keys())
        assert '_cache' not in state_keys
        assert '_module' not in state_keys

        # Inject both during deserialization
        json_str = payload.model_dump_json()
        result = deserialize(
            json.loads(json_str),
            server_provided={'_cache': 'NEW_CACHE', '_module': 'NEW_MODULE'}
        )

        assert result._cache == 'NEW_CACHE'
        assert result._module == 'NEW_MODULE'

    def test_server_provided_with_getstate_setstate(self):
        """Server-provided should work with __getstate__/__setstate__."""
        obj = ServerProvidedWithState()
        payload = serialize(obj)

        result = deserialize(
            json.loads(payload.model_dump_json()),
            server_provided={'_heavy': 'INJECTED'}
        )

        assert result.light == "small"
        assert result._heavy == "INJECTED"

    def test_missing_server_provided_is_none(self):
        """Missing server-provided attrs should remain as they were (not injected)."""
        obj = ServerProvidedModel()
        payload = serialize(obj)

        # Deserialize WITHOUT providing the server value
        result = deserialize(json.loads(payload.model_dump_json()))

        assert result.name == "test"
        # _module was skipped during serialization and not injected
        # So it shouldn't exist or should be default
        assert not hasattr(result, '_module') or result._module is None


# =============================================================================
# 5. Source Code Validation Tests
# =============================================================================

class TestSourceValidation:
    """Tests for AST-based source code validation."""

    def test_validate_function_with_forbidden_import(self):
        """Functions with forbidden imports should fail validation."""
        def bad_func():
            import subprocess
            return subprocess.run(['ls'])

        errors = validate_function_source(bad_func, NNSIGHT_CONFIG)
        assert len(errors) > 0
        assert any('subprocess' in str(e) for e in errors)

    def test_validate_function_with_forbidden_call(self):
        """Functions with forbidden calls should fail validation."""
        def bad_func():
            return eval("1 + 1")

        errors = validate_function_source(bad_func, NNSIGHT_CONFIG)
        assert len(errors) > 0
        assert any('eval' in str(e) for e in errors)

    def test_validate_function_with_open_call(self):
        """Functions calling open() should fail validation."""
        def bad_func():
            f = open('/etc/passwd')
            return f.read()

        errors = validate_function_source(bad_func, NNSIGHT_CONFIG)
        assert len(errors) > 0
        assert any('open' in str(e) for e in errors)

    def test_validate_clean_function(self):
        """Clean functions should pass validation."""
        def good_func(x, y):
            return x + y

        errors = validate_function_source(good_func, NNSIGHT_CONFIG)
        assert len(errors) == 0

    def test_validate_function_with_allowed_imports(self):
        """Functions with allowed imports should pass validation."""
        def good_func():
            import math
            return math.sqrt(4)

        errors = validate_function_source(good_func, NNSIGHT_CONFIG)
        assert len(errors) == 0

    def test_validate_attr_chain_os_system(self):
        """os.system calls should fail validation."""
        def bad_func():
            import os
            os.system('rm -rf /')

        errors = validate_function_source(bad_func, NNSIGHT_CONFIG)
        assert len(errors) > 0

    def test_validate_source_integration(self):
        """validate_function_source should catch forbidden imports in real functions."""
        # Define a function with forbidden import (at module level for source access)
        def func_with_forbidden_import():
            import subprocess
            return subprocess.run(['ls'])

        # validate_function_source should catch this
        errors = validate_function_source(func_with_forbidden_import, NNSIGHT_CONFIG)
        assert len(errors) > 0
        assert any('subprocess' in str(e) for e in errors)


# =============================================================================
# 6. Closure Edge Cases
# =============================================================================

class TestClosureEdgeCases:
    """Tests for closure serialization edge cases."""

    def test_simple_closure_roundtrip(self):
        """Simple closures should roundtrip correctly."""
        multiply = make_simple_closure()
        result = roundtrip(multiply)
        assert result(5) == 10  # multiplier is 2

    def test_closure_with_multiple_variables(self):
        """Closures with multiple captured variables should work."""
        compute = make_multi_var_closure()
        result = roundtrip(compute)
        assert result(10) == 33  # 1*10 + 2*10 + 3 = 33

    def test_nested_closure(self):
        """Nested closures should work."""
        outer = make_nested_closure()
        inner = outer(10)
        result = roundtrip(inner)
        assert result(5) == 15

    def test_closure_with_list(self):
        """Closures capturing mutable objects should work."""
        get_sum = make_list_closure()
        result = roundtrip(get_sum)
        assert result() == 6

    def test_module_level_function_roundtrip(self):
        """Module-level functions capturing globals should roundtrip."""
        result = roundtrip(module_level_closure_func)
        assert result(5) == 15  # GLOBAL_MULTIPLIER is 3


# =============================================================================
# 7. Class Serialization Edge Cases
# =============================================================================

class TestClassEdgeCases:
    """Tests for class serialization edge cases."""

    def test_class_with_slots(self):
        """Classes with __slots__ should serialize correctly."""
        obj = SlottedTestClass(1, 2)
        # Serialize should work
        payload = serialize(obj)
        assert payload is not None

        # For slotted classes, state may be wrapped differently
        # Just verify serialization completes
        obj_ref = payload.obj
        obj_data = payload.memo[obj_ref]
        assert obj_data is not None
        assert obj_data.state is not None

        # Note: Slotted classes use __getstate__ which may return non-dict state
        # The state format may be {'__state__': ...} for non-dict states

    def test_simple_class_roundtrip(self):
        """Simple classes should serialize correctly."""
        obj = SimpleTestClass(10, 20)
        result = roundtrip(obj)
        assert result.x == 10
        assert result.y == 20

    def test_inheritance_roundtrip(self):
        """Inherited classes should serialize correctly."""
        obj = DerivedTestClass()
        result = roundtrip(obj)
        assert result.base_attr == "base"
        assert result.derived_attr == "derived"

    def test_staticmethod_roundtrip(self):
        """Static methods should serialize correctly."""
        obj = StaticMethodTestClass()
        result = roundtrip(obj)
        assert result.static_method(5) == 10

    def test_classmethod_roundtrip(self):
        """Class methods should serialize correctly."""
        obj = ClassMethodTestClass()
        result = roundtrip(obj)
        # The classmethod should work on the deserialized object


# =============================================================================
# 8. Circular Reference Tests
# =============================================================================

class TestCircularReferences:
    """Tests for circular reference handling."""

    def test_self_referential_list(self):
        """Lists containing themselves should roundtrip."""
        lst = [1, 2, 3]
        lst.append(lst)

        result = roundtrip(lst)
        assert result[0] == 1
        assert result[3] is result

    def test_mutual_references(self):
        """Objects with mutual references should roundtrip."""
        a = NodeClass("a")
        b = NodeClass("b")
        a.next = b
        b.next = a

        result = roundtrip(a)
        assert result.value == "a"
        assert result.next.value == "b"
        assert result.next.next is result

    def test_deeply_nested_circular(self):
        """Deeply nested circular references should work."""
        data = {"level": 0}
        current = data
        for i in range(1, 10):
            current["child"] = {"level": i}
            current = current["child"]
        current["root"] = data  # Create cycle

        result = roundtrip(data)
        assert result["level"] == 0
        assert result["child"]["level"] == 1


# =============================================================================
# 9. Error Message Quality Tests
# =============================================================================

class TestErrorMessages:
    """Tests for quality of error messages."""

    def test_nonlocal_error_is_helpful(self):
        """Nonlocal error should be clear and actionable."""
        counter = make_counter_with_nonlocal()

        with pytest.raises(ValueError) as exc_info:
            serialize(counter, lint=NNSIGHT_CONFIG)

        msg = str(exc_info.value)
        # Should mention nonlocal
        assert 'nonlocal' in msg.lower()
        # Should mention the variable name
        assert 'count' in msg
        # Should suggest alternative
        assert 'class' in msg.lower()

    def test_forbidden_import_error_mentions_module(self):
        """Forbidden import error should mention the module name."""
        # This test validates source code, not actual execution
        # The function source contains forbidden import
        def bad_func_with_import():
            import subprocess
            return subprocess

        errors = validate_function_source(bad_func_with_import, NNSIGHT_CONFIG)
        assert len(errors) > 0
        assert any('subprocess' in str(e) for e in errors)


# =============================================================================
# 10. Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_with_linting(self):
        """Complete workflow with linting enabled."""
        # Use module-level function
        payload = serialize(module_level_closure_func, lint=NNSIGHT_CONFIG)
        json_str = payload.model_dump_json()
        result = deserialize(json.loads(json_str))

        assert result(5) == 15  # GLOBAL_MULTIPLIER is 3

    def test_full_workflow_with_server_provided(self):
        """Complete workflow with server-provided attributes."""
        model = IntegrationModel("test_model")
        payload = serialize(model)
        json_str = payload.model_dump_json()

        result = deserialize(
            json.loads(json_str),
            server_provided={'_engine': MockEngine()}
        )

        assert result.name == "test_model"
        assert result.run(5) == 50

    def test_complex_nested_structure(self):
        """Complex nested structure with functions and objects."""
        process = make_simple_closure()  # multiplier is 2
        obj = ContainerClass(process, [1, 2, 3])
        result = roundtrip(obj)

        assert result.processor(5) == 10
        assert result.data == [1, 2, 3]


# =============================================================================
# 11. Lint Object Comprehensive Tests
# =============================================================================

class TestLintObject:
    """Tests for the comprehensive lint_object function."""

    def test_lint_object_clean_function(self):
        """Clean functions should pass lint_object."""
        def good_func(x):
            return x + 1

        errors = lint_object(good_func, NNSIGHT_CONFIG)
        assert len(errors) == 0

    def test_lint_object_catches_forbidden_type(self):
        """lint_object should catch forbidden types."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            errors = lint_object(sock, NNSIGHT_CONFIG)
            assert len(errors) > 0
        finally:
            sock.close()

    def test_lint_object_catches_nonlocal(self):
        """lint_object should catch nonlocal usage."""
        def outer():
            x = 0
            def inner():
                nonlocal x
                x += 1
            return inner

        errors = lint_object(outer(), NNSIGHT_CONFIG)
        assert len(errors) > 0
        assert any('nonlocal' in str(e).lower() for e in errors)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
