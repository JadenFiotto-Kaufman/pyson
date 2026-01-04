"""Tests for pyson serialization/deserialization."""

import pytest
from pyson import serialize, deserialize
from cloudpickle import register_pickle_by_value
import json
import sys


def roundtrip(obj):
    """Serialize and deserialize an object, returning the result."""
    payload = serialize(obj)
    json_str = payload.model_dump_json()
    return deserialize(json.loads(json_str))


# ============================================================================
# Module-level classes for object tests (must be importable for deserialization)
# ============================================================================


class SimpleObject:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class StatefulObject:
    def __init__(self, value):
        self.value = value
        self._cache = "should not be serialized"

    def __getstate__(self):
        return {"value": self.value}

    def __setstate__(self, state):
        self.value = state["value"]
        self._cache = "restored"


class InnerObject:
    def __init__(self, val):
        self.val = val


class OuterObject:
    def __init__(self, inner):
        self.inner = inner


# ============================================================================
# Module-level functions for function tests (must have source accessible)
# ============================================================================

GLOBAL_VALUE = 100


def simple_add(a, b):
    return a + b


def greet_with_default(name, greeting="Hello"):
    return f"{greeting}, {name}!"


def func_with_kwonly(a, *, b=10, c=20):
    return a + b + c


def use_global(x):
    return x + GLOBAL_VALUE


def documented_func():
    """This is a docstring."""
    pass


def annotated_func(x: int, y: str) -> bool:
    return True


def func_with_attr():
    pass


func_with_attr.custom_attr = "hello"


def double_it(x):
    return x * 2


def helper_func(x):
    return x + 1


def main_using_helper(x):
    return helper_func(x) * 2


def make_adder(n):
    def adder(x):
        return x + n

    return adder


def make_linear(m, b):
    def linear(x):
        return m * x + b

    return linear


def make_counter(start):
    data = {"count": start}

    def increment():
        data["count"] += 1
        return data["count"]

    return increment


DEFAULT_LIST = [1, 2, 3]


def func_with_list_default(x, lst=DEFAULT_LIST):
    return x + sum(lst)


# ============================================================================
# Module-level classes for dynamic class tests
# ============================================================================


class EmptyClass:
    """An empty class with just a docstring."""

    pass


class ClassWithAttributes:
    """A class with class-level attributes."""

    class_attr = 42
    class_list = [1, 2, 3]
    class_str = "hello"


class ClassWithMethods:
    """A class with various methods."""

    def instance_method(self, x):
        return x * 2

    @classmethod
    def class_method(cls, x):
        return x + 10

    @staticmethod
    def static_method(x):
        return x - 5


class ClassWithInit:
    """A class with __init__ and instance attributes."""

    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value


class BaseClass:
    """A base class for inheritance tests."""

    base_attr = "from base"

    def base_method(self):
        return "base"


class DerivedClass(BaseClass):
    """A derived class that inherits from BaseClass."""

    derived_attr = "from derived"

    def derived_method(self):
        return "derived"

    def base_method(self):
        return "overridden"


class MultipleInheritanceA:
    attr_a = "A"

    def method_a(self):
        return "A"


class MultipleInheritanceB:
    attr_b = "B"

    def method_b(self):
        return "B"


class MultipleInheritanceDerived(MultipleInheritanceA, MultipleInheritanceB):
    attr_c = "C"

    def method_c(self):
        return "C"


class ClassWithProperty:
    """A class with a property."""

    def __init__(self):
        self._value = 0

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v


class ClassWithSlots:
    """A class with __slots__."""

    __slots__ = ["x", "y"]

    def __init__(self, x, y):
        self.x = x
        self.y = y


class ClassWithClassMethod:
    """A class with a classmethod that uses cls."""

    counter = 0

    @classmethod
    def increment(cls):
        cls.counter += 1
        return cls.counter


# Register this module so functions and classes defined here are serialized by value
# Must be done after module is loaded, so we use sys.modules
register_pickle_by_value(sys.modules[__name__])


# ============================================================================
# Tests
# ============================================================================


class TestPrimitives:
    """Test serialization of primitive types."""

    def test_int(self):
        assert roundtrip(42) == 42
        assert roundtrip(-1) == -1
        assert roundtrip(0) == 0

    def test_float(self):
        assert roundtrip(3.14) == 3.14
        assert roundtrip(-0.5) == -0.5
        assert roundtrip(0.0) == 0.0

    def test_bool(self):
        assert roundtrip(True) is True
        assert roundtrip(False) is False

    def test_str(self):
        assert roundtrip("hello") == "hello"
        assert roundtrip("") == ""
        assert roundtrip("unicode: 你好 🎉") == "unicode: 你好 🎉"

    def test_none(self):
        assert roundtrip(None) is None


class TestCollections:
    """Test serialization of collection types."""

    def test_list(self):
        assert roundtrip([1, 2, 3]) == [1, 2, 3]
        assert roundtrip([]) == []
        assert roundtrip([1, "two", 3.0, True, None]) == [1, "two", 3.0, True, None]

    def test_tuple(self):
        assert roundtrip((1, 2, 3)) == (1, 2, 3)
        assert roundtrip(()) == ()
        assert roundtrip((1, "two", 3.0)) == (1, "two", 3.0)

    def test_dict(self):
        assert roundtrip({1: 2, 3: 4}) == {1: 2, 3: 4}
        assert roundtrip({}) == {}
        assert roundtrip({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested_collections(self):
        nested = {"list": [1, 2, 3], "tuple": (4, 5, 6), "dict": {"a": "b"}}
        result = roundtrip(nested)
        assert result == nested
        assert isinstance(result["tuple"], tuple)


class TestReferenceCycles:
    """Test handling of reference cycles."""

    def test_list_self_reference(self):
        x = [1, 2, 3]
        x.append(x)
        result = roundtrip(x)
        assert result[:3] == [1, 2, 3]
        assert result[3] is result  # Self-reference preserved

    def test_nested_cycle(self):
        x = [1, 2, 3]
        y = (x,)
        x.append(y)
        x.append(x)
        result = roundtrip(x)
        assert result[:3] == [1, 2, 3]
        assert isinstance(result[3], tuple)
        assert result[3][0] is result  # Cycle through tuple
        assert result[4] is result  # Direct self-reference

    def test_dict_cycle(self):
        d = {"a": 1}
        d["self"] = d
        result = roundtrip(d)
        assert result["a"] == 1
        assert result["self"] is result

    def test_shared_references(self):
        shared = [1, 2, 3]
        container = [shared, shared, shared]
        result = roundtrip(container)
        # All three should be the same object
        assert result[0] is result[1]
        assert result[1] is result[2]


class TestObjects:
    """Test serialization of custom objects."""

    def test_simple_object(self):
        obj = SimpleObject(1, "two")
        result = roundtrip(obj)
        assert result.x == 1
        assert result.y == "two"

    def test_object_with_getstate_setstate(self):
        obj = StatefulObject(42)
        result = roundtrip(obj)
        assert result.value == 42
        assert result._cache == "restored"

    def test_nested_objects(self):
        obj = OuterObject(InnerObject(123))
        result = roundtrip(obj)
        assert result.inner.val == 123


class TestDynamicClasses:
    """Test serialization of dynamic classes (by value)."""

    def test_empty_class(self):
        result = roundtrip(EmptyClass)
        assert result.__name__ == "EmptyClass"
        assert result.__doc__ == "An empty class with just a docstring."
        # Should be able to instantiate
        obj = result()
        assert isinstance(obj, result)

    def test_class_with_attributes(self):
        result = roundtrip(ClassWithAttributes)
        assert result.class_attr == 42
        assert result.class_list == [1, 2, 3]
        assert result.class_str == "hello"

    def test_class_with_methods(self):
        result = roundtrip(ClassWithMethods)
        obj = result()
        assert obj.instance_method(5) == 10
        assert result.class_method(5) == 15
        assert result.static_method(10) == 5

    def test_class_with_init(self):
        result = roundtrip(ClassWithInit)
        obj = result(42)
        assert obj.value == 42
        assert obj.get_value() == 42

    def test_class_inheritance(self):
        result = roundtrip(DerivedClass)
        # Check class attributes
        assert result.base_attr == "from base"
        assert result.derived_attr == "from derived"
        # Check methods
        obj = result()
        assert obj.base_method() == "overridden"
        assert obj.derived_method() == "derived"
        # Check inheritance chain (base class is also serialized by value,
        # so we check the deserialized base, not the original)
        assert len(result.__bases__) == 1
        base = result.__bases__[0]
        assert base.__name__ == "BaseClass"
        assert base.base_attr == "from base"
        # The base method should be callable from base class
        base_obj = base()
        assert base_obj.base_method() == "base"

    def test_multiple_inheritance(self):
        result = roundtrip(MultipleInheritanceDerived)
        obj = result()
        assert obj.method_a() == "A"
        assert obj.method_b() == "B"
        assert obj.method_c() == "C"
        assert result.attr_a == "A"
        assert result.attr_b == "B"
        assert result.attr_c == "C"

    def test_class_with_property(self):
        result = roundtrip(ClassWithProperty)
        obj = result()
        assert obj.value == 0
        obj.value = 42
        assert obj.value == 42

    def test_class_same_multiple_times(self):
        container = [ClassWithAttributes, ClassWithAttributes]
        result = roundtrip(container)
        # Both should be the same class object
        assert result[0] is result[1]
        assert result[0].class_attr == 42

    def test_class_with_classmethod(self):
        result = roundtrip(ClassWithClassMethod)
        # Reset counter for test
        result.counter = 0
        assert result.increment() == 1
        assert result.increment() == 2

    def test_class_metadata(self):
        result = roundtrip(ClassWithMethods)
        assert result.__name__ == "ClassWithMethods"
        assert result.__module__ == "pyson.test"
        assert "A class with various methods" in result.__doc__

    def test_instantiate_deserialized_class(self):
        result = roundtrip(ClassWithInit)
        obj1 = result(10)
        obj2 = result(20)
        assert obj1.value == 10
        assert obj2.value == 20
        assert obj1.get_value() != obj2.get_value()

    def test_class_with_complex_attribute(self):
        # Test that class attributes that are themselves complex objects work
        result = roundtrip(ClassWithAttributes)
        # class_list should be a list that we can modify
        assert isinstance(result.class_list, list)
        assert result.class_list == [1, 2, 3]


class TestByReference:
    """Test serialization of classes/functions by reference (non-registered modules)."""

    def test_builtin_class_by_reference(self):
        # Built-in classes like int, str, list should be serialized by reference
        result = roundtrip(int)
        assert result is int

        result = roundtrip(str)
        assert result is str

        result = roundtrip(list)
        assert result is list

    def test_stdlib_class_by_reference(self):
        # Standard library classes should be serialized by reference
        from collections import OrderedDict
        from datetime import datetime

        result = roundtrip(OrderedDict)
        assert result is OrderedDict

        result = roundtrip(datetime)
        assert result is datetime

    def test_stdlib_exception_by_reference(self):
        # Exception classes should be serialized by reference
        result = roundtrip(ValueError)
        assert result is ValueError

        result = roundtrip(TypeError)
        assert result is TypeError

    def test_object_instance_with_by_ref_class(self):
        # Objects of classes that are serialized by reference
        # The class reference is preserved, instances are reconstructed
        obj = SimpleObject(10, "test")
        result = roundtrip(obj)
        # SimpleObject is by-value since pyson.test is registered
        assert result.x == 10
        assert result.y == "test"

    def test_class_reference_preserved_identity(self):
        # When a class is serialized by reference, multiple references
        # should resolve to the same class
        container = [int, int, str, str]
        result = roundtrip(container)
        assert result[0] is result[1]
        assert result[0] is int
        assert result[2] is result[3]
        assert result[2] is str

    def test_mixed_by_value_and_by_reference(self):
        # A dynamic class can have attributes that reference stdlib classes
        result = roundtrip(ClassWithInit)
        # ClassWithInit is by-value, but it can still work with stdlib types
        obj = result(42)
        assert isinstance(obj.value, int)
        assert obj.value == 42


class TestFunctions:
    """Test serialization of functions."""

    def test_simple_function(self):
        result = roundtrip(simple_add)
        assert result(2, 3) == 5

    def test_function_with_defaults(self):
        result = roundtrip(greet_with_default)
        assert result("World") == "Hello, World!"
        assert result("World", "Hi") == "Hi, World!"

    def test_function_with_kwonly_defaults(self):
        result = roundtrip(func_with_kwonly)
        assert result(1) == 31
        assert result(1, b=5) == 26
        assert result(1, b=5, c=10) == 16

    def test_function_with_globals(self):
        result = roundtrip(use_global)
        assert result(5) == 105

    def test_function_with_closure(self):
        add_5 = make_adder(5)
        result = roundtrip(add_5)
        assert result(10) == 15

    def test_function_with_multiple_closure_vars(self):
        f = make_linear(2, 3)  # y = 2x + 3
        result = roundtrip(f)
        assert result(0) == 3
        assert result(1) == 5
        assert result(10) == 23

    def test_function_with_docstring(self):
        result = roundtrip(documented_func)
        assert result.__doc__ == "This is a docstring."

    def test_function_with_annotations(self):
        result = roundtrip(annotated_func)
        assert result.__annotations__.get("x") == int
        assert result.__annotations__.get("y") == str
        assert result.__annotations__.get("return") == bool

    def test_function_with_custom_attribute(self):
        result = roundtrip(func_with_attr)
        assert result.custom_attr == "hello"

    def test_same_function_multiple_times(self):
        container = [double_it, double_it]
        result = roundtrip(container)
        # Both should be the same function object (shared reference)
        assert result[0] is result[1]
        assert result[0](5) == 10

    def test_function_referencing_other_function(self):
        result = roundtrip(main_using_helper)
        assert result(5) == 12  # (5 + 1) * 2


class TestComplexScenarios:
    """Test complex combinations of types and references."""

    def test_mixed_container(self):
        x = [1, 2, 3]
        z = (1, 2, 3)
        w = {1: 2, 3: 4}
        v = 1.0
        u = True
        t = "hello"

        thing = [x, z, w, v, u, t]
        thing.append(thing)

        result = roundtrip(thing)
        assert result[0] == [1, 2, 3]
        assert result[1] == (1, 2, 3)
        assert result[2] == {1: 2, 3: 4}
        assert result[3] == 1.0
        assert result[4] is True
        assert result[5] == "hello"
        assert result[6] is result  # Self-reference

    def test_function_with_complex_default(self):
        result = roundtrip(func_with_list_default)
        assert result(10) == 16  # 10 + 1 + 2 + 3

    def test_closure_over_mutable(self):
        counter = make_counter(0)
        # Call original a few times
        counter()
        counter()

        # Roundtrip - the closure captures the dict state at serialization time
        result = roundtrip(counter)
        # The deserialized function has its own copy of data starting at 2
        assert result() == 3  # Continues from captured state


class TestCustomSerializers:
    """Test custom serializers for pandas and numpy."""

    def test_numpy_array_1d(self):
        import pyson.custom  # noqa: F401
        import numpy as np

        arr = np.array([1, 2, 3, 4, 5])
        result = roundtrip(arr)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, arr)
        assert result.dtype == arr.dtype

    def test_numpy_array_2d(self):
        import pyson.custom  # noqa: F401
        import numpy as np

        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = roundtrip(arr)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, arr)
        assert result.shape == (2, 3)

    def test_numpy_array_float(self):
        import pyson.custom  # noqa: F401
        import numpy as np

        arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        result = roundtrip(arr)
        assert np.array_equal(result, arr)
        assert result.dtype == np.float64

    def test_numpy_array_empty(self):
        import pyson.custom  # noqa: F401
        import numpy as np

        arr = np.array([])
        result = roundtrip(arr)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_numpy_array_multidimensional(self):
        import pyson.custom  # noqa: F401
        import numpy as np

        arr = np.zeros((2, 3, 4))
        result = roundtrip(arr)
        assert result.shape == (2, 3, 4)
        assert np.array_equal(result, arr)

    def test_pandas_series(self):
        import pyson.custom  # noqa: F401
        import pandas as pd

        s = pd.Series([1, 2, 3], index=["a", "b", "c"], name="my_series")
        result = roundtrip(s)
        assert isinstance(result, pd.Series)
        assert result.name == "my_series"
        assert list(result.index) == ["a", "b", "c"]
        assert list(result.values) == [1, 2, 3]

    def test_pandas_series_float(self):
        import pyson.custom  # noqa: F401
        import pandas as pd

        s = pd.Series([1.1, 2.2, 3.3])
        result = roundtrip(s)
        assert result.dtype == s.dtype
        assert s.equals(result)

    def test_pandas_dataframe(self):
        import pyson.custom  # noqa: F401
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.5, 2.5, 3.5]})
        result = roundtrip(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b", "c"]
        assert df.equals(result)

    def test_pandas_dataframe_with_index(self):
        import pyson.custom  # noqa: F401
        import pandas as pd

        df = pd.DataFrame({"col1": [10, 20], "col2": [30, 40]}, index=["row1", "row2"])
        result = roundtrip(df)
        assert list(result.index) == ["row1", "row2"]
        assert df.equals(result)

    def test_pandas_dataframe_dtypes_preserved(self):
        import pyson.custom  # noqa: F401
        import pandas as pd

        df = pd.DataFrame({"ints": [1, 2], "floats": [1.0, 2.0], "strs": ["a", "b"]})
        result = roundtrip(df)
        assert result["ints"].dtype == df["ints"].dtype
        assert result["floats"].dtype == df["floats"].dtype
        assert result["strs"].dtype == df["strs"].dtype

    def test_mixed_numpy_pandas(self):
        import pyson.custom  # noqa: F401
        import numpy as np
        import pandas as pd

        arr = np.array([1, 2, 3])
        series = pd.Series([4, 5, 6])
        df = pd.DataFrame({"x": [7, 8, 9]})

        container = {"array": arr, "series": series, "dataframe": df}
        result = roundtrip(container)

        assert np.array_equal(result["array"], arr)
        assert result["series"].equals(series)
        assert result["dataframe"].equals(df)

    def test_torch_tensor_1d(self):
        import pyson.custom  # noqa: F401
        import torch

        t = torch.tensor([1.0, 2.0, 3.0])
        result = roundtrip(t)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, t)

    def test_torch_tensor_2d(self):
        import pyson.custom  # noqa: F401
        import torch

        t = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = roundtrip(t)
        assert result.shape == (2, 3)
        assert torch.equal(result, t)

    def test_torch_tensor_float64(self):
        import pyson.custom  # noqa: F401
        import torch

        t = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64)
        result = roundtrip(t)
        assert result.dtype == torch.float64
        assert torch.equal(result, t)

    def test_torch_tensor_int32(self):
        import pyson.custom  # noqa: F401
        import torch

        t = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = roundtrip(t)
        assert result.dtype == torch.int32
        assert torch.equal(result, t)

    def test_torch_tensor_bfloat16(self):
        import pyson.custom  # noqa: F401
        import torch

        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        result = roundtrip(t)
        assert result.dtype == torch.bfloat16
        assert torch.equal(result, t)

    def test_torch_tensor_multidimensional(self):
        import pyson.custom  # noqa: F401
        import torch

        t = torch.zeros((2, 3, 4, 5))
        result = roundtrip(t)
        assert result.shape == (2, 3, 4, 5)
        assert torch.equal(result, t)

    def test_torch_sparse_tensor(self):
        import pyson.custom  # noqa: F401
        import torch

        # Create a sparse COO tensor
        indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
        values = torch.tensor([1.0, 2.0, 3.0])
        t = torch.sparse_coo_tensor(indices, values, (3, 3))

        result = roundtrip(t)
        assert result.is_sparse
        # Compare dense representations
        assert torch.equal(result.to_dense(), t.to_dense())

    def test_torch_quantized_tensor(self):
        import pyson.custom  # noqa: F401
        import torch

        # Create a quantized tensor
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        qt = torch.quantize_per_tensor(t, scale=0.1, zero_point=0, dtype=torch.qint8)

        result = roundtrip(qt)
        assert result.is_quantized
        assert result.q_scale() == qt.q_scale()
        assert result.q_zero_point() == qt.q_zero_point()
        # Compare dequantized values
        assert torch.allclose(result.dequantize(), qt.dequantize())

    def test_torch_tensor_shared_reference(self):
        import pyson.custom  # noqa: F401
        import torch

        t = torch.tensor([1, 2, 3])
        container = [t, t]
        result = roundtrip(container)
        # Both should be the same tensor object
        assert result[0] is result[1]

    def test_mixed_torch_numpy_pandas(self):
        import pyson.custom  # noqa: F401
        import torch
        import numpy as np
        import pandas as pd

        tensor = torch.tensor([1.0, 2.0, 3.0])
        arr = np.array([4, 5, 6])
        series = pd.Series([7, 8, 9])

        container = {"tensor": tensor, "array": arr, "series": series}
        result = roundtrip(container)

        assert torch.equal(result["tensor"], tensor)
        assert np.array_equal(result["array"], arr)
        assert result["series"].equals(series)


class TestPersistentSerialization:
    """Test persistent serialization with custom dump functions."""

    def test_persistent_simple(self):
        from pyson import serialize, deserialize, register_persistent
        from pyson.serialize import persistent_table

        # Create a simple class with persistent serialization
        class CachedObject:
            def __init__(self, key, value):
                self.key = key
                self.value = value

        # Storage for persistent objects
        storage = {}

        def dump_cached(obj):
            storage[obj.key] = obj
            return obj.key

        register_persistent(CachedObject, dump_cached)

        obj = CachedObject("my-key", 42)
        payload = serialize(obj)
        json_str = payload.model_dump_json()

        # Deserialize with persistent_objects
        result = deserialize(json.loads(json_str), persistent_objects=storage)

        assert result.key == "my-key"
        assert result.value == 42
        # Should be the same object from storage
        assert result is storage["my-key"]

        # Cleanup
        del persistent_table[CachedObject]

    def test_persistent_in_container(self):
        from pyson import serialize, deserialize, register_persistent
        from pyson.serialize import persistent_table

        class ExternalResource:
            def __init__(self, resource_id, data):
                self.resource_id = resource_id
                self.data = data

        storage = {}

        def dump_resource(obj):
            storage[obj.resource_id] = obj
            return obj.resource_id

        register_persistent(ExternalResource, dump_resource)

        res = ExternalResource("res-123", {"value": 100})
        container = {"resource": res, "extra": [1, 2, 3]}

        payload = serialize(container)
        json_str = payload.model_dump_json()
        result = deserialize(json.loads(json_str), persistent_objects=storage)

        assert result["extra"] == [1, 2, 3]
        assert result["resource"].resource_id == "res-123"
        assert result["resource"].data == {"value": 100}

        # Cleanup
        del persistent_table[ExternalResource]

    def test_persistent_shared_reference(self):
        from pyson import serialize, deserialize, register_persistent
        from pyson.serialize import persistent_table

        class SingletonLike:
            def __init__(self, name):
                self.name = name

        storage = {}

        def dump_singleton(obj):
            storage[obj.name] = obj
            return obj.name

        register_persistent(SingletonLike, dump_singleton)

        obj = SingletonLike("the-one")
        container = [obj, obj, obj]

        payload = serialize(container)
        json_str = payload.model_dump_json()
        result = deserialize(json.loads(json_str), persistent_objects=storage)

        # All three should be the same object
        assert result[0] is result[1]
        assert result[1] is result[2]
        assert result[0].name == "the-one"

        # Cleanup
        del persistent_table[SingletonLike]

    def test_persistent_takes_priority_over_dispatch(self):
        from pyson import serialize, deserialize, register_persistent
        from pyson.serialize import persistent_table

        class SpecialList(list):
            pass

        storage = {}

        def dump_special(obj):
            key = f"special-{id(obj)}"
            storage[key] = SpecialList(obj)
            return key

        register_persistent(SpecialList, dump_special)

        obj = SpecialList([1, 2, 3])
        payload = serialize(obj)

        # Check that it was serialized as persistent, not as a list
        json_data = json.loads(payload.model_dump_json())
        memo = json_data["memo"]
        # Find the root object
        root_ref = str(json_data["obj"])
        root_entry = memo[root_ref]
        assert root_entry["type"] == "persistent"
        assert "persistent_id" in root_entry

        # Verify deserialization works
        result = deserialize(json_data, persistent_objects=storage)
        assert isinstance(result, SpecialList)
        assert list(result) == [1, 2, 3]

        # Cleanup
        del persistent_table[SpecialList]

    def test_persistent_id_attribute(self):
        """Test that objects with _persistent_id attribute are serialized persistently."""
        from pyson import serialize, deserialize

        class ResourceWithId:
            def __init__(self, resource_id, data):
                self._persistent_id = resource_id  # Magic attribute
                self.data = data

        storage = {}
        obj = ResourceWithId("resource-abc", {"value": 42})
        storage[obj._persistent_id] = obj

        payload = serialize(obj)
        json_str = payload.model_dump_json()

        # Check it was serialized as persistent
        json_data = json.loads(json_str)
        memo = json_data["memo"]
        root_ref = str(json_data["obj"])
        root_entry = memo[root_ref]
        assert root_entry["type"] == "persistent"
        assert root_entry["persistent_id"] == "resource-abc"

        # Deserialize
        result = deserialize(json_data, persistent_objects=storage)
        assert result is obj
        assert result.data == {"value": 42}

    def test_persistent_id_attribute_in_container(self):
        """Test _persistent_id objects inside containers."""
        from pyson import serialize, deserialize

        class Entity:
            def __init__(self, entity_id):
                self._persistent_id = entity_id
                self.entity_id = entity_id

        storage = {}
        e1 = Entity("entity-1")
        e2 = Entity("entity-2")
        storage[e1._persistent_id] = e1
        storage[e2._persistent_id] = e2

        container = {"entities": [e1, e2], "primary": e1}
        payload = serialize(container)
        json_str = payload.model_dump_json()

        result = deserialize(json.loads(json_str), persistent_objects=storage)

        assert result["entities"][0] is e1
        assert result["entities"][1] is e2
        assert result["primary"] is e1  # Shared reference preserved

    def test_persistent_id_attribute_priority(self):
        """Test that _persistent_id takes priority over persistent_table."""
        from pyson import serialize, deserialize, register_persistent
        from pyson.serialize import persistent_table

        class PriorityTest:
            def __init__(self, pid):
                self._persistent_id = pid

        # Register a dump function that would return a different ID
        register_persistent(PriorityTest, lambda obj: "from-table")

        storage = {}
        obj = PriorityTest("from-attribute")
        storage["from-attribute"] = obj

        payload = serialize(obj)
        json_data = json.loads(payload.model_dump_json())

        # Should use _persistent_id, not the dump function
        memo = json_data["memo"]
        root_ref = str(json_data["obj"])
        assert memo[root_ref]["persistent_id"] == "from-attribute"

        # Cleanup
        del persistent_table[PriorityTest]


class TestModuleSerialization:
    """Test module serialization (by reference and by value)."""

    def test_module_ref(self):
        """Test that installed modules are serialized by reference."""
        import json as json_mod

        payload = serialize(json_mod)
        json_str = payload.model_dump_json()
        json_data = json.loads(json_str)

        # Check it's a module_ref
        memo = json_data["memo"]
        root_ref = str(json_data["obj"])
        assert memo[root_ref]["type"] == "module_ref"
        assert memo[root_ref]["name"] == "json"

        # Deserialize and verify
        result = roundtrip(json_mod)
        assert result is json_mod

    def test_module_in_function_globals(self):
        """Test that modules in function globals are serialized correctly."""
        import math

        def uses_math(x):
            return math.sqrt(x)

        result = roundtrip(uses_math)
        assert result(16) == 4.0

    def test_module_ref_pandas(self):
        """Test pandas module serialization by reference."""
        import pandas as pd

        payload = serialize(pd)
        json_str = payload.model_dump_json()
        json_data = json.loads(json_str)

        memo = json_data["memo"]
        root_ref = str(json_data["obj"])
        assert memo[root_ref]["type"] == "module_ref"
        assert memo[root_ref]["name"] == "pandas"

        result = roundtrip(pd)
        assert result is pd


class TestFunctionRefSerialization:
    """Test function by-reference serialization."""

    def test_stdlib_function_ref(self):
        """Test that stdlib functions are serialized by reference."""
        from os.path import join

        payload = serialize(join)
        json_str = payload.model_dump_json()
        json_data = json.loads(json_str)

        memo = json_data["memo"]
        root_ref = str(json_data["obj"])
        assert memo[root_ref]["type"] == "function_ref"
        assert (
            memo[root_ref]["module"] == "posixpath"
            or memo[root_ref]["module"] == "ntpath"
        )
        assert memo[root_ref]["name"] == "join"

        result = roundtrip(join)
        assert result("a", "b") == join("a", "b")

    def test_installed_package_function_ref(self):
        """Test that functions from installed packages are by reference."""
        from json import dumps

        payload = serialize(dumps)
        json_str = payload.model_dump_json()
        json_data = json.loads(json_str)

        memo = json_data["memo"]
        root_ref = str(json_data["obj"])
        assert memo[root_ref]["type"] == "function_ref"
        assert memo[root_ref]["name"] == "dumps"

        result = roundtrip(dumps)
        assert result({"a": 1}) == '{"a": 1}'

    def test_function_by_value_vs_ref(self):
        """Test that local functions are by-value while stdlib are by-ref."""
        from math import sqrt

        def local_func(x):
            return x * 2

        # Local function should be by value
        payload_local = serialize(local_func)
        json_local = json.loads(payload_local.model_dump_json())
        memo_local = json_local["memo"]
        root_local = str(json_local["obj"])
        assert memo_local[root_local]["type"] == "function"

        # Stdlib function should be by reference
        payload_stdlib = serialize(sqrt)
        json_stdlib = json.loads(payload_stdlib.model_dump_json())
        memo_stdlib = json_stdlib["memo"]
        root_stdlib = str(json_stdlib["obj"])
        assert memo_stdlib[root_stdlib]["type"] == "function_ref"

    def test_function_ref_with_qualname(self):
        """Test function reference with qualified name (class method)."""
        # json.JSONEncoder.encode is a method on a class
        from json import JSONEncoder

        result = roundtrip(JSONEncoder.encode)
        encoder = JSONEncoder()
        assert result(encoder, {"a": 1}) == '{"a": 1}'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
