# pyson

**JSON-serializable Python object serialization.**

Unlike pickle, pyson produces human-readable JSON that can be inspected, modified, and transmitted over JSON-based protocols. It preserves object identity, handles circular references, and supports complex types including functions with closures.

## Installation

```bash
pip install pydantic cloudpickle
# Optional: for numpy/pandas/torch support
pip install numpy pandas torch
```

## Quick Start

```python
from pyson import serialize, deserialize
import json

# Serialize any Python object
data = {"users": [1, 2, 3], "nested": {"key": "value"}}
payload = serialize(data)
json_str = payload.model_dump_json()

# Deserialize back
result = deserialize(json.loads(json_str))
assert result == data
```

## Features

### Primitives & Collections

```python
# All JSON-native types
serialize(42)
serialize(3.14)
serialize("hello")
serialize(True)
serialize(None)

# Collections with full fidelity
serialize([1, 2, 3])           # Lists
serialize((1, 2, 3))           # Tuples (preserved as tuples!)
serialize({"a": 1, "b": 2})    # Dicts with any serializable keys
```

### Circular References & Shared Identity

```python
# Self-referencing structures work
x = [1, 2, 3]
x.append(x)  # Circular reference
result = roundtrip(x)
assert result[3] is result  # Identity preserved!

# Shared references maintained
shared = {"data": [1, 2, 3]}
container = [shared, shared, shared]
result = roundtrip(container)
assert result[0] is result[1] is result[2]  # Same object
```

### Custom Objects

```python
class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age

user = User("Alice", 30)
result = roundtrip(user)
assert result.name == "Alice"
assert result.age == 30
```

Supports `__getstate__`/`__setstate__` for custom serialization:

```python
class CachedObject:
    def __init__(self, value):
        self.value = value
        self._cache = {}  # Don't serialize this
    
    def __getstate__(self):
        return {"value": self.value}
    
    def __setstate__(self, state):
        self.value = state["value"]
        self._cache = {}  # Reinitialize
```

### Functions (with Source Code)

Functions from `__main__` or registered modules are serialized with their source code:

```python
from cloudpickle import register_pickle_by_value
import mymodule
register_pickle_by_value(mymodule)

def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

result = roundtrip(greet)
assert result("World") == "Hello, World!"
```

#### Closures

```python
def make_adder(n):
    def adder(x):
        return x + n
    return adder

add_5 = make_adder(5)
result = roundtrip(add_5)
assert result(10) == 15  # Closure preserved!
```

#### Globals, Defaults, Annotations

```python
MULTIPLIER = 10

def process(x: int, factor=2) -> int:
    """Multiply x by factor and MULTIPLIER."""
    return x * factor * MULTIPLIER

result = roundtrip(process)
assert result(5) == 100
assert result.__doc__ == "Multiply x by factor and MULTIPLIER."
assert result.__annotations__ == {"x": int, "factor": int, "return": int}
```

### Dynamic Classes

Classes from `__main__` or registered modules are serialized by value:

```python
class MyClass:
    """A dynamic class."""
    class_attr = 42
    
    def __init__(self, value):
        self.value = value
    
    def double(self):
        return self.value * 2
    
    @classmethod
    def from_string(cls, s):
        return cls(int(s))
    
    @staticmethod
    def helper(x):
        return x + 1

ResultClass = roundtrip(MyClass)
obj = ResultClass(10)
assert obj.double() == 20
assert ResultClass.class_attr == 42
```

### Properties, Methods, Descriptors

```python
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5/9

ResultClass = roundtrip(Temperature)
temp = ResultClass(100)
assert temp.fahrenheit == 212
```

## Custom Serializers

Register custom serializers for any type:

```python
from pyson import register_serializer, SerializedType
from typing import Literal

class ComplexType(SerializedType):
    type: Literal["complex"] = "complex"
    real: float
    imag: float

    @classmethod
    def serialize(cls, obj: complex, context):
        return cls(real=obj.real, imag=obj.imag)

    def deserialize(self, referenceID, context):
        result = complex(self.real, self.imag)
        context.memo[referenceID] = result
        return result

register_serializer(complex, ComplexType)

# Now complex numbers work!
result = roundtrip(3+4j)
assert result == 3+4j
```

## Persistent Serialization

For large objects or external resources, serialize as just an ID.

### Using `_persistent_id` Attribute

The simplest way: add a `_persistent_id` attribute to your object:

```python
from pyson import serialize, deserialize

class Resource:
    def __init__(self, resource_id, data):
        self._persistent_id = resource_id  # Magic attribute!
        self.data = data

resource = Resource("res-123", heavy_data)
payload = serialize(resource)
# Payload just contains: {"type": "persistent", "persistent_id": "res-123"}

# On deserialize, provide the objects mapping
result = deserialize(payload, persistent_objects={"res-123": resource})
assert result is resource  # Same object!
```

### Using `register_persistent`

For types you don't control, register a dump function:

```python
from pyson import serialize, deserialize, register_persistent

class LargeModel:
    def __init__(self, path):
        self.path = path
        self.weights = load_weights(path)  # Large data

# Register: only store the path
register_persistent(LargeModel, lambda m: m.path)

model = LargeModel("/models/gpt.pt")
payload = serialize(model)

# On deserialize, provide the objects mapping
models = {"/models/gpt.pt": model}
result = deserialize(json.loads(payload.model_dump_json()), persistent_objects=models)
assert result is model  # Same object!
```

**Note:** `_persistent_id` takes priority over `register_persistent` if both are present.

## NumPy, Pandas & PyTorch

Import `pyson.custom` to enable support:

```python
import pyson.custom  # Registers serializers for numpy, pandas, torch
from pyson import serialize, deserialize

# NumPy arrays
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
result = roundtrip(arr)
assert np.array_equal(result, arr)
assert result.dtype == np.float32

# Pandas DataFrames
import pandas as pd
df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
result = roundtrip(df)
assert df.equals(result)

# Pandas Series
series = pd.Series([1, 2, 3], index=["a", "b", "c"], name="values")
result = roundtrip(series)
assert series.equals(result)

# PyTorch Tensors
import torch
tensor = torch.randn(3, 4)
result = roundtrip(tensor)
assert torch.equal(result, tensor)
```

### Special Tensor Types

```python
# Sparse tensors (preserved as sparse)
indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
values = torch.tensor([1.0, 2.0, 3.0])
sparse = torch.sparse_coo_tensor(indices, values, (3, 3))
result = roundtrip(sparse)
assert result.is_sparse

# Quantized tensors (preserves scale/zero_point)
t = torch.tensor([1.0, 2.0, 3.0])
qt = torch.quantize_per_tensor(t, scale=0.1, zero_point=0, dtype=torch.qint8)
result = roundtrip(qt)
assert result.is_quantized
assert result.q_scale() == 0.1

# bfloat16 (preserved exactly)
bf16 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
result = roundtrip(bf16)
assert result.dtype == torch.bfloat16
```

## API Reference

### Core Functions

```python
from pyson import serialize, deserialize

# Serialize to Payload (Pydantic model)
payload = serialize(obj)
json_str = payload.model_dump_json()

# Deserialize from dict
result = deserialize(json.loads(json_str))

# With persistent objects
result = deserialize(data, persistent_objects={"id": obj})
```

### Registration Functions

```python
from pyson import register_serializer, register_persistent

# Custom type serializer
register_serializer(MyType, MyTypeSerializer)
register_serializer((Type1, Type2), SharedSerializer)  # Multiple types

# Persistent serialization
register_persistent(MyType, dump_fn)  # dump_fn(obj) -> str
```

### Base Classes

```python
from pyson import SerializedType

class MySerializer(SerializedType):
    type: Literal["my_type"] = "my_type"
    # ... fields
    
    @classmethod
    def serialize(cls, obj, context):
        # context.serialize(nested_obj) for nested objects
        return cls(...)
    
    def deserialize(self, referenceID, context):
        # context.deserialize(ref_id) for nested objects
        result = ...
        context.memo[referenceID] = result  # Important for cycles!
        return result
```

## Comparison with Pickle

| Feature | pyson | pickle |
|---------|-------|--------|
| Output format | JSON (human-readable) | Binary |
| Cross-language | ✅ (JSON) | ❌ |
| Inspectable | ✅ | ❌ |
| Functions | Source code | Bytecode |
| Security | Safer (no arbitrary code) | ⚠️ Unsafe |
| Custom serializers | ✅ `register_serializer` | ✅ `__reduce__` |
| Circular refs | ✅ | ✅ |

## License

MIT

