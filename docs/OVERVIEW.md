# pyson: JSON-Serializable Python Object Serialization

## Overview

pyson is a Python serialization library that produces human-readable JSON output
while preserving object identity, circular references, and complex types. Unlike
pickle (which produces opaque binary output), pyson's output can be inspected,
debugged, and transmitted over JSON-based protocols.

## Key Features

### 1. Human-Readable JSON Output
All serialized data is valid JSON, viewable in any JSON editor or debugger:

```python
from pyson import serialize
import json

data = {"users": [{"name": "Alice", "scores": [95, 87, 92]}]}
payload = serialize(data)
print(json.dumps(json.loads(payload.model_dump_json()), indent=2))
```

### 2. Pydantic-Based Type System
pyson uses Pydantic models for type validation and serialization:
- Strong typing with discriminated unions
- Automatic JSON schema generation
- Runtime type validation
- Easy extension via custom SerializedType subclasses

### 3. Object Identity Preservation
Objects are tracked by their `id()` in a memo table:
- Circular references are handled correctly
- Shared objects maintain identity across references
- The same object serialized twice produces the same reference ID

### 4. Source-Based Function Serialization
Functions are serialized as source code (not bytecode):
- Readable and debuggable
- Platform-independent
- Inspectable before execution
- Uses cloudpickle's `_function_getstate` for robust extraction

### 5. Configurable Validation (Linting)
Optional validation prevents serialization of unsafe content:
- Forbidden types (DataFrames, file handles, etc.)
- Nonlocal closures (which lose shared state)
- Dangerous imports and function calls

## Benefits for nnsight

pyson was designed with nnsight's remote execution use case in mind:

### 1. Secure Remote Execution
The linting system validates code before it's transmitted to remote servers:

```python
from pyson import serialize
from pyson.lint import NNSIGHT_CONFIG

# Will raise with helpful error message if unsafe
payload = serialize(intervention_function, lint=NNSIGHT_CONFIG)
```

Validation includes:
- **Forbidden modules**: `os`, `subprocess`, `socket`, etc.
- **Forbidden calls**: `eval()`, `exec()`, `compile()`, etc.
- **Forbidden types**: DataFrames, matplotlib figures, file objects

### 2. Server-Provided Attributes
Large AI model instances shouldn't be serialized and transmitted. Classes can
mark attributes that the server will provide:

```python
class StandardizedTransformer(LanguageModel):
    _server_provided = frozenset({'_module', '_tokenizer', '_model'})
```

During serialization, these attributes are skipped. During deserialization on
the server, they're injected from the execution environment:

```python
result = deserialize(
    payload_dict,
    server_provided={
        '_module': server_model_module,
        '_tokenizer': server_tokenizer,
    }
)
```

### 3. Clear Error Messages
Instead of cryptic serialization failures, pyson provides actionable guidance:

```
Cannot serialize DataFrame:
pandas.DataFrame cannot be serialized.
Convert to a tensor before serialization:
  tensor_data = torch.tensor(df.values)
```

### 4. Persistent Object References
Large objects stored externally can be serialized by reference only:

```python
from pyson import register_persistent

register_persistent(LargeModel, lambda m: m.model_id)
```

Or use the `_persistent_id` attribute for per-instance control:

```python
model._persistent_id = "gpt-3.5-turbo"
```

### 5. Extensible Type System
Custom serializers can be registered for any type:

```python
from pyson import register_serializer, SerializedType
from typing import Literal

class TensorType(SerializedType):
    type: Literal["tensor"] = "tensor"
    data: list[float]
    shape: list[int]
    dtype: str

    @classmethod
    def serialize(cls, obj, context):
        return cls(
            data=obj.tolist(),
            shape=list(obj.shape),
            dtype=str(obj.dtype),
        )

    def deserialize(self, referenceID, context):
        import torch
        result = torch.tensor(self.data, dtype=getattr(torch, self.dtype))
        result = result.reshape(self.shape)
        context.memo[referenceID] = result
        return result

register_serializer(torch.Tensor, TensorType)
```

## Comparison with Alternatives

| Feature | pyson | pickle | cloudpickle |
|---------|-------|--------|-------------|
| Output format | JSON | Binary | Binary |
| Human-readable | Yes | No | No |
| Cross-platform | Yes | Limited | Limited |
| Security validation | Yes (with lint) | No | No |
| Server-provided attrs | Yes | No | No |
| Helpful error messages | Yes | No | No |
| Type system | Pydantic | None | None |
| Custom serializers | Easy | Complex | Complex |

## Architecture

```
serialize(obj)
    |
    v
SerializationContext
    |
    +-- Check memo (already serialized?)
    |
    +-- Check lint config (forbidden type?)
    |
    +-- Check persistent_table (serialize by ID only?)
    |
    +-- Dispatch to SerializedType based on type
    |       |
    |       +-- PrimitiveType (int, float, bool, str, None)
    |       +-- ListType, TupleType, DictType
    |       +-- ObjectType (generic objects)
    |       +-- FunctionType / FunctionRefType
    |       +-- ClassRefType / DynamicClassType
    |       +-- MethodType, PropertyType
    |       +-- PersistentType
    |
    v
Payload(memo={...}, obj=root_ref_id)
    |
    v
.model_dump_json() -> JSON string
```

## Getting Started

```python
from pyson import serialize, deserialize
import json

# Basic serialization
payload = serialize({"key": [1, 2, 3]})
json_str = payload.model_dump_json()

# Deserialization
result = deserialize(json.loads(json_str))

# With linting (for remote execution)
from pyson.lint import NNSIGHT_CONFIG
payload = serialize(my_function, lint=NNSIGHT_CONFIG)
```
