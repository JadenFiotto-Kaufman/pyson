# API Additions for nnsight Integration

This document describes the new API features added to pyson for nnsight's remote
execution use case.

## Security Model: Defense in Depth

pyson's security approach uses two complementary layers:

| Layer | Location | Purpose |
|-------|----------|---------|
| **pyson linting** | Client-side | Early feedback with helpful error messages |
| **RestrictedPython** | Server-side | Runtime sandboxing and enforcement |

### Why Two Layers?

**Client-side linting** provides a better user experience:
- Catches common mistakes before network transfer
- Gives actionable guidance ("Convert DataFrame to tensor")
- Fast feedback loop during development

**Server-side RestrictedPython** provides actual security:
- Cannot be bypassed by malicious clients
- Enforces restrictions at runtime
- Prevents execution of dangerous code even if it passes linting

### Important Security Note

**Linting is NOT a security boundary.** A malicious client can always craft
payloads that bypass client-side validation. The linting layer exists purely
for user experience - to help well-intentioned users fix their code before
submission.

**All security enforcement must happen server-side** using RestrictedPython
or equivalent sandboxing. See section 7 for the server-side helper.

## 1. Configurable Linting (`lint` parameter)

The `serialize()` function now accepts an optional `lint` parameter that enables
validation of serialized content.

### Basic Usage

```python
from pyson import serialize
from pyson.lint import NNSIGHT_CONFIG

# Validate against nnsight's default forbidden items
payload = serialize(my_function, lint=NNSIGHT_CONFIG)
```

### Custom Configuration

```python
from pyson.lint import LintConfig

config = LintConfig(
    forbidden_modules=frozenset({'os', 'subprocess', 'socket'}),
    forbidden_calls=frozenset({'eval', 'exec', 'compile'}),
    forbidden_attr_chains=frozenset({('torch', 'save'), ('torch', 'load')}),
    forbidden_classes={'DataFrame': 'Convert to tensor: torch.tensor(df.values)'},
    reject_nonlocal=True,
    validate_source=True,
)

payload = serialize(my_function, lint=config)
```

### LintConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `forbidden_modules` | `FrozenSet[str]` | `frozenset()` | Module prefixes that cannot be imported |
| `forbidden_calls` | `FrozenSet[str]` | `frozenset()` | Function names that cannot be called |
| `forbidden_attr_chains` | `FrozenSet[Tuple[str,...]]` | `frozenset()` | Attribute chains like `('torch', 'save')` |
| `forbidden_classes` | `Dict[str, str]` | `{}` | Class names with helpful error messages |
| `reject_nonlocal` | `bool` | `True` | Reject functions with `nonlocal` declarations |
| `validate_source` | `bool` | `True` | Parse and validate function source code |

### Pre-configured: NNSIGHT_CONFIG

The `NNSIGHT_CONFIG` preset includes all forbidden items from nnsight's source
serialization validation:

```python
from pyson.lint import NNSIGHT_CONFIG, FORBIDDEN_MODULE_PREFIXES, FORBIDDEN_CALLS

# View what's forbidden
print(FORBIDDEN_MODULE_PREFIXES)  # os, subprocess, socket, etc.
print(FORBIDDEN_CALLS)            # eval, exec, compile, etc.
```

## 2. Server-Provided Attributes

For remote execution, certain object attributes should not be serialized because
they represent large resources (like AI models) that exist on the server. Instead,
the server injects these values during deserialization.

### Defining Server-Provided Attributes

Add a `_server_provided` class attribute with a frozenset of attribute names:

```python
class LanguageModel:
    _server_provided = frozenset({'_module', '_tokenizer'})

    def __init__(self, model_name):
        self.model_name = model_name  # This gets serialized
        self._module = load_model(model_name)  # This is skipped
        self._tokenizer = load_tokenizer(model_name)  # This is skipped
```

### How It Works

**During serialization:**
- Attributes in `_server_provided` are skipped
- The skipped attribute names are recorded in the payload

**During deserialization:**
- Pass a `server_provided` dict with the server's values
- These are injected into the reconstructed objects

```python
# Client side
payload = serialize(my_model)
send_to_server(payload.model_dump_json())

# Server side
payload_dict = json.loads(received_json)
result = deserialize(
    payload_dict,
    server_provided={
        '_module': server_model_module,
        '_tokenizer': server_tokenizer,
    }
)
```

### Inheritance

Server-provided attributes are collected from the entire class hierarchy (MRO):

```python
class BaseModel:
    _server_provided = frozenset({'_cache'})

class LanguageModel(BaseModel):
    _server_provided = frozenset({'_module', '_tokenizer'})

# LanguageModel instances skip: _cache, _module, _tokenizer
```

## 3. Nonlocal Rejection

Functions using the `nonlocal` keyword cannot be correctly serialized because
multiple closures sharing the same cell would lose their shared state.

### The Problem

```python
def make_counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    def decrement():
        nonlocal count
        count -= 1
        return count
    return increment, decrement

inc, dec = make_counter()
inc()  # 1
inc()  # 2
dec()  # 1 - they share `count`
```

After serialization/deserialization, `inc` and `dec` would each have their own
copy of `count` instead of sharing it.

### Detection

When linting is enabled with `reject_nonlocal=True`, pyson detects nonlocal
usage and raises a helpful error:

```
ValueError: Function 'increment' uses 'nonlocal count'.
Functions with nonlocal cannot be serialized because multiple closures
sharing the same cell would lose shared state.

Refactor to use a class:
    class Counter:
        def __init__(self): self.count = 0
        def increment(self): self.count += 1; return self.count
```

### Detection Method

The `check_nonlocal()` function in `pyson.lint` walks the function's AST to find
`nonlocal` statements:

```python
from pyson.lint import check_nonlocal

def my_func():
    nonlocal x
    x += 1

check_nonlocal(my_func)  # Returns {'x'}
```

## 4. Forbidden Type Detection

Certain types cannot be meaningfully serialized for remote execution. When
linting is enabled, these are detected early with helpful error messages.

### How It Works

The `forbidden_classes` field in `LintConfig` maps class names to error messages:

```python
config = LintConfig(
    forbidden_classes={
        'DataFrame': 'pandas.DataFrame cannot be serialized.\n'
                     'Convert to a tensor before serialization:\n'
                     '  tensor_data = torch.tensor(df.values)',
        'Figure': 'matplotlib Figure cannot be serialized.\n'
                  'Extract the data you need instead of the figure.',
    }
)
```

### Pre-configured Types

`NNSIGHT_CONFIG` includes helpful messages for:
- `DataFrame` - pandas DataFrames
- `Series` - pandas Series
- `Figure` - matplotlib figures
- `Axes` - matplotlib axes
- `TextIOWrapper`, `BufferedReader`, etc. - file handles
- `module` - module objects (should be imported by name)
- `Lock`, `RLock`, `Semaphore`, etc. - threading primitives

### Error Output

```
ValueError: Cannot serialize DataFrame:
pandas.DataFrame cannot be serialized.
Convert to a tensor before serialization:
  tensor_data = torch.tensor(df.values)
```

## 5. Source Code Validation

When `validate_source=True` in the lint config, function source code is parsed
and validated for forbidden patterns.

### What's Validated

1. **Import statements** - Checked against `forbidden_modules`
2. **Function calls** - Checked against `forbidden_calls`
3. **Attribute access chains** - Checked against `forbidden_attr_chains`

### Example

```python
def unsafe_function():
    import os  # Forbidden module
    os.system('rm -rf /')  # Forbidden call pattern

# With linting enabled:
serialize(unsafe_function, lint=NNSIGHT_CONFIG)
# Raises: ValueError: Function 'unsafe_function' imports forbidden module: os
```

## 6. Persistent Object Serialization

Objects that should be serialized by reference only (not by value) can use
persistent serialization.

### Registration-Based

```python
from pyson import register_persistent

# Register a type with a function that returns its ID
register_persistent(LargeModel, lambda m: m.model_id)

# Serialization uses only the ID
model = LargeModel("gpt-3.5-turbo")
payload = serialize(model)

# Deserialization requires the object to be provided
result = deserialize(
    payload_dict,
    persistent_objects={"gpt-3.5-turbo": model}
)
```

### Attribute-Based

For per-instance control, set `_persistent_id` on the object:

```python
model = LargeModel()
model._persistent_id = "custom-model-v1"

payload = serialize(model)  # Uses "custom-model-v1" as the ID
```

## Complete Example

```python
from pyson import serialize, deserialize
from pyson.lint import NNSIGHT_CONFIG
import json

# Define a model class with server-provided attributes
class MyLanguageModel:
    _server_provided = frozenset({'_module', '_tokenizer'})

    def __init__(self, name):
        self.name = name
        self._module = None  # Set by server
        self._tokenizer = None  # Set by server

    def generate(self, prompt):
        return self._module.generate(self._tokenizer.encode(prompt))

# Define an intervention function
def my_intervention(model, hidden_states):
    # This function uses the model but doesn't serialize it
    return hidden_states * 2

# Client: Serialize with validation
model = MyLanguageModel("gpt2")
payload = serialize(
    {"model": model, "intervention": my_intervention},
    lint=NNSIGHT_CONFIG
)
json_str = payload.model_dump_json()

# Server: Deserialize with server-provided values
server_module = load_actual_model("gpt2")
server_tokenizer = load_tokenizer("gpt2")

payload_dict = json.loads(json_str)
data = deserialize(
    payload_dict,
    server_provided={
        '_module': server_module,
        '_tokenizer': server_tokenizer,
    }
)

# Now data["model"]._module is the server's loaded model
result = data["intervention"](data["model"], hidden_states)
```

## 7. Server-Side RestrictedPython Integration

For secure server-side execution, pyson provides a helper module that integrates
with RestrictedPython. This ensures that deserialized functions execute in a
sandboxed environment.

### Installation

RestrictedPython is an optional dependency:

```bash
pip install RestrictedPython
```

### Basic Usage

```python
from pyson.restricted import deserialize_restricted
import json

# Receive payload from client
payload_dict = json.loads(received_json)

# Deserialize with RestrictedPython enforcement
data = deserialize_restricted(
    payload_dict,
    server_provided={
        '_module': server_model,
        '_tokenizer': server_tokenizer,
    }
)

# Functions are now wrapped to execute in restricted mode
result = data["intervention"](hidden_states)
```

### How It Works

1. **Deserialization** proceeds normally, reconstructing objects
2. **Functions** serialized by value (FunctionType) are wrapped in a restricted executor
3. **When called**, the wrapper compiles the source with `compile_restricted` and executes with safe builtins

### What RestrictedPython Blocks

RestrictedPython prevents:
- Attribute access starting with `_` (no `obj._private`)
- Direct `__dict__` access
- `getattr`/`setattr`/`delattr` on arbitrary objects
- Iteration that could cause denial of service
- Many other potentially dangerous patterns

### Custom Guards

You can provide custom guards for RestrictedPython:

```python
from RestrictedPython import safe_globals
from RestrictedPython.Guards import guarded_iter_unpack_sequence

custom_globals = {
    **safe_globals,
    '_iter_unpack_sequence_': guarded_iter_unpack_sequence,
    # Add allowed modules/functions
    'torch': torch,
    'numpy': np,
}

data = deserialize_restricted(
    payload_dict,
    restricted_globals=custom_globals,
)
```

### Error Handling

RestrictedPython compilation errors are raised as `RestrictedExecutionError`:

```python
from pyson.restricted import RestrictedExecutionError

try:
    result = my_function(args)
except RestrictedExecutionError as e:
    print(f"Code violated security policy: {e}")
```

### Complete Server Example

```python
from pyson.restricted import deserialize_restricted, RestrictedExecutionError
from pyson import deserialize
import json
import torch

def handle_intervention_request(json_payload: str, model, tokenizer):
    """Process a client intervention request securely."""
    payload_dict = json.loads(json_payload)

    try:
        # Deserialize with security enforcement
        data = deserialize_restricted(
            payload_dict,
            server_provided={
                '_module': model,
                '_tokenizer': tokenizer,
            },
            restricted_globals={
                'torch': torch,  # Allow torch operations
            }
        )

        # Execute the intervention
        with torch.no_grad():
            result = data["intervention"](data["input"])

        return {"status": "success", "result": result.tolist()}

    except RestrictedExecutionError as e:
        return {"status": "error", "message": f"Security violation: {e}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

### When to Use Each Layer

| Scenario | Use Linting? | Use RestrictedPython? |
|----------|--------------|----------------------|
| Development/testing | Yes | Optional |
| Production client | Yes | N/A (client-side) |
| Production server | N/A | **Required** |
| Trusted internal use | Optional | Optional |

The linting layer helps users write correct code. The RestrictedPython layer
ensures security regardless of what code is submitted.
