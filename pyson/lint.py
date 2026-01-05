"""
Static source code linting for pyson serialization.

This module provides configurable static analysis to enforce host-chosen
standards on serialized code. It performs AST-based validation to detect
patterns that the host application wants to reject, providing clear and
actionable error messages to users.

Purpose:
    This is a CLIENT-SIDE linting tool for user experience, NOT a security
    boundary. It helps users fix their code before submission by catching
    common issues early with helpful suggestions.

    For server-side security enforcement, see pyson.restricted which provides
    RestrictedPython integration for runtime sandboxing.

Checks performed:
    1. Forbidden module imports (configurable list)
    2. Forbidden function calls (e.g., eval, exec, open)
    3. Forbidden attribute chains (e.g., os.system, subprocess.run)
    4. Forbidden types with helpful error messages
    5. Nonlocal closure variables (cannot preserve shared cell semantics)

Configuration:
    The LintConfig class allows customizing which patterns are forbidden.
    Use the predefined NNSIGHT_CONFIG for nnsight-compatible validation,
    or create a custom config for other use cases.

    >>> from pyson import serialize
    >>> from pyson.lint import LintConfig, NNSIGHT_CONFIG
    >>>
    >>> # Use nnsight's configuration
    >>> payload = serialize(obj, lint=NNSIGHT_CONFIG)
    >>>
    >>> # Or create a custom configuration
    >>> my_config = LintConfig(
    ...     forbidden_modules={'mymodule'},
    ...     reject_nonlocal=True,
    ... )
    >>> payload = serialize(obj, lint=my_config)
"""

from __future__ import annotations

import ast
import builtins
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Type, Union


# =============================================================================
# Forbidden Module Prefixes
# =============================================================================
# Objects from these modules will be immediately rejected with a clear message.
# These represent OS/system resources, database connections, or test frameworks
# that cannot be meaningfully serialized.

FORBIDDEN_MODULE_PREFIXES: FrozenSet[str] = frozenset({
    # Test frameworks - these leak into scope during testing
    '_pytest',
    'pytest',
    'unittest',

    # OS/System resources - represent state that cannot be transferred
    'socket',           # Network sockets
    'multiprocessing',  # Process resources, queues, locks
    'asyncio',          # Async event loops, tasks
    'concurrent',       # Thread pools, futures
    'queue',            # Thread communication queues
    'subprocess',       # OS subprocesses

    # Database connections - represent network/session state
    'sqlite3',
    'sqlalchemy',
    'pymongo',
    'redis',
    'psycopg',          # PostgreSQL
    'psycopg2',
    'mysql',
    'pymysql',

    # Logging - handlers often have file references
    'logging',
})


# =============================================================================
# Forbidden Function Calls
# =============================================================================
# These function names are never allowed in serializable code.

FORBIDDEN_CALLS: FrozenSet[str] = frozenset({
    'open',
    'exec',
    'eval',
    'compile',
    'input',
    '__import__',
})


# =============================================================================
# Forbidden Attribute Chains
# =============================================================================
# Attribute chains that indicate dangerous operations.
# Format: tuple of attribute names. Matching is prefix-based.

FORBIDDEN_ATTR_CHAINS: FrozenSet[Tuple[str, ...]] = frozenset({
    # os module - block process and filesystem operations
    ('os', 'system'), ('os', 'popen'), ('os', 'spawn'), ('os', 'spawnl'),
    ('os', 'spawnle'), ('os', 'spawnlp'), ('os', 'spawnlpe'), ('os', 'spawnv'),
    ('os', 'spawnve'), ('os', 'spawnvp'), ('os', 'spawnvpe'),
    ('os', 'exec'), ('os', 'execl'), ('os', 'execle'), ('os', 'execlp'),
    ('os', 'execlpe'), ('os', 'execv'), ('os', 'execve'), ('os', 'execvp'),
    ('os', 'execvpe'),
    ('os', 'fork'), ('os', 'forkpty'), ('os', 'kill'), ('os', 'killpg'),
    ('os', 'remove'), ('os', 'unlink'), ('os', 'rmdir'), ('os', 'removedirs'),
    ('os', 'rename'), ('os', 'renames'), ('os', 'replace'),
    ('os', 'mkdir'), ('os', 'makedirs'), ('os', 'symlink'), ('os', 'link'),
    ('os', 'chdir'), ('os', 'chroot'), ('os', 'chmod'), ('os', 'chown'),
    ('os', 'lchown'), ('os', 'chflags'), ('os', 'lchflags'),
    ('os', 'open'), ('os', 'fdopen'), ('os', 'read'), ('os', 'write'),
    ('os', 'truncate'), ('os', 'ftruncate'),

    # subprocess module
    ('subprocess', 'run'), ('subprocess', 'call'), ('subprocess', 'Popen'),
    ('subprocess', 'check_call'), ('subprocess', 'check_output'),
    ('subprocess', 'getoutput'), ('subprocess', 'getstatusoutput'),

    # socket module (any access)
    ('socket',),

    # urllib module (any access)
    ('urllib',),

    # requests module (any access)
    ('requests',),

    # shutil module (file operations)
    ('shutil',),

    # pathlib - block I/O operations but allow path manipulation
    ('pathlib', 'Path', 'read_text'), ('pathlib', 'Path', 'write_text'),
    ('pathlib', 'Path', 'read_bytes'), ('pathlib', 'Path', 'write_bytes'),
    ('pathlib', 'Path', 'open'), ('pathlib', 'Path', 'unlink'),
    ('pathlib', 'Path', 'rmdir'), ('pathlib', 'Path', 'rename'),
    ('pathlib', 'Path', 'replace'), ('pathlib', 'Path', 'symlink_to'),
    ('pathlib', 'Path', 'link_to'), ('pathlib', 'Path', 'mkdir'),
    ('pathlib', 'Path', 'touch'), ('pathlib', 'Path', 'chmod'),
    ('pathlib', 'Path', 'lchmod'),
})


# =============================================================================
# Forbidden Classes with Messages
# =============================================================================
# Specific classes that cannot be serialized, with helpful error messages.

FORBIDDEN_CLASSES: Dict[str, str] = {
    # Pandas - massive dependency explosion, convert to tensor instead
    'pandas.core.frame.DataFrame': (
        "pandas.DataFrame cannot be serialized.\n"
        "DataFrames have complex internal state that cannot be transferred.\n"
        "\n"
        "Convert to a tensor before serialization:\n"
        "  tensor_data = torch.tensor(df.values)\n"
        "Or extract specific columns:\n"
        "  values = df['column'].tolist()"
    ),
    'pandas.core.series.Series': (
        "pandas.Series cannot be serialized.\n"
        "\n"
        "Convert to a tensor or list:\n"
        "  tensor_data = torch.tensor(series.values)\n"
        "  values = series.tolist()"
    ),

    # Matplotlib - contains rendering state and callbacks
    'matplotlib.figure.Figure': (
        "matplotlib.Figure cannot be serialized.\n"
        "Figures contain rendering state and callbacks that cannot be transferred.\n"
        "\n"
        "If you need to pass image data, save to bytes first:\n"
        "  import io\n"
        "  buf = io.BytesIO()\n"
        "  fig.savefig(buf, format='png')\n"
        "  image_bytes = buf.getvalue()"
    ),
    'matplotlib.axes._axes.Axes': (
        "matplotlib.Axes cannot be serialized.\n"
        "Axes contain rendering state bound to a Figure.\n"
        "\n"
        "Access the data you need before serialization instead."
    ),
    'matplotlib.axes._subplots.Axes': (
        "matplotlib.Axes cannot be serialized.\n"
        "Axes contain rendering state bound to a Figure.\n"
        "\n"
        "Access the data you need before serialization instead."
    ),
    'matplotlib.axes._subplots.AxesSubplot': (
        "matplotlib.Axes cannot be serialized.\n"
        "Axes contain rendering state bound to a Figure.\n"
        "\n"
        "Access the data you need before serialization instead."
    ),

    # PIL/Pillow - convert to tensor
    'PIL.Image.Image': (
        "PIL.Image cannot be serialized.\n"
        "\n"
        "Convert to a tensor first:\n"
        "  from torchvision import transforms\n"
        "  tensor = transforms.ToTensor()(image)"
    ),

    # Scipy sparse matrices - convert to dense or use specific format
    'scipy.sparse._csr.csr_matrix': (
        "scipy.sparse.csr_matrix cannot be serialized.\n"
        "\n"
        "Convert to a dense tensor:\n"
        "  tensor = torch.tensor(sparse_matrix.toarray())\n"
        "Or extract the CSR components if you need sparse format."
    ),
    'scipy.sparse._csc.csc_matrix': (
        "scipy.sparse.csc_matrix cannot be serialized.\n"
        "\n"
        "Convert to a dense tensor:\n"
        "  tensor = torch.tensor(sparse_matrix.toarray())"
    ),

    # File handles
    '_io.TextIOWrapper': (
        "File handles cannot be serialized.\n"
        "\n"
        "Read the content first:\n"
        "  content = f.read()"
    ),
    '_io.BufferedReader': (
        "File handles cannot be serialized.\n"
        "\n"
        "Read the content first:\n"
        "  data = f.read()"
    ),
    '_io.BufferedWriter': (
        "File handles cannot be serialized.\n"
        "\n"
        "Write operations cannot be transferred to remote execution."
    ),
}


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class LintConfig:
    """
    Configuration for serialization linting.

    This class allows customizing which patterns are forbidden during
    serialization. Use the predefined NNSIGHT_CONFIG for nnsight-compatible
    validation, or create a custom config for other use cases.

    Attributes:
        forbidden_modules: Set of module prefixes that cannot be serialized.
            Objects from these modules will be rejected.
        forbidden_calls: Set of function names that cannot be called in
            serialized code.
        forbidden_attr_chains: Set of attribute chain tuples that indicate
            dangerous operations.
        forbidden_classes: Dict mapping fully qualified class names to
            error messages with suggestions.
        reject_nonlocal: If True, reject functions that use `nonlocal`.
        validate_source: If True, validate function/class source code
            for forbidden imports and calls.

    Example:
        >>> config = LintConfig(
        ...     forbidden_modules={'mypackage'},
        ...     forbidden_calls={'dangerous_function'},
        ...     reject_nonlocal=True,
        ... )
        >>> payload = serialize(obj, lint=config)
    """

    forbidden_modules: FrozenSet[str] = field(default_factory=frozenset)
    forbidden_calls: FrozenSet[str] = field(default_factory=frozenset)
    forbidden_attr_chains: FrozenSet[Tuple[str, ...]] = field(default_factory=frozenset)
    forbidden_classes: Dict[str, str] = field(default_factory=dict)
    reject_nonlocal: bool = True
    validate_source: bool = True

    def merge_with_defaults(self) -> 'LintConfig':
        """
        Create a new config that includes default forbidden items.

        Returns a new LintConfig with this config's items merged with
        the module-level defaults.
        """
        return LintConfig(
            forbidden_modules=FORBIDDEN_MODULE_PREFIXES | self.forbidden_modules,
            forbidden_calls=FORBIDDEN_CALLS | self.forbidden_calls,
            forbidden_attr_chains=FORBIDDEN_ATTR_CHAINS | self.forbidden_attr_chains,
            forbidden_classes={**FORBIDDEN_CLASSES, **self.forbidden_classes},
            reject_nonlocal=self.reject_nonlocal,
            validate_source=self.validate_source,
        )


# Pre-configured LintConfig for nnsight remote execution
# This includes all the forbidden patterns from nnsight's source-serialization
NNSIGHT_CONFIG = LintConfig(
    forbidden_modules=FORBIDDEN_MODULE_PREFIXES,
    forbidden_calls=FORBIDDEN_CALLS,
    forbidden_attr_chains=FORBIDDEN_ATTR_CHAINS,
    forbidden_classes=FORBIDDEN_CLASSES,
    reject_nonlocal=True,
    validate_source=True,
)


# =============================================================================
# Error Classes
# =============================================================================


@dataclass
class LintError:
    """A linting error with location and suggestion."""
    message: str
    line: Optional[int] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        parts = []
        if self.line:
            parts.append(f"Line {self.line}: {self.message}")
        else:
            parts.append(self.message)
        if self.suggestion:
            parts.append(f"  Suggestion: {self.suggestion}")
        return "\n".join(parts)


class SerializationLintError(Exception):
    """Raised when code fails linting for serialization."""

    def __init__(self, message: str, errors: List[LintError] = None):
        self.errors = errors or []
        super().__init__(message)


# =============================================================================
# AST Validation
# =============================================================================

class _CodeValidator(ast.NodeVisitor):
    """AST visitor that checks for forbidden patterns."""

    def __init__(self, config: LintConfig, source_file: str = "<unknown>"):
        self.config = config
        self.errors: List[LintError] = []
        self.source_file = source_file

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module_name = alias.name
            root = module_name.split('.')[0]
            if root in self.config.forbidden_modules or module_name in self.config.forbidden_modules:
                self.errors.append(LintError(
                    f"Import of '{module_name}' is not allowed for serialization",
                    line=node.lineno,
                    suggestion=f"Remove this import or use an allowed alternative"
                ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ''
        root = module.split('.')[0]
        if root in self.config.forbidden_modules or module in self.config.forbidden_modules:
            self.errors.append(LintError(
                f"Import from '{module}' is not allowed for serialization",
                line=node.lineno
            ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check direct forbidden calls: eval(...), exec(...), open(...)
        if isinstance(node.func, ast.Name) and node.func.id in self.config.forbidden_calls:
            self.errors.append(LintError(
                f"Call to '{node.func.id}()' is not allowed",
                line=node.lineno,
                suggestion=f"Remove the {node.func.id}() call"
            ))

        # Check attribute chains: os.system(...), subprocess.run(...)
        if isinstance(node.func, ast.Attribute):
            chain = self._get_attr_chain(node.func)
            if chain:
                chain_tuple = tuple(chain)
                # Check for exact matches or prefix matches
                for forbidden in self.config.forbidden_attr_chains:
                    if len(forbidden) <= len(chain_tuple):
                        if chain_tuple[:len(forbidden)] == forbidden:
                            self.errors.append(LintError(
                                f"Call to '{'.'.join(chain)}()' is not allowed",
                                line=node.lineno
                            ))
                            break

        self.generic_visit(node)

    def _get_attr_chain(self, node: ast.Attribute) -> List[str]:
        """Get attribute chain like ['os', 'path', 'join']."""
        chain = [node.attr]
        current = node.value
        while isinstance(current, ast.Attribute):
            chain.insert(0, current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            chain.insert(0, current.id)
        return chain


class _NonlocalFinder(ast.NodeVisitor):
    """AST visitor to find nonlocal statements in a function body."""

    def __init__(self):
        self.nonlocal_names: Set[str] = set()

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)
        self.generic_visit(node)


# =============================================================================
# Public API
# =============================================================================

def get_full_class_name(obj: Any) -> str:
    """Get the fully qualified class name of an object."""
    cls = type(obj) if not isinstance(obj, type) else obj
    module = getattr(cls, '__module__', '')
    name = getattr(cls, '__qualname__', cls.__name__)
    return f"{module}.{name}" if module else name


def check_forbidden_type(obj: Any, config: LintConfig = None) -> Optional[str]:
    """
    Check if an object's type is forbidden for serialization.

    Args:
        obj: The object to check.
        config: Optional LintConfig. If None, uses NNSIGHT_CONFIG.

    Returns:
        Error message if type is forbidden, None otherwise.
    """
    if config is None:
        config = NNSIGHT_CONFIG

    full_name = get_full_class_name(obj)

    # Check exact class match
    if full_name in config.forbidden_classes:
        return config.forbidden_classes[full_name]

    # Check base classes
    cls = type(obj) if not isinstance(obj, type) else obj
    for base in cls.__mro__[1:]:
        base_name = get_full_class_name(base)
        if base_name in config.forbidden_classes:
            return config.forbidden_classes[base_name]

    # Check module prefix
    module = getattr(cls, '__module__', '') or ''
    for prefix in config.forbidden_modules:
        if module.startswith(prefix) or module.startswith(f"_{prefix}"):
            return _get_module_category_message(prefix, full_name)

    return None


def _get_module_category_message(prefix: str, class_name: str) -> str:
    """Generate a helpful message based on the forbidden module category."""
    if prefix in ('_pytest', 'pytest', 'unittest'):
        return (
            f"'{class_name}' is from test framework '{prefix}'.\n"
            f"Test fixtures and objects cannot be serialized.\n"
            f"\n"
            f"This usually happens when test fixtures leak into the scope.\n"
            f"Remove the test object from variables being serialized."
        )
    elif prefix in ('socket', 'asyncio', 'concurrent', 'multiprocessing', 'queue', 'subprocess'):
        return (
            f"'{class_name}' represents OS/system resources.\n"
            f"Objects from '{prefix}' cannot be serialized because they\n"
            f"represent resources (processes, sockets, locks) that cannot\n"
            f"be transferred between systems."
        )
    elif prefix in ('sqlite3', 'sqlalchemy', 'pymongo', 'redis', 'psycopg', 'psycopg2', 'mysql', 'pymysql'):
        return (
            f"'{class_name}' is a database connection/cursor.\n"
            f"Database connections cannot be serialized because they\n"
            f"represent network session state.\n"
            f"\n"
            f"Fetch the data you need before serialization."
        )
    elif prefix == 'logging':
        return (
            f"'{class_name}' is from the logging module.\n"
            f"Logging handlers often contain file references that\n"
            f"cannot be serialized.\n"
            f"\n"
            f"Use the data directly instead of the logger."
        )
    else:
        return f"'{class_name}' is from forbidden module '{prefix}'."


def check_nonlocal(func: Callable) -> Set[str]:
    """
    Check if a function uses nonlocal closure variables.

    Args:
        func: The function to check.

    Returns:
        Set of nonlocal variable names, empty if none.
    """
    try:
        source = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(source))
    except (OSError, TypeError, SyntaxError):
        return set()

    finder = _NonlocalFinder()
    finder.visit(tree)
    return finder.nonlocal_names


def validate_function_source(func: Callable, config: LintConfig = None) -> List[LintError]:
    """
    Validate a function's source code for serialization safety.

    Args:
        func: The function to validate.
        config: Optional LintConfig. If None, uses NNSIGHT_CONFIG.

    Returns:
        List of LintError objects, empty if validation passes.
    """
    if config is None:
        config = NNSIGHT_CONFIG

    try:
        source = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(source))
    except (OSError, TypeError, SyntaxError) as e:
        return [LintError(f"Could not parse source: {e}")]

    source_file = getattr(func, '__code__', None)
    source_file = source_file.co_filename if source_file else "<unknown>"

    validator = _CodeValidator(config, source_file)
    validator.visit(tree)
    return validator.errors


def validate_class_source(cls: type, config: LintConfig = None) -> List[LintError]:
    """
    Validate a class and all its methods for serialization safety.

    Args:
        cls: The class to validate.
        config: Optional LintConfig. If None, uses NNSIGHT_CONFIG.

    Returns:
        List of LintError objects, empty if validation passes.
    """
    if config is None:
        config = NNSIGHT_CONFIG

    errors = []
    try:
        source = inspect.getsource(cls)
        tree = ast.parse(textwrap.dedent(source))
        validator = _CodeValidator(config)
        validator.visit(tree)
        errors.extend(validator.errors)
    except (OSError, TypeError, SyntaxError) as e:
        errors.append(LintError(f"Could not parse class source: {e}"))
    return errors


def lint_object(obj: Any, config: LintConfig = None) -> List[LintError]:
    """
    Comprehensive lint check for an object.

    Checks:
    1. Forbidden type
    2. Nonlocal closures (for functions, if config.reject_nonlocal)
    3. Forbidden imports/calls in source (for functions/classes, if config.validate_source)

    Args:
        obj: The object to lint.
        config: Optional LintConfig. If None, uses NNSIGHT_CONFIG.

    Returns:
        List of LintError objects, empty if validation passes.
    """
    if config is None:
        config = NNSIGHT_CONFIG

    errors = []

    # Check forbidden type
    forbidden_msg = check_forbidden_type(obj, config)
    if forbidden_msg:
        errors.append(LintError(forbidden_msg))
        return errors  # Early return - no point checking source

    # Check functions
    if callable(obj) and hasattr(obj, '__code__'):
        # Check nonlocal
        if config.reject_nonlocal:
            nonlocal_names = check_nonlocal(obj)
            if nonlocal_names:
                errors.append(LintError(
                    f"Function '{obj.__name__}' uses 'nonlocal {', '.join(sorted(nonlocal_names))}'.\n"
                    f"Functions with nonlocal cannot be serialized because multiple closures\n"
                    f"sharing the same cell would lose shared state.\n"
                    f"\n"
                    f"Refactor to use a class:\n"
                    f"    class Counter:\n"
                    f"        def __init__(self): self.count = 0\n"
                    f"        def increment(self): self.count += 1; return self.count"
                ))

        # Check source
        if config.validate_source:
            errors.extend(validate_function_source(obj, config))

    # Check classes
    elif isinstance(obj, type):
        if config.validate_source:
            errors.extend(validate_class_source(obj, config))

    return errors
