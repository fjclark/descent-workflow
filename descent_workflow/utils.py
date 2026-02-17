import importlib
from typing import Any, Callable


def get_fn(fn_name: str) -> Callable[..., Any]:
    """Get a function from a module according to the config."""
    *module_names, function_name = fn_name.rsplit(".")
    module_name = ".".join(module_names)
    # "module" is protected in a Snakefile
    module_ = importlib.import_module(module_name)
    function: Callable[..., Any] = getattr(module_, function_name)
    return function
