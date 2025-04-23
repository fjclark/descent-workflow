from typing import Callable
import importlib


def get_fn(fn_name: str) -> Callable:
    """
    Get a function from a module according to the config
    """
    module_name, function_name = fn_name.rsplit(".", 1)
    # "module" is protected in a Snakefile
    module_ = importlib.import_module(module_name)
    function = getattr(module_, function_name)
    return function
