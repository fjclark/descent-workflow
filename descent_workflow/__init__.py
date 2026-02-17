"""Workflow package for descent force field optimization.

This package contains modules for:
- Data acquisition and processing (get_data, get_data_byte_dance, filter)
- Force field parameterization (parameterise, models)
- Training and optimization (train)
- Benchmarking and analysis (benchmark, plot_benchmark, run_yammbs_script)
- Utilities (utils, convert_ff)
"""

__all__ = [
    "benchmark",
    "convert_ff",
    "filter",
    "get_data",
    "get_data_byte_dance",
    "models",
    "parameterise",
    "plot_benchmark",
    "run_yammbs_script",
    "train",
    "utils",
]
