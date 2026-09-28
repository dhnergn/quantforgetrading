"""Core module: temporal semantics, instruments, configuration, reproducibility.

The platform subpackage is re-exported here so callers can ask about host
capabilities without knowing which module owns the answer.
"""
from quantforge.core.platform import (
    MOBILE_CLIENT,
    Platform,
    app_dir,
    clear_resource_limits,
    config_dir,
    current,
    data_dir,
    default_bind_host,
    describe,
    ensure_dirs,
    is_mobile,
    is_read_only,
    log_dir,
    make_read_only,
    set_resource_limits,
    state_dir,
    temp_dir,
)

__all__ = [
    "MOBILE_CLIENT",
    "Platform",
    "app_dir",
    "clear_resource_limits",
    "config_dir",
    "current",
    "data_dir",
    "default_bind_host",
    "describe",
    "ensure_dirs",
    "is_mobile",
    "is_read_only",
    "log_dir",
    "make_read_only",
    "set_resource_limits",
    "state_dir",
    "temp_dir",
]
