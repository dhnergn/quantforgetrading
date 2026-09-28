"""Platform abstraction: one place that knows what the host OS can do.

Every OS-specific decision in the platform is made here, once, and every other
module asks this module instead of testing ``sys.platform`` itself. Two reasons:

* ``sys.platform`` checks scattered through the codebase drift, and a missed
  one is an ``AttributeError`` at the worst possible moment.
* The important part is not *which* OS, but *what capability* is missing. The
  sandbox needs a CPU limit; on Windows that limit does not exist. Saying
  "this host cannot set RLIMIT_CPU" is useful. Saying "win32" three call sites
  away is not.

Host support, and the honest scope of it:

==================  ==========================================================
Platform            Status
==================  ==========================================================
Linux               Supported and tested
macOS               Supported and tested
Windows             Supported and tested (no POSIX resource limits)
iOS / Android       NOT a host. See ``MOBILE_CLIENT`` below.
==================  ==========================================================

Mobile platforms are excluded because the compute stack cannot load there: no
numpy, pandas, scipy, or scikit-learn binary wheel exists for iOS, and iOS
additionally forbids the JIT those libraries need and forbids process forking.
This is a platform constraint, not a missing feature, and no amount of porting
removes it. Mobile is supported as a *client* of a server - see
``quantforge.api.mobile``.

Naming: the canonical API is ``current()``, ``app_dir()``, ``log_dir()``,
``set_resource_limits()`` and friends. The ``get_*`` spellings from an earlier
draft are kept as aliases so both conventions work against one module rather
than two competing ones.
"""
from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

__all__ = [
    "Platform", "current", "get_platform", "describe", "get_platform_config",
    "is_mobile", "is_windows", "MOBILE_CLIENT",
    "resource_limits_supported", "set_resource_limits", "clear_resource_limits",
    "remove_resource_limits",
    "app_dir", "get_app_data_dir", "data_dir", "get_data_dir",
    "log_dir", "get_logs_dir", "config_dir", "get_config_path",
    "state_dir", "temp_dir", "get_temp_dir", "ensure_dirs",
    "make_read_only", "is_read_only", "default_bind_host", "get_optimal_workers",
    "file_lock", "file_lock_backend",
]

#: Guidance for the mobile case, surfaced by ``/health`` and the CLI so a user
#: gets a specific answer rather than an ImportError.
MOBILE_CLIENT = (
    "iOS and Android are not supported as compute hosts: no numpy, pandas, "
    "scipy, or scikit-learn binary wheels exist for iOS, and iOS additionally "
    "forbids the JIT those libraries require and forbids process forking. Run "
    "the platform on Linux, macOS, or Windows and reach it from a phone with "
    "quantforge.api.mobile.QuantForgeClient, which is pure-Python with no "
    "compiled dependencies and therefore ships inside a mobile app."
)

IS_WINDOWS = sys.platform.startswith("win")
IS_DARWIN = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")
#: Covers iOS, Android, and anything else that is not a desktop/server OS.
#: iOS reports as "darwin", so it is detected from the interpreter build string.
IS_MOBILE = (
    sys.platform in ("ios", "android")
    or "android" in sys.platform.lower()
    or (IS_DARWIN and "iPhone" in sys.version)
)


@dataclass(frozen=True, slots=True)
class Platform:
    """What this host supports, expressed as capabilities, not OS names."""

    name: str
    is_windows: bool
    is_darwin: bool
    is_linux: bool
    is_mobile: bool
    python_version: str
    executable_suffix: str
    path_sep: str
    supports_resource_limits: bool
    supports_fork: bool
    supports_symlinks_unprivileged: bool
    case_sensitive_paths: bool
    max_path_length: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def supports_read_only_files(self) -> bool:
        """Whether a 0400 file is actually read-only.

        POSIX permission bits are honoured. Windows largely ignores them, so a
        "read-only" authorisation file there is still writable by anything
        running as the same user - the protection the live-trading gate relies
        on does not exist, and the code that writes it says so rather than
        reporting hardening that was never applied.
        """
        return not self.is_windows

    @property
    def supports_rename_over_existing(self) -> bool:
        """Whether an atomic replace-over-existing works.

        ``os.replace`` is atomic on POSIX. On Windows it fails if the
        destination is open, which is a routine condition for a file a running
        process holds open - the audit log, for instance.
        """
        return True

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "python_version": self.python_version,
            "executable_suffix": self.executable_suffix,
            "path_separator": self.path_sep,
            "case_sensitive_paths": self.case_sensitive_paths,
            "max_path_length": self.max_path_length,
            "supports_resource_limits": self.supports_resource_limits,
            "supports_read_only_files": self.supports_read_only_files,
            "supports_fork": self.supports_fork,
            "is_mobile": self.is_mobile,
            "notes": list(self.notes),
        }


def current() -> Platform:
    """Describe the running host."""
    notes: list[str] = []
    if IS_WINDOWS:
        notes.append(
            "CPU and memory limits are unavailable: the resource module is "
            "POSIX-only. Static source screening and the wall-clock timeout "
            "still apply, which is a weaker guarantee than on Linux or macOS."
        )
        notes.append(
            "File mode bits are not enforced, so the live-trading authorisation "
            "is not protected by filesystem permissions here. Its signature "
            "still detects tampering on the next check."
        )
    if IS_DARWIN:
        notes.append("APFS is case-insensitive by default; path case is not significant.")
    if IS_MOBILE:
        notes.append(MOBILE_CLIENT)

    return Platform(
        name=sys.platform,
        is_windows=IS_WINDOWS,
        is_darwin=IS_DARWIN,
        is_linux=IS_LINUX,
        is_mobile=IS_MOBILE,
        python_version=sys.version.split()[0],
        executable_suffix=".exe" if IS_WINDOWS else "",
        path_sep=os.sep,
        # RLIMIT_CPU / RLIMIT_AS / RLIMIT_NOFILE / RLIMIT_FSIZE are POSIX-only.
        supports_resource_limits=not IS_WINDOWS and os.name == "posix",
        supports_fork=hasattr(os, "fork"),
        supports_symlinks_unprivileged=not IS_WINDOWS,
        case_sensitive_paths=not IS_DARWIN,
        # Windows MAX_PATH is 260 without long-path opt-in; POSIX PATH_MAX is
        # commonly 4096.
        max_path_length=260 if IS_WINDOWS else 4096,
        notes=tuple(notes),
    )


def is_mobile() -> bool:
    return IS_MOBILE


def is_windows() -> bool:
    return IS_WINDOWS


def get_optimal_workers() -> int:
    """Worker count for parallel work.

    ``os.cpu_count()`` can return ``None`` in a restricted container, and 0 is
    not a valid thread count, so both are handled rather than propagated into
    an allocator that then raises.
    """
    try:
        n = os.cpu_count() or 1
    except NotImplementedError:
        n = 1
    return max(1, n)


# ==========================================================================
# Resource limits
# ==========================================================================


def resource_limits_supported() -> bool:
    return current().supports_resource_limits


def set_resource_limits(
    max_cpu_seconds: Optional[float] = None,
    max_address_space_mb: Optional[int] = None,
    max_open_files: Optional[int] = None,
    max_file_size_mb: Optional[int] = None,
) -> dict:
    """Apply OS resource limits where they exist.

    Returns a report of what was applied and what was skipped. **A skip is
    reported, never silent**: a caller that believes it bounded CPU on Windows
    when it did not is relying on a sandbox that does not sandbox.
    """
    if not current().supports_resource_limits:
        return {
            "applied": {},
            "skipped": {
                "cpu_seconds": "RLIMIT_CPU is POSIX-only; unavailable on this host",
                "address_space_mb": "RLIMIT_AS is POSIX-only; unavailable on this host",
                "open_files": "RLIMIT_NOFILE is POSIX-only; unavailable on this host",
                "file_size_mb": "RLIMIT_FSIZE is POSIX-only; unavailable on this host",
            },
            "supported": False,
            "note": (
                "Resource limits are a defence-in-depth control, not a "
                "boundary. Their absence does not make the sandbox unsafe on "
                "its own terms - it makes it weaker, and the caller is told."
            ),
        }

    import resource as _res  # POSIX only; reached only when supported

    applied: dict = {}
    skipped: dict = {}

    def _set(name: int, value: Optional[int], label: str) -> None:
        if value is None:
            skipped[label] = "not requested"
            return
        try:
            _, hard = _res.getrlimit(name)
            # Never raise a hard limit we lack the privilege to raise.
            if hard != _res.RLIM_INFINITY and int(value) > hard:
                skipped[label] = (
                    f"requested {value} exceeds the hard limit {hard}; lowering "
                    "privileges to raise it is not attempted"
                )
                return
            _res.setrlimit(name, (int(value), hard))
            applied[label] = int(value)
        except (ValueError, OSError, AttributeError) as exc:
            skipped[label] = f"{type(exc).__name__}: {exc}"

    _set(_res.RLIMIT_CPU, int(max_cpu_seconds) if max_cpu_seconds else None, "cpu_seconds")
    _set(
        _res.RLIMIT_AS,
        int(max_address_space_mb * 1024 * 1024) if max_address_space_mb else None,
        "address_space_mb",
    )
    _set(_res.RLIMIT_NOFILE, max_open_files, "open_files")
    _set(
        _res.RLIMIT_FSIZE,
        int(max_file_size_mb * 1024 * 1024) if max_file_size_mb else None,
        "file_size_mb",
    )
    return {"applied": applied, "skipped": skipped, "supported": True, "note": None}


def clear_resource_limits() -> None:
    """Restore limits to infinity. Only for a controlled shutdown path."""
    if not current().supports_resource_limits:
        return
    import resource as _res

    for name in (_res.RLIMIT_CPU, _res.RLIMIT_AS, _res.RLIMIT_FSIZE,
                 _res.RLIMIT_NOFILE):
        try:
            _res.setrlimit(name, (_res.RLIM_INFINITY, _res.RLIM_INFINITY))
        except (ValueError, OSError, AttributeError):
            continue


remove_resource_limits = clear_resource_limits


# ==========================================================================
# Filesystem locations
# ==========================================================================


def _resolve(env_var: str, windows_base: str, darwin_base: str,
             xdg_var: str, xdg_subpath: str, fallback_name: str) -> Path:
    """Resolve a per-user directory following this OS's convention.

    Order: an explicit environment variable (so a deployment can place state on
    a mounted volume), then the OS convention. The fallback is the *user*
    directory rather than the system one - writing under ``C:\\ProgramData`` or
    ``/etc`` needs elevation, and a research tool should not require it.
    """
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser()
    if IS_WINDOWS:
        base = os.environ.get(windows_base) or os.path.expanduser("~")
        return Path(base) / "QuantForge" / fallback_name
    if IS_DARWIN:
        return Path(os.path.expanduser("~")) / darwin_base / "QuantForge" / fallback_name
    xdg = os.environ.get(xdg_var)
    if xdg:
        return Path(xdg) / "quantforge" / fallback_name
    return Path(os.path.expanduser("~")) / ".local" / xdg_subpath / "quantforge" / fallback_name


def app_dir() -> Path:
    """Root for all QuantForge state."""
    return _resolve(
        "QUANTFORGE_HOME", "LOCALAPPDATA",
        "Library/Application Support", "XDG_DATA_HOME", "share", "",
    )


def data_dir() -> Path:
    return _resolve(
        "QUANTFORGE_DATA", "LOCALAPPDATA",
        "Library/Application Support", "XDG_DATA_HOME", "share", "data",
    )


def log_dir() -> Path:
    return _resolve(
        "QUANTFORGE_LOGS", "LOCALAPPDATA",
        "Library/Logs", "XDG_STATE_HOME", "state", "logs",
    )


def config_dir() -> Path:
    return _resolve(
        "QUANTFORGE_CONFIG", "APPDATA",
        "Library/Preferences", "XDG_CONFIG_HOME", "config", "",
    )


def state_dir() -> Path:
    """Mutable runtime state: databases, caches."""
    return _resolve(
        "QUANTFORGE_STATE", "LOCALAPPDATA",
        "Library/Application Support", "XDG_STATE_HOME", "state", "state",
    )


def temp_dir() -> Path:
    """A writable temporary directory, honouring TMPDIR on POSIX."""
    return Path(tempfile.gettempdir())


def ensure_dirs() -> dict:
    """Create the standard directories, reporting what was created."""
    created = {}
    for label, fn in (("data", data_dir), ("logs", log_dir),
                      ("config", config_dir), ("state", state_dir)):
        p = fn()
        try:
            p.mkdir(parents=True, exist_ok=True)
            created[label] = str(p)
        except OSError as exc:
            created[label] = f"FAILED: {exc}"
    return created


# --- aliases from the earlier `get_*` naming ------------------------------
get_app_data_dir = app_dir
get_data_dir = data_dir
get_logs_dir = log_dir
get_config_path = config_dir
get_temp_dir = temp_dir


# ==========================================================================
# File permissions
# ==========================================================================


def make_read_only(path: Path) -> dict:
    """Best-effort read-only protection for a file.

    On POSIX this is a real protection. On Windows the mode bits are largely
    advisory, so the return value says the file is **not** protected rather
    than reporting hardening that does not exist.
    """
    if not current().supports_read_only_files:
        return {
            "protected": False,
            "mechanism": None,
            "detail": (
                "file permission bits are not enforced on this platform, so the "
                "file remains writable by the owning user. The live-trading gate "
                "still verifies the signature, so tampering is DETECTED on the "
                "next check, but it is not PREVENTED. Run the platform on Linux "
                "or macOS for filesystem-level protection."
            ),
        }
    import stat as _stat

    result = {"protected": False, "mechanism": "chmod 0400", "detail": ""}
    try:
        Path(path).chmod(_stat.S_IRUSR)
        result["protected"] = True
        result["detail"] = "owner read-only; enforced by the filesystem"
    except OSError as exc:
        result["detail"] = f"chmod failed: {exc}"
    return result


def is_read_only(path: Path) -> bool:
    if not current().supports_read_only_files:
        return False
    import stat as _stat

    try:
        mode = os.stat(path).st_mode
    except OSError:
        return False
    return not bool(mode & (_stat.S_IWUSR | _stat.S_IWGRP | _stat.S_IWOTH))


# ==========================================================================
# File locking
# ==========================================================================


def file_lock_backend() -> str:
    """Which locking primitive this host offers: ``fcntl``, ``msvcrt``, or none."""
    if os.name == "nt":
        try:
            import msvcrt  # noqa: F401
            return "msvcrt"
        except ImportError:
            return "none"
    try:
        import fcntl  # noqa: F401
        return "fcntl"
    except ImportError:
        return "none"


#: Per-lockfile in-process locks, so two threads in one process do not both
#: reach for the OS primitive and contend with themselves.
_PROCESS_LOCAL_LOCKS: dict = {}


@contextmanager
def file_lock(target: Path, timeout: float = 5.0, poll_interval: float = 0.01):
    """Hold an advisory lock while writing to ``target``.

    A ``threading.Lock`` protects threads inside one process. It does not
    protect against a second *process* writing the same file, and running the
    API and a worker as separate processes is an ordinary deployment rather
    than an exotic one. Two writers interleaving half a JSON line corrupt the
    audit trail, which is exactly the record that has to be trustworthy.

    A sidecar ``<target>.lock`` is used rather than locking the target itself,
    so the lock survives a rotation that replaces the target.

    Degrades honestly: where a host offers no primitive, threads are still
    serialised but the result says the lock does **not** cover other processes.
    """
    import threading
    import time

    target = Path(target)
    lock_path = target.with_name(target.name + ".lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        yield {"locked": False, "backend": "none", "detail": f"mkdir failed: {exc}"}
        return

    backend = file_lock_backend()
    thread_lock = _PROCESS_LOCAL_LOCKS.setdefault(str(lock_path), threading.Lock())
    if not thread_lock.acquire(timeout=max(0.0, timeout)):
        raise TimeoutError(
            f"timed out after {timeout}s waiting for the in-process lock on "
            f"{lock_path.name}"
        )

    handle = None
    try:
        try:
            handle = open(lock_path, "a+b")
        except OSError as exc:
            yield {"locked": False, "backend": "none", "detail": str(exc)}
            return

        if backend == "fcntl":
            import fcntl

            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"timed out after {timeout}s acquiring an fcntl lock "
                            f"on {lock_path.name}"
                        )
                    time.sleep(poll_interval)
            try:
                yield {"locked": True, "backend": "fcntl", "detail": None}
            finally:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass

        elif backend == "msvcrt":
            import msvcrt

            # msvcrt.locking locks a byte range from the current position, so
            # the target region must exist: a zero-length file cannot be
            # locked, which is why a byte is written first.
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            deadline = time.monotonic() + timeout
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"timed out after {timeout}s acquiring an msvcrt "
                            f"lock on {lock_path.name}"
                        )
                    time.sleep(poll_interval)
            try:
                yield {"locked": True, "backend": "msvcrt", "detail": None}
            finally:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass

        else:
            yield {
                "locked": False,
                "backend": "none",
                "detail": (
                    "this host provides no file-locking primitive, so the lock "
                    "serialises threads only and does NOT protect against a "
                    "second process writing the same file"
                ),
            }
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        thread_lock.release()


# ==========================================================================
# Networking
# ==========================================================================


def default_bind_host() -> str:
    """Loopback by default.

    An authenticated trading API should not become network-reachable because
    someone bound it to 0.0.0.0 for convenience. Choosing that is a deliberate
    act, so the default is loopback on every platform.
    """
    return "127.0.0.1"


def describe() -> dict:
    """Platform report, for diagnostics and the /health endpoint."""
    out = current().as_dict()
    out["directories"] = {
        "app": str(app_dir()),
        "data": str(data_dir()),
        "logs": str(log_dir()),
        "config": str(config_dir()),
        "state": str(state_dir()),
        "temp": str(temp_dir()),
    }
    out["supported_hosts"] = ["linux", "darwin", "win32"]
    out["optimal_workers"] = get_optimal_workers()
    out["mobile"] = {
        "is_mobile_host": IS_MOBILE,
        "guidance": MOBILE_CLIENT,
    }
    return out


def get_platform() -> str:
    return sys.platform


def get_platform_config() -> dict:
    """Flat capability map, for callers that want a dict rather than objects."""
    p = current()
    return {
        "platform": p.name,
        "is_windows": p.is_windows,
        "is_darwin": p.is_darwin,
        "is_linux": p.is_linux,
        "is_mobile": p.is_mobile,
        "supports_resource_limits": p.supports_resource_limits,
        "supports_read_only_files": p.supports_read_only_files,
        "optimal_workers": get_optimal_workers(),
        "executable_suffix": p.executable_suffix,
        "path_separator": p.path_sep,
        "case_sensitive_paths": p.case_sensitive_paths,
    }
