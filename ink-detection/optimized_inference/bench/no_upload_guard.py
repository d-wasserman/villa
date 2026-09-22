"""
Network and upload guard for the benchmark harness.

Enforces the harness ground rules in code rather than by convention:

* ``boto3.client``, ``boto3.resource`` and ``boto3.session.Session.client`` /
  ``.resource`` raise, so no AWS client can be created at all.
* s3fs write-mode opens and mutating methods raise.
* Every outbound socket connection is checked. Loopback and Unix sockets are
  allowed (DataLoader workers, Inductor compile workers), *except* connections
  to a configured HTTP(S) proxy, which count as external. Any other host must
  be in the allow list, or the connection raises ``NetworkBlockedError``.

All attempts are recorded; ``network_report()`` returns them for the
results file. The allow list is exported through ``BENCH_ALLOW_HOSTS`` so
spawned child processes that call ``install()`` apply the same policy.
"""
from __future__ import annotations

import functools
import ipaddress
import os
import socket
import threading
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlsplit

ENV_ALLOW_HOSTS = "BENCH_ALLOW_HOSTS"
PROXY_ENV_VARS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")


class NetworkBlockedError(RuntimeError):
    """An outbound connection to a host outside the allow list."""


class UploadBlockedError(RuntimeError):
    """An attempt to create an AWS client or write through s3fs."""


_lock = threading.Lock()
_installed = False
_allow_hosts: Set[str] = set()
_proxy_endpoints: Set[Tuple[str, int]] = set()
_attempts: Dict[Tuple[str, int], Dict[str, object]] = {}
_orig_connect = None
_orig_connect_ex = None


def _parse_proxy_endpoints() -> Set[Tuple[str, int]]:
    endpoints = set()
    for name in PROXY_ENV_VARS:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        parts = urlsplit(raw if "://" in raw else f"http://{raw}")
        if parts.hostname:
            default_port = 443 if parts.scheme == "https" else 80
            endpoints.add((parts.hostname.lower(), parts.port or default_port))
    return endpoints


def _is_loopback(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _check(address, family) -> None:
    if family == getattr(socket, "AF_UNIX", object()):
        return
    if not isinstance(address, tuple) or len(address) < 2:
        return
    host, port = str(address[0]).lower(), int(address[1])
    is_proxy = (host, port) in _proxy_endpoints or (
        _is_loopback(host) and any(_is_loopback(h) and p == port for h, p in _proxy_endpoints)
    )
    if is_proxy:
        allowed = "proxy" in _allow_hosts
    else:
        allowed = _is_loopback(host) or host in _allow_hosts
    if not _is_loopback(host) or is_proxy:
        with _lock:
            entry = _attempts.setdefault((host, port), {"host": host, "port": port, "proxy": is_proxy, "allowed": allowed, "count": 0})
            entry["count"] = int(entry["count"]) + 1
    if not allowed:
        kind = "proxy " if is_proxy else ""
        raise NetworkBlockedError(
            f"bench no-upload guard blocked a connection to {kind}{host}:{port}; "
            f"allowed hosts: {sorted(_allow_hosts) or 'none (offline workload)'}"
        )


def _guarded_connect(self, address):
    _check(address, self.family)
    return _orig_connect(self, address)


def _guarded_connect_ex(self, address):
    _check(address, self.family)
    return _orig_connect_ex(self, address)


def _blocked(what: str):
    def _raise(*_args, **_kwargs):
        raise UploadBlockedError(f"bench no-upload guard: {what} is disabled in the benchmark harness")

    return _raise


def _patch_boto3() -> None:
    try:
        import boto3
        import boto3.session
    except ImportError:
        return
    boto3.client = _blocked("boto3.client")
    boto3.resource = _blocked("boto3.resource")
    boto3.session.Session.client = _blocked("boto3 Session.client")
    boto3.session.Session.resource = _blocked("boto3 Session.resource")


_S3FS_WRITE_METHODS = (
    "_put_file", "_pipe_file", "_rm", "_rm_file", "_mkdir", "_makedirs",
    "_cp_file", "_copy", "_mv", "put", "put_file", "pipe", "pipe_file",
    "rm", "rm_file", "mkdir", "makedirs", "copy", "cp_file", "mv", "touch",
)


def _patch_s3fs() -> None:
    try:
        import s3fs
    except ImportError:
        return
    cls = s3fs.S3FileSystem
    if getattr(cls, "_bench_guarded", False):
        return
    orig_open = cls._open

    @functools.wraps(orig_open)
    def _guarded_open(self, path, mode="rb", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            raise UploadBlockedError(f"bench no-upload guard: s3fs open({path!r}, mode={mode!r}) is disabled")
        return orig_open(self, path, mode, *args, **kwargs)

    cls._open = _guarded_open
    for name in _S3FS_WRITE_METHODS:
        if hasattr(cls, name):
            setattr(cls, name, _blocked(f"s3fs {name}"))
    cls._bench_guarded = True


def install(allow_hosts: Optional[Iterable[str]] = None) -> None:
    """
    Install the guard (idempotent). ``allow_hosts`` replaces the allow list;
    when omitted, the list is read from ``BENCH_ALLOW_HOSTS`` (comma separated).
    Use the special host ``proxy`` to permit connections through the
    configured HTTP(S) proxy.
    """
    global _installed, _orig_connect, _orig_connect_ex, _allow_hosts, _proxy_endpoints
    if allow_hosts is None:
        allow_hosts = [h for h in os.environ.get(ENV_ALLOW_HOSTS, "").split(",") if h.strip()]
    with _lock:
        _allow_hosts = {h.strip().lower() for h in allow_hosts}
        _proxy_endpoints = _parse_proxy_endpoints()
        os.environ[ENV_ALLOW_HOSTS] = ",".join(sorted(_allow_hosts))
        if not _installed:
            _orig_connect = socket.socket.connect
            _orig_connect_ex = socket.socket.connect_ex
            socket.socket.connect = _guarded_connect
            socket.socket.connect_ex = _guarded_connect_ex
            _installed = True
    _patch_boto3()
    _patch_s3fs()


def worker_init(_worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: installs the guard inside spawned workers."""
    install()


def is_active() -> bool:
    if not _installed or socket.socket.connect is not _guarded_connect:
        return False
    try:
        import boto3

        try:
            boto3.client("s3")
        except UploadBlockedError:
            pass
        else:
            return False
    except ImportError:
        pass
    return True


def reset_attempts() -> None:
    with _lock:
        _attempts.clear()


def network_report() -> Dict[str, object]:
    with _lock:
        attempts: List[Dict[str, object]] = sorted(
            (dict(v) for v in _attempts.values()), key=lambda e: (str(e["host"]), int(e["port"]))
        )
        return {
            "guard_active": is_active(),
            "allow_hosts": sorted(_allow_hosts),
            "hosts_contacted": [f"{e['host']}:{e['port']}" for e in attempts if e["allowed"]],
            "hosts_blocked": [f"{e['host']}:{e['port']}" for e in attempts if not e["allowed"]],
            "attempts": attempts,
        }
