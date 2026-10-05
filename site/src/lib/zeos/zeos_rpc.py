"""Generic RPC glue for the ZEOS kernel worker (src/workers/zeosKernel.worker.ts).

Loaded into Pyodide as the module ``_zeos_rpc``. The worker turns each
postMessage request into one call here; everything crosses as JSON text so the
JavaScript side never holds a Python proxy.

Arguments (JSON, decoded with ``_decode``) may contain markers:

``{"$model": name}``   the SyncModelWorker attached under ``name``
``{"$handle": n}``     a Python object an earlier call returned
``{"$bytes": b64}``    bytes

Results are encoded with ``_encode``: JSON-able values pass through, ``bytes``
become ``{"$bytes": b64}``, and anything else is kept in a handle table and
returned as ``{"$handle": n, "type": "<class name>"}``. Release handles with
``release`` when done.

Python code can stream events to the page with ``emit(kind, data)``.
"""

from __future__ import annotations

import base64
import importlib
import json
import traceback
from typing import Any

import _zeos_host  # registered from JavaScript: emit(kind, json_text)

__all__ = ["emit", "register_model", "model", "call", "call_method", "get_attr", "exec_source", "release"]

_models: dict[str, Any] = {}
_handles: dict[int, Any] = {}
_next_handle = 1
_globals: dict[str, Any] = {"__name__": "__zeos_exec__"}


def emit(kind: str, data: Any = None) -> None:
    """Post ``{type: 'event', kind, data}`` to the page."""
    _zeos_host.emit(kind, json.dumps(_encode_value(data)))


def register_model(name: str, worker: Any) -> None:
    _models[name] = worker


def model(name: str = "default") -> Any:
    """The SyncModelWorker attached under ``name``, for use from ``exec`` code."""
    try:
        return _models[name]
    except KeyError:
        raise KeyError(f"no model attached as {name!r}; attached: {sorted(_models)}") from None


def _hook(obj: dict[str, Any]) -> Any:
    if len(obj) == 1:
        if "$model" in obj:
            return model(obj["$model"])
        if "$bytes" in obj:
            return base64.b64decode(obj["$bytes"])
    if "$handle" in obj:
        n = int(obj["$handle"])
        if n not in _handles:
            raise KeyError(f"unknown or released handle {n}")
        return _handles[n]
    return obj


def _decode(text: str | None, default: Any) -> Any:
    if not text:
        return default
    return json.loads(text, object_hook=_hook)


def _keep(value: Any) -> dict[str, Any]:
    global _next_handle
    n = _next_handle
    _next_handle += 1
    _handles[n] = value
    return {"$handle": n, "type": type(value).__name__}


def _encode_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"$bytes": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, (list, tuple)):
        return [_encode_value(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _encode_value(v) for k, v in value.items()}
    return _keep(value)


def _ok(value: Any) -> str:
    return json.dumps({"ok": True, "value": _encode_value(value)})


def _err(exc: BaseException) -> str:
    return json.dumps(
        {
            "ok": False,
            "error": {
                "name": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            },
        }
    )


def call(module: str, fn: str, args_json: str = "", kwargs_json: str = "") -> str:
    """``module.fn(*args, **kwargs)``; ``fn`` may be dotted (``Class.method``)."""
    try:
        target: Any = importlib.import_module(module)
        for part in fn.split("."):
            target = getattr(target, part)
        return _ok(target(*_decode(args_json, []), **_decode(kwargs_json, {})))
    except BaseException as exc:  # noqa: BLE001 - every failure goes back to JS
        return _err(exc)


def call_method(handle: int, method: str, args_json: str = "", kwargs_json: str = "") -> str:
    try:
        obj = _hook({"$handle": handle})
        return _ok(getattr(obj, method)(*_decode(args_json, []), **_decode(kwargs_json, {})))
    except BaseException as exc:  # noqa: BLE001
        return _err(exc)


def get_attr(handle: int, name: str) -> str:
    try:
        return _ok(getattr(_hook({"$handle": handle}), name))
    except BaseException as exc:  # noqa: BLE001
        return _err(exc)


def exec_source(source: str) -> str:
    """Run ``source`` in one persistent namespace; the value of a trailing
    expression is the result, as in a REPL."""
    try:
        from pyodide.code import eval_code

        _globals.setdefault("zeos_rpc", importlib.import_module("_zeos_rpc"))
        return _ok(eval_code(source, _globals))
    except BaseException as exc:  # noqa: BLE001
        return _err(exc)


def release(handle: int) -> str:
    _handles.pop(int(handle), None)
    return _ok(None)
