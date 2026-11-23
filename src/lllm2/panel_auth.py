"""Panel session secret and the checks on paths that panel requests name.

The panel listens on TCP, where any local process or account can connect, so
the HTTP API trusts only a client that presents the per-start secret: in the
``X-LLLM2-Token`` header, or in the HttpOnly cookie that the browser receives
when it opens the session link. The secret is written to a 0600 file in the
state directory and printed by ``lllm2 panel-url``; ``/api/status`` never
returns it.

Settings that arrive over HTTP may name only engines that the owner
configured (``LLLM2_ENGINE_ROOTS`` or ``llama-server`` on ``PATH``) and chat
templates under the models directory or the packaged templates, because the
panel executes the engine on this machine and a remote launch uploads the
template.
"""

from __future__ import annotations

import hmac
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import config
from .discovery import engine_paths

SESSION_FILE = "panel-session.json"
TOKEN_HEADER = "X-LLLM2-Token"
TEMPLATE_DIR = Path(__file__).with_name("templates")
TEMPLATE_SUFFIXES = {".jinja", ".jinja2", ".j2", ".tmpl", ".txt"}
TEMPLATE_MAX_BYTES = 1_000_000


def cookie_name(port: int) -> str:
    """Name the session cookie per port: browsers share cookies across ports."""
    return f"lllm2_session_{port}"


def session_path(state_dir: Path | None = None) -> Path:
    return (state_dir or config.STATE_DIR) / SESSION_FILE


def panel_urls(host: str, port: int, token: str, hostname: str) -> list[str]:
    """Return the session links for a panel, loopback first.

    Args:
        host: The address the panel binds to.
        port: The panel port.
        token: The session secret.
        hostname: This machine's name, used for a panel bound to 0.0.0.0.

    Returns:
        One link per address a browser can use.

    >>> panel_urls("127.0.0.1", 8082, "s3cret", "gpu")
    ['http://127.0.0.1:8082/?token=s3cret']
    >>> panel_urls("0.0.0.0", 8082, "s3cret", "gpu")[1]
    'http://gpu:8082/?token=s3cret'
    """
    urls = [f"http://127.0.0.1:{port}/?token={token}"]
    if host != "127.0.0.1":
        lan = hostname if host == "0.0.0.0" else host
        urls.append(f"http://{lan}:{port}/?token={token}")
    return urls


def write_session(record: dict[str, Any], state_dir: Path | None = None) -> Path:
    """Write the session record, readable only by this user.

    The file is created 0600 before any secret is written to it, so no other
    account can read it in between.
    """
    path = session_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, json.dumps(record).encode())
    finally:
        os.close(fd)
    os.replace(tmp, path)
    return path


def read_session(state_dir: Path | None = None) -> dict[str, Any]:
    """Read the running panel's session record.

    Raises:
        FileNotFoundError: No panel has written a session for this state directory.
    """
    return json.loads(session_path(state_dir).read_text())


def remove_session(token: str, state_dir: Path | None = None) -> None:
    """Remove the session file if it still belongs to this panel run."""
    try:
        if read_session(state_dir).get("token") == token:
            session_path(state_dir).unlink()
    except (OSError, ValueError):
        pass


def cookie_token(cookie_header: str | None, port: int) -> str | None:
    """Return this panel's session cookie value from a ``Cookie`` header.

    >>> cookie_token("a=1; lllm2_session_8082=abc", 8082)
    'abc'
    >>> cookie_token("lllm2_session_8083=abc", 8082) is None
    True
    """
    name = cookie_name(port)
    for part in (cookie_header or "").split(";"):
        key, _, value = part.strip().partition("=")
        if key == name:
            return value
    return None


def same_secret(candidate: str | None, token: str) -> bool:
    """Compare a presented secret with the panel's in constant time."""
    return bool(candidate) and hmac.compare_digest(
        str(candidate).encode(), token.encode()
    )


def _named_paths(data: Any, key: str) -> Iterator[str]:
    """Yield every non-empty string stored under ``key`` anywhere in ``data``."""
    if isinstance(data, dict):
        for k, v in data.items():
            if k == key and isinstance(v, str) and v:
                yield v
            else:
                yield from _named_paths(v, key)
    elif isinstance(data, list):
        for item in data:
            yield from _named_paths(item, key)


def check_engine(value: str, allowed: set[Path] | None = None) -> None:
    """Reject an engine path that the owner did not configure.

    Raises:
        ValueError: The path is not a discovered engine.
    """
    allowed = engine_paths() if allowed is None else allowed
    try:
        resolved = Path(value).expanduser().resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError("Invalid engine path.") from e
    if resolved not in allowed:
        raise ValueError(
            "The panel runs only engines found under LLLM2_ENGINE_ROOTS or "
            "llama-server on PATH. Add this build's directory or file to "
            "LLLM2_ENGINE_ROOTS and restart the panel."
        )


def check_template(value: str) -> None:
    """Reject a chat template outside the models directory or packaged templates.

    Raises:
        ValueError: The path is outside the allowed folders, is not a regular
            template file, or is too large.
    """
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as e:
        raise ValueError("Chat template file does not exist.") from e
    roots = [config.MODELS_DIR.expanduser().resolve(), TEMPLATE_DIR.resolve()]
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError(
            f"Chat templates must be under {roots[0]} or the packaged templates."
        )
    if not path.is_file():
        raise ValueError("Chat template must be a regular file.")
    if path.suffix.lower() not in TEMPLATE_SUFFIXES:
        raise ValueError(
            "Chat template must end in one of: "
            + ", ".join(sorted(TEMPLATE_SUFFIXES))
            + "."
        )
    if path.stat().st_size > TEMPLATE_MAX_BYTES:
        raise ValueError("Chat template is larger than 1 MB.")


def check_request_paths(data: Any) -> None:
    """Check every engine and chat template path a panel request names.

    Settings travel under ``settings``, as benchmark ``combinations`` and as a
    top-level ``engine``, so the whole payload is searched.

    Raises:
        ValueError: A path fails :func:`check_engine` or :func:`check_template`.
    """
    engines = list(_named_paths(data, "engine"))
    if engines:
        allowed = engine_paths()
        for value in engines:
            check_engine(value, allowed)
    for value in _named_paths(data, "chat_template"):
        check_template(value)
