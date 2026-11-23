"""The panel's real HTTP handler refuses clients without the session secret.

These tests run ``make_server`` on a free loopback port and talk to it with
urllib, as another local process would. They cover gilesknap/lllm2#129 (the
token handed out by ``/api/status``) and the panel-supplied engine path that
the panel would otherwise execute on this machine.
"""

import json
import os
import stat
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from lllm2 import config, panel_auth
from lllm2.app import make_server
from test_panel_fake_engine import app, no_local_gpu  # noqa: F401  (fixtures)

SECRET = "s3cret-for-tests"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Notices:
    def notice(self):
        return None


@pytest.fixture
def panel(app):  # noqa: F811  (pytest fixture injection)
    app.update_check = Notices()
    app.selected_hardware = lambda query: {"gpus": []}
    server = make_server("127.0.0.1", 0, app, token=SECRET)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", server
    server.shutdown()
    server.server_close()
    thread.join(5)


def call(url, path, data=None, headers=None):
    """Return (status, headers, body) for one request, without following redirects."""
    request = urllib.request.Request(
        url + path,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="GET" if data is None else "POST",
    )
    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(request, timeout=10) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


AUTH = {panel_auth.TOKEN_HEADER: SECRET}


def test_status_needs_the_secret_and_never_returns_it(panel):
    url, _ = panel
    status, _, body = call(url, "/api/status")
    assert status == 401
    assert SECRET not in body.decode()
    status, _, body = call(url, "/api/status", headers=AUTH)
    assert status == 200
    assert "token" not in json.loads(body)
    assert SECRET not in body.decode()


@pytest.mark.parametrize("path", ["/api/results", "/api/results/export"])
def test_result_reads_need_the_secret(panel, path):
    url, _ = panel
    assert call(url, path)[0] == 401
    assert call(url, path, headers=AUTH)[0] == 200


def test_post_needs_the_secret(panel, tmp_path):
    url, _ = panel
    request = {"directory": str(tmp_path)}
    assert call(url, "/api/files", request)[0] == 401
    assert (
        call(url, "/api/files", request, {panel_auth.TOKEN_HEADER: "wrong"})[0] == 401
    )
    assert call(url, "/api/files", request, AUTH)[0] == 200


def test_session_link_sets_a_strict_cookie_that_unlocks_the_panel(panel):
    url, server = panel
    port = server.server_address[1]
    assert call(url, "/")[0] == 401
    assert call(url, "/?token=wrong")[0] == 401
    status, headers, _ = call(url, f"/?token={SECRET}")
    assert status == 303
    assert headers["Location"] == "/"
    cookie = headers["Set-Cookie"]
    assert cookie.startswith(f"{panel_auth.cookie_name(port)}={SECRET};")
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    jar = {"Cookie": cookie.split(";")[0]}
    assert call(url, "/", headers=jar)[0] == 200
    assert call(url, "/api/status", headers=jar)[0] == 200
    # A cookie for another port's panel does not unlock this one.
    other = {"Cookie": f"{panel_auth.cookie_name(port + 1)}={SECRET}"}
    assert call(url, "/api/status", headers=other)[0] == 401


def test_cross_origin_post_is_refused_even_with_the_secret(panel, tmp_path):
    url, _ = panel
    status, _, _ = call(
        url,
        "/api/files",
        {"directory": str(tmp_path)},
        AUTH | {"Origin": "http://evil.example"},
    )
    assert status == 403


def test_foreign_host_is_refused(panel):
    url, _ = panel
    assert call(url, "/api/status", headers=AUTH | {"Host": "evil.example"})[0] == 403


def test_panel_never_runs_an_unconfigured_engine(panel, tmp_path, monkeypatch):
    """A client with the secret still cannot make the panel run any binary."""
    url, _ = panel
    monkeypatch.setattr(config, "ENGINE_ROOTS", [tmp_path / "engines"])
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    marker = tmp_path / "ran"
    payload = tmp_path / "payload.sh"
    payload.write_text(f"#!/bin/sh\ntouch {marker}\n")
    payload.chmod(0o755)
    model = str(config.MODELS_DIR / "example" / "model.gguf")
    for path, request in [
        ("/api/start", {"settings": {"model": model, "engine": str(payload)}}),
        ("/api/capabilities", {"settings": {"model": model, "engine": str(payload)}}),
        ("/api/launch/select", {"model": model, "engine": str(payload)}),
        (
            "/api/benchmark",
            {
                "settings": {"model": model},
                "combinations": [{"model": model, "engine": str(payload)}],
            },
        ),
    ]:
        status, _, body = call(url, path, request, AUTH)
        assert status == 400, (path, body)
        assert "LLLM2_ENGINE_ROOTS" in json.loads(body)["error"]
    assert not marker.exists()


def test_panel_refuses_a_chat_template_outside_the_models_folder(panel, tmp_path):
    url, _ = panel
    secret = tmp_path / "id_ed25519"
    secret.write_text("PRIVATE KEY")
    secret.chmod(0o600)
    model = str(config.MODELS_DIR / "example" / "model.gguf")
    request = {"settings": {"model": model, "chat_template": str(secret)}}
    status, _, body = call(url, "/api/start", request, AUTH)
    assert status == 400
    assert "Chat templates must be under" in json.loads(body)["error"]


def test_configured_engines_pass_the_engine_check(tmp_path, monkeypatch):
    root = tmp_path / "engines"
    binary = root / "b1" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    custom = tmp_path / "custom-llama-server"
    custom.write_text("#!/bin/sh\n")
    custom.chmod(0o755)
    monkeypatch.setattr(config, "ENGINE_ROOTS", [root, custom])
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    panel_auth.check_request_paths({"settings": {"engine": str(binary)}})
    panel_auth.check_request_paths({"engine": str(custom)})
    link = tmp_path / "link"
    link.symlink_to(binary)
    panel_auth.check_request_paths({"engine": str(link)})
    with pytest.raises(ValueError, match="LLLM2_ENGINE_ROOTS"):
        panel_auth.check_request_paths({"engine": "/bin/sh"})


def test_templates_in_models_folder_or_package_pass(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", models)
    good = models / "chat.jinja"
    good.write_text("{{ messages }}")
    panel_auth.check_template(str(good))
    panel_auth.check_template(str(panel_auth.TEMPLATE_DIR / "qwen3.8-27b.jinja"))
    wrong_suffix = models / "chat.key"
    wrong_suffix.write_text("x")
    with pytest.raises(ValueError, match="must end in"):
        panel_auth.check_template(str(wrong_suffix))
    big = models / "big.jinja"
    big.write_bytes(b"x" * (panel_auth.TEMPLATE_MAX_BYTES + 1))
    with pytest.raises(ValueError, match="larger"):
        panel_auth.check_template(str(big))
    escape = models / "escape.jinja"
    outside = tmp_path / "outside.jinja"
    outside.write_text("x")
    escape.symlink_to(outside)
    with pytest.raises(ValueError, match="must be under"):
        panel_auth.check_template(str(escape))


def test_session_file_is_private_and_removed_only_by_its_owner(tmp_path):
    record = {"token": "abc", "urls": ["http://127.0.0.1:8082/?token=abc"]}
    path = panel_auth.write_session(record, tmp_path)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert panel_auth.read_session(tmp_path) == record
    panel_auth.remove_session("other", tmp_path)
    assert path.exists()
    panel_auth.remove_session("abc", tmp_path)
    assert not path.exists()


def test_panel_url_prints_the_running_session(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from lllm2.cli import app as cli

    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    runner = CliRunner()
    missing = runner.invoke(cli, ["panel-url"])
    assert missing.exit_code == 1
    urls = ["http://127.0.0.1:8082/?token=abc", "http://gpu:8082/?token=abc"]
    panel_auth.write_session({"token": "abc", "urls": urls}, Path(tmp_path))
    shown = runner.invoke(cli, ["panel-url"])
    assert shown.exit_code == 0
    assert shown.output.split() == urls
    assert runner.invoke(cli, ["panel-url", "--token"]).output.strip() == "abc"
