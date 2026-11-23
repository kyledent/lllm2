# Paths and ports

Set environment variables before starting the workbench or a client wrapper.

| Variable | Default | Contents |
| --- | --- | --- |
| `LLLM2_MODELS_DIR` | `~/models` | GGUF checkpoints and downloads. |
| `LLLM2_STATE_DIR` | `~/.local/state/lllm2` | `workbench.sqlite3`, the panel lock, the running panel's `panel-session.json` (session link and secret, mode 0600), and the remote files `remote-calls.json` (owned serve calls, mode 0600) and `remote-probes.json` (GPU probe results). |
| `LLLM2_ENGINE_HOME` | `~/.local/share/lllm2/engines` | Newly compiled engines. |
| `LLLM2_ENGINE_ROOTS` | `LLLM2_ENGINE_HOME` | Colon-separated engine search roots; setting it replaces the defaults. |
| `LLLM2_ENGINE_PORT` | `1920` | Model server port on loopback. A remote backend's local proxy uses the same port. |
| `LLLM2_IDLE_TIMEOUT_MINUTES` | `30` | Default idle stop for a remote backend, in whole minutes from 0 to 1440; `0` or `off` disables it. For `lllm2 launch`, `--idle-timeout` overrides it, and it overrides saved settings. The panel fills its idle field from it. lllm2 reads it at startup. |
| `LLLM2_UPDATE_CHECK` | `1` | `0` or `off` stops the panel checking GitHub once a day for a newer lllm2 release. The check runs in the background and fails silently offline. |

The panel defaults to `127.0.0.1:8082`. Its controls use `/api/*`; model
clients connect to the separate llama.cpp server on port 1920. Protocol
support depends on the selected engine build.

The panel creates a new session secret each time it starts. Open the link it
prints, or run `lllm2 panel-url`, once per browser: the link sets an HttpOnly,
SameSite=Strict cookie for that port. Other API clients send the secret, from
`lllm2 panel-url --token`, in an `X-LLLM2-Token` header. Every `/api/*`
request without one gets HTTP 401. Settings sent to the panel may name only an
engine under `LLLM2_ENGINE_ROOTS` or `llama-server` on `PATH`, and a chat
template under `LLLM2_MODELS_DIR` or lllm2's packaged templates (a `.jinja`,
`.jinja2`, `.j2`, `.tmpl` or `.txt` file up to 1 MB); `lllm2 launch` on the
command line is not restricted.
