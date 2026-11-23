# Access over SSH

Forward the panel and API ports from your client machine:

```bash
ssh -N -L 8082:127.0.0.1:8082 -L 1920:127.0.0.1:1920 user@gpu-host
```

Run `ssh user@gpu-host lllm2 panel-url` and open the `http://127.0.0.1:8082/?token=...`
link it prints on your client machine; the tunnel carries it to the panel.
`lllm2 panel --host 0.0.0.0` also exposes the panel on a trusted LAN. The
session link still guards it, but there is no TLS, so anyone who can watch
the network can copy the link or cookie and control the workbench.
