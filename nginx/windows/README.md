# Windows nginx (front door)

The nginx that runs on the Windows host (`C:\nginx`) in front of the WSL stack. It has two server blocks:

- **`lan.example.com`**: the LAN hostname, with a public certificate. It serves Open WebUI at `/`, the avatar at `/avatar/`, Sparkboard at `/sparkboard/`, and the model APIs at `/llama/` and `/inkling/`. The model APIs go through Sparkboard's [model director](../../README.md#model-director). Everything except Open WebUI and the avatar is LAN-only.
- **`origin.example.com`**: the public origin behind CloudFront. It serves only the avatar, refuses anything that didn't come through CloudFront (`X-Origin-Verify`), and bans brute-force attempts through authgate.

`avatar-app.conf` is included by both server blocks. It holds the avatar page and its services, with a Google sign-in for anyone not on the LAN.

## Before you use it

Replace the placeholders:

| Placeholder | What goes there |
|---|---|
| `lan.example.com`, `origin.example.com`, `www.example.com` | Your hostnames |
| `C:/nginx-certs/...pem` | Your certificate and key (for example from win-acme) |
| `<CLOUDFRONT_ORIGIN_SECRET>` | The custom origin header value set in your CloudFront distribution |
| `<LLAMA_API_KEY>`, `<LENS_BRIDGE_KEY>` | The values from `.env`. nginx adds them for the avatar, so the keys never reach the browser |
| `192.168.0.0/16` | Your LAN range |

Copy both files to `C:\nginx\conf\`. Then test the config and reload nginx from an administrator prompt:

```powershell
C:\nginx\nginx.exe -t -p C:/nginx/
C:\nginx\nginx.exe -s reload -p C:/nginx/
```

Or run `ops/windows-setup.ps1 -NginxDir C:\nginx` once, which adds an `nginx-reload` task. After that, `schtasks.exe /Run /TN nginx-reload` reloads nginx from WSL without an administrator prompt.

The keys sit in plain text in `avatar-app.conf`, so when you rotate `LLAMA_API_KEY` or `LENS_BRIDGE_KEY`, update this file too.
