# NVIDIA Jetson (vn-desktop) Remote Access

This skill describes how to connect to the NVIDIA Jetson Orin used for the cube filling machine (macchina riempimento cubi) project.

## Device Info

- **Hostname**: `vn-desktop`
- **User**: `vn`
- **Architecture**: `aarch64` (ARM64)
- **OS**: Ubuntu 20.04 (L4T / JetPack)
- **Local IP** (may change): `192.168.1.118` or `192.168.1.155`
- **Tailscale IP**: `100.118.77.39`

## SSH Access (Preferred)

The Jetson has **Tailscale** installed and active with SSH enabled.

```bash
ssh vn@100.118.77.39
```

- Password is stored in the secret `JETSON_SSH_PASSWORD`
- Use `sshpass` for non-interactive access:
  ```bash
  sshpass -p "$JETSON_SSH_PASSWORD" ssh -o StrictHostKeyChecking=no vn@100.118.77.39 "<command>"
  ```
- Tailscale starts automatically at boot — no tunnel or port forwarding needed.

## Splashtop (Remote Desktop)

Splashtop Streamer is installed and deployed on the Jetson:
- Service name: `SRStreamer`
- Starts automatically at boot
- The Jetson appears as `vn-desktop` in the Splashtop account
- Use Splashtop for GUI-based tasks that cannot be done over SSH

## Devin Browser on the Jetson

The Jetson has a custom Chromium launcher for Devin:
- **Desktop icon**: `~/Desktop/devin.desktop`
- **Terminal alias**: type `devin` to launch
- **Key flag**: `--disable-gpu` is required or the page stays blank
- The full command:
  ```
  chromium-browser --user-data-dir=$HOME/.config/devin-chromium --disable-gpu --disable-extensions https://app.devin.ai
  ```

## Fallback: Pinggy Tunnel

If Tailscale is down, ask Andrea to run this on the Jetson terminal:
```bash
ssh -p 443 -o ServerAliveInterval=30 -R0:localhost:22 tcp@free.pinggy.io
```
This gives a temporary `tcp://host:port` address (expires in 60 min).

## Important Notes

- The Jetson is ARM64 — only install `arm64`/`aarch64` packages
- `curl` was manually installed (not present by default)
- The router cannot be modified for port forwarding
- Standard `apt install tailscale` does NOT work on the Jetson; Tailscale was installed via the official install script which adds the correct APT repo
- Splashtop Streamer uses the **Raspberry Pi ARM64** `.deb` package from Splashtop's download page
