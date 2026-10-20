# Studio: local and authenticated HTTPS access

NexuML Studio runs over your selected, already-installed NexuML. See the
[Studio guide](https://github.com/NexuFed/NexuML/blob/main/studio/README.md) for installation, editing, training and runtime selection.
Its API interface is **2**; neither mode changes Python execution or security semantics.

## Local/offline default

```sh
nexuml-studio --port 3000 --api-port 8000 /path/to/work
```

Both processes bind to loopback by default. The browser opens `http://127.0.0.1:3000`
and sends HTTP/WebSocket traffic through that same origin. Prebuilt UI startup requires
no internet access or dependency installation. The private API never needs browser access.
The launcher stops only services it owns; an attached API is never stopped.

## Existing authenticated HTTPS reverse proxy

```sh
nexuml-studio --port 3000 --public-origin https://studio.example --no-open /path/to/work
```

`--public-origin` (or `NEXUML_STUDIO_PUBLIC_ORIGIN`) opts into hosting and defaults the UI
listen address to `0.0.0.0`. Choose a private IPv4 interface with `--host` (or
`NEXUML_STUDIO_HOST`) if needed. Non-loopback hosting without an exact HTTPS public origin
is rejected. CLI flags take precedence. Origins have no credentials, trailing slash,
path, query or fragment. The API always remains loopback-only; its configured origin
remains `http://127.0.0.1:<UI port>`, including in attach mode.

Configure the proxy, including NetBird, outside Studio:

- One `/` target to the **UI port**, with **Pass Host Header enabled**.
- HTTPS, WebSocket forwarding, and authentication/access restrictions for trusted operators.
- No API target or externally reachable API port.
- Firewall/private-network restrictions on direct UI-port access so the proxy's access
  controls cannot be bypassed. Local access remains available when the UI listen interface
  includes loopback.

All browser API requests and observations use the public origin; HTTPS uses WSS.
The Next proxy forwards to the runtime-selected API port. It preserves browser bearer
authorization and WebSocket first-frame auth, rewrites upstream Host/Origin to their
local values, and strips cookies and forwarded-proxy metadata. Bootstrap returns the
browser origin and an in-memory runtime credential, never the private API address.
Host, optional Origin and Fetch Metadata must match configured settings; forwarded
Host/Proto do not establish trust. Python token/origin/interface validation is unchanged.

This is trusted-operator hosting, **not public/multi-tenant SaaS**. Anyone admitted to
Studio can obtain its runtime credential and execute trusted local Python. Studio adds
no account system and does not mutate NetBird settings. Production and development need
the same external access restrictions. For dev/HMR commands, see the Studio guide; the
allowlist includes only `127.0.0.1` and the explicitly configured public hostname.
