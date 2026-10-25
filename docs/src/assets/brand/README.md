# Corporate source

Observed 2026-10-22 at https://nexufed.eu1.netbird.services/.

- Wordmark: `/images/brand/nexufed-wordmark.webp` (966 × 217).
  SHA-256: `f4144f862a5e032ea3e9aa7380a64787c024cfecbdf8d5c52c456ab9f03ac270`.
- Geist Latin variable font: `/_next/static/media/caa3a2e1cccd8315-s.p.0wgildi0cnwt9.woff2`.
  SHA-256: `9b6f5ff45b278c744b5f379a2c4ecbaf858a842b8eaf82ac8d21b699ca16c608`.
  Upstream: https://github.com/vercel/geist-font (SIL Open Font License).
- Served CSS: `/_next/static/chunks/[root-of-the-server]__1moo6zs._.css`.
  Tokens: charcoal `#101010`, surface `#181818`, white `#f3f3f3`, muted `#aaa`,
  blue `#0da2e7`, teal `#20cb98`, line `#303030`. Body Geist, 16px, 1.65 leading;
  paragraphs ≤65ch, restrained heading tracking, blue visible keyboard focus.

## Documentation application

Keep the native sidebar/article/TOC grid. Use Geist for prose and headings; keep
code in Starlight's monospace stack. Left-align content, limit reading width, and
use surface steps rather than decorative cards. The recognizable wordmark and
blue active states provide identity; no promotional hero or animation.

The corporate display-heading sizes/tracking are too tight for API documentation:
use smaller, less tightly tracked article headings. Light mode uses darker blue
and teal for contrast, not the corporate dark-background accent values unchanged.
The UI guidance search favored minimal developer documentation, but its generic
IBM Plex/green palette was rejected in favor of this actual corporate source.
