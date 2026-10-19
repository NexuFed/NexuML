import { publicOrigin } from "./bin/connection.mjs";

const origin = publicOrigin(process.env.NEXUML_STUDIO_PUBLIC_ORIGIN);

export default {
  experimental: { proxyTimeout: 90000 },
  // No remote images, ISR or cached API responses: the local runtime works offline.
  images: { unoptimized: true },
  poweredByHeader: false,
  allowedDevOrigins: ["127.0.0.1", ...(origin ? [new URL(origin).hostname] : [])],
};
