import { copyFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

// Release-only asset copy, never executed by the installed launcher.
await copyFile(fileURLToPath(new URL("../../LICENSE",import.meta.url)),fileURLToPath(new URL("../LICENSE",import.meta.url)));
