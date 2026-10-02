import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir:"tests/browser",timeout:120000,workers:1,
  use:{baseURL:process.env.STUDIO_TEST_URL ?? "http://127.0.0.1:41240",viewport:{width:1440,height:1000},
    // Traces contain bearer headers/first-frame tokens; do not record them.
    trace:"off",screenshot:"only-on-failure"},
});
