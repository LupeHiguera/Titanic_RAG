const { defineConfig } = require("@playwright/test");
module.exports = defineConfig({
  testDir: "./Testing/UI",
  testMatch: "**/*.spec.cjs",
  timeout: 20000,
  fullyParallel: true,
  workers: 2,
  reporter: "list",
  use: {
    headless: true,
    launchOptions: process.env.CHROMIUM_PATH
      ? { executablePath: process.env.CHROMIUM_PATH }
      : {},
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1440, height: 1080 } } },
    {
      name: "mobile",
      use: {
        viewport: { width: 390, height: 844 },
        isMobile: true,
        hasTouch: true,
      },
    },
  ],
});
