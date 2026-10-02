const js = require("@eslint/js");
const globals = require("globals");
module.exports = [
  { ignores: ["node_modules/**", "test-results/**", "playwright-report/**"] },
  {
    files: ["static/*.js", "Testing/UI/*.cjs", "*.config.cjs"],
    ...js.configs.recommended,
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
  },
];
