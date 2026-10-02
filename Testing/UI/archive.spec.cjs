const { test, expect } = require("@playwright/test");
const AxeBuilder = require("@axe-core/playwright").default;
const fs = require("node:fs");
const path = require("node:path");
const fixtures = require("./fixtures.cjs");
const root = path.resolve(__dirname, "../..");

async function openArchive(page, options = {}) {
  const requests = [];
  const forbidden = [];
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (url.origin !== "http://archive.test") {
      forbidden.push(url.href);
      return route.abort();
    }
    const assets = {
      "/": ["static/index.html", "text/html"],
      "/static/archive.css": ["static/archive.css", "text/css"],
      "/static/archive.js": ["static/archive.js", "text/javascript"],
    };
    if (assets[url.pathname]) {
      const [file, contentType] = assets[url.pathname];
      return route.fulfill({
        body: fs.readFileSync(path.join(root, file)),
        contentType,
      });
    }
    if (url.pathname === "/witnesses") {
      return options.witnesses
        ? options.witnesses(route)
        : route.fulfill({ json: { witnesses: fixtures.witnesses } });
    }
    if (["/search", "/search/contradictions"].includes(url.pathname)) {
      const body = route.request().postDataJSON();
      requests.push({ endpoint: url.pathname, ...body });
      if (options.search) return options.search(route, body);
      return route.fulfill({
        json: url.pathname.endsWith("/contradictions")
          ? {
              query: body.query,
              contradictions: fixtures.contradictions,
              total_contradictions: 1,
            }
          : { query: body.query, results: fixtures.results, total_results: 3 },
      });
    }
    forbidden.push(url.pathname);
    return route.abort();
  });
  await page.goto("http://archive.test/");
  return { requests, forbidden, errors };
}
async function search(page, query = "ice warnings") {
  await page.locator("#query").fill(query);
  await page.locator("#searchForm").evaluate((form) => form.requestSubmit());
}
async function filters(page) {
  await page.locator("#filtersPanel").evaluate((details) => {
    details.open = true;
  });
}
async function noOverflow(page) {
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
}
async function screenshot(page, testInfo, name) {
  if (!process.env.UI_SCREENSHOT_DIR) return;
  await page.evaluate(() => {
    document.activeElement?.blur();
    window.scrollTo(0, 0);
    if (document.querySelector("#fixtureBanner")) return;
    const banner = document.createElement("div");
    banner.id = "fixtureBanner";
    banner.textContent =
      "OFFLINE DESIGN REVIEW · Illustrative test fixtures, not historical quotations";
    banner.style.cssText =
      "background:#85452e;color:white;padding:8px 18px;font:11px/1.5 Arial;text-align:center;";
    document.body.prepend(banner);
  });
  fs.mkdirSync(process.env.UI_SCREENSHOT_DIR, { recursive: true });
  await page.screenshot({
    path: path.join(
      process.env.UI_SCREENSHOT_DIR,
      `${testInfo.project.name}-${name}.png`,
    ),
    fullPage: true,
  });
}

test("initial layout and accessible search do not contact live dependencies", async ({
  page,
}, testInfo) => {
  const state = await openArchive(page);
  await expect(page.locator("#witnessName")).toBeEnabled();
  await expect(
    page.getByRole("heading", { name: "The Titanic inquiries." }),
  ).toBeVisible();
  await noOverflow(page);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await screenshot(page, testInfo, "initial");
  expect(state.forbidden).toEqual([]);
  expect(state.errors).toEqual([]);
});

test("search shows citations, highlights, safe text, and expandable excerpts", async ({
  page,
}, testInfo) => {
  const state = await openArchive(page);
  await search(page);
  await expect(page.locator(".evidence")).toHaveCount(3);
  await expect(page.locator(".evidence").first()).toContainText(
    "Printed page 8",
  );
  await expect(page.locator(".evidence mark").first()).toHaveText("ice");
  const detail = page.locator(".excerpt-disclosure").first();
  await detail.locator("summary").click();
  await expect(detail).toHaveAttribute("open", "");
  await expect(page.locator(".preview").first()).toBeHidden();
  await detail.locator("summary").click();
  await expect(page.locator(".preview").first()).toBeVisible();
  await page.locator(".search-details summary").first().click();
  await expect(page.locator(".search-details").first()).toContainText(
    "Similarity: 0.724",
  );
  await page.locator(".search-details summary").first().click();
  await noOverflow(page);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await screenshot(page, testInfo, "evidence");
  expect(state.requests[0]).toMatchObject({
    query: "ice warnings",
    witness_name: null,
    top_k: 5,
    similarity_threshold: 0.4,
  });
  expect(state.errors).toEqual([]);
});

test("witness selection clears immediately on edit and clear; keyboard selection works", async ({
  page,
}) => {
  const state = await openArchive(page);
  await filters(page);
  await page.locator("#witnessName").fill("Ismay");
  await page.locator("#witnessName").press("ArrowDown");
  await expect(page.locator("#witnessName")).toHaveAttribute(
    "aria-activedescendant",
    "witness-0",
  );
  await page.locator("#witnessName").press("Enter");
  await search(page);
  await expect(page.locator(".evidence")).toHaveCount(3);
  expect(state.requests[0].witness_name).toBe("J. Bruce Ismay");
  await page.locator("#witnessName").fill("Fleet");
  await page.locator("#searchForm").evaluate((form) => form.requestSubmit());
  expect(state.requests).toHaveLength(1); // Unselected text cannot reuse Ismay or silently broaden the search.
  await expect(page.locator("#witnessName")).toHaveJSProperty(
    "validationMessage",
    "Choose a witness from the list, or clear this field.",
  );
  await page.getByRole("option", { name: "Frederick Fleet" }).click();
  await page.locator("#searchForm").evaluate((form) => form.requestSubmit());
  await expect.poll(() => state.requests.length).toBe(2);
  expect(state.requests[1].witness_name).toBe("Frederick Fleet");
  await page.getByRole("button", { name: "Clear witness" }).click();
  await page.locator("#searchForm").evaluate((form) => form.requestSubmit());
  await expect.poll(() => state.requests.length).toBe(3);
  expect(state.requests[2].witness_name).toBeNull();
  await expect(page.locator("#witnessName")).toHaveAttribute(
    "aria-expanded",
    "false",
  );
});

for (const lateError of [false, true]) {
  test(`late ${lateError ? "error" : "success"} cannot replace newer preset results`, async ({
    page,
  }) => {
    const pending = [];
    // Ignore abort deliberately: sequence protection must work even when cancellation loses the race.
    await page.addInitScript(() => {
      const original = window.fetch;
      window.fetch = (url, options) =>
        original(url, options ? { ...options, signal: undefined } : options);
    });
    const state = await openArchive(page, {
      search: (route, body) => {
        pending.push({ route, body });
      },
    });
    await page.getByRole("button", { name: "Ship speed", exact: true }).click();
    await expect.poll(() => pending.length).toBe(1);
    await page
      .getByRole("button", { name: "Ice warnings", exact: true })
      .click();
    await expect.poll(() => pending.length).toBe(2);
    await expect(page.locator("#resultsCount")).toBeEmpty();
    await expect(page.locator("#resultsSection")).toHaveAttribute(
      "aria-busy",
      "true",
    );
    await pending[1].route.fulfill({
      json: { query: pending[1].body.query, results: fixtures.results },
    });
    await expect(page.locator(".evidence")).toHaveCount(3);
    await pending[0].route.fulfill(
      lateError
        ? { status: 500, json: { detail: "stale error" } }
        : { json: { query: "stale success", results: [] } },
    );
    await page.waitForLoadState("networkidle");
    await expect(page.locator("#resultsContext")).toContainText(
      "Were ice warnings received by wireless?",
    );
    await expect(page.locator(".evidence")).toHaveCount(3);
    await expect(page.locator("#searchBtn")).toHaveText("Search →");
    expect(state.errors).toEqual([]);
  });
}

test("repeated submission deduplicates; edits cancel pending work", async ({
  page,
}) => {
  const pending = [];
  const state = await openArchive(page, {
    search: (route) => {
      pending.push(route);
    },
  });
  await search(page);
  await expect.poll(() => pending.length).toBe(1);
  await page.locator("#searchForm").evaluate((form) => form.requestSubmit());
  expect(state.requests).toHaveLength(1);
  await page.locator("#query").fill("different question");
  await expect(page.locator("#resultsSection")).toHaveAttribute(
    "aria-busy",
    "false",
  );
  await expect(page.locator("#resultsTitle")).toHaveText("Search paused");
  await pending[0]
    .fulfill({ json: { results: fixtures.results } })
    .catch(() => {});
  await expect(page.locator(".evidence")).toHaveCount(0);
});

test("comparison mode preserves controls, attribution and cautious language", async ({
  page,
}, testInfo) => {
  const state = await openArchive(page);
  await page.getByLabel("Compare accounts", { exact: true }).check();
  await filters(page);
  await page.locator("#sourceType").selectOption("british_inquiry");
  await page.locator("#topK").selectOption("10");
  await page.locator("#minConfidence").fill("0.75");
  await search(page, "when were messages received?");
  await expect(page.locator(".comparison")).toHaveCount(1);
  expect(state.requests[0]).toMatchObject({
    endpoint: "/search/contradictions",
    min_confidence: 0.75,
    source_type: "british_inquiry",
    top_k: 10,
  });
  await expect(page.locator(".comparison")).toContainText(
    "Same witness, across both inquiries",
  );
  await expect(page.locator(".comparison")).toContainText("Printed page 440");
  await expect(page.locator(".claim-label")).toHaveCount(2);
  await page.locator(".excerpt-disclosure summary").first().click();
  await expect(page.locator(".excerpt-disclosure").first()).toHaveAttribute(
    "open",
    "",
  );
  await page.locator(".excerpt-disclosure summary").first().click();
  await noOverflow(page);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.locator("#sourceType").selectOption("");
  await page.locator("#searchForm").evaluate((form) => form.requestSubmit());
  await expect(page.locator(".comparison")).toHaveCount(1);
  await screenshot(page, testInfo, "comparison");
  await page.locator("#resetFilters").click();
  await expect(page.locator("#sourceType")).toHaveValue("");
  await expect(page.locator("#minConfidence")).toHaveValue("0.6");
});

test("empty results and comparisons avoid claiming agreement", async ({
  page,
}, testInfo) => {
  await openArchive(page, {
    search: (route) =>
      route.fulfill({ json: { results: [], contradictions: [] } }),
  });
  await search(page);
  await expect(
    page.getByRole("heading", { name: "No matching passages." }),
  ).toBeVisible();
  await page.getByLabel("Compare accounts", { exact: true }).check();
  await page.locator("#searchForm").evaluate((form) => form.requestSubmit());
  await expect(page.locator("#resultsContent")).toContainText(
    "This does not establish that the accounts agree.",
  );
  await screenshot(page, testInfo, "empty");
});

test("errors reset counts and expose retry; HTML and validation errors stay readable", async ({
  page,
}, testInfo) => {
  let attempt = 0;
  await openArchive(page, {
    search: (route) => {
      attempt += 1;
      if (attempt === 1)
        return route.fulfill({
          status: 429,
          json: { detail: "Rate limit reached. Please try again shortly." },
        });
      if (attempt === 2)
        return route.fulfill({
          status: 502,
          contentType: "text/html",
          body: "<h1>Bad gateway</h1>",
        });
      if (attempt === 3)
        return route.fulfill({
          status: 422,
          json: { detail: [{ msg: "invalid" }] },
        });
      return route.fulfill({ json: { results: fixtures.results } });
    },
  });
  await search(page);
  await expect(page.getByRole("alert")).toContainText("Rate limit reached");
  await expect(page.locator("#resultsCount")).toBeEmpty();
  await screenshot(page, testInfo, "error");
  await page.getByRole("button", { name: "Try again", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("unreadable response");
  await page.getByRole("button", { name: "Try again", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("could not be completed");
  await page.getByRole("button", { name: "Try again", exact: true }).click();
  await expect(page.locator(".evidence")).toHaveCount(3);
});

test("witness loading failure leaves searching usable", async ({ page }) => {
  await openArchive(page, {
    witnesses: (route) =>
      route.fulfill({ status: 503, json: { detail: "unavailable" } }),
  });
  await filters(page);
  await expect(page.locator("#witnessHint")).toContainText(
    "Witness names are unavailable",
  );
  await expect(page.locator("#witnessName")).toBeDisabled();
  await search(page);
  await expect(page.locator(".evidence")).toHaveCount(3);
});

test("untrusted response text is escaped and narrow layouts do not overflow", async ({
  page,
}) => {
  await openArchive(page, {
    search: (route) =>
      route.fulfill({
        json: {
          results: [
            {
              ...fixtures.results[0],
              witness_name: "<img src=x onerror=alert(1)>",
              content: "<script>window.fixtureExecuted = true</script> **ice**",
            },
          ],
        },
      }),
  });
  await page.setViewportSize({ width: 320, height: 850 });
  await search(page);
  await expect(page.locator(".witness-name")).toHaveText(
    "<img src=x onerror=alert(1)>",
  );
  await expect(page.locator(".evidence img, .evidence script")).toHaveCount(0);
  expect(await page.evaluate(() => window.fixtureExecuted)).toBeUndefined();
  await noOverflow(page);
});

test("witness keyboard navigation starts at either end and wraps", async ({
  page,
}) => {
  await openArchive(page);
  await filters(page);
  const input = page.locator("#witnessName");
  await input.fill("i");
  await input.press("ArrowUp");
  await expect(
    page.locator("#witnessDropdown").getByRole("option", { selected: true }),
  ).toHaveText("J. Bruce Ismay");
  await input.press("ArrowDown");
  await expect(
    page.locator("#witnessDropdown").getByRole("option", { selected: true }),
  ).toHaveText("Charles Herbert Lightoller");
  await input.press("ArrowUp");
  await input.press("Enter");
  await expect(input).toHaveValue("J. Bruce Ismay");
  await expect(input).toHaveAttribute("aria-expanded", "false");
});

test("loading can be cancelled explicitly and announced", async ({
  page,
}, testInfo) => {
  let pending;
  await openArchive(page, {
    search: (route) => {
      pending = route;
    },
  });
  await search(page);
  await expect(page.locator("#resultsSection")).toHaveAttribute(
    "aria-busy",
    "true",
  );
  await expect(
    page.getByRole("button", { name: "Cancel search" }),
  ).toBeVisible();
  await screenshot(page, testInfo, "loading");
  await page.getByRole("button", { name: "Cancel search" }).click();
  await expect(page.locator("#searchStatus")).toContainText("Search cancelled");
  await expect(page.locator("#resultsSection")).toHaveAttribute(
    "aria-busy",
    "false",
  );
  await pending.fulfill({ json: { results: [] } }).catch(() => {});
});
