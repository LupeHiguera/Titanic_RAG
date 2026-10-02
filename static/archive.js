"use strict";

const $ = (id) => document.getElementById(id);
let availableWitnesses = [];
let selectedWitness = "";
let activeRequest = null;
let requestVersion = 0;

// Install interactions before the asynchronous witness list has loaded.
setupWitnessAutocomplete();
setupSearch();
loadWitnesses();
if (window.matchMedia("(max-width: 700px)").matches)
  $("filtersPanel").open = false;

async function loadWitnesses() {
  $("witnessName").disabled = true;
  try {
    const response = await fetch("/witnesses");
    if (!response.ok) throw new Error("Witness list unavailable");
    const data = await response.json();
    if (!Array.isArray(data.witnesses)) throw new Error("Invalid witness list");
    availableWitnesses = data.witnesses.filter(
      (name) => typeof name === "string",
    );
    $("witnessName").disabled = false;
    $("witnessHint").textContent = "Choose a name to apply this filter.";
  } catch {
    $("witnessHint").textContent =
      "Witness names are unavailable. You can still search both inquiries. Reload to try again.";
  }
}

function setupWitnessAutocomplete() {
  const input = $("witnessName");
  const list = $("witnessDropdown");
  let filtered = [];
  let highlight = -1;

  function hide() {
    list.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
    highlight = -1;
  }
  function render() {
    filtered = availableWitnesses.filter((name) =>
      name.toLowerCase().includes(input.value.trim().toLowerCase()),
    );
    if (!input.value.trim()) {
      hide();
      return;
    }
    highlight = -1;
    input.removeAttribute("aria-activedescendant");
    list.innerHTML = filtered
      .map(
        (name, index) =>
          `<li id="witness-${index}" role="option" aria-selected="false" data-index="${index}">${escapeHtml(name)}</li>`,
      )
      .join("");
    if (!filtered.length) {
      hide();
      $("witnessHint").textContent =
        "No matching name. Edit or clear this filter.";
      return;
    }
    list.hidden = false;
    input.setAttribute("aria-expanded", "true");
  }
  function pick(name) {
    selectedWitness = name;
    input.value = name;
    input.setCustomValidity("");
    $("clearWitness").hidden = false;
    $("witnessHint").textContent = "Witness filter selected.";
    hide();
    input.focus();
    cancelPending();
  }
  input.addEventListener("invalid", () => {
    $("filtersPanel").open = true;
  });
  input.addEventListener("input", () => {
    // Invalidate immediately: a rapid submit must never reuse the old name.
    selectedWitness = "";
    input.setCustomValidity(
      input.value.trim()
        ? "Choose a witness from the list, or clear this field."
        : "",
    );
    $("clearWitness").hidden = !input.value;
    $("witnessHint").textContent = "Choose a name to apply this filter.";
    cancelPending();
    render();
  });
  input.addEventListener("focus", () => {
    if (!selectedWitness) render();
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      hide();
      return;
    }
    if (event.key === "Tab") {
      hide();
      return;
    }
    if (event.key === "Enter" && !list.hidden && highlight >= 0) {
      event.preventDefault();
      pick(filtered[highlight]);
      return;
    }
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    if (list.hidden) render();
    if (list.hidden || !filtered.length) return;
    const direction = event.key === "ArrowDown" ? 1 : -1;
    highlight =
      highlight < 0
        ? direction > 0
          ? 0
          : filtered.length - 1
        : (highlight + direction + filtered.length) % filtered.length;
    list
      .querySelectorAll("[role=option]")
      .forEach((option, index) =>
        option.setAttribute("aria-selected", String(index === highlight)),
      );
    const option = $(`witness-${highlight}`);
    input.setAttribute("aria-activedescendant", option.id);
    option.scrollIntoView({ block: "nearest" });
  });
  list.addEventListener("pointerdown", (event) => event.preventDefault());
  list.addEventListener("click", (event) => {
    const option = event.target.closest("[data-index]");
    if (option) pick(filtered[Number(option.dataset.index)]);
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".autocomplete-wrap")) hide();
  });
  $("clearWitness").addEventListener("click", () => {
    clearWitness();
    hide();
    input.focus();
    cancelPending();
  });
  $("resetFilters").addEventListener("click", hide);
}

function clearWitness() {
  selectedWitness = "";
  $("witnessName").value = "";
  $("witnessName").setCustomValidity("");
  $("clearWitness").hidden = true;
  if (!$("witnessName").disabled)
    $("witnessHint").textContent = "Choose a name to apply this filter.";
}

function setupSearch() {
  $("searchForm").addEventListener("submit", (event) => {
    event.preventDefault();
    performSearch();
  });
  document.querySelectorAll("[data-query]").forEach((button) =>
    button.addEventListener("click", () => {
      $("query").value = button.dataset.query;
      $("searchForm").requestSubmit();
    }),
  );
  $("query").addEventListener("input", cancelPending);
  for (const id of ["sourceType", "topK", "threshold", "minConfidence"])
    $(id).addEventListener("input", cancelPending);
  document.querySelectorAll("[name=mode]").forEach((radio) =>
    radio.addEventListener("change", () => {
      const compare = $("contradictionsToggle").checked;
      $("confidenceControl").hidden = !compare;
      $("modeHint").textContent = compare
        ? "Find possible disagreements. Model-generated comparisons should be checked against the testimony."
        : "Find passages by topic, question, or detail.";
      cancelPending();
    }),
  );
  $("minConfidence").addEventListener("input", () => {
    $("minConfidenceValue").value = Number($("minConfidence").value).toFixed(2);
  });
  $("resetFilters").addEventListener("click", () => {
    clearWitness();
    $("sourceType").value = "";
    $("topK").value = "5";
    $("threshold").value = "0.4";
    $("minConfidence").value = "0.6";
    $("minConfidenceValue").value = "0.60";
    cancelPending();
    $("searchStatus").textContent =
      "Filters reset. Search again to update the results.";
  });
}

function cancelPending() {
  if (!activeRequest) return;
  requestVersion += 1;
  activeRequest.controller.abort();
  activeRequest = null;
  $("resultsSection").setAttribute("aria-busy", "false");
  $("searchBtn").innerHTML = 'Search <span aria-hidden="true">→</span>';
  $("searchStatus").textContent =
    "Search cancelled. Submit your updated query when ready.";
  $("resultsTitle").textContent = "Search paused";
  $("resultsContent").innerHTML =
    '<div class="state-message"><h3>The search has changed.</h3><p>Run the updated search to read the matching testimony.</p></div>';
}

async function performSearch() {
  const query = $("query").value.trim();
  if (!query) {
    $("query").focus();
    return;
  }
  const compare = $("contradictionsToggle").checked;
  const body = {
    query,
    top_k: Number($("topK").value),
    similarity_threshold: Number($("threshold").value),
    witness_name: selectedWitness || null,
    source_type: $("sourceType").value || null,
  };
  if (compare) body.min_confidence = Number($("minConfidence").value);
  const signature = JSON.stringify(body) + compare;
  if (activeRequest?.signature === signature) return;
  activeRequest?.controller.abort();
  const version = ++requestVersion;
  const controller = new AbortController();
  activeRequest = { controller, signature };
  const context = [
    selectedWitness,
    body.source_type ? formatSource(body.source_type) : "Both inquiries",
  ]
    .filter(Boolean)
    .join(" · ");
  $("resultsTitle").textContent = compare
    ? "Comparing accounts"
    : "Searching the record";
  $("resultsCount").textContent = "";
  $("resultsContext").textContent = `“${query}” · ${context}`;
  $("resultsSection").setAttribute("aria-busy", "true");
  $("searchBtn").textContent = "Searching…";
  $("searchStatus").textContent = compare
    ? "Comparing retrieved testimony. This may take a little longer."
    : "Retrieving matching passages…";
  $("resultsContent").innerHTML =
    '<div class="state-message"><div class="loading-rule" aria-hidden="true"></div><h3>Consulting the record…</h3><p>You can change the query or choose another starting point.</p><button type="button" class="text-button" id="cancelSearch">Cancel search</button></div>';
  $("cancelSearch").addEventListener("click", cancelPending);
  try {
    const response = await fetch(
      compare ? "/search/contradictions" : "/search",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      },
    );
    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error(
        "The archive returned an unreadable response. Please try again.",
      );
    }
    if (version !== requestVersion) return;
    if (!response.ok)
      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : "The search could not be completed. Please try again.",
      );
    const records = compare ? data.contradictions : data.results;
    if (!Array.isArray(records))
      throw new Error(
        "The archive returned an incomplete response. Please try again.",
      );
    $("resultsTitle").textContent = compare
      ? "Possible disagreements"
      : "Witness testimony";
    const noun = compare ? "comparison" : "passage";
    const resultLabel = `${records.length} ${noun}${records.length === 1 ? "" : "s"}`;
    $("resultsCount").textContent = resultLabel;
    $("searchStatus").textContent = `Search complete. ${resultLabel} returned.`;
    if (!records.length) {
      $("resultsContent").innerHTML = compare
        ? '<div class="state-message"><h3>No comparisons returned.</h3><p>This does not establish that the accounts agree. The search may lack comparable passages, or checks may be unavailable. Try another question or adjust the confidence setting.</p></div>'
        : '<div class="state-message"><h3>No matching passages.</h3><p>Try a broader question, clear the witness filter, or lower the minimum similarity in Search settings.</p></div>';
      return;
    }
    $("resultsContent").innerHTML = records
      .map(compare ? renderComparison : renderEvidence)
      .join("");
  } catch (error) {
    if (version !== requestVersion || error.name === "AbortError") return;
    $("resultsTitle").textContent = "Search unavailable";
    $("searchStatus").textContent = "Search failed. No results are shown.";
    $("resultsContent").innerHTML =
      `<div class="state-message error" role="alert"><h3>We couldn’t complete this search.</h3><p>${escapeHtml(error.message || "Please try again.")}</p><button type="button" class="text-button" id="retrySearch">Try again</button></div>`;
    $("retrySearch").addEventListener("click", () =>
      $("searchForm").requestSubmit(),
    );
  } finally {
    // Aborted/late requests must not clear a newer request's loading state.
    if (version === requestVersion) {
      activeRequest = null;
      $("resultsSection").setAttribute("aria-busy", "false");
      $("searchBtn").innerHTML = 'Search <span aria-hidden="true">→</span>';
    }
  }
}

function citation(source, page) {
  return `<p class="citation">${escapeHtml(formatSource(source))}<br><span class="page">${page ? `Printed page ${escapeHtml(page)}` : "Page not supplied"}</span></p>`;
}
function excerpt(text, highlights = false) {
  const format = highlights ? formatHighlights : escapeHtml;
  if (text.length <= 320)
    return `<blockquote class="excerpt">${format(text)}</blockquote>`;
  const preview = text.slice(0, 300).replace(/\s+\S*$/, "") + "…";
  return `<blockquote class="excerpt preview">${format(preview)}</blockquote><details class="excerpt-disclosure"><summary>Read full excerpt</summary><blockquote class="excerpt">${format(text)}</blockquote></details>`;
}
function renderEvidence(result, index) {
  return `<article class="evidence"><header><span class="record-number">PASSAGE ${String(index + 1).padStart(2, "0")}</span><h3 class="witness-name">${escapeHtml(result.witness_name)}</h3>${result.role ? `<p class="witness-role">${escapeHtml(result.role)}</p>` : ""}${citation(result.source_type, result.page_number)}</header><div class="evidence-text">${excerpt(String(result.content || ""), true)}<details class="search-details"><summary>Search details</summary><p>Similarity: ${escapeHtml(result.similarity_score)} · Relevance: ${escapeHtml(result.relevance_score)}</p>${result.explanation ? `<p>${escapeHtml(result.explanation)}</p>` : ""}<p>Search scores indicate relevance, not historical reliability.</p></details></div></article>`;
}
function renderComparison(comparison, index) {
  const confidence = Number(comparison.confidence);
  return `<article class="comparison"><div class="comparison-header"><span class="comparison-label">Possible conflict ${String(index + 1).padStart(2, "0")}</span><span>Model confidence: ${Number.isFinite(confidence) ? Math.round(confidence * 100) + "%" : "not supplied"}</span></div>${comparison.same_person ? '<span class="same-person">Same witness, across both inquiries</span>' : ""}<div class="comparison-pair">${comparisonSide(comparison, "a")}${comparisonSide(comparison, "b")}</div><div class="comparison-note"><strong>Why the model flagged this</strong>${escapeHtml(comparison.explanation)}</div></article>`;
}
function comparisonSide(comparison, side) {
  return `<section class="comparison-side"><h3 class="witness-name">${escapeHtml(comparison[`witness_${side}`])}</h3><p class="witness-role">${escapeHtml(comparison[`role_${side}`])}</p>${citation(comparison[`source_${side}`], comparison[`page_${side}`])}<p class="claim-label">Model paraphrase</p><p class="claim">${escapeHtml(comparison[`claim_${side}`])}</p><div class="evidence-text">${excerpt(String(comparison[`chunk_${side}`] || ""))}</div></section>`;
}
function formatSource(type) {
  return (
    {
      us_inquiry: "US Senate Inquiry",
      british_inquiry: "British Inquiry",
      other: "Other source",
    }[type] ||
    type ||
    "Source not supplied"
  );
}
function escapeHtml(value) {
  const element = document.createElement("span");
  element.textContent = value == null ? "" : String(value);
  return element.innerHTML;
}
function formatHighlights(value) {
  return escapeHtml(value).replace(/\*\*(.+?)\*\*/g, "<mark>$1</mark>");
}
