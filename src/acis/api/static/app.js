/* ACIS UI — vanilla, offline, no dependencies (D15).
 *
 * Everything shown comes from the API. The architecture (models, channels, ranker, device) is what `/v1/system`
 * reports the running engine has loaded; ranks and channel evidence are the engine's own signals for this query;
 * timings are its stage timings; every code block is the text re-read from the content store by hash (INV-1); every
 * accuracy number is a ledger row's value, formatted by the server. Nothing here ranks, re-scores, estimates or
 * fills in a value the backend did not return.
 */
"use strict";
const $ = (id) => document.getElementById(id);

/* -- helpers ----------------------------------------------------------------------------------------------- */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const KEYWORDS = new Set(("False None True and as assert async await break class continue def del elif else except " +
  "finally for from global if import in is lambda nonlocal not or pass raise return try while with yield").split(" "));
function highlight(source) {
  const out = [];
  const re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(\b\d+\.?\d*\b)|([A-Za-z_][A-Za-z0-9_]*)/g;
  let last = 0, m, prevDef = false;
  while ((m = re.exec(source)) !== null) {
    out.push(escapeHtml(source.slice(last, m.index)));
    const [text, comment, str, num, word] = m;
    if (comment) out.push(`<span class="tok-com">${escapeHtml(text)}</span>`);
    else if (str) out.push(`<span class="tok-str">${escapeHtml(text)}</span>`);
    else if (num) out.push(`<span class="tok-num">${escapeHtml(text)}</span>`);
    else if (word && KEYWORDS.has(word)) out.push(`<span class="tok-kw">${escapeHtml(text)}</span>`);
    else if (word && prevDef) out.push(`<span class="tok-def">${escapeHtml(text)}</span>`);
    else out.push(escapeHtml(text));
    prevDef = word === "def" || word === "class";
    last = m.index + text.length;
  }
  out.push(escapeHtml(source.slice(last)));
  return out.join("");
}
async function api(path, body) {
  const response = await fetch(path, body
    ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }
    : {});
  const text = await response.text();
  let payload; try { payload = JSON.parse(text); } catch { payload = { message: text || response.statusText }; }
  if (!response.ok) {
    const detail = payload.message || (Array.isArray(payload.detail) ? payload.detail.map((d) => d.msg).join("; ") : payload.detail);
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return payload;
}
async function apiText(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.text();
}
function segValue(id) { return document.querySelector(`#${id} button.on`).dataset.value; }
function bindSeg(id, onChange) {
  $(id).addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    $(id).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    if (onChange) onChange(b.dataset.value);
  });
}
function codeBlock(source, lines = 14) {
  const count = source.split("\n").length;
  const id = `c${Math.random().toString(36).slice(2, 9)}`;
  const more = count > lines ? `<button class="more" data-target="${id}" type="button">show all ${count} lines</button>` : "";
  return `<pre class="code" id="${id}">${highlight(source)}</pre>${more}`;
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("button.more"); if (!b) return;
  const pre = $(b.dataset.target); pre.classList.toggle("expanded");
  b.textContent = pre.classList.contains("expanded") ? "collapse" : `show all ${pre.textContent.split("\n").length} lines`;
});
function showError(target, err) { $(target).innerHTML = `<div class="error">${escapeHtml(err.message || err)}</div>`; }
function notice(target, kind, html) {
  const el = $(target);
  if (!html) { el.hidden = true; el.innerHTML = ""; return; }
  el.hidden = false; el.className = `notice ${kind}`; el.innerHTML = html;
}
const fmtInt = (n) => (Number.isFinite(Number(n)) && n !== null && n !== undefined ? Number(n).toLocaleString() : "unknown");

/* -- custom dropdown ----------------------------------------------------------------------------------------
 * A button and a listbox, in place of the native <select>: styled like everything else, keyboard-operable
 * (↑ ↓ Home End Enter Esc, type-ahead by first letter), and its value lives in `data-value`. */
const dropdowns = {};
function dropdown(id, { options = [], value, onChange } = {}) {
  const root = $(id);
  const state = dropdowns[id] || { root, options: [], value: null, onChange: null, active: 0 };
  dropdowns[id] = state;
  if (!root.querySelector(".dd-button")) {
    root.innerHTML = `<button type="button" class="dd-button" aria-haspopup="listbox" aria-expanded="false"><span class="dd-label"></span><span class="dd-chev"></span></button><ul class="dd-list" role="listbox" tabindex="-1" hidden></ul>`;
    const button = root.querySelector(".dd-button"), list = root.querySelector(".dd-list");
    button.addEventListener("click", () => (root.classList.contains("open") ? close(id) : open(id)));
    button.addEventListener("keydown", (e) => {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(e.key)) { e.preventDefault(); open(id); }
    });
    list.addEventListener("keydown", (e) => keyNav(id, e));
    list.addEventListener("click", (e) => {
      const li = e.target.closest("li"); if (!li) return;
      choose(id, li.dataset.value); close(id); button.focus();
    });
  }
  state.options = options.map((o) => (typeof o === "object" ? o : { value: String(o), label: String(o) }));
  if (onChange) state.onChange = onChange;
  const wanted = value ?? root.dataset.value ?? state.value;
  const found = state.options.find((o) => o.value === String(wanted)) || state.options[0];
  setValue(id, found ? found.value : "", false);
  root.querySelector(".dd-button").disabled = state.options.length === 0;
  return state;
}
function setValue(id, value, fire = true) {
  const s = dropdowns[id]; const option = s.options.find((o) => o.value === value);
  s.value = option ? option.value : ""; s.root.dataset.value = s.value;
  s.root.querySelector(".dd-label").textContent = option ? option.label : "none";
  if (fire && s.onChange) s.onChange(s.value);
}
function choose(id, value) { if (dropdowns[id].value !== value) setValue(id, value); }
function ddValue(id) { return dropdowns[id] ? dropdowns[id].value : $(id).dataset.value; }
function renderList(id) {
  const s = dropdowns[id]; const list = s.root.querySelector(".dd-list");
  list.innerHTML = s.options.map((o, i) => `<li role="option" data-value="${escapeHtml(o.value)}" aria-selected="${o.value === s.value}" class="dd-option${i === s.active ? " active" : ""}"><span>${escapeHtml(o.label)}</span>${o.hint ? `<span class="hint">${escapeHtml(o.hint)}</span>` : ""}</li>`).join("");
}
function open(id) {
  Object.keys(dropdowns).forEach((other) => other !== id && close(other));
  const s = dropdowns[id]; if (!s.options.length) return;
  s.active = Math.max(0, s.options.findIndex((o) => o.value === s.value));
  renderList(id); s.root.classList.add("open");
  const list = s.root.querySelector(".dd-list"); list.hidden = false; list.focus();
  s.root.querySelector(".dd-button").setAttribute("aria-expanded", "true");
}
function close(id) {
  const s = dropdowns[id]; if (!s) return;
  s.root.classList.remove("open"); s.root.querySelector(".dd-list").hidden = true;
  s.root.querySelector(".dd-button").setAttribute("aria-expanded", "false");
}
function keyNav(id, e) {
  const s = dropdowns[id]; const n = s.options.length;
  if (e.key === "Escape") { close(id); s.root.querySelector(".dd-button").focus(); return; }
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(id, s.options[s.active].value); close(id); s.root.querySelector(".dd-button").focus(); return; }
  if (e.key === "ArrowDown") s.active = (s.active + 1) % n;
  else if (e.key === "ArrowUp") s.active = (s.active - 1 + n) % n;
  else if (e.key === "Home") s.active = 0;
  else if (e.key === "End") s.active = n - 1;
  else if (e.key.length === 1) {
    const i = s.options.findIndex((o, j) => j > s.active && o.label.toLowerCase().startsWith(e.key.toLowerCase()));
    s.active = i >= 0 ? i : Math.max(0, s.options.findIndex((o) => o.label.toLowerCase().startsWith(e.key.toLowerCase())));
  } else return;
  e.preventDefault(); renderList(id);
}
document.addEventListener("click", (e) => { if (!e.target.closest(".dd")) Object.keys(dropdowns).forEach(close); });

/* -- theme ------------------------------------------------------------------------------------------------- */
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("acis-theme", theme); } catch (e) { /* storage may be unavailable */ }
}
$("theme").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

/* -- latency (this browser session) ------------------------------------------------------------------------ */
const latencies = [];
function recordLatency(ms, serverMs) {
  latencies.push(ms);
  const sorted = [...latencies].sort((a, b) => a - b);
  const pct = (p) => sorted[Math.min(sorted.length - 1, Math.round((p / 100) * (sorted.length - 1)))];
  $("latency").innerHTML = `last query <b>${ms.toFixed(0)} ms</b>` +
    (serverMs != null ? ` (engine ${serverMs.toFixed(0)} ms)` : "") +
    ` · this session p50 <b>${pct(50).toFixed(0)} ms</b> · p95 <b>${pct(95).toFixed(0)} ms</b> · n=${sorted.length}`;
}

/* -- corpus pill: always the corpus the current tab searches ------------------------------------------------ */
const corpus = { search: null, versions: null, evolution: null, evaluation: "recorded evaluation · dev split", system: "live view of this server" };
function currentTab() { return document.querySelector(".tab.active").dataset.view; }
function showCorpus() {
  const text = corpus[currentTab()];
  $("pill-corpus").textContent = text || "no corpus loaded";
  $("pill-corpus").classList.toggle("ok", !!text);
}
function setCorpus(tab, text) { corpus[tab] = text; showCorpus(); }

/* -- what this server runs (GET /v1/system) ---------------------------------------------------------------- */
let sys = null;
const shortName = (name) => String(name || "").split("/").pop();
const millions = (n) => (Number.isFinite(Number(n)) && n ? `${Math.round(Number(n) / 1e6)}M` : null);
function encoderOf(role) { return sys ? sys.encoders.find((e) => e.role === role) : null; }
function stateBadge(e) {
  if (!e) return "";
  if (e.state === "serving") return '<span class="pl-state serving">serving</span>';
  if (e.state === "optional") return `<span class="pl-state optional" title="${escapeHtml(e.reason || "")}">optional</span>`;
  return `<span class="pl-state off" title="${escapeHtml(e.reason || "")}">${escapeHtml(e.state || "off")}</span>`;
}

/* The pipeline: every node is a component the server reports; after a search each node shows what it did. */
function pipelineNodes() {
  const primary = encoderOf("primary dense channel");
  const second = encoderOf("second dense channel");
  const lex = (sys && sys.lexical) || {};
  const cand = (sys && sys.candidates) || {};
  const rk = (sys && sys.ranker) || {};
  const nFeat = rk.features, nGroups = rk.groups ? Object.keys(rk.groups).length : null;
  return {
    understand: { kicker: "1 · Query", title: "Query understanding", desc: "normalise · extract identifiers · detect query type" },
    channels: [
      { id: "dense", color: "c-dense", title: primary ? shortName(primary.name) : "dense encoder", state: primary,
        desc: primary ? [`semantic`, primary.dim && `${primary.dim}-d`, millions(primary.params) && `${millions(primary.params)} params`, cand.dense_k && `top ${cand.dense_k}`].filter(Boolean).join(" · ") : "" },
      { id: "dense2", color: "c-dense2", title: second ? shortName(second.name || second.key) : "second encoder", state: second,
        desc: second ? ["semantic, instruction-aware", second.state === "serving" && cand.aux_k ? `top ${cand.aux_k}` : null].filter(Boolean).join(" · ") : "" },
      { id: "bm25", color: "c-bm25", title: "BM25", desc: [`keywords, code-aware tokenizer`, cand.lexical_k && `top ${cand.lexical_k}`].filter(Boolean).join(" · ") },
      { id: "symbols", color: "c-symbols", title: "Exact identifiers", desc: [lex.symbols && lex.symbols.distinct_symbols ? `${fmtInt(lex.symbols.distinct_symbols)} indexed` : "identifier index", cand.symbol_k && `top ${cand.symbol_k}`].filter(Boolean).join(" · ") },
    ],
    union: { kicker: "3 · Candidates", title: "Candidate union", desc: `${cand.union_cap ? `up to ${cand.union_cap}` : "union"} · every candidate scored by every channel` },
    features: { kicker: "4 · Evidence", title: nFeat ? `${nFeat} ranking signals` : "Ranking signals", desc: nGroups ? `${nGroups} families: similarity, keywords, identifiers, I/O literals, agreement, length` : "per-candidate features" },
    rank: { kicker: "5 · Ranking", title: rk.loaded ? "Learned ranking" : "Ranking", desc: rk.loaded ? `LightGBM LambdaRank · ${rk.rounds} trees · trained on ${fmtInt(rk.trained_on_queries)} queries` : "no ranker loaded" },
    evidence: { kicker: "6 · Results", title: "Verified code", desc: "re-read from the content store by hash · calibrated confidence" },
  };
}
function nodeHtml(key, n, extra = "") {
  return `<div class="pl-node ${extra}" data-node="${key}"><div class="pl-kicker">${escapeHtml(n.kicker || "")}</div><div class="pl-title">${escapeHtml(n.title)}${n.badge || ""}</div><div class="pl-desc">${escapeHtml(n.desc || "")}</div><div class="pl-live"></div></div>`;
}
function renderPipeline(target) {
  const n = pipelineNodes();
  const channels = n.channels.map((c) => `<div class="pl-node" data-node="ch-${c.id}"><div class="pl-title"><i class="dotc ${c.color}"></i>${escapeHtml(c.title)}${c.state && c.state.state !== "serving" ? stateBadge(c.state) : ""}</div><div class="pl-desc">${escapeHtml(c.desc)}</div><div class="pl-live"></div></div>`).join("");
  $(target).innerHTML = [
    nodeHtml("understand", n.understand),
    '<div class="pl-arrow"></div>',
    `<div><div class="pl-group-kicker">2 · Retrieval channels</div><div class="pl-channels">${channels}</div></div>`,
    '<div class="pl-arrow"></div>', nodeHtml("union", n.union),
    '<div class="pl-arrow"></div>', nodeHtml("features", n.features),
    '<div class="pl-arrow"></div>', nodeHtml("rank", n.rank),
    '<div class="pl-arrow"></div>', nodeHtml("evidence", n.evidence),
  ].join("");
}
const ORDERING = {
  ltr: "Learned ranking (LambdaRank)",
  fusion: "Hybrid score fusion",
  identifier_first: "Exact identifier match first",
  ltr_abstained: "Semantic order kept",
  dense: "Semantic similarity only",
  lexical: "BM25 only",
};
function lightPipeline(target, r) {
  const root = $(target); if (!root || !root.children.length) return;
  const e = r.explanation || {}, c = e.candidates || {}, t = r.timings_ms || {};
  const ms = (...keys) => keys.reduce((a, k) => a + (t[`stage.${k}`] || 0), 0);
  const set = (key, live, lit = true) => {
    const node = root.querySelector(`[data-node="${key}"]`); if (!node) return;
    node.classList.toggle("lit", lit && live !== null); node.classList.toggle("idle", live === null);
    node.querySelector(".pl-live").textContent = live || "";
  };
  const symbols = (e.query_symbols || []).length;
  set("understand", `${ms("route", "route_encode", "encode", "encode_wait").toFixed(0)} ms${symbols ? ` · ${symbols} identifier${symbols > 1 ? "s" : ""}` : ""}`);
  const used = new Set(e.channels || []);
  const single = e.channel === "dense" || e.channel === "lexical";
  const chan = (id, from, stages) => {
    if (single) return set(`ch-${id}`, (e.channel === "dense" && id === "dense") || (e.channel === "lexical" && id === "bm25") ? `${ms(...stages).toFixed(0)} ms` : null);
    set(`ch-${id}`, used.has(id) ? `${fmtInt(c[from])} found · ${ms(...stages).toFixed(1)} ms` : null);
  };
  chan("dense", "from_dense", ["dense"]); chan("dense2", "from_dense2", ["encode2", "dense2"]);
  chan("bm25", "from_bm25", ["bm25"]); chan("symbols", "from_symbols", ["symbols"]);
  set("union", single ? null : c.candidates != null ? `${fmtInt(c.candidates)} candidates` : null);
  const ordering = e.ordering || (single ? e.channel : null);
  set("features", ordering === "ltr" ? `${ms("features").toFixed(1)} ms` : null);
  const rankNode = root.querySelector('[data-node="rank"] .pl-title');
  if (rankNode) rankNode.textContent = ordering && ordering !== "ltr" && !single ? ORDERING[ordering] || "Ranking" : (sys && sys.ranker && sys.ranker.loaded ? "Learned ranking" : "Ranking");
  // The description under the title says what ran for THIS query: the ranker only when it ordered the list.
  const rankDesc = root.querySelector('[data-node="rank"] .pl-desc');
  if (rankDesc) {
    const rk = (sys && sys.ranker) || {};
    const w = e.generic_aux_weight, a = e.generic_alpha;
    const fusionDesc = `${e.second_encoder ? `gte + ${shortName(e.second_encoder)} cosines` : "dense cosine"}`
      + `${w ? ` (second-encoder share ${w})` : ""} + BM25${a != null ? ` · α ${a}` : ""} · the learned ranker is used for problem statements`;
    rankDesc.textContent = single ? ""
      : ordering === "ltr" ? `LightGBM LambdaRank · ${rk.rounds || ""} trees · trained on ${fmtInt(rk.trained_on_queries)} queries`
      : ordering === "fusion" ? fusionDesc
      : ordering === "identifier_first" ? `units containing the identifier verbatim first, then ${fusionDesc}`
      : ordering === "ltr_abstained" ? "the ranker abstained (too little evidence fired): semantic order kept"
      : rankDesc.textContent;
  }
  set("rank", single ? null : ordering === "ltr" ? `ordered ${fmtInt(c.candidates)} · ${ms("ranker").toFixed(1)} ms`
    : ordering === "fusion" || ordering === "identifier_first" ? `short query · ${ms("fusion").toFixed(1)} ms` : ORDERING[ordering] || "");
  const facts = e.confidence || {};
  const verdict = facts.identifier_matches ? "exact identifier match" : r.no_strong_match ? "weak match" : `${r.confidence} confidence`;
  set("evidence", `${r.results.length} results · ${verdict}`);
}

/* -- recorded accuracy (ledger) ---------------------------------------------------------------------------- */
async function loadKpis() {
  // Server-rendered from the ledger row (like the evaluation panel) and inserted verbatim: no number is built here.
  try { $("kpis").innerHTML = await apiText("/v1/benchmarks/p0/headline"); }
  catch (err) { $("kpis").innerHTML = `<div class="kpi-foot">Recorded accuracy unavailable: ${escapeHtml(err.message)}</div>`; }
}

/* -- P0 / P1 hit rendering --------------------------------------------------------------------------------- */
const STAGES = { route: ["query analysis", "--stage-route"], route_encode: ["encode query", "--stage-encode"], encode: ["encode query", "--stage-encode"],
  encode_wait: ["queued for the model", "--stage-wait"], dense: ["dense search", "--stage-dense"], bm25: ["BM25", "--stage-bm25"],
  features: ["features", "--stage-features"], ranker: ["learned ranker", "--stage-ranker"], fusion: ["fusion", "--stage-fusion"],
  encode2: ["encode (2nd encoder)", "--stage-encode"], dense2: ["dense search (2nd encoder)", "--stage-dense"], symbols: ["identifiers", "--stage-bm25"] };
function renderTrace(t) {
  const merged = {};
  for (const [k, v] of Object.entries(t)) {
    if (!k.startsWith("stage.")) continue;
    const [label, color] = STAGES[k.slice(6)] || [k.slice(6), "--muted"];
    merged[label] = merged[label] || { ms: 0, color }; merged[label].ms += v;
  }
  const steps = Object.entries(merged).map(([label, s]) =>
    `<span class="step"><i style="background:var(${s.color})"></i>${escapeHtml(label)} ${s.ms.toFixed(1)} ms</span>`);
  return `<span class="trace">${steps.join('<span class="arrow">→</span>')}</span>`;
}
/* Names the unit defines, read from its stored text for display (the ranking never sees this). */
function definedNames(source) {
  const out = [];
  for (const m of source.matchAll(/^[ \t]*(?:async[ \t]+)?(def|class)[ \t]+([A-Za-z_]\w*)/gm)) {
    const label = m[1] === "class" ? `class ${m[2]}` : `${m[2]}()`;
    if (!out.includes(label)) out.push(label);
  }
  return out;
}
const CHANNEL_META = {
  dense: { color: "c-dense", label: () => `Semantic · ${shortName((encoderOf("primary dense channel") || {}).name) || "dense"}`, rank: "dense_rank", found: "found_dense" },
  dense2: { color: "c-dense2", label: () => `Semantic · ${shortName((encoderOf("second dense channel") || {}).name) || "encoder 2"}`, rank: "dense2_rank", found: "found_dense2" },
  bm25: { color: "c-bm25", label: () => "Keywords · BM25", rank: "bm25_rank", found: "found_bm25" },
  symbols: { color: "c-symbols", label: () => "Identifiers", rank: null, found: "found_symbols" },
};
function renderHits(target, hits, explanation) {
  const e = explanation || {};
  const active = (e.channels && e.channels.length ? e.channels : ["dense"]).filter((c) => CHANNEL_META[c]);
  const single = e.channel === "dense" || e.channel === "lexical";
  $(target).innerHTML = hits.map((h, i) => {
    const s = h.signals || {};
    const names = definedNames(h.source);
    const shown = names.slice(0, 4).map((n) => `<span class="fn">${escapeHtml(n)}</span>`).join("") + (names.length > 4 ? `<span class="unit-key">+${names.length - 4} more</span>` : "");
    const title = `<div class="fns">${shown || '<span class="fn">program</span>'}<span class="unit-key">${escapeHtml(h.unit.key || h.unit.unit_id)}</span></div>`;
    const sigs = [];
    for (const ch of ["dense", "dense2", "bm25"]) {
      const m = CHANNEL_META[ch]; const rank = s[m.rank];
      if (rank == null) continue;
      const on = s[m.found] === 1;
      sigs.push(`<span class="sig ${on ? "on" : ""}" title="${on ? "retrieved by this channel for this query" : "outside this channel's candidate list; scored by it anyway"}"><i class="${m.color}"></i>${escapeHtml(m.label())} <b>#${rank}</b></span>`);
    }
    const matched = (e.hit_symbols || [])[i] || [];
    if (matched.length) sigs.push(`<span class="sig on" title="identifiers from the query that this unit contains"><i class="c-symbols"></i>Identifiers · ${matched.slice(0, 4).map((x) => `<code>${escapeHtml(x)}</code>`).join(" ")}</span>`);
    const found = active.filter((c) => s[CHANNEL_META[c].found] === 1);
    const dots = single ? "" : active.map((c) => `<i class="${s[CHANNEL_META[c].found] === 1 ? "on" : ""}" title="${escapeHtml(CHANNEL_META[c].label())}"></i>`).join("");
    const agree = single ? `<div class="label">single channel</div>`
      : `<div class="dots">${dots}</div><div class="label">found by ${found.length} of ${active.length} channel${active.length > 1 ? "s" : ""}</div>`;
    const tech = [
      s.similarity != null ? `cosine <code>${s.similarity.toFixed(4)}</code>` : "",
      s.similarity2 != null ? `cosine (2nd encoder) <code>${s.similarity2.toFixed(4)}</code>` : "",
      s.bm25 != null ? `BM25 <code>${s.bm25.toFixed(2)}</code>` : "",
      `content hash <code>${escapeHtml(h.unit.body_hash.slice(0, 16))}</code>`,
      `${fmtInt(h.unit.n_bytes)} bytes`,
      h.unit.version && h.unit.version !== "v0" && h.unit.version !== "-" ? `version <code>${escapeHtml(h.unit.version)}</code>` : "",
    ].filter(Boolean).map((x) => `<span>${x}</span>`).join("");
    return `<article class="hit ${i === 0 ? "top" : ""}">
      <div class="hit-head">
        <div class="rank" title="ACIS rank">${h.rank}</div>
        <div>${title}<div class="signals">${sigs.join("")}</div></div>
        <div class="agree">${agree}${i === 0 ? `<div class="conf">${confidenceBadge(RESPONSE_OF[target])}</div>` : ""}</div>
      </div>
      ${codeBlock(h.source)}
      <div class="hit-tech">${tech}</div>
    </article>`;
  }).join("");
}
const RESPONSE_OF = {};
function confidenceBadge(r) {
  if (!r) return "";
  const e = r.explanation || {};
  const facts = e.confidence || {};
  const title = escapeHtml(e.confidence_basis || "");
  if (facts.identifier_matches) return `<span class="badge good" title="${title}">exact identifier match</span>`;
  if (facts.calibrated === false) return `<span class="badge" title="${title}">confidence not calibrated</span>`;
  if (r.no_strong_match) return `<span class="badge warn" title="${title}">weak match</span>`;
  const kind = r.confidence === "high" ? "good" : r.confidence === "medium" ? "accent" : "warn";
  return `<span class="badge ${kind}" title="${title}">${escapeHtml(r.confidence)} confidence</span>`;
}
const CHANNEL_NAMES = () => ({
  dense: shortName((encoderOf("primary dense channel") || {}).name) || "dense",
  dense2: shortName((encoderOf("second dense channel") || {}).name) || "encoder 2",
  bm25: "BM25", symbols: "identifiers",
});
function renderSummary(target, r, wallMs) {
  const e = r.explanation || {};
  const c = e.candidates || {};
  const names = CHANNEL_NAMES();
  const ordering = e.ordering || e.channel;
  $(target).hidden = false;
  const from = (e.channels || []).map((ch) => names[ch] || ch).join(", ");
  $(target).innerHTML = [
    `<span class="lead">${r.results.length} results</span>`,
    `<span>ranked by <b>${escapeHtml(ORDERING[ordering] || e.ordered_by || "-")}</b></span>`,
    c.candidates != null && !["dense", "lexical"].includes(e.channel) ? `<span>over <b>${fmtInt(c.candidates)}</b> candidates from <b>${escapeHtml(from)}</b></span>` : "",
    `<span><b>${r.timings_ms.total.toFixed(0)} ms</b> in the engine</span>`,
    ...(r.degradations || []).map((d) => `<span class="badge bad" title="a fallback this request took">${escapeHtml(d)}</span>`),
  ].join("");
  const tech = $(target.replace("summary", "tech"));
  if (!tech) return;
  const rd = e.route_decision || {};
  const conf = e.confidence || {};
  const rows = [
    ["Stages", renderTrace(r.timings_ms)],
    ["Ordered by", escapeHtml(e.ordered_by || "-")],
    ["Query type", `${escapeHtml(r.route)}${rd.reason ? ` — ${escapeHtml(rd.reason)}` : ""}${e.category ? ` · category <code>${escapeHtml(e.category)}</code>` : ""}`],
    ["Identifiers in query", (e.query_symbols || []).length ? e.query_symbols.map((x) => `<code>${escapeHtml(x)}</code>`).join(" ") : "none"],
    ["Candidates", Object.entries(c).map(([k, v]) => `${escapeHtml(k.replace("from_", "from "))} ${fmtInt(v)}`).join(" · ") || "-"],
    ["Confidence", `${escapeHtml(r.confidence)}${conf.p != null ? ` · calibrated P(top result relevant) ${Number(conf.p).toFixed(2)}` : ""}${e.confidence_basis ? ` — ${escapeHtml(e.confidence_basis)}` : ""}`],
    ["Short-query fusion weight", e.generic_alpha != null ? `α = ${escapeHtml(e.generic_alpha)} (dense share)` : "not set"],
    ["Encoder", escapeHtml(e.encoder || "-")],
    ["Snapshot", `<code>${escapeHtml(r.snapshot.id)}</code> · ${escapeHtml(r.snapshot.version)}`],
    ["Wall time", `${wallMs.toFixed(0)} ms (browser → server → browser)`],
  ];
  tech.hidden = false;
  $(`${tech.id}-body`).innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}
function renderContext(target, r, { expectRepo, versionsById }) {
  const snap = r.snapshot;
  const repoLabel = snap.repo_id === "-" ? "APPS P0 corpus" : `repository <b>${escapeHtml(snap.repo_id)}</b>`;
  const owned = expectRepo === snap.repo_id && (!versionsById || versionsById[snap.id] !== undefined);
  $(target).hidden = false;
  $(target).innerHTML = `<span>answered from ${repoLabel}</span>` +
    (snap.version && snap.version !== "v0" ? `<span>version <b>${escapeHtml(snap.version)}</b></span>` : "") +
    `<span><b>${fmtInt(snap.n_units)}</b> units searched</span>` +
    (owned ? `<span class="badge good" title="snapshot ${escapeHtml(snap.id)} belongs to the corpus selected on this tab">verified source</span>`
           : `<span class="badge bad">does not match the selection</span>`);
  return owned;
}

/* -- P0: search the APPS corpus ---------------------------------------------------------------------------- */
async function searchP0(e) {
  e?.preventDefault();
  const query = $("q-search").value.trim(); if (!query) return;
  $("go-search").disabled = true; $("empty-search").hidden = true;
  const t0 = performance.now();
  try {
    const r = await api("/v1/search", { query, repo_id: "-", top_k: Number(ddValue("topk-search")) || 10, mode: segValue("channel"), explain: true });
    const wall = performance.now() - t0;
    RESPONSE_OF["results-search"] = r;
    renderContext("context-search", r, { expectRepo: "-" });
    notice("notice-search", "warn", r.no_strong_match
      ? "<b>Low confidence.</b> The calibrated estimate that the top result is relevant is below 30 % for this query (see Technical details for the basis). The results are still the best-ranked matches; treat them as suggestions."
      : "");
    renderHits("results-search", r.results, r.explanation);
    renderSummary("summary-search", r, wall);
    lightPipeline("pipeline", r);
    setCorpus("search", `APPS corpus · ${fmtInt(r.snapshot.n_units)} units`);
    recordLatency(wall, r.timings_ms.total);
  } catch (err) { showError("results-search", err); $("summary-search").hidden = true; $("context-search").hidden = true; $("tech-search").hidden = true; }
  finally { $("go-search").disabled = false; }
}

/* -- P1: pinned versions ----------------------------------------------------------------------------------- */
let versionsState = { repo: null, byId: {}, list: [], loading: null };
async function loadVersions() {
  const repo = ddValue("repo-versions"); if (!repo) return;
  $("go-versions").disabled = true;                       // no search while the version list belongs to another repo
  $("selector-versions").value = "";                     // a selector typed for another repository is not carried over
  const loading = api(`/v1/repos/${encodeURIComponent(repo)}/versions`);
  versionsState.loading = loading;
  try {
    const { versions } = await loading;
    if (versionsState.loading !== loading) return;       // a newer selection superseded this one
    versionsState = { repo, byId: Object.fromEntries(versions.map((v) => [v.snapshot_id, v.label])), list: versions, loading: null };
    dropdown("version-versions", {
      options: [{ value: "latest", label: "latest (active)" }, ...versions.map((v) => ({ value: v.label, label: v.label, hint: `${fmtInt(v.n_units)} units${v.active ? " · active" : ""}` }))],
      value: "latest",
    });
    $("vtable-repo").textContent = repo;
    $("vtable").innerHTML = versions.map((v) => `<tr class="${v.active ? "active" : ""}"><td>${escapeHtml(v.label)}</td><td>${fmtInt(v.n_units)}</td><td class="snap">${escapeHtml(String(v.snapshot_id).slice(0, 12))}</td><td>${v.active ? '<span class="badge good">active</span>' : ""}</td></tr>`).join("");
    const active = versions.find((v) => v.active);
    setCorpus("versions", `${repo} · ${active ? active.label : "?"} active · ${fmtInt(active ? active.n_units : 0)} units`);
  } catch (err) { showError("results-versions", err); }
  finally { if (versionsState.repo === repo) $("go-versions").disabled = false; }
}
async function searchP1(e) {
  e?.preventDefault();
  const query = $("q-versions").value.trim(); if (!query) return;
  const repo = ddValue("repo-versions");
  if (!repo || versionsState.repo !== repo) return;       // the version list is still loading for this repository
  const version = $("selector-versions").value.trim() || ddValue("version-versions");
  $("empty-versions").hidden = true; $("go-versions").disabled = true;
  const t0 = performance.now();
  try {
    const r = await api("/v1/search", { query, repo_id: repo, version, top_k: 10, mode: "auto", explain: true });
    const wall = performance.now() - t0;
    const owned = renderContext("context-versions", r, { expectRepo: repo, versionsById: versionsState.byId });
    if (!owned) {
      notice("notice-versions", "bad", `<b>Refusing to show these results:</b> the snapshot that answered (<code>${escapeHtml(r.snapshot.id)}</code>, repository <code>${escapeHtml(r.snapshot.repo_id)}</code>) is not a version of <code>${escapeHtml(repo)}</code>.`);
      $("results-versions").innerHTML = ""; $("summary-versions").hidden = true; return;
    }
    RESPONSE_OF["results-versions"] = r;
    notice("notice-versions", "warn", r.no_strong_match ? "<b>No strong match</b> in this version. Results are shown ranked; treat them as suggestions." : "");
    renderHits("results-versions", r.results, r.explanation);
    renderSummary("summary-versions", r, wall);
    setCorpus("versions", `${repo} · ${r.snapshot.version} · ${fmtInt(r.snapshot.n_units)} units`);
    recordLatency(wall, r.timings_ms.total);
  } catch (err) { showError("results-versions", err); $("summary-versions").hidden = true; }
  finally { $("go-versions").disabled = false; }
}
async function commit() {
  const repo = ddValue("repo-versions"); if (!repo) return;
  const box = $("commit-result"); box.hidden = false; box.className = "commit-result"; box.textContent = "committing and rebuilding incrementally…";
  $("commit").disabled = true;
  try {
    const r = await api(`/v1/repos/${encodeURIComponent(repo)}/commit`, { edits: Number($("commit-edits").value), seed: Date.now() % 100000 });
    box.innerHTML = `<b>${escapeHtml(r.version)}</b> searchable in <b>${r.seconds_searchable.toFixed(2)} s</b> · ` +
      `<b>${r.units_new}</b> unit(s) embedded, ${fmtInt(r.units_reused)} reused of ${fmtInt(r.units_total)}<br>` +
      `<span class="muted">edited: ${r.changed.map(escapeHtml).join(", ") || "none"}</span>`;
    await loadVersions();
  } catch (err) { box.className = "commit-result error"; box.textContent = err.message; }
  finally { $("commit").disabled = false; }
}
async function rollback() {
  const repo = ddValue("repo-versions"); if (!repo) return;
  try { await api(`/v1/repos/${encodeURIComponent(repo)}/rollback`, {}); await loadVersions(); }
  catch (err) { const box = $("commit-result"); box.hidden = false; box.className = "commit-result error"; box.textContent = err.message; }
}

/* -- Bonus: across every version --------------------------------------------------------------------------- */
function renderGroups(groups, labels) {
  $("results-evolution").innerHTML = groups.map((g, i) => {
    const byVersion = Object.fromEntries(g.timeline.map((s) => [s.version, s]));
    const steps = labels.map((v) => {
      const s = byVersion[v];
      const cls = !s ? "absent" : `${s.changed ? "changed" : ""} ${v === g.best.version ? "best" : ""}`;
      const title = !s ? `${v} · not present` : `${v} · ${s.relation}${s.inferred ? " (inferred)" : ""}${s.same_content_as ? ` · same content as ${s.same_content_as} (revert)` : ""}`;
      return `<span class="tl-step"><span class="dot ${cls}" title="${escapeHtml(title)}"></span><span class="vlabel">${escapeHtml(v)}</span></span>`;
    }).join('<span class="rail"></span>');
    const order = Object.fromEntries(labels.map((v, j) => [v, j]));   // the repository's own version order
    const members = [...g.members].sort((a, b) => (order[a.version] ?? 1e9) - (order[b.version] ?? 1e9))
      .map((m) => `<code>${escapeHtml(m.version)}</code>`).join(" ");
    const names = g.best.source ? definedNames(g.best.source).slice(0, 3).map((n) => `<span class="fn">${escapeHtml(n)}</span>`).join("") : "";
    return `<article class="hit">
      <div class="hit-head">
        <div class="rank">${i + 1}</div>
        <div><div class="fns">${names}<span class="unit-key">${escapeHtml(g.best.key)}</span></div>
          <div class="chips"><span class="badge accent">best revision ${escapeHtml(g.best.version)}</span>
          <span class="badge">${g.n_revisions} revision(s) · ${escapeHtml(g.span[0])} → ${escapeHtml(g.span[1])}</span>
          ${g.inferred ? '<span class="badge warn" title="at least one link in this lineage was inferred rather than exact">contains an inferred link</span>' : ""}
          ${g.timeline.filter((s) => s.same_content_as).map((s) => `<span class="badge" title="the content at ${escapeHtml(s.version)} is byte-identical to ${escapeHtml(s.same_content_as)}">${escapeHtml(s.version)} reverts to ${escapeHtml(s.same_content_as)}</span>`).join("")}
          <span class="badge" title="lineage id from the lineage index">${escapeHtml(g.lineage_id)}</span></div></div>
        <div class="sim"><div class="sim-label">history</div></div>
      </div>
      <div class="timeline">${steps}</div>
      <div class="members">matched in ${members}</div>
      ${g.best.source ? codeBlock(g.best.source) : ""}
    </article>`;
  }).join("");
}
function renderFlat(hits) {
  const seen = new Map();
  $("results-evolution").innerHTML = hits.map((h) => {
    const first = seen.get(h.unit.key);
    if (first == null) seen.set(h.unit.key, h.rank);
    const dup = first != null ? `<span class="badge warn">same unit as #${first}</span>` : "";
    return `<article class="hit">
      <div class="hit-head"><div class="rank">${h.rank}</div>
        <div><div class="fns">${definedNames(h.source).slice(0, 3).map((n) => `<span class="fn">${escapeHtml(n)}</span>`).join("")}<span class="unit-key">${escapeHtml(h.unit.key)}</span></div>
          <div class="chips"><span class="badge accent">${escapeHtml(h.unit.version)}</span>${dup}
          <span class="badge">${escapeHtml(h.unit.body_hash.slice(0, 10))}</span></div></div>
        <div class="sim"></div></div>
      ${codeBlock(h.source, 8)}
    </article>`;
  }).join("");
}
async function searchBonus(e) {
  e?.preventDefault();
  const query = $("q-evolution").value.trim(); if (!query) return;
  const repo = ddValue("repo-evolution"); if (!repo) return;
  const flat = segValue("evo-mode") === "flat";
  $("empty-evolution").hidden = true; $("go-evolution").disabled = true;
  const t0 = performance.now();
  try {
    const [r, vs] = await Promise.all([
      api("/v1/evolve", { query, repo_id: repo, top_k: 10, flat }),
      api(`/v1/repos/${encodeURIComponent(repo)}/versions`),
    ]);
    const wall = performance.now() - t0;
    const labels = vs.versions.map((v) => v.label);
    if (flat) renderFlat(r.flat_results); else renderGroups(r.groups, labels);
    const facts = Object.fromEntries((r.degradations || []).map((d) => d.split("=")));
    $("context-evolution").hidden = false;
    $("context-evolution").innerHTML = `<span>answered from repository <b>${escapeHtml(repo)}</b></span><span><b>${labels.length}</b> versions searched</span>` +
      (facts.lineages ? `<span><b>${fmtInt(facts.lineages)}</b> lineages in the index</span>` : "");
    $("summary-evolution").hidden = false;
    $("summary-evolution").innerHTML = [
      `<span><b>${flat ? "flat: every matching revision" : "grouped by the lineage index: one answer per lineage"}</b></span>`,
      facts.flat_duplicate_rate ? `<span title="share of flat hits that repeat a lineage already listed">flat duplicate rate <b>${(100 * Number(facts.flat_duplicate_rate)).toFixed(0)}%</b></span>` : "",
      `<span><b>${wall.toFixed(0)} ms</b> wall</span>`,
    ].join("");
    setCorpus("evolution", `${repo} · all ${labels.length} versions`);
    recordLatency(wall, null);
  } catch (err) { showError("results-evolution", err); }
  finally { $("go-evolution").disabled = false; }
}

/* -- P0 evaluation (recorded) ------------------------------------------------------------------------------ */
const evalState = { offset: 0, limit: 25, total: 0 };
const RECORDED_ORDER = { "ltr.applied": "learned ranking", "ltr.abstained": "semantic (ranker abstained)", "fusion.applied": "hybrid fusion", "fusion.identifier_first": "identifier match" };
async function loadEvaluation() {
  $("load-eval").disabled = true;
  try {
    // Server-rendered from the ledger row; inserted exactly as received — no number is computed on this page.
    const html = await apiText("/v1/benchmarks/p0/panel");
    $("eval-panel").innerHTML = html;
    evalState.offset = 0;
    await loadEvalRows();
  } catch (err) { $("eval-panel").innerHTML = `<div class="error">${escapeHtml(err.message)}</div>`; }
  finally { $("load-eval").disabled = false; }
}
async function loadEvalRows() {
  try {
    const page = await api(`/v1/benchmarks/p0/queries?only=${encodeURIComponent(ddValue("eval-filter"))}&offset=${evalState.offset}&limit=${evalState.limit}`);
    evalState.total = page.total;
    $("eval-count").textContent = page.total ? `${fmtInt(evalState.offset + 1)}–${fmtInt(Math.min(page.total, evalState.offset + evalState.limit))} of ${fmtInt(page.total)}` : "none";
    $("eval-prev").disabled = evalState.offset === 0;
    $("eval-next").disabled = evalState.offset + evalState.limit >= page.total;
    const rank = (v) => (v == null ? '<span class="badge warn">not in top 100</span>' : v <= 10 ? `<b>${v}</b>` : String(v));
    $("eval-rows").innerHTML = page.records.map((r) => `<tr data-q="${escapeHtml(r.query_id)}">
      <td class="id">${escapeHtml(r.query_id)}</td><td>${escapeHtml(RECORDED_ORDER[r.ordered_by] || r.ordered_by)}</td>
      <td>${rank(r.rank_full)}</td><td>${rank(r.rank_dense)}</td><td class="head">${escapeHtml(r.query_head)}</td></tr>`).join("");
  } catch (err) { $("eval-rows").innerHTML = `<tr><td colspan="5" class="muted">${escapeHtml(err.message)}</td></tr>`; $("eval-count").textContent = ""; }
}
async function showQuery(qid) {
  const box = $("eval-detail"); box.hidden = false; box.innerHTML = '<div class="muted">loading…</div>';
  try {
    const r = await api(`/v1/benchmarks/p0/queries/${encodeURIComponent(qid)}`);
    const gold = new Set(r.gold);
    const top = r.top10_full.map((d, i) => `<li class="${gold.has(d) ? "gold" : ""}"><span>${i + 1}.</span><span>${escapeHtml(d)}</span>${gold.has(d) ? "<span>relevant</span>" : ""}</li>`).join("");
    const goldCode = (r.gold_code || []).map((g) => g.available ? codeBlock(g.source, 16) : `<div class="muted">${escapeHtml(g.doc_id)} (P0 corpus not loaded)</div>`).join("");
    box.innerHTML = `<h3>Query <code>${escapeHtml(r.query_id)}</code> · fold ${r.fold}</h3>
      <p class="muted">Ranked by <b>${escapeHtml(RECORDED_ORDER[r.ordered_by] || r.ordered_by)}</b> (query type ${escapeHtml(r.route.route)}: ${escapeHtml(r.route.reason)}).
      Relevant document at rank <b>${r.rank_full ?? "beyond 100"}</b> by ACIS, <b>${r.rank_dense ?? "beyond 100"}</b> by dense similarity alone.</p>
      <div class="detail-grid">
        <div><h3>Statement (first 600 characters of ${fmtInt(r.query_chars)})</h3><div class="statement">${escapeHtml(r.query_excerpt)}</div>
          <h3 style="margin-top:14px">ACIS top 10</h3><ol class="toplist">${top}</ol></div>
        <div><h3>Relevant document: ${r.gold.map((g) => `<code>${escapeHtml(g)}</code>`).join(" ")}</h3>${goldCode}</div>
      </div>`;
  } catch (err) { box.innerHTML = `<div class="error">${escapeHtml(err.message)}</div>`; }
}

/* -- Architecture tab (GET /v1/system) --------------------------------------------------------------------- */
const kv = (pairs) => `<div class="kv">${pairs.filter(([, v]) => v !== null && v !== undefined && v !== "").map(([k, v]) => `<div><span>${escapeHtml(k)}</span><b>${v}</b></div>`).join("")}</div>`;
function renderSystem() {
  if (!sys) return;
  renderPipeline("pipeline-system");
  $("sys-models").innerHTML = sys.encoders.map((e) => {
    const note = e.state === "optional" ? `<div class="muted" style="margin:6px 0 0">Second semantic view of every query, instruction-aware per query type. ${escapeHtml(e.reason || "")}.</div>` : "";
    return `<div class="model"><div class="model-head"><span class="model-name">${escapeHtml(e.name || e.key)}</span><span>${stateBadge(e) || ""}</span></div>
      <div class="model-role">${escapeHtml(e.role)}</div>
      ${kv([["parameters", millions(e.params)], ["dimensions", e.dim], ["max tokens", e.max_tokens], ["weights", e.size_mb ? `${fmtInt(Math.round(e.size_mb))} MB` : null],
            ["licence", e.licence ? escapeHtml(e.licence) : null], ["commit", e.commit ? `<code>${escapeHtml(String(e.commit).slice(0, 10))}</code>` : null],
            ["pooling", e.pooling ? escapeHtml(e.pooling) : null], ["instructions", e.instruction_aware === undefined ? null : e.instruction_aware ? "per query type" : "none"]])}
      ${note}</div>`;
  }).join("");
  const rk = sys.ranker || {};
  $("sys-ranker").innerHTML = rk.loaded ? `${kv([["model", escapeHtml(rk.kind)], ["trees", rk.rounds], ["features", rk.features], ["trained on", `${fmtInt(rk.trained_on_queries)} queries`],
      ["group dropout", rk.group_dropout != null ? `${Math.round(rk.group_dropout * 100)}%` : null], ["evaluated", "out of fold, 5 folds"]])}
      <div class="feature-groups">${Object.entries(rk.groups || {}).map(([g, names]) => `<div><b>${escapeHtml(g)}</b>${names.map((n) => `<code>${escapeHtml(n)}</code>`).join(" ")}</div>`).join("")}</div>
      <p class="muted" style="margin:10px 0 0">Candidates from every channel are scored on every signal; LambdaRank learns how to weigh them from 5,000 labelled dev queries. Group dropout during training keeps any single family of evidence from becoming load-bearing, so an unfamiliar query degrades gracefully.</p>`
    : '<div class="muted">No ranker is loaded in this process.</div>';
  $("sys-bakeoff").innerHTML = sys.bakeoff_html;   // server-rendered from the ledger rows, inserted verbatim
  const lex = sys.lexical || {}, bm = lex.bm25 || {}, sy = lex.symbols || {}, cand = sys.candidates || {};
  $("sys-lexical").innerHTML = kv([["BM25", escapeHtml(bm.method || "-")], ["k1 · b", `${bm.k1} · ${bm.b}`], ["tokenizer", escapeHtml(bm.tokenizer || "-")], ["stemmer", escapeHtml(bm.stemmer || "-")],
    ["identifiers indexed", fmtInt(sy.distinct_symbols)], ["names defined", fmtInt(sy.defined_names)]]) +
    `<p class="muted" style="margin:10px 0 0">The identifier channel matches code-shaped names in the query (<code>heapq.heappush</code>, <code>sum_intervals</code>, <code>UnionFind</code>) exactly against every identifier in the corpus, weighted by rarity.</p>` +
    kv([["semantic top", cand.dense_k], ["BM25 top", cand.lexical_k], ["identifier top", cand.symbol_k], ["2nd encoder top", cand.aux_k || null], ["union cap", cand.union_cap]]);
  const dv = sys.device || {};
  $("sys-runtime").innerHTML = kv([["device", "CPU"], ["numeric profile", escapeHtml(dv.numeric_profile)], ["model threads", dv.threads], ["BLAS threads", dv.blas_threads],
    ["GPU at query time", dv.gpu_at_query_time ? "yes" : "none"], ["network", dv.network_at_query_time ? "yes" : "none"],
    ["corpus", sys.corpus.loaded ? `${fmtInt(sys.corpus.units)} units` : "not loaded"], ["config", `<code>${escapeHtml(sys.config_hash)}</code>`]]) +
    `<p class="muted" style="margin:10px 0 0">${escapeHtml(sys.candidates.search)}.</p>`;
}
async function loadSystem() {
  try {
    sys = await api("/v1/system");
    renderPipeline("pipeline");
    renderSystem();
    const channels = ["primary dense channel", "second dense channel"].map(encoderOf).filter((e) => e && e.state === "serving").map((e) => shortName(e.name));
    $("pill-stack").textContent = [...channels, "BM25", "identifiers", sys.ranker.loaded ? "LambdaRank" : null].filter(Boolean).join(" + ");
    $("pill-stack").classList.add("ok");
    $("pill-profile").textContent = `CPU · ${sys.device.numeric_profile} · ${sys.device.threads} threads · offline`;
  } catch (err) { $("pill-stack").textContent = `system view unavailable: ${err.message}`; $("pill-stack").classList.add("warn"); }
}

/* Deep links for a demo runbook: `#tab=system` opens a tab; `#q=<query>` runs that search on load (on the Search tab,
 * or on `#tab=versions` / `#tab=evolution` when one of those is named). */
function openFromFragment() {
  const params = new URLSearchParams(location.hash.slice(1));
  const tab = params.get("tab");
  if (tab) { const button = document.querySelector(`.tab[data-view="${CSS.escape(tab)}"]`); if (button) button.click(); }
  const q = params.get("q");
  const view = tab && ["versions", "evolution"].includes(tab) ? tab : "search";
  const version = params.get("version");                 // `#tab=versions&version=v1&q=…` pins that version
  if (view === "versions" && version && dropdowns["version-versions"].options.some((o) => o.value === version)) setValue("version-versions", version, false);
  const mode = params.get("mode");                       // `#tab=evolution&mode=flat&q=…` shows every revision
  if (view === "evolution" && ["grouped", "flat"].includes(mode)) document.querySelectorAll("#evo-mode button").forEach((b) => b.classList.toggle("on", b.dataset.value === mode));
  if (q) { $(`q-${view}`).value = q; $(`form-${view}`).requestSubmit(); }
}

/* -- start-up ---------------------------------------------------------------------------------------------- */
async function init() {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${t.dataset.view}`));
    showCorpus();
    if (t.dataset.view === "evaluation" && !$("eval-panel").dataset.loaded) { $("eval-panel").dataset.loaded = "1"; loadEvaluation(); }
    if (t.dataset.view === "system") loadSystem();
  }));
  bindSeg("channel"); bindSeg("evo-mode");
  $("form-search").addEventListener("submit", searchP0);
  $("form-versions").addEventListener("submit", searchP1);
  $("form-evolution").addEventListener("submit", searchBonus);
  for (const id of ["q-search", "q-versions", "q-evolution"]) {
    $(id).addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $(id).form.requestSubmit(); }
    });
  }
  $("commit").addEventListener("click", commit);
  $("rollback").addEventListener("click", rollback);
  $("load-eval").addEventListener("click", loadEvaluation);
  $("eval-prev").addEventListener("click", () => { evalState.offset = Math.max(0, evalState.offset - evalState.limit); loadEvalRows(); });
  $("eval-next").addEventListener("click", () => { evalState.offset += evalState.limit; loadEvalRows(); });
  $("eval-rows").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-q]"); if (tr) showQuery(tr.dataset.q); });
  dropdown("topk-search", { options: ["10", "25", "50"], value: "10" });
  dropdown("eval-filter", {
    options: [{ value: "all", label: "all queries" }, { value: "missed", label: "relevant doc not in top 10" },
      { value: "improved", label: "ranked higher than dense" }, { value: "worsened", label: "ranked lower than dense" },
      { value: "generic", label: "short / non-statement queries" }],
    value: "all", onChange: () => { evalState.offset = 0; loadEvalRows(); },
  });
  dropdown("repo-versions", { options: [], onChange: loadVersions });
  dropdown("repo-evolution", { options: [], onChange: (repo) => setCorpus("evolution", `${repo} · all versions`) });
  dropdown("version-versions", { options: [] });
  loadKpis();
  try {
    const [health, ready] = await Promise.all([api("/healthz"), api("/readyz")]);
    const needed = ["repo_identity", "benchmarks_p0", "calibrated_confidence", "system_view", "hit_evidence"];
    const missing = needed.filter((f) => !(health.api_features || []).includes(f));
    if (missing.length) {
      document.querySelector("main").insertAdjacentHTML("afterbegin", `<div class="notice bad"><b>This server process is older than this page.</b> It was started before an upgrade and lacks: ${missing.map(escapeHtml).join(", ")}. Restart <code>acis serve</code>; until then some panels cannot show real values.</div>`);
    }
    await loadSystem();
    setCorpus("search", ready.p0_corpus ? `APPS corpus · ${fmtInt(ready.p0_units)} units` : null);
    const repos = health.repos || [];
    const preferred = repos.includes("apps-history") ? "apps-history" : repos[0];
    dropdown("repo-evolution", { options: repos, value: preferred });
    if (preferred) setCorpus("evolution", `${preferred} · all versions`);
    dropdown("repo-versions", { options: repos, value: preferred });
    if (preferred) await loadVersions();
    openFromFragment();
  } catch (err) { $("pill-stack").textContent = `API unavailable: ${err.message}`; $("pill-stack").classList.add("warn"); }
}
init();
