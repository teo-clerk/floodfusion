// floodfusion Arena (v1.2): every number comes from window.FLOODFUSION_ARENA, built from the committed reports by
// scripts/build_site.py. Sections whose report does not exist yet say so instead of drawing anything.
(() => {
  "use strict";

  const DATA = window.FLOODFUSION_ARENA;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const byId = (id) => document.getElementById(id);

  function setAttrs(node, attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      if (value == null || value === false) continue;
      if (key === "style" && typeof value === "object") Object.assign(node.style, value);
      else if (key === "text") node.textContent = value;
      else node.setAttribute(key, value === true ? "" : value);
    }
    return node;
  }
  function append(node, children) {
    for (const child of children.flat()) {
      if (child != null && child !== false) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }
  const h = (tag, attrs = {}, ...children) => append(setAttrs(document.createElement(tag), attrs), children);
  const s = (tag, attrs = {}, ...children) => append(setAttrs(document.createElementNS(SVG_NS, tag), attrs), children);
  const MINUS = "−";
  const fixed = (x, digits = 3) => (x == null || Number.isNaN(x) ? "—" : x.toFixed(digits).replace("-", MINUS));

  // ---------------------------------------------------------------- shared UI
  const tooltip = byId("tooltip");
  function showTip(text, x, y) {
    tooltip.textContent = text;
    tooltip.hidden = false;
    const pad = 12, rect = tooltip.getBoundingClientRect();
    tooltip.style.left = `${Math.min(x + pad, window.innerWidth - rect.width - pad)}px`;
    tooltip.style.top = `${Math.max(pad, y - rect.height - pad)}px`;
  }
  const hideTip = () => { tooltip.hidden = true; };
  function withTip(node, text) {
    node.addEventListener("pointerenter", (e) => showTip(text, e.clientX, e.clientY));
    node.addEventListener("pointermove", (e) => showTip(text, e.clientX, e.clientY));
    node.addEventListener("pointerleave", hideTip);
    node.addEventListener("focus", () => { const r = node.getBoundingClientRect(); showTip(text, r.right, r.top); });
    node.addEventListener("blur", hideTip);
    return node;
  }

  function radioGroup(container, options, onSelect, initial = 0) {
    const buttons = options.map(([value, label]) =>
      h("button", { type: "button", role: "radio", "data-value": value }, label));
    container.replaceChildren(...buttons);
    const select = (i) => {
      buttons.forEach((b, j) => { b.setAttribute("aria-checked", String(i === j)); b.tabIndex = i === j ? 0 : -1; });
      onSelect(options[i][0]);
    };
    buttons.forEach((b, i) => {
      b.addEventListener("click", () => select(i));
      b.addEventListener("keydown", (e) => {
        const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
        if (!step) return;
        e.preventDefault();
        const next = (i + step + buttons.length) % buttons.length;
        buttons[next].focus();
        select(next);
      });
    });
    select(initial);
  }

  function initTheme() {   // shared with the explainer: same key, same values
    const root = document.documentElement;
    try {
      const saved = localStorage.getItem("floodfusion-theme");
      if (saved === "light" || saved === "dark") root.dataset.theme = saved;
    } catch { /* storage unavailable: follow the system theme */ }
    byId("theme-toggle").addEventListener("click", () => {
      const dark = root.dataset.theme ? root.dataset.theme === "dark"
        : window.matchMedia("(prefers-color-scheme: dark)").matches;
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("floodfusion-theme", root.dataset.theme); } catch { /* not persisted */ }
    });
  }

  // ---------------------------------------------------------------- stats
  function stat(value, label) {
    const pending = value == null;
    return h("div", { class: "stat" },
      h("p", { class: pending ? "stat-value stat-pending" : "stat-value" }, pending ? "pending" : value),
      h("p", { class: "stat-label" }, label));
  }
  function initStats() {
    const sb = DATA.scoreboard, blind = DATA.blind;
    const table = sb.status === "ok" ? sb.tables.find((t) => t.unit === "activation" && !t.gfm_common) : null;
    byId("arena-stats").replaceChildren(
      stat(table ? String(Object.keys(table.models).length) : null, "models on the board"),
      stat(table ? String(table.units.length) : null, "Copernicus flood activations with reference flood"),
      stat(sb.status === "ok" ? `${sb.gfm_pairs}/${sb.n_pairs}` : null, "radar-drawn pairs GFM covers (≥ 95 %)"),
      stat(`${blind.k_activations ?? 0}/20`, "blind-test activations scored"),
    );
  }

  // ---------------------------------------------------------------- scoreboard
  const METRICS = [
    ["iou_tau", "IoU at τ*"], ["iou_selfcal", "Self-calibrated"], ["iou_tau_binned", "τ*, binned"],
    ["iou_oracle", "Oracle τ"], ["ap", "AP"],
  ];
  const FAMILY_COLOURS = [
    [/gfm/, "--c-gfm"], [/tm|terramind/, "--c-tm"], [/r4|unet/, "--c-sar"], [/r2s|r2$|baseline/, "--c-phys"],
    [/v12|ctx|row2[0-2]/, "--c-v12"], [/r9|r11|fusion/, "--c-fusion"],
  ];
  const colourOf = (key) => `var(${(FAMILY_COLOURS.find(([re]) => re.test(key)) || [null, "--muted"])[1]})`;

  function scoreboardChart(container, table, metric) {
    const rows = Object.entries(table.models)
      .map(([key, row]) => ({ key, ...row, value: row[metric], ci: (row.ci || {})[metric] || [NaN, NaN] }))
      .filter((r) => r.value != null && !Number.isNaN(r.value))
      .sort((a, b) => b.value - a.value);
    const W = Math.max(320, container.clientWidth || 760), rowH = 30, top = 26, left = Math.min(260, W * 0.42);
    const H = top + rows.length * rowH + 30, right = 56;
    const max = Math.max(0.1, ...rows.map((r) => (Number.isFinite(r.ci[1]) ? r.ci[1] : r.value)));
    const x = (v) => left + (v / max) * (W - left - right);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
      "aria-label": `${METRICS.find(([m]) => m === metric)[1]} per model, macro over ${table.units.length} ${table.unit}s` });
    for (let v = 0; v <= max + 1e-9; v += 0.1) {
      svg.append(s("line", { x1: x(v), x2: x(v), y1: top - 6, y2: H - 24, style: { stroke: "var(--c-grid)" } }),
                 s("text", { x: x(v), y: H - 8, "text-anchor": "middle" }, v.toFixed(1)));
    }
    rows.forEach((r, i) => {
      const y = top + i * rowH, c = colourOf(r.key);
      const bar = s("rect", { x: left, y: y + 6, width: Math.max(1, x(r.value) - left), height: rowH - 14, rx: 3,
        tabindex: 0, style: { fill: c, opacity: 0.85 } });
      const ci = Number.isFinite(r.ci[0]) ? `95 % interval ${fixed(r.ci[0])} – ${fixed(r.ci[1])}` : "no interval (fewer than 2 units)";
      withTip(bar, `${r.name}: ${fixed(r.value)} (${ci}; ${r.k} ${table.unit}s)`);
      svg.append(s("text", { x: left - 8, y: y + rowH / 2 + 4, "text-anchor": "end" }, r.name), bar);
      if (Number.isFinite(r.ci[0])) {
        svg.append(s("line", { x1: x(r.ci[0]), x2: x(r.ci[1]), y1: y + rowH / 2, y2: y + rowH / 2,
                               style: { stroke: "var(--text)", "stroke-width": 1.5 } }));
      }
      svg.append(s("text", { x: x(Math.max(r.value, Number.isFinite(r.ci[1]) ? r.ci[1] : r.value)) + 6, y: y + rowH / 2 + 4,
                             class: "num" }, fixed(r.value)));
    });
    container.replaceChildren(svg);
  }

  function initScoreboard() {
    const sb = DATA.scoreboard, chart = byId("scoreboard-chart"), note = byId("scoreboard-note");
    if (sb.status !== "ok") {
      chart.replaceChildren(h("p", { class: "notice" }, "The v1.2 scores are being computed: this board fills in when ",
        h("code", {}, "reports/v12_cems_preview.json"), " exists."));
      chart.closest(".panel").querySelector(".arena-controls").style.display = "none";
      return;
    }
    const state = { metric: "iou_tau", unit: "activation", gfm: false };
    const draw = () => {
      const table = sb.tables.find((t) => t.unit === state.unit && t.gfm_common === state.gfm);
      scoreboardChart(chart, table, state.metric);
      note.textContent = `Macro over ${table.units.length} ${state.unit}s with reference flood${state.gfm
        ? ", restricted to those Copernicus GFM covers on at least 95 % of the scored pixels, so every model faces the same floods"
        : ""}. Bars: mean; whiskers: 95 % bootstrap interval over ${state.unit}s. Exploratory: this benchmark was ` +
        "already seen in v1.1; v1.2's confirmatory test is the blind one below.";
    };
    radioGroup(byId("metric"), METRICS, (m) => { state.metric = m; draw(); });
    radioGroup(byId("unit"), [["activation", "Per activation"], ["cluster", "Per cluster"]], (u) => { state.unit = u; draw(); });
    radioGroup(byId("scope"), [["all", "All floods"], ["gfm", "Floods GFM covers"]], (v) => { state.gfm = v === "gfm"; draw(); });
    let t = null;
    window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(draw, 150); });
  }

  // ---------------------------------------------------------------- maps
  function initMaps() {
    const maps = DATA.maps;
    if (maps.status !== "ok" || !maps.pairs.length) {
      byId("maps-pending").hidden = false;
      const panel = byId("arena-swipe").closest(".panel");
      for (const node of panel.querySelectorAll(".arena-controls, .swipe, .footnote")) node.style.display = "none";
      return;
    }
    const pairSel = byId("pair"), leftSel = byId("left"), rightSel = byId("right");
    const left = byId("arena-left"), right = byId("arena-right"), range = byId("arena-range");
    const option = (value, label) => h("option", { value }, label);
    pairSel.replaceChildren(...maps.pairs.map((p, i) => option(String(i), `${p.activation} · AOI ${p.aoi} · ${p.product_time.slice(0, 10)}`)));
    let pair = null;
    const tag = (key) => {
      const layer = pair.layers[key];
      return layer.iou == null ? layer.name : `${layer.name} · IoU ${fixed(layer.iou)}`;
    };
    const move = () => {
      right.style.clipPath = `inset(0 0 0 ${range.value}%)`;
      byId("arena-handle").style.left = `${range.value}%`;
      range.setAttribute("aria-valuetext", `${Math.round(range.value)} % ${tag(leftSel.value)}, ` +
        `${Math.round(100 - range.value)} % ${tag(rightSel.value)}`);
    };
    const show = () => {
      for (const [img, sel, tagId] of [[left, leftSel, "arena-tag-left"], [right, rightSel, "arena-tag-right"]]) {
        const layer = pair.layers[sel.value];
        Object.assign(img, { src: layer.src, alt: `${tag(sel.value)}: ${pair.activation} AOI ${pair.aoi}` });
        byId(tagId).textContent = tag(sel.value);
      }
      move();
    };
    const choosePair = () => {
      pair = maps.pairs[Number(pairSel.value)];
      const keys = Object.keys(pair.layers);
      const opts = () => keys.map((k) => option(k, pair.layers[k].name));
      const keepL = keys.includes(leftSel.value) ? leftSel.value : "radar";
      const keepR = keys.includes(rightSel.value) ? rightSel.value : keys.find((k) => !["radar", "cems"].includes(k)) || "cems";
      leftSel.replaceChildren(...opts());
      rightSel.replaceChildren(...opts());
      leftSel.value = keepL;
      rightSel.value = keepR;
      byId("pair-facts").textContent = `${pair.activation} AOI ${pair.aoi}: Copernicus map from ${pair.sensors.join(", ")}` +
        ` (${pair.product_time.slice(0, 16).replace("T", " ")} UTC), Sentinel-1 at ${pair.pass_time.slice(0, 16).replace("T", " ")}` +
        ` UTC; ${fixed(pair.flood_km2, 1)} km² of reference flood on this tile.`;
      show();
    };
    pairSel.addEventListener("change", choosePair);
    leftSel.addEventListener("change", show);
    rightSel.addEventListener("change", show);
    range.addEventListener("input", move);
    choosePair();
  }

  // ---------------------------------------------------------------- errata
  function initErrata() {
    byId("errata-cards").replaceChildren(...DATA.errata.map((e) =>
      h("article", { class: "panel erratum" },
        h("p", { class: `badge ${e.resolved ? "badge-good" : "badge-warn"}` }, e.status),
        h("h3", {}, e.title),
        h("p", {}, e.summary),
        e.result ? h("p", { class: "erratum-result" }, e.result) : null,
        h("p", { class: "cite" }, h("a", { href: e.link }, "The erratum, with its evidence")))));
  }

  // ---------------------------------------------------------------- blind test
  function progress(label, value, target) {
    const share = Math.min(1, value / target);
    return h("div", { class: "blind-progress" },
      h("span", { class: "blind-label" }, `${label}: ${value} of ${target}`),
      h("span", { class: "blind-track", role: "img", "aria-label": `${value} of ${target} ${label}` },
        h("span", { class: "blind-fill", style: { width: `${100 * share}%` } })));
  }
  function initBlind() {
    const b = DATA.blind, panel = byId("blind-panel");
    const rows = (b.floods || []).map((f) => h("tr", {}, h("td", {}, f.code), h("td", {}, f.date), h("td", {}, f.status)));
    const parts = [
      h("p", {}, h("b", {}, b.status_text)),
      progress("activations", b.k_activations ?? 0, 20),
      b.k_clusters === null
        ? h("p", { class: "footnote" }, "Clusters: not counted here (the runner's activation records are local).")
        : progress("clusters", b.k_clusters ?? 0, 15),
      rows.length ? h("div", { class: "table-wrap" }, h("table", { class: "data-table" },
        h("thead", {}, h("tr", {}, h("th", {}, "Pair"), h("th", {}, "Event"), h("th", {}, "Status"))),
        h("tbody", {}, rows))) : null,
      h("p", { class: "footnote" }, "Scores of the blind floods are sealed: a running score invites stopping when it ",
        "looks good. The verdicts are computed once, at the stopping rule."),
    ];
    panel.replaceChildren(...parts.filter(Boolean));   // a DOM node list prints null as text
  }

  // ---------------------------------------------------------------- start
  initTheme();
  if (!DATA) { byId("no-data").hidden = false; return; }
  document.querySelectorAll("[data-bind=version]").forEach((n) => { n.textContent = DATA.version; });
  initStats();
  initScoreboard();
  initMaps();
  initErrata();
  initBlind();
})();
