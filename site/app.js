/* floodfusion explainer: renders the interactive parts from window.FLOODFUSION, which
   scripts/build_site.py writes from the committed reports. No numbers live in this file. */
(() => {
  "use strict";

  const DATA = window.FLOODFUSION;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const byId = (id) => document.getElementById(id);

  // ------------------------------------------------------------------ DOM helpers
  function setAttrs(node, attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      if (value == null || value === false) continue;
      if (key === "style") {
        for (const [prop, v] of Object.entries(value)) node.style.setProperty(prop, v);
      } else if (key.startsWith("on")) {
        node.addEventListener(key.slice(2), value);
      } else {
        node.setAttribute(key, value === true ? "" : String(value));
      }
    }
    return node;
  }
  function append(node, children) {
    for (const child of children.flat()) {
      if (child != null) node.append(child instanceof Node ? child : String(child));
    }
    return node;
  }
  const h = (tag, attrs = {}, ...children) => append(setAttrs(document.createElement(tag), attrs), children);
  const s = (tag, attrs = {}, ...children) => append(setAttrs(document.createElementNS(SVG_NS, tag), attrs), children);
  // role="img" makes children presentational, so charts with focusable parts are groups instead.
  const svgRoot = (width, height, label, role = "img") =>
    s("svg", { viewBox: `0 0 ${width} ${height}`, role, "aria-label": label });

  // ------------------------------------------------------------------ responsive charts
  // Charts are drawn at their container's pixel width so text keeps its size; they redraw on resize.
  const redraws = [];
  const widthOf = (container) => Math.max(300, Math.round(container.clientWidth || 760));
  function chartSlot(draw) {
    let args = null;
    const run = () => { if (args) draw(...args); };
    redraws.push(run);
    return (...latest) => { args = latest; run(); };
  }
  let resizeTimer = 0, lastWidth = window.innerWidth;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (window.innerWidth === lastWidth) return;
      lastWidth = window.innerWidth;
      redraws.forEach((run) => run());
    }, 150);
  });

  // ------------------------------------------------------------------ formatting
  const MINUS = "−";
  const fixed = (x, digits = 3) => (x == null || Number.isNaN(x) ? "—" : x.toFixed(digits).replace("-", MINUS));
  const signed = (x, digits = 3) => (x == null ? "—" : (x < 0 ? MINUS : "+") + Math.abs(x).toFixed(digits));
  const linear = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
  const utc = (text) => Date.parse(`${text}:00Z`);
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const dayLabel = (ms) => { const d = new Date(ms); return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`; };
  const timeLabel = (ms) => {
    const d = new Date(ms);
    return `${dayLabel(ms)} ${String(d.getUTCHours()).padStart(2, "0")}:${String(d.getUTCMinutes()).padStart(2, "0")}`;
  };
  const range = (values, digits) => {
    const lo = Math.min(...values), hi = Math.max(...values);
    return lo.toFixed(digits) === hi.toFixed(digits) ? lo.toFixed(digits) : `${lo.toFixed(digits)}–${hi.toFixed(digits)}`;
  };

  // ------------------------------------------------------------------ tooltip
  const tooltip = byId("tooltip");
  function placeTip(x, y) {
    const pad = 12, rect = tooltip.getBoundingClientRect();
    tooltip.style.left = `${Math.min(window.innerWidth - rect.width - pad, Math.max(pad, x + 14))}px`;
    tooltip.style.top = `${y - rect.height - 14 < pad ? y + 18 : y - rect.height - 14}px`;
  }
  function showTip(text, x, y) {
    tooltip.textContent = text;
    tooltip.hidden = false;
    placeTip(x, y);
  }
  const hideTip = () => { tooltip.hidden = true; };
  function withTip(node, text) {
    node.addEventListener("pointerenter", (e) => showTip(text, e.clientX, e.clientY));
    node.addEventListener("pointermove", (e) => placeTip(e.clientX, e.clientY));
    node.addEventListener("pointerleave", hideTip);
    node.addEventListener("focus", () => {
      const r = node.getBoundingClientRect();
      showTip(text, r.left + r.width / 2, r.top);
    });
    node.addEventListener("blur", hideTip);
    return node;
  }

  // ------------------------------------------------------------------ accessible controls
  /** Radio buttons with roving tabindex and arrow keys; calls onSelect(button). */
  function radioGroup(buttons, onSelect, initial = 0) {
    const select = (target, focus = false) => {
      for (const b of buttons) {
        const on = b === target;
        b.setAttribute("aria-checked", String(on));
        b.tabIndex = on ? 0 : -1;
      }
      if (focus) target.focus();
      onSelect(target);
    };
    buttons.forEach((b, i) => {
      b.addEventListener("click", () => select(b));
      b.addEventListener("keydown", (e) => {
        const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
        if (step) {
          e.preventDefault();
          select(buttons[(i + step + buttons.length) % buttons.length], true);
        }
      });
    });
    select(buttons[initial]);
  }

  /** Tabs (role="tab") with arrow keys; calls onSelect(button). */
  function tabs(buttons, onSelect, initial = 0) {
    const select = (target, focus = false) => {
      for (const b of buttons) {
        const on = b === target;
        b.setAttribute("aria-selected", String(on));
        b.tabIndex = on ? 0 : -1;
      }
      if (focus) target.focus();
      onSelect(target);
    };
    buttons.forEach((b, i) => {
      b.addEventListener("click", () => select(b));
      b.addEventListener("keydown", (e) => {
        const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
        if (step) {
          e.preventDefault();
          select(buttons[(i + step + buttons.length) % buttons.length], true);
        }
      });
    });
    select(buttons[initial]);
  }

  /** Toggle chips (aria-pressed); keeps at least one on; calls onChange(set of keys). */
  function toggleChips(container, items, initiallyOn, onChange) {
    const on = new Set(initiallyOn);
    for (const item of items) {
      const chip = h("button", { type: "button", class: "chip", "aria-pressed": String(on.has(item.key)),
                                  style: { "--chip-color": item.color } },
                     h("span", { class: "swatch", "aria-hidden": "true" }), item.label);
      chip.addEventListener("click", () => {
        if (on.has(item.key) && on.size === 1) return;
        on.has(item.key) ? on.delete(item.key) : on.add(item.key);
        chip.setAttribute("aria-pressed", String(on.has(item.key)));
        onChange(on);
      });
      container.append(chip);
    }
    onChange(on);
  }

  // ------------------------------------------------------------------ theme and navigation
  function initTheme() {
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

  function initNav() {
    const links = new Map([...document.querySelectorAll(".topnav a")].map((a) => [a.hash.slice(1), a]));
    if (!("IntersectionObserver" in window)) return;
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          links.forEach((a, id) => a.classList.toggle("active", id === entry.target.id));
        }
      }
    }, { rootMargin: "-45% 0px -50% 0px" });
    links.forEach((_, id) => { const section = byId(id); if (section) observer.observe(section); });
  }

  function initTerms() {
    for (const term of document.querySelectorAll("a.term")) {
      const entry = document.querySelector(`${term.hash} dd`);
      if (entry) withTip(term, entry.textContent);
    }
  }

  function initCopy() {
    for (const button of document.querySelectorAll("[data-copy]")) {
      button.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(byId(button.dataset.copy).textContent);
          button.textContent = "Copied";
        } catch {
          button.textContent = "Select and copy";
        }
        setTimeout(() => { button.textContent = "Copy"; }, 1800);
      });
    }
  }

  // ------------------------------------------------------------------ model palette
  const S1F11_COLORS = { 1: "--c-otsu", 2: "--c-phys", 3: "--c-mndwi", 4: "--c-sar", 9: "--c-fusion", 11: "--c-noopt",
                         cmp_terramind_small: "--c-tm" };
  const s1Color = (key) => `var(${S1F11_COLORS[key] || "--muted"})`;
  const forliStyle = (name) => {
    if (name.startsWith("physics")) return { label: "Physics baseline", color: "var(--c-phys)" };
    if (name.includes("SAR only")) return { label: "Fusion, no optical", color: "var(--c-noopt)" };
    return { label: "Fusion", color: "var(--c-fusion)" };
  };
  const STRESS_SERIES = {
    "row 2 baseline": { label: "Physics baseline", color: "var(--c-phys)" },
    "row 4 U-Net SAR": { label: "U-Net, SAR only", color: "var(--c-sar)" },
    "row 9 fusion": { label: "Fusion", color: "var(--c-fusion)" },
    "row 11 fusion, SAR-only": { label: "Fusion, optical removed", color: "var(--c-noopt)" },
  };
  const stressStyle = (name) => STRESS_SERIES[name] || { label: name, color: "var(--muted)" };

  // ------------------------------------------------------------------ derived facts
  function forliFacts(fo) {
    const full = Math.max(...fo.passes.map((p) => p.tiles));
    const fullPasses = fo.passes.filter((p) => p.tiles === full);
    const byOptical = [...fullPasses].sort((a, b) => a.optical_tiles - b.optical_tiles);
    const nearest = (tag) => Object.entries(fo.offset_h[tag]).sort((a, b) => Math.abs(a[1]) - Math.abs(b[1]))[0][0];
    const model = (prefix) => fo.models.find((m) => m.startsWith(prefix));
    const names = { fusion: model("fusion ("), noOpt: fo.models.find((m) => m.includes("SAR only")), base: model("physics") };
    const iou = (name, pass) => fo.iou.common[name][pass.tag][nearest(pass.tag)].iou_tau;
    const latest = [...fo.products].sort((a, b) => utc(a.time) - utc(b.time)).at(-1);
    return { optPass: byOptical.at(-1), sarPass: byOptical[0], fullPasses, nearest, names, iou, latest };
  }

  const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
  const word = (n) => WORDS[n] ?? String(n);
  const capital = (text) => text.charAt(0).toUpperCase() + text.slice(1);

  function bindings(D, forli) {
    const s1 = D.sen1floods11;
    const row = (key) => s1.rows.find((r) => r.key === key);
    const gap = (key) => s1.gaps.find((g) => g.key === key);
    const g94 = gap("9-4");
    const { optPass, sarPass, names, iou, latest, nearest } = forli;
    const reference = D.forli.products.find((p) => p.tag === nearest(optPass.tag));
    const synthetic = D.synthetic;
    const gapTm = gap("9-cmp_terramind_small");
    const area = D.forli.area_km2;
    const trained = s1.protocol_gap.filter((p) => ["4", "9", "cmp_terramind_small"].includes(p.row));
    const gate = D.synthetic.gate;
    return {
      version: `v${D.version}`,
      nEvents: String(s1.events.length),
      fusionIoU: fixed(row("9").iou),
      sarIoU: fixed(row("4").iou),
      noOptIoU: fixed(row("11").iou),
      tmIoU: fixed(row("cmp_terramind_small").iou),
      fusionWins: `${g94.wins}/${g94.n}`,
      fusionHolm: fixed(g94.p_holm),
      fusionGain: fixed(g94.delta, 2),
      forliGain: signed(iou(names.fusion, optPass) - iou(names.noOpt, optPass), 2),
      forliGainPlain: fixed(iou(names.fusion, optPass) - iou(names.noOpt, optPass), 2),
      activation: D.forli.activation,
      nProducts: word(D.forli.products.length),
      forliOptDate: `${dayLabel(utc(optPass.time))} ${optPass.time.slice(0, 4)}`,
      forliOptDay: dayLabel(utc(optPass.time)),
      forliSarDay: dayLabel(utc(sarPass.time)),
      forliOptTiles: `${optPass.optical_tiles} of ${optPass.tiles}`,
      forliRefSensor: reference.sensor,
      forliRefDay: dayLabel(utc(reference.time)),
      tmWins: `${gapTm.wins} of ${gapTm.n}`,
      nLandscapes: word(D.synthetic.stress.landscape_seeds.length),
      nSeedsSynth: capital(word(Math.max(...synthetic.rows.map((r) => r.seeds.length)))),
      nBasinsSynth: word(Math.max(...synthetic.gaps.map((g) => g.n))),
      commonKm2: D.forli.common_km2.toLocaleString("en"),
      forliFusionOpt: fixed(iou(names.fusion, optPass)),
      forliNoOptOpt: fixed(iou(names.noOpt, optPass)),
      forliBaseSar: fixed(iou(names.base, sarPass)),
      forliFusionSar: fixed(iou(names.fusion, sarPass)),
      forliAreaFusion: fixed(area.models[names.fusion][optPass.tag], 1),
      forliAreaNoOpt: fixed(area.models[names.noOpt][optPass.tag], 1),
      forliAreaCems: fixed(area.cems[latest.tag], 1),
      conformalRange: range(s1.conformal.map((c) => c.held), 0),
      noDropEvents: String(s1.label_change.filter((e) => e.median_dvv_db > -1).length),
      protocolTrained: signed(Math.max(...trained.map((p) => p.iou_tau.delta))),
      gateStart: fixed(gate.start, 2),
      gateClearRange: range(gate.by_cloud.map((g) => g.clear), 2),
    };
  }

  function applyBindings(values) {
    for (const node of document.querySelectorAll("[data-bind]")) {
      node.textContent = values[node.dataset.bind] ?? "?";
    }
  }

  // ------------------------------------------------------------------ 1 · radar explorer
  function initExplorer() {
    const lo = -30, hi = 5;
    const pct = (db) => ((db - lo) / (hi - lo)) * 100;
    const ticks = byId("db-ticks");
    for (let db = lo; db <= hi; db += 5) {
      ticks.append(h("span", { style: { left: `${pct(db)}%` } }, db > 0 ? `+${db}` : fixed(db, 0)));
    }
    const band = byId("db-range");
    const infos = [...document.querySelectorAll("[data-info]")];
    radioGroup([...document.querySelectorAll("#surface-buttons [data-surface]")], (button) => {
      const min = Number(button.dataset.min), max = Number(button.dataset.max);
      band.style.left = `${pct(min)}%`;
      band.style.width = `${pct(max) - pct(min)}%`;
      for (const info of infos) info.hidden = info.dataset.info !== button.dataset.surface;
    });
  }

  // ------------------------------------------------------------------ 2 · pipeline
  function initPipeline() {
    tabs([...document.querySelectorAll("#pipeline-tabs [role=tab]")], (tab) => {
      for (const panel of document.querySelectorAll(".step-panel")) {
        panel.hidden = panel.id !== tab.getAttribute("aria-controls");
      }
    });
    const input = byId("tau-r");
    const update = () => {
      const r = Number(input.value);
      byId("tau-r-out").textContent = r.toFixed(2).replace(/\.?0+$/, "");
      byId("tau-r-text").textContent = byId("tau-r-out").textContent;
      byId("tau-value").textContent = (1 / (1 + r)).toFixed(3);
    };
    input.addEventListener("input", update);
    update();
  }

  // ------------------------------------------------------------------ 3 · fusion gate
  function initGate(gate) {
    const svg = byId("gate-diagram");
    const box = (x, y, w, label, sub, extra = {}) => s("g", extra,
      s("rect", { x, y, width: w, height: 44, rx: 9, style: { fill: "var(--surface-2)", stroke: "var(--border)" } }),
      s("text", { x: x + w / 2, y: y + (sub ? 19 : 27), "text-anchor": "middle" }, label),
      sub ? s("text", { x: x + w / 2, y: y + 34, "text-anchor": "middle", class: "small" }, sub) : null);
    const line = (d, extra = {}) => s("path", { d, fill: "none", "marker-end": "url(#arrow)",
                                                  style: { stroke: "var(--muted)", "stroke-width": "1.6" }, ...extra });
    svg.append(
      s("defs", {}, s("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7,
                                   orient: "auto-start-reverse" },
                      s("path", { d: "M0 0 L10 5 L0 10 z", style: { fill: "var(--muted)" } }))),
      box(4, 24, 128, "Radar", "VV, VH, change, terrain"),
      box(152, 24, 104, "SAR encoder", null),
      box(388, 24, 108, "Decoder", "flood probability"),
      box(4, 180, 128, "Optical", "G, NIR, SWIR, cloud"),
      box(152, 180, 104, "Optical encoder", null),
      line("M132 46 H150"), line("M256 46 H344"), line("M372 46 H386"),
      line("M132 202 H150"),
      s("text", { x: 300, y: 38, "text-anchor": "middle" }, "F", s("tspan", { "baseline-shift": "sub", "font-size": "9" }, "s")),
    );
    const attention = s("g", {},
      box(280, 180, 112, "Cross-attention", "Q opt · K,V SAR"),
      line("M256 202 H278"),
      line("M204 68 C 204 140, 312 120, 312 178"),
      s("text", { x: 238, y: 128, class: "small" }, "K, V"),
      s("text", { x: 263, y: 196, class: "small" }, "Q"),
      line("M358 180 V130"));
    const gateLine = line("M358 96 V62");
    const gateNode = s("g", {},
      s("circle", { cx: 358, cy: 113, r: 17, style: { fill: "var(--surface)", stroke: "var(--warn)", "stroke-width": "2" } }),
      s("text", { x: 358, y: 117, "text-anchor": "middle" }, "× g"));
    const plus = s("g", {},
      s("circle", { cx: 358, cy: 46, r: 13, style: { fill: "var(--surface)", stroke: "var(--text)", "stroke-width": "1.6" } }),
      s("text", { x: 358, y: 51, "text-anchor": "middle", style: { "font-size": "16px" } }, "+"));
    svg.append(attention, gateLine, gateNode, plus);

    const slider = byId("gate-slider");
    slider.value = gate.start.toFixed(2);
    const clear = gate.by_cloud.map((g) => g.clear);
    const measured = byId("gate-measured");
    measured.style.left = `${Math.min(...clear) * 100}%`;
    measured.style.width = `${Math.max((Math.max(...clear) - Math.min(...clear)) * 100, 1)}%`;
    const update = () => {
      const g = Number(slider.value);
      byId("gate-out").textContent = g.toFixed(2);
      attention.style.opacity = String(0.2 + 0.8 * g);
      gateLine.style.strokeWidth = String(1.2 + 4 * g);
      gateLine.style.stroke = g === 0 ? "var(--border)" : "var(--warn)";
      byId("gate-note").textContent = g === 0
        ? "g = 0: the optical path is cut and the output is exactly the SAR-only network's."
        : `g = ${g.toFixed(2)}: ${Math.round(g * 100)} % of the attention output is added to the radar features. `
          + `Shaded under the slider: the trained gate's mean over clear pixels (synthetic), ${range(clear, 2)}.`;
    };
    slider.addEventListener("input", update);
    update();
  }

  // ------------------------------------------------------------------ 4 · ten floods
  function floodChart(container, s1, keys) {
    const rows = s1.rows.filter((r) => keys.has(r.key));
    const labels = [...s1.events, "All ten"];
    const W = widthOf(container), left = 92, right = 18, top = 30, rowH = 30, gapBeforeAll = 14;
    const H = top + rowH * labels.length + gapBeforeAll + 8;
    const x = linear(0, 1, left, W - right);
    const svg = svgRoot(W, H, `IoU per flood for ${rows.map((r) => r.label).join(", ")}`);
    for (let t = 0; t <= 1.0001; t += 0.2) {
      svg.append(s("line", { x1: x(t), x2: x(t), y1: top - 6, y2: H - 8, class: "grid", style: { stroke: "var(--c-grid)" } }),
                 s("text", { x: x(t), y: top - 12, "text-anchor": "middle" }, t.toFixed(1)));
    }
    labels.forEach((event, i) => {
      const isAll = i === labels.length - 1;
      const y = top + rowH * (i + 0.5) + (isAll ? gapBeforeAll : 0);
      if (isAll) svg.append(s("line", { x1: 0, x2: W, y1: y - rowH / 2 - gapBeforeAll / 2, y2: y - rowH / 2 - gapBeforeAll / 2,
                                         style: { stroke: "var(--border)" } }));
      svg.append(s("text", { x: left - 10, y: y + 4, "text-anchor": "end", class: "label",
                             style: isAll ? { "font-weight": "700" } : {} }, event));
      const values = rows.map((r) => ({ r, v: isAll ? r.iou : r.per_event[event] })).filter((d) => d.v != null);
      if (values.length > 1) {
        const vs = values.map((d) => d.v);
        svg.append(s("line", { x1: x(Math.min(...vs)), x2: x(Math.max(...vs)), y1: y, y2: y,
                               style: { stroke: "var(--border)", "stroke-width": "4", "stroke-linecap": "round" } }));
      }
      for (const { r, v } of values) {
        svg.append(withTip(s("circle", { cx: x(v), cy: y, r: isAll ? 7.5 : 6.5, class: "dot", style: { fill: s1Color(r.key) } }),
                           `${event} · ${r.label}: IoU ${fixed(v)}`));
      }
    });
    container.replaceChildren(svg);
  }

  function floodTable(table, s1) {
    const head = h("tr", {}, h("th", {}, "Model"), h("th", { class: "num" }, "IoU at τ* [95 % CI]"),
                   h("th", { class: "num" }, "IoU, best threshold"), h("th", { class: "num" }, "AP"), h("th", { class: "num" }, "Seeds"));
    const body = s1.rows.map((r) => h("tr", { class: r.key === "9" ? "highlight" : null },
      h("td", {}, h("span", { class: "model-cell" }, h("span", { class: "swatch", style: { background: s1Color(r.key) } }), r.label)),
      h("td", { class: "num" }, `${fixed(r.iou)} [${fixed(r.ci[0], 2)}, ${fixed(r.ci[1], 2)}]`),
      h("td", { class: "num" }, r.oracle == null ? "—" : fixed(r.oracle)),
      h("td", { class: "num" }, fixed(r.ap)),
      h("td", { class: "num" }, r.seeds.length ? String(r.seeds.length) : "—")));
    table.replaceChildren(h("thead", {}, head), h("tbody", {}, body));
  }

  function gapBar(gap, bound) {
    const W = 260, H = 26, x = linear(-bound, bound, 8, W - 8);
    const tone = gap.significant ? "var(--good)" : "var(--muted)";
    return s("svg", { viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" },
      s("line", { x1: x(0), x2: x(0), y1: 2, y2: H - 2, style: { stroke: "var(--muted)", "stroke-dasharray": "3 3" } }),
      s("line", { x1: x(gap.ci[0]), x2: x(gap.ci[1]), y1: H / 2, y2: H / 2,
                  style: { stroke: tone, "stroke-width": "3", "stroke-linecap": "round" } }),
      s("circle", { cx: x(gap.delta), cy: H / 2, r: 6, style: { fill: tone, stroke: "var(--surface)", "stroke-width": "2" } }));
  }

  function gapList(list, gaps, labelOf, unit) {
    const bound = Math.ceil(Math.max(...gaps.flatMap((g) => g.ci.map(Math.abs))) * 10) / 10;
    list.replaceChildren(...gaps.map((g) => {
      const [b, a] = g.key.split("-");
      const verdict = g.significant == null ? h("span", { class: "badge ns" }, "no verdict")
        : h("span", { class: g.significant ? "badge sig" : "badge ns" }, g.significant ? "significant" : "not significant");
      return h("li", {},
        h("div", { class: "gap-q" }, g.question, h("small", {}, `${labelOf(b)} − ${labelOf(a)}`)),
        h("div", { class: "gap-bar" }, gapBar(g, bound)),
        h("div", { class: "gap-verdict" }, h("b", {}, `${signed(g.delta)} IoU`),
          `better in ${g.wins}/${g.n} ${unit}`, h("br"), verdict,
          g.p_holm == null ? null : h("span", { class: "muted" }, ` p = ${fixed(g.p_holm)}`)));
    }));
  }

  function protocolTable(table, note, s1) {
    const label = (key) => s1.rows.find((r) => r.key === key)?.label ?? key;
    const byRow = new Map();
    for (const p of s1.protocol_gap) byRow.set(p.row, [...(byRow.get(p.row) || []), p]);
    const head = h("tr", {}, h("th", {}, "Model"), h("th", { class: "num" }, "Seeds"),
                   h("th", { class: "num" }, "Δ IoU at τ*"), h("th", { class: "num" }, "Δ AP"), h("th", { class: "num" }, "Holm p"));
    const rows = [...byRow].map(([key, runs]) => {
      const span = (values) => {
        const lo = Math.min(...values), hi = Math.max(...values);
        return lo === hi ? signed(lo) : `${signed(lo)} to ${signed(hi)}`;
      };
      return h("tr", {}, h("td", {}, label(key)), h("td", { class: "num" }, String(runs.length)),
        h("td", { class: "num" }, span(runs.map((r) => r.iou_tau.delta))),
        h("td", { class: "num" }, span(runs.map((r) => r.ap.delta))),
        h("td", { class: "num" }, `≥ ${fixed(Math.min(...runs.map((r) => r.iou_tau.p_holm)))}`));
    });
    table.replaceChildren(h("thead", {}, head), h("tbody", {}, rows));
    const trained = s1.protocol_gap.filter((p) => p.row !== "11").map((p) => p.iou_tau.delta);
    const noOpt = s1.protocol_gap.filter((p) => p.row === "11").map((p) => p.iou_tau.delta);
    const anySignificant = s1.protocol_gap.some((p) => p.iou_tau.p_holm < 0.05);
    note.textContent = `At τ*, the trained models gain at most ${signed(Math.max(...trained))}`
      + (noOpt.length ? `; the fusion model run without optical gains up to ${signed(Math.max(...noOpt))}` : "")
      + `. ${anySignificant ? "Some differences are significant." : "None is significant."} `
      + `Measured on the ${s1.protocol_gap[0].n} floods that have official test chips.`;
  }

  function conformalChart(container, note, run, events) {
    const W = widthOf(container), left = 82, right = 40, top = 26, rowH = 26;
    const H = top + rowH * events.length + 6;
    const max = Math.max(0.3, ...Object.values(run.fnr)) * 1.08;
    const x = linear(0, max, left, W - right);
    const svg = svgRoot(W, H, `Mean per-chip miss rate per flood; the target is ${run.alpha}`, "group");
    for (let t = 0; t <= max; t += 0.05) {
      svg.append(s("text", { x: x(t), y: top - 10, "text-anchor": "middle" }, t.toFixed(2)));
    }
    events.forEach((event, i) => {
      const v = run.fnr[event], y = top + rowH * i;
      const over = v > run.alpha;
      svg.append(s("text", { x: left - 10, y: y + rowH / 2 + 4, "text-anchor": "end", class: "label" }, event),
        withTip(s("rect", { x: x(0), y: y + 5, width: Math.max(x(v) - x(0), 1), height: rowH - 10, rx: 3, tabindex: 0,
                            "aria-label": `${event}: ${fixed(v)}`, style: { fill: over ? "var(--warn)" : "var(--good)" } }),
                `${event}: mean miss rate ${fixed(v)} (${over ? "above" : "within"} α = ${run.alpha})`));
    });
    svg.append(s("line", { x1: x(run.alpha), x2: x(run.alpha), y1: top - 4, y2: H, style: { stroke: "var(--text)", "stroke-dasharray": "4 3" } }),
               s("text", { x: x(run.alpha) + 4, y: H - 4, class: "label" }, `α = ${run.alpha}`));
    container.replaceChildren(svg);
    note.textContent = `${run.held} of ${run.folds} held-out floods stay within α for this run.`;
  }

  function registrationGrid(container, registration) {
    const SOURCE = { label: "from label edges", optical: "from SWIR", none: "not moved" };
    container.replaceChildren(...registration.map((r) => {
      // shift_pixels: out[r, c] = in[r + dy, c + dx], so content moves by (-dy, -dx); rows grow southwards.
      const mx = -r.dx * 9, my = -r.dy * 9;
      const svg = s("svg", { viewBox: "-36 -36 72 72", "aria-hidden": "true" },
        s("circle", { r: 9, style: { fill: "none", stroke: "var(--c-grid)" } }),
        s("circle", { r: 18, style: { fill: "none", stroke: "var(--c-grid)" } }),
        s("circle", { r: 27, style: { fill: "none", stroke: "var(--c-grid)" } }),
        s("line", { x1: -33, x2: 33, y1: 0, y2: 0, style: { stroke: "var(--c-grid)" } }),
        s("line", { y1: -33, y2: 33, x1: 0, x2: 0, style: { stroke: "var(--c-grid)" } }),
        s("text", { x: 0, y: -27, "text-anchor": "middle", style: { "font-size": "8px", fill: "var(--muted)" } }, "N"),
        r.dx === 0 && r.dy === 0
          ? s("circle", { r: 3.5, style: { fill: "var(--muted)" } })
          : s("line", { x1: 0, y1: 0, x2: mx, y2: my,
                        "marker-end": `url(#reg-arrow-${r.source === "label" ? "label" : "other"})`,
                        style: { stroke: r.source === "label" ? "var(--accent)" : "var(--warn)", "stroke-width": "3" } }));
      const moves = [r.dy ? `${Math.abs(r.dy)} ${r.dy > 0 ? "N" : "S"}` : "", r.dx ? `${Math.abs(r.dx)} ${r.dx > 0 ? "W" : "E"}` : ""]
        .filter(Boolean).join(" ");
      const where = moves ? `${moves} px` : "0 px";
      return h("div", { class: "arrow-card" }, svg, h("b", {}, r.event),
        h("small", {}, `${where} · ${SOURCE[r.source] || r.source}`));
    }));
    const marker = (id, color) => s("marker", { id, viewBox: "0 0 10 10", refX: 6, refY: 5, markerWidth: 4, markerHeight: 4,
                                                orient: "auto" }, s("path", { d: "M0 0 L10 5 L0 10 z", style: { fill: color } }));
    container.append(s("svg", { width: 0, height: 0, style: { position: "absolute" }, "aria-hidden": "true" },
      s("defs", {}, marker("reg-arrow-label", "var(--accent)"), marker("reg-arrow-other", "var(--warn)"))));
  }

  function changeChart(container, change) {
    const W = widthOf(container), left = 82, right = 24, top = 44, rowH = 26;
    const H = top + rowH * change.length + 22;
    const values = change.map((c) => c.median_dvv_db);
    const lo = Math.floor(Math.min(-4, ...values)), hi = Math.ceil(Math.max(1, ...values));
    const x = linear(lo, hi, left, W - right);
    const svg = svgRoot(W, H, "Median change in VV backscatter on labelled flood pixels, per event", "group");
    for (let t = lo; t <= hi; t += 1) {
      svg.append(s("text", { x: x(t), y: top - 10, "text-anchor": "middle" }, t > 0 ? `+${t}` : fixed(t, 0)));
    }
    change.forEach((c, i) => {
      const y = top + rowH * i, v = c.median_dvv_db, dropped = v <= -1;
      svg.append(s("text", { x: left - 10, y: y + rowH / 2 + 4, "text-anchor": "end", class: "label" }, c.event),
        withTip(s("rect", { x: Math.min(x(0), x(v)), y: y + 5, width: Math.max(Math.abs(x(v) - x(0)), 1), height: rowH - 10,
                            rx: 3, tabindex: 0, "aria-label": `${c.event}: ${signed(v, 2)} dB`,
                            style: { fill: dropped ? "var(--c-cems)" : "var(--c-otsu)" } }),
                `${c.event}: median ΔVV ${signed(v, 2)} dB; dark before ${fixed(c.dark_before, 2)}, after ${fixed(c.dark_after, 2)}`));
    });
    svg.append(s("line", { x1: x(0), x2: x(0), y1: top - 4, y2: H - 18, style: { stroke: "var(--muted)" } }),
               s("line", { x1: x(-1), x2: x(-1), y1: top - 4, y2: H - 18, style: { stroke: "var(--text)", "stroke-dasharray": "4 3" } }),
               s("text", { x: x(-1), y: H - 4, "text-anchor": "middle", class: "label" }, "−1 dB"),
               s("text", { x: left, y: 12 }, "median ΔVV on labelled flood (dB)"));
    container.replaceChildren(svg);
  }

  const drawFlood = chartSlot(floodChart), drawConformal = chartSlot(conformalChart), drawChange = chartSlot(changeChart);

  function initFloods(s1) {
    const label = (key) => s1.rows.find((r) => r.key === key)?.label ?? key;
    toggleChips(byId("flood-models"), s1.rows.map((r) => ({ key: r.key, label: r.label, color: s1Color(r.key) })),
                ["4", "9", "cmp_terramind_small"], (keys) => drawFlood(byId("flood-chart"), s1, keys));
    floodTable(byId("flood-table"), s1);
    gapList(byId("flood-gaps"), s1.gaps, label, "floods");
    protocolTable(byId("protocol-table"), byId("protocol-note"), s1);
    const select = byId("conformal-run");
    s1.conformal.forEach((c, i) => select.append(h("option", { value: String(i) }, `${label(c.row)} · seed ${c.seed}`)));
    const draw = () => drawConformal(byId("conformal-chart"), byId("conformal-note"), s1.conformal[Number(select.value)], s1.events);
    select.value = String(Math.max(0, s1.conformal.findIndex((c) => c.row === "9")));
    select.addEventListener("change", draw);
    draw();
    registrationGrid(byId("registration-grid"), s1.registration);
    drawChange(byId("change-chart"), s1.label_change);
  }

  // ------------------------------------------------------------------ 5 · Forlì
  function initSwipe(fo, facts) {
    const product = fo.products.find((p) => p.tag === facts.nearest(facts.optPass.tag));
    const when = timeLabel(utc(facts.optPass.time));
    const layers = {
      vv: { src: "assets/forli_vv.png", tag: `Sentinel-1 VV · ${when}`, alt: `Sentinel-1 VV radar image of Forlì, ${when} UTC` },
      fusion: { src: "assets/forli_fusion.png", tag: `floodfusion · ${when}`, alt: `floodfusion flood map (red) of Forlì, ${when} UTC` },
      cems: { src: "assets/forli_cems.png", tag: `CEMS · ${product.sensor} · ${timeLabel(utc(product.time))}`,
              alt: `Copernicus EMS delineation (blue) from ${product.sensor}, ${timeLabel(utc(product.time))} UTC` },
    };
    const left = byId("swipe-left"), right = byId("swipe-right"), input = byId("swipe-range");
    let pair = null;
    const move = () => {
      right.style.clipPath = `inset(0 0 0 ${input.value}%)`;
      byId("swipe-handle").style.left = `${input.value}%`;
      if (pair) {
        input.setAttribute("aria-valuetext",
          `${Math.round(input.value)} % ${pair[0].tag} on the left, ${Math.round(100 - input.value)} % ${pair[1].tag} on the right`);
      }
    };
    radioGroup([...document.querySelectorAll("#swipe-modes button")], (button) => {
      const l = layers[button.dataset.left], r = layers[button.dataset.right];
      pair = [l, r];
      move();
      Object.assign(left, { src: l.src, alt: l.alt });
      Object.assign(right, { src: r.src, alt: r.alt });
      byId("swipe-tag-left").textContent = l.tag;
      byId("swipe-tag-right").textContent = r.tag;
    });
    input.addEventListener("input", move);
    move();
  }

  function recessionChart(container, fo, facts, models) {
    const W = widthOf(container), H = 300, left = 44, right = 16, top = 34, bottom = 36;
    const products = [...fo.products].sort((a, b) => utc(a.time) - utc(b.time));
    const times = [...products.map((p) => utc(p.time)), ...fo.passes.map((p) => utc(p.time))];
    const day = 86400000;
    const t0 = Math.floor(Math.min(...times) / day) * day, t1 = Math.ceil(Math.max(...times) / day) * day;
    const values = [...Object.values(fo.area_km2.cems), ...Object.values(fo.area_km2.models).flatMap(Object.values)];
    const yMax = Math.ceil(Math.max(...values) / 20) * 20;
    const x = linear(t0, t1, left, W - right), y = linear(0, yMax, H - bottom, top);
    const svg = svgRoot(W, H, "Flooded area over time on the common region: CEMS delineations and floodfusion passes", "group");
    for (let v = 0; v <= yMax; v += 20) {
      svg.append(s("line", { x1: left, x2: W - right, y1: y(v), y2: y(v), style: { stroke: "var(--c-grid)" } }),
                 s("text", { x: left - 6, y: y(v) + 4, "text-anchor": "end" }, String(v)));
    }
    const dayStep = W < 460 ? 2 : 1;
    for (let t = t0, i = 0; t <= t1; t += day, i += 1) {
      if (i % dayStep === 0) svg.append(s("text", { x: x(t), y: H - bottom + 18, "text-anchor": "middle" }, dayLabel(t)));
    }
    svg.append(s("text", { x: left, y: 14 }, "flooded km²"));
    for (const p of fo.passes) {
      const px = x(utc(p.time)), partial = !facts.fullPasses.includes(p);
      svg.append(s("line", { x1: px, x2: px, y1: top - 6, y2: H - bottom, style: { stroke: "var(--border)", "stroke-dasharray": "3 4" } }),
                 withTip(s("text", { x: px, y: top - 10, "text-anchor": "middle", tabindex: 0, style: { "font-size": "10px" } },
                               partial ? "S1 (part)" : `S1 · S2 ${p.optical_tiles}/${p.tiles}`),
                         `Sentinel-1 pass ${timeLabel(utc(p.time))}, orbit ${p.orbit}: ${partial
                           ? `covered ${p.tiles} of ${Math.max(...fo.passes.map((q) => q.tiles))} tiles, scored on its own coverage only`
                           : `Sentinel-2 on ${p.optical_tiles} of ${p.tiles} tiles`}`));
    }
    const cemsPoints = products.map((p) => [x(utc(p.time)), y(fo.area_km2.cems[p.tag])]);
    svg.append(s("polyline", { points: cemsPoints.map((pt) => pt.join(",")).join(" "),
                               style: { fill: "none", stroke: "var(--c-cems)", "stroke-width": "2.5" } }));
    products.forEach((p, i) => {
      const [cx, cy] = cemsPoints[i];
      svg.append(withTip(s("rect", { x: cx - 6, y: cy - 6, width: 12, height: 12, rx: 2, tabindex: 0,
                                     "aria-label": `CEMS ${p.sensor}: ${fixed(fo.area_km2.cems[p.tag], 1)} km²`,
                                     style: { fill: "var(--c-cems)", stroke: "var(--surface)", "stroke-width": "2" } }),
                         `CEMS ${p.tag} · ${p.sensor} · ${timeLabel(utc(p.time))}: ${fixed(fo.area_km2.cems[p.tag], 1)} km²`));
    });
    for (const model of fo.models.filter((m) => models.has(m))) {
      const style = forliStyle(model);
      const pts = facts.fullPasses.filter((p) => fo.area_km2.models[model]?.[p.tag] != null)
        .map((p) => ({ p, cx: x(utc(p.time)), cy: y(fo.area_km2.models[model][p.tag]) }));
      svg.append(s("polyline", { points: pts.map((d) => `${d.cx},${d.cy}`).join(" "),
                                 style: { fill: "none", stroke: style.color, "stroke-width": "1.5", "stroke-dasharray": "5 4" } }));
      for (const d of pts) {
        const km2 = fo.area_km2.models[model][d.p.tag];
        svg.append(withTip(s("circle", { cx: d.cx, cy: d.cy, r: 6.5, class: "dot", tabindex: 0,
                                         "aria-label": `${style.label}, ${timeLabel(utc(d.p.time))}: ${fixed(km2, 1)} km²`,
                                         style: { fill: style.color } }),
                           `${style.label} · ${timeLabel(utc(d.p.time))}: ${fixed(km2, 1)} km²`));
      }
    }
    container.replaceChildren(svg);
  }

  function forliMatrix(table, fo, facts, model, region) {
    const products = [...fo.products].sort((a, b) => utc(a.time) - utc(b.time));
    const color = forliStyle(model).color;
    const head = h("tr", {}, h("th", {}, "Sentinel-1 pass"),
      ...products.map((p) => h("th", { class: "pass" }, dayLabel(utc(p.time)), h("small", {}, p.sensor))));
    const rows = fo.passes.map((pass) => {
      const nearest = facts.nearest(pass.tag);
      return h("tr", {},
        h("th", { class: "pass", scope: "row" }, timeLabel(utc(pass.time)),
          h("small", {}, `S2 on ${pass.optical_tiles}/${pass.tiles} tiles`)),
        ...products.map((p) => {
          const cell = fo.iou[region][model]?.[pass.tag]?.[p.tag];
          const v = cell?.iou_tau;
          const offset = fo.offset_h[pass.tag][p.tag];
          const td = h("td", { class: `cell${p.tag === nearest ? " nearest" : ""}`,
                               style: v == null ? {} : { background: `color-mix(in srgb, ${color} ${Math.round(v * 80)}%, transparent)` } },
            v == null ? "—" : fixed(v), h("small", {}, `${offset >= 0 ? "+" : MINUS}${Math.abs(Math.round(offset))} h`));
          return withTip(td, v == null
            ? `${p.tag}: not scored (the pass and this product share no scored pixels here)`
            : `${p.tag} (${p.sensor}), ${Math.abs(Math.round(offset))} h ${offset >= 0 ? "before" : "after"} the pass: IoU ${fixed(v)}`
              + (cell.iou_oracle != null ? `; at the best threshold ${fixed(cell.iou_oracle)}` : ""));
        }));
    });
    table.replaceChildren(h("thead", {}, head), h("tbody", {}, rows));
  }

  const drawRecession = chartSlot(recessionChart);

  function initForli(fo, facts) {
    initSwipe(fo, facts);
    toggleChips(byId("recession-models"), fo.models.map((m) => ({ key: m, ...forliStyle(m) })), fo.models,
                (models) => drawRecession(byId("recession-chart"), fo, facts, models));
    let model = fo.models[0], region = "common";
    const draw = () => forliMatrix(byId("forli-matrix"), fo, facts, model, region);
    const modelBox = byId("matrix-model");
    for (const m of fo.models) modelBox.append(h("button", { type: "button", role: "radio", "data-model": m }, forliStyle(m).label));
    radioGroup([...modelBox.querySelectorAll("button")], (b) => { model = b.dataset.model; draw(); });
    radioGroup([...document.querySelectorAll("#matrix-region button")], (b) => { region = b.dataset.region; draw(); });
  }

  // ------------------------------------------------------------------ 6 · synthetic
  const FACTORS = {
    wind_speed_ms: { label: "Wind speed", unit: "m/s",
                     text: "Wind roughens open water until it looks like land. A fixed threshold cannot adapt; the learned models partly can." },
    cloud_fraction: { label: "Cloud cover", unit: "",
                      text: "Cloud only removes optical. The fusion model falls back to radar, and the radar-only methods are untouched." },
    incidence: { label: "Incidence angle", unit: "°",
                 text: "Across Sentinel-1's range of viewing angles every method holds steady on these landscapes." },
    harvest_fraction: { label: "Harvested fields", unit: "",
                        text: "Fields harvested between the two images look like change. The change-based baseline loses a little; the learned models barely notice." },
  };

  function stressChart(container, factorData, index, showBand) {
    const keys = Object.keys(factorData);
    const numeric = keys.every((k) => !Number.isNaN(Number(k)));
    const series = Object.keys(factorData[keys[0]]);
    const W = widthOf(container), H = 310, left = 46, right = 18, top = 30, bottom = 40;
    const xs = numeric ? keys.map(Number) : keys.map((_, i) => i);
    const x = linear(Math.min(...xs), Math.max(...xs), left + 10, W - right - 10);
    const all = keys.flatMap((k) => series.map((name) => factorData[k][name].max));
    const yMax = Math.max(0.8, Math.ceil(Math.max(...all) * 10) / 10);
    const y = linear(0, yMax, H - bottom, top);
    const svg = svgRoot(W, H, "IoU of each model as one factor varies");
    for (let v = 0; v <= yMax + 1e-9; v += 0.2) {
      svg.append(s("line", { x1: left, x2: W - right, y1: y(v), y2: y(v), style: { stroke: "var(--c-grid)" } }),
                 s("text", { x: left - 8, y: y(v) + 4, "text-anchor": "end" }, v.toFixed(1)));
    }
    keys.forEach((k, i) => svg.append(s("text", { x: x(xs[i]), y: H - bottom + 18, "text-anchor": "middle" }, k)));
    svg.append(s("text", { x: left, y: 12 }, "IoU"));
    svg.append(s("line", { x1: x(xs[index]), x2: x(xs[index]), y1: top, y2: H - bottom,
                           style: { stroke: "var(--text)", "stroke-dasharray": "4 3", opacity: "0.6" } }));
    for (const name of series) {
      const { label, color } = stressStyle(name);
      const pts = keys.map((k, i) => ({ k, cx: x(xs[i]), d: factorData[k][name] }));
      if (showBand) {
        const upper = pts.map((p) => `${p.cx},${y(p.d.max)}`), lower = pts.map((p) => `${p.cx},${y(p.d.min)}`).reverse();
        svg.append(s("polygon", { points: [...upper, ...lower].join(" "), style: { fill: color, opacity: "0.1" } }));
      }
      svg.append(s("polyline", { points: pts.map((p) => `${p.cx},${y(p.d.mean)}`).join(" "),
                                 style: { fill: "none", stroke: color, "stroke-width": "2.5" } }));
      pts.forEach((p, i) => svg.append(withTip(
        s("circle", { cx: p.cx, cy: y(p.d.mean), r: i === index ? 6.5 : 4, class: "dot", style: { fill: color } }),
        `${label} at ${p.k}: ${fixed(p.d.mean)} [${fixed(p.d.min, 2)}, ${fixed(p.d.max, 2)}]`)));
    }
    container.replaceChildren(svg);
  }

  const drawStress = chartSlot(stressChart);

  function initStress(stress) {
    const results = stress.results;
    const factorBox = byId("stress-factors");
    const slider = byId("stress-slider"), band = byId("stress-band");
    let factor = Object.keys(results)[0];
    const draw = () => {
      const data = results[factor], keys = Object.keys(data), index = Number(slider.value);
      const meta = FACTORS[factor] || { label: factor, unit: "", text: "" };
      byId("stress-slider-label").textContent = `${meta.label}: ${keys[index]}${meta.unit ? ` ${meta.unit}` : ""}`;
      drawStress(byId("stress-chart"), data, index, band.checked);
      byId("stress-readout").replaceChildren(...Object.entries(data[keys[index]]).map(([name, d]) => {
        const { label, color } = stressStyle(name);
        return h("div", { class: "readout", style: { "--c": color } }, h("span", {}, label), h("b", {}, fixed(d.mean)),
                 h("span", {}, `${fixed(d.min, 2)} to ${fixed(d.max, 2)} over ${stress.landscape_seeds.length} landscapes`));
      }));
      const first = data[keys[0]], last = data[keys.at(-1)];
      const moves = Object.keys(first).map((name) => `${stressStyle(name).label} ${fixed(first[name].mean)} → ${fixed(last[name].mean)}`);
      byId("stress-insight").textContent = `${meta.text} From ${keys[0]} to ${keys.at(-1)}${meta.unit ? ` ${meta.unit}` : ""}: ${moves.join("; ")}.`;
    };
    for (const key of Object.keys(results)) {
      factorBox.append(h("button", { type: "button", role: "radio", "data-factor": key }, FACTORS[key]?.label ?? key));
    }
    radioGroup([...factorBox.querySelectorAll("button")], (b) => {
      factor = b.dataset.factor;
      slider.max = String(Object.keys(results[factor]).length - 1);
      slider.value = "0";
      draw();
    });
    slider.addEventListener("input", draw);
    band.addEventListener("change", draw);
  }

  const ABLATION_GROUP = (key) => {
    const n = Number(key);
    if (n <= 3) return "var(--c-phys)";
    if (n === 6) return "var(--c-mndwi)";
    if (n <= 5) return "var(--c-sar)";
    if (n === 13) return "var(--c-otsu)";
    return "var(--c-fusion)";
  };
  const ABLATION_NOTES = {
    6: "Scored on cloud-free test chips only, a different test set, so it is excluded from the paired comparisons.",
    11: "The row-9 checkpoint, evaluated with optical removed.",
    13: "Chips of the same basins in training and test: the score it reports is inflated on purpose.",
  };

  function ablationDetail(row) {
    const seeds = row.seeds.length ? `${row.seeds.length} seeds` : "no training";
    const cost = row.vram_gib ? ` · peak VRAM ${row.vram_gib.toFixed(2)} GiB` : "";
    const speed = row.ms_per_chip ? ` · ${row.ms_per_chip.toFixed(1)} ms per chip (${row.device.toUpperCase()})` : "";
    const parts = [h("h4", {}, `Row ${row.key}: ${row.label}`), h("p", {}, row.question),
      h("p", { class: "muted" }, `IoU ${fixed(row.iou)} [${fixed(row.ci[0], 2)}, ${fixed(row.ci[1], 2)}] · ${seeds}${cost}${speed}`)];
    if (ABLATION_NOTES[row.key]) parts.push(h("p", { class: "muted" }, ABLATION_NOTES[row.key]));
    byId("ablation-detail").replaceChildren(...parts);
  }

  function ablationChart(container, rows, selected, onSelect) {
    const lo = Math.floor(Math.min(...rows.map((r) => r.ci[0])) * 20) / 20;
    const hi = Math.ceil(Math.max(...rows.map((r) => r.ci[1])) * 20) / 20;
    const W = widthOf(container), left = Math.min(250, Math.round(W * 0.44)), right = 54, top = 24, rowH = 30;
    const maxChars = Math.floor((left - 18) / 7);
    const H = top + rowH * rows.length + 6;
    const x = linear(lo, hi, left, W - right);
    const svg = svgRoot(W, H, "Synthetic ablation: IoU of each row with its interval over test basins", "group");
    const step = W < 520 ? 0.1 : 0.05;
    for (let t = lo; t <= hi + 1e-9; t += step) {
      svg.append(s("line", { x1: x(t), x2: x(t), y1: top - 6, y2: H, style: { stroke: "var(--c-grid)" } }),
                 s("text", { x: x(t), y: top - 10, "text-anchor": "middle" }, t.toFixed(2)));
    }
    rows.forEach((row, i) => {
      const y = top + rowH * i + rowH / 2, color = ABLATION_GROUP(row.key);
      const name = `${row.key}. ${row.label}`;
      const group = s("g", { tabindex: 0, role: "button", "data-key": row.key, class: row.key === selected ? "selected-row" : null,
                             "aria-pressed": String(row.key === selected),
                             "aria-label": `Row ${row.key}, ${row.label}: IoU ${fixed(row.iou)}`, style: { cursor: "pointer" } },
        s("rect", { class: "bg", x: 0, y: y - rowH / 2, width: W, height: rowH, rx: 6, style: { fill: "transparent" } }),
        s("text", { x: left - 12, y: y + 4, "text-anchor": "end", class: "label" },
          name.length > maxChars ? `${name.slice(0, maxChars - 1)}…` : name),
        s("line", { x1: x(row.ci[0]), x2: x(row.ci[1]), y1: y, y2: y,
                    style: { stroke: color, "stroke-width": "3", "stroke-linecap": "round", opacity: "0.55" } }),
        s("circle", { cx: x(row.iou), cy: y, r: 6.5, class: "dot", style: { fill: color } }),
        s("text", { x: W - right + 8, y: y + 4, class: "label" }, fixed(row.iou)));
      group.addEventListener("click", () => onSelect(row, false));
      group.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(row, true); }
      });
      svg.append(group);
    });
    container.replaceChildren(svg);
  }

  const drawAblation = chartSlot(ablationChart);

  function initAblation(synthetic) {
    const rows = synthetic.rows.filter((r) => !r.supplementary);
    const container = byId("ablation-chart");
    const select = (row, keyboard = false) => {
      ablationDetail(row);
      drawAblation(container, rows, row.key, select);
      if (keyboard) container.querySelector(`[data-key="${row.key}"]`)?.focus();  // the redraw replaced the row
    };
    select(rows.find((r) => r.key === "9") || rows[0]);
  }

  // ------------------------------------------------------------------ start
  initTheme();
  initNav();
  initCopy();
  if (!DATA) {
    byId("no-data").hidden = false;
    return;
  }
  const facts = forliFacts(DATA.forli);
  applyBindings(bindings(DATA, facts));
  initTerms();
  initExplorer();
  initPipeline();
  initGate(DATA.synthetic.gate);
  initFloods(DATA.sen1floods11);
  initForli(DATA.forli, facts);
  initStress(DATA.synthetic.stress);
  initAblation(DATA.synthetic);
})();
