// floodfusion Live: the maps come from window.FLOODFUSION_LIVE, built by scripts/build_site.py from the live watcher's
// index (the live-data branch). A map's polygons are fetched from this site; the base map is OpenStreetMap's tiles.
(() => {
  "use strict";

  const DATA = window.FLOODFUSION_LIVE;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const TILE = 256;
  const TILE_URL = (z, x, y) => `https://tile.openstreetmap.org/${z}/${x}/${y}.png`;
  const MAX_ZOOM = 15;
  const MAX_TILES = 48;
  const byId = (id) => document.getElementById(id);

  function setAttrs(node, attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      if (value == null || value === false) continue;
      if (key === "text") node.textContent = value;
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
  const km2 = (x) => (x == null ? "—" : `${x.toLocaleString("en-GB", { maximumFractionDigits: x < 10 ? 2 : 1 })} km²`);
  const utc = (iso) => (iso ? `${iso.slice(0, 10)} ${iso.slice(11, 16)} UTC` : "—");
  const KIND = { showcase: "seen flood", live: "live", manual: "on request" };

  // ---------------------------------------------------------------- theme (as on the other pages)
  function theme() {
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

  // ---------------------------------------------------------------- Web Mercator
  const worldX = (lon, z) => ((lon + 180) / 360) * TILE * 2 ** z;
  function worldY(lat, z) {
    const phi = (Math.max(-85.0511, Math.min(85.0511, lat)) * Math.PI) / 180;
    return ((1 - Math.log(Math.tan(phi) + 1 / Math.cos(phi)) / Math.PI) / 2) * TILE * 2 ** z;
  }
  function fitZoom([west, south, east, north], width, height) {
    for (let z = MAX_ZOOM; z > 2; z -= 1) {
      const w = worldX(east, z) - worldX(west, z), hgt = worldY(south, z) - worldY(north, z);
      if (w * 1.15 <= width && hgt * 1.15 <= height) return z;
    }
    return 3;
  }

  // ---------------------------------------------------------------- one map
  let current = null;

  function ringPath(ring, project) {
    return ring.map(([lon, lat], i) => `${i ? "L" : "M"}${project(lon, lat).join(" ")}`).join("") + "Z";
  }

  async function drawFlood(entry, svg, project, token) {
    let features = 0;
    for (const url of entry.files) {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
      const geojson = await response.json();
      if (token !== current) return null;
      for (const feature of geojson.features || []) {
        const g = feature.geometry || {};
        const polygons = g.type === "Polygon" ? [g.coordinates] : g.type === "MultiPolygon" ? g.coordinates : [];
        for (const polygon of polygons) {
          svg.append(s("path", { class: "flood-poly", d: polygon.map((r) => ringPath(r, project)).join("") }));
          features += 1;
        }
      }
    }
    return features;
  }

  function render(entry) {
    const view = byId("map-view");
    const width = view.clientWidth || 640, height = view.clientHeight || 420;
    const z = fitZoom(entry.bbox, width, height);
    const [west, south, east, north] = entry.bbox;
    const cx = (worldX(west, z) + worldX(east, z)) / 2, cy = (worldY(north, z) + worldY(south, z)) / 2;
    const left = cx - width / 2, top = cy - height / 2;
    const project = (lon, lat) => [(worldX(lon, z) - left).toFixed(1), (worldY(lat, z) - top).toFixed(1)];
    const tiles = [];
    for (let x = Math.floor(left / TILE); x <= Math.floor((left + width) / TILE); x += 1) {
      for (let y = Math.floor(top / TILE); y <= Math.floor((top + height) / TILE); y += 1) {
        if (y < 0 || y >= 2 ** z || tiles.length >= MAX_TILES) continue;
        const wrapped = ((x % 2 ** z) + 2 ** z) % 2 ** z;
        tiles.push(h("img", { src: TILE_URL(z, wrapped, y), alt: "", loading: "lazy", decoding: "async",
                              style: `left:${(x * TILE - left).toFixed(1)}px;top:${(y * TILE - top).toFixed(1)}px` }));
      }
    }
    const [x0, y0] = project(west, north), [x1, y1] = project(east, south);
    const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, "aria-hidden": "true" },
      s("rect", { class: "aoi-box", x: x0, y: y0, width: (x1 - x0).toFixed(1), height: (y1 - y0).toFixed(1) }));
    view.replaceChildren(...tiles, svg);
    view.setAttribute("aria-label", `Flood mapped in ${entry.aoi_name || entry.name} on ${utc(entry.pass_time)}`);
    const token = (current = {});
    const note = byId("map-note");
    note.textContent = "Loading the flood polygons…";
    drawFlood(entry, svg, project, token).then((n) => {
      if (n != null) note.textContent = n ? `${n.toLocaleString("en-GB")} flood polygons.` : "No flood mapped on this pass.";
    }).catch((err) => { note.textContent = `The flood polygons could not be loaded (${err.message}).`; });
  }

  function details(entry) {
    const rows = [
      ["Flood", entry.code ? `${entry.name} (${entry.code})` : "A manual request"],
      ["Area", entry.aoi_name ? `${entry.aoi_name} (AOI ${entry.aoi})` : entry.bbox.map((v) => v.toFixed(3)).join(", ")],
      ["Sentinel-1 pass", `${utc(entry.pass_time)} · orbit ${entry.orbit} ${entry.orbit_state}`],
      ["Flood mapped", `${km2(entry.flood_km2)} of ${km2(entry.valid_km2)} mapped`],
      ["Map made", entry.kind === "live" ? `${utc(entry.mapped_at)}, ${entry.delay_hours} h after the pass`
        : utc(entry.mapped_at)],
      ["Tiles", entry.skipped_tiles ? `${entry.tiles} mapped, ${entry.skipped_tiles} without a usable pass` : `${entry.tiles}`],
    ];
    const links = entry.files.map((url, i) => h("a", { href: url, download: "" }, `flood polygons${entry.files.length > 1 ? ` ${i + 1}` : ""}`));
    const items = entry.files.map((url) => url.replace(/flood\.geojson$/, "item.json"));
    byId("map-details").replaceChildren(
      ...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]),
      h("dt", {}, "Downloads"),
      h("dd", {}, ...links.flatMap((a, i) => [i ? " · " : "", a, " · ", h("a", { href: items[i] }, "STAC item")])));
  }

  function select(entry, button) {
    for (const b of byId("map-list").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === button));
    details(entry);
    render(entry);
  }

  function list(maps) {
    const ul = byId("map-list");
    const buttons = maps.map((m) => {
      const button = h("button", { type: "button", "aria-pressed": "false" },
        h("span", { class: "map-title" }, m.aoi_name ? `${m.aoi_name}` : "Manual map"),
        h("span", { class: "map-meta" }, `${m.code || ""} ${utc(m.pass_time)}`.trim()),
        h("span", { class: "map-meta" }, km2(m.flood_km2), " · ", h("span", { class: "badge" }, KIND[m.kind] || m.kind)));
      button.addEventListener("click", () => select(m, button));
      return button;
    });
    ul.replaceChildren(...buttons.map((b) => h("li", {}, b)));
    return buttons;
  }

  function stats(data) {
    const codes = new Set(data.maps.map((m) => m.code).filter(Boolean));
    const cards = [
      [String(data.maps.length), "maps published"],
      [String(codes.size), "Copernicus floods"],
      [data.updated_at ? utc(data.updated_at) : "not yet", "last run"],
      [data.blind_floods ? "on" : "off", "blind floods"],
    ];
    byId("live-stats").replaceChildren(...cards.map(([v, k]) =>
      h("div", { class: "stat" }, h("span", { class: "stat-value" }, v), h("span", { class: "stat-label" }, k))));
    byId("blind-status").textContent = data.blind_floods
      ? "Blind floods are on: new Copernicus flood activations are mapped as they come, the model's map alone."
      : "Blind floods are off: the job requests no activation from EMSR899 on and maps no pass from 18 July 2026 on.";
  }

  function main() {
    theme();
    if (!DATA) { byId("no-data").hidden = false; return; }
    stats(DATA);
    if (!DATA.maps.length) { byId("no-maps").hidden = false; byId("live-panel").hidden = true; return; }
    const buttons = list(DATA.maps);
    select(DATA.maps[0], buttons[0]);
    let pending = null;
    window.addEventListener("resize", () => {
      clearTimeout(pending);
      pending = setTimeout(() => {
        const i = buttons.findIndex((b) => b.getAttribute("aria-pressed") === "true");
        if (i >= 0) render(DATA.maps[i]);
      }, 200);
    });
  }

  main();
})();
