#!/usr/bin/env python3
"""Generate an interactive browser tool for classifying head-echo measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "reports" / "head_echo_gaussian_fit_summary_retained_ext0p5.csv"
DEFAULT_OUT = ROOT / "reports" / "figures" / "head_echo_measurement_selector.html"

STATION_LABELS = {
    "beverly": "Beverly, MA",
    "clinton": "Clinton, MA",
    "pomfret": "Pomfret, VT",
}


def load_measurements(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    df = df[df["success"].astype(str).str.lower() == "true"].copy()
    numeric = [
        "idx",
        "time_rel_s",
        "time_unix",
        "fit_range_m",
        "seed_range_m",
        "range_correction_m",
        "fit_doppler_hz",
        "seed_doppler_hz",
        "doppler_correction_hz",
        "sigma_range_m",
        "sigma_doppler_hz",
        "range_doppler_correlation",
        "amplitude_db",
        "cost",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["station", "time_rel_s", "idx"])
    rows = []
    for row in df.to_dict(orient="records"):
        rows.append(
            {
                "station": row["station"],
                "station_label": STATION_LABELS.get(row["station"], row["station"]),
                "idx": int(row["idx"]),
                "time_rel_s": float(row["time_rel_s"]),
                "time_unix": float(row["time_unix"]),
                "range_km": float(row["fit_range_m"]) / 1000.0,
                "seed_range_km": float(row["seed_range_m"]) / 1000.0,
                "range_correction_m": float(row["range_correction_m"]),
                "doppler_hz": float(row["fit_doppler_hz"]),
                "seed_doppler_hz": float(row["seed_doppler_hz"]),
                "doppler_correction_hz": float(row["doppler_correction_hz"]),
                "sigma_range_km": float(row["sigma_range_m"]) / 1000.0,
                "sigma_doppler_hz": float(row["sigma_doppler_hz"]),
                "rho": float(row["range_doppler_correlation"]),
                "amplitude_db": float(row["amplitude_db"]),
                "cost": float(row["cost"]),
            }
        )
    return rows


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Head Echo Measurement Selector</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f5;
      --panel: #ffffff;
      --ink: #181818;
      --muted: #666;
      --line: #d6d6d0;
      --real: #16803c;
      --reject: #b8202e;
      --unknown: #7c7c86;
      --select: #f1a208;
    }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      padding: 14px 18px 10px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 {
      font-size: 20px;
      line-height: 1.2;
      margin: 0 0 10px;
      font-weight: 680;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    button, select, label.file {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 7px 10px;
      font-size: 14px;
      line-height: 1.1;
      cursor: pointer;
    }
    button.primary { border-color: #1f5fbf; background: #1f5fbf; color: #fff; }
    button.real { border-color: var(--real); color: var(--real); }
    button.reject { border-color: var(--reject); color: var(--reject); }
    button.unknown { color: var(--muted); }
    button:disabled { opacity: 0.45; cursor: default; }
    input[type="file"] { display: none; }
    main { padding: 14px 18px 24px; }
    .status {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 8px 0 12px;
      color: var(--muted);
      font-size: 14px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
    }
    .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
    .dot.real { background: var(--real); }
    .dot.reject { background: var(--reject); }
    .dot.unknown { background: var(--unknown); }
    .dot.selected { background: var(--select); }
    .plots {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    .plot-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px;
    }
    .plot {
      width: 100%;
      height: 390px;
    }
    .help {
      margin-top: 12px;
      max-width: 980px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    code {
      background: #eeeeea;
      border-radius: 4px;
      padding: 1px 4px;
    }
  </style>
</head>
<body>
  <header>
    <h1>Head-echo range/Doppler measurement selector</h1>
    <div class="toolbar">
      <button id="prevStation">Previous Station</button>
      <select id="stationSelect"></select>
      <button id="nextStation">Next Station</button>
      <button id="markReal" class="real">Mark Selected Real</button>
      <button id="markReject" class="reject">Reject Selected</button>
      <button id="markUnknown" class="unknown">Set Selected Unknown</button>
      <button id="clearSelection">Clear Selection</button>
      <button id="exportCsv" class="primary">Export CSV</button>
      <button id="exportJson">Export JSON</button>
      <label class="file">Import JSON<input id="importJson" type="file" accept="application/json,.json" /></label>
      <button id="resetAll">Reset All</button>
    </div>
  </header>
  <main>
    <div id="status" class="status"></div>
    <div class="plots">
      <div class="plot-card"><div id="rangePlot" class="plot"></div></div>
      <div class="plot-card"><div id="dopplerPlot" class="plot"></div></div>
    </div>
    <div class="help">
      Click points to add/remove them from the current selection. Drag box/lasso on
      either panel to select points. Then use <code>Mark Selected Real</code>,
      <code>Reject Selected</code>, or <code>Set Selected Unknown</code>. Your
      work is saved in browser local storage. Export CSV when you are happy.
    </div>
  </main>
  <script>
    const rows = __ROWS_JSON__;
    const storageKey = "new_england_bolide_head_echo_selection_v1";
    const stationOrder = [...new Set(rows.map(r => r.station))];
    const stationLabels = Object.fromEntries(rows.map(r => [r.station, r.station_label]));
    const colors = {unknown: "#7c7c86", real: "#16803c", reject: "#b8202e"};
    let station = stationOrder[0];
    let state = {};
    let selected = new Set();

    function rowKey(r) { return `${r.station}:${r.idx}`; }

    function defaultState() {
      const out = {};
      for (const r of rows) out[rowKey(r)] = "unknown";
      return out;
    }

    function loadState() {
      state = defaultState();
      try {
        const saved = JSON.parse(localStorage.getItem(storageKey) || "{}");
        for (const [key, val] of Object.entries(saved)) {
          if (key in state && ["unknown", "real", "reject"].includes(val)) state[key] = val;
        }
      } catch (_) {}
    }

    function saveState() {
      localStorage.setItem(storageKey, JSON.stringify(state));
    }

    function stationRows() {
      return rows.filter(r => r.station === station);
    }

    function traceFor(kind, yField, yErrField, title) {
      const sr = stationRows();
      return ["unknown", "real", "reject"].map(cls => {
        const rr = sr.filter(r => state[rowKey(r)] === cls);
        return {
          type: "scattergl",
          mode: "markers",
          name: cls,
          x: rr.map(r => r.time_rel_s),
          y: rr.map(r => r[yField]),
          customdata: rr.map(r => [r.station, r.idx, rowKey(r), r.amplitude_db, r.cost, r.range_correction_m, r.doppler_correction_hz]),
          marker: {
            size: rr.map(r => selected.has(rowKey(r)) ? 11 : 7),
            color: selected.has(rowKey(rr[0] || {})) ? undefined : colors[cls],
            line: {
              width: rr.map(r => selected.has(rowKey(r)) ? 2 : 0.5),
              color: rr.map(r => selected.has(rowKey(r)) ? "#f1a208" : "#ffffff")
            }
          },
          error_y: {
            type: "data",
            array: rr.map(r => r[yErrField]),
            visible: true,
            thickness: 0.8,
            color: colors[cls],
          },
          hovertemplate:
            `${stationLabels[station]}<br>` +
            `class=${cls}<br>` +
            `t=%{x:.3f} s<br>` +
            `${title}=%{y:.3f}<br>` +
            `amp=%{customdata[3]:.2f} dB<br>` +
            `cost=%{customdata[4]:.3f}<br>` +
            `range corr=%{customdata[5]:+.1f} m<br>` +
            `Doppler corr=%{customdata[6]:+.1f} Hz<br>` +
            `idx=%{customdata[1]}<extra></extra>`
        };
      });
    }

    function plotLayout(yTitle) {
      return {
        margin: {l: 72, r: 20, t: 30, b: 48},
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        hovermode: "closest",
        dragmode: "lasso",
        legend: {orientation: "h", x: 0, y: 1.12},
        xaxis: {title: "Seconds after 18:06:20 UTC", zeroline: false},
        yaxis: {title: yTitle, zeroline: false},
      };
    }

    function updateStatus() {
      const sr = stationRows();
      const counts = {unknown: 0, real: 0, reject: 0};
      for (const r of sr) counts[state[rowKey(r)]]++;
      document.getElementById("status").innerHTML = `
        <span class="pill"><strong>${stationLabels[station]}</strong></span>
        <span class="pill"><span class="dot unknown"></span>unknown ${counts.unknown}</span>
        <span class="pill"><span class="dot real"></span>real ${counts.real}</span>
        <span class="pill"><span class="dot reject"></span>rejected ${counts.reject}</span>
        <span class="pill"><span class="dot selected"></span>selected ${selected.size}</span>
      `;
    }

    function render() {
      updateStatus();
      const rangeData = traceFor("range", "range_km", "sigma_range_km", "range km");
      const dopplerData = traceFor("doppler", "doppler_hz", "sigma_doppler_hz", "Doppler Hz");
      Plotly.react("rangePlot", rangeData, plotLayout("Bistatic range (km)"), {responsive: true, displaylogo: false});
      Plotly.react("dopplerPlot", dopplerData, plotLayout("Doppler (Hz)"), {responsive: true, displaylogo: false});
      wirePlot("rangePlot");
      wirePlot("dopplerPlot");
    }

    function wirePlot(id) {
      const el = document.getElementById(id);
      el.removeAllListeners?.("plotly_click");
      el.removeAllListeners?.("plotly_selected");
      el.on("plotly_click", ev => {
        for (const p of ev.points || []) {
          const key = p.customdata?.[2];
          if (!key) continue;
          if (selected.has(key)) selected.delete(key);
          else selected.add(key);
        }
        render();
      });
      el.on("plotly_selected", ev => {
        selected.clear();
        for (const p of ev?.points || []) {
          const key = p.customdata?.[2];
          if (key) selected.add(key);
        }
        render();
      });
    }

    function setSelected(value) {
      for (const key of selected) {
        if (key in state) state[key] = value;
      }
      selected.clear();
      saveState();
      render();
    }

    function exportRows(format) {
      const out = rows.map(r => ({
        station: r.station,
        idx: r.idx,
        time_rel_s: r.time_rel_s,
        fit_range_m: r.range_km * 1000.0,
        fit_doppler_hz: r.doppler_hz,
        sigma_range_m: r.sigma_range_km * 1000.0,
        sigma_doppler_hz: r.sigma_doppler_hz,
        amplitude_db: r.amplitude_db,
        cost: r.cost,
        selection: state[rowKey(r)],
        real: state[rowKey(r)] === "real",
      }));
      let blob;
      let filename;
      if (format === "json") {
        blob = new Blob([JSON.stringify(out, null, 2)], {type: "application/json"});
        filename = "head_echo_measurement_selection.json";
      } else {
        const cols = Object.keys(out[0]);
        const csv = [cols.join(",")].concat(out.map(r => cols.map(c => JSON.stringify(r[c] ?? "")).join(","))).join("\n");
        blob = new Blob([csv], {type: "text/csv"});
        filename = "head_echo_measurement_selection.csv";
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    function setupControls() {
      const select = document.getElementById("stationSelect");
      for (const st of stationOrder) {
        const opt = document.createElement("option");
        opt.value = st;
        opt.textContent = stationLabels[st];
        select.appendChild(opt);
      }
      select.value = station;
      select.addEventListener("change", () => {
        station = select.value;
        selected.clear();
        render();
      });
      document.getElementById("prevStation").onclick = () => {
        const i = stationOrder.indexOf(station);
        station = stationOrder[(i - 1 + stationOrder.length) % stationOrder.length];
        select.value = station;
        selected.clear();
        render();
      };
      document.getElementById("nextStation").onclick = () => {
        const i = stationOrder.indexOf(station);
        station = stationOrder[(i + 1) % stationOrder.length];
        select.value = station;
        selected.clear();
        render();
      };
      document.getElementById("markReal").onclick = () => setSelected("real");
      document.getElementById("markReject").onclick = () => setSelected("reject");
      document.getElementById("markUnknown").onclick = () => setSelected("unknown");
      document.getElementById("clearSelection").onclick = () => { selected.clear(); render(); };
      document.getElementById("exportCsv").onclick = () => exportRows("csv");
      document.getElementById("exportJson").onclick = () => exportRows("json");
      document.getElementById("resetAll").onclick = () => {
        if (confirm("Reset all classifications to unknown?")) {
          state = defaultState();
          selected.clear();
          saveState();
          render();
        }
      };
      document.getElementById("importJson").addEventListener("change", ev => {
        const file = ev.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
          const imported = JSON.parse(reader.result);
          for (const row of imported) {
            const key = `${row.station}:${row.idx}`;
            const val = row.selection || (row.real ? "real" : "unknown");
            if (key in state && ["unknown", "real", "reject"].includes(val)) state[key] = val;
          }
          saveState();
          render();
        };
        reader.readAsText(file);
      });
    }

    loadState();
    setupControls();
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    rows = load_measurements(args.summary)
    html = HTML_TEMPLATE.replace("__ROWS_JSON__", json.dumps(rows, separators=(",", ":")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(args.out)
    print(f"{len(rows)} measurements")


if __name__ == "__main__":
    main()
