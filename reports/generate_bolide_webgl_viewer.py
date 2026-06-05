#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from station_config import STATION_ORDER, load_all_station_configs, short_station_name, transmitter_config


DEFAULT_HEAD = "products/aod_position_from_merged/aod_position_summary.json"
DEFAULT_TRAIL = "products/specular_trail_positions/specular_trail_position_summary.json"
DEFAULT_BALLISTIC = "products/ballistic_head_trail_fit/ballistic_head_trail_fit.json"
DEFAULT_OUT = "reports/figures/bolide_webgl_volume_viewer.html"
DEFAULT_HAYSTACK_LOGO = "/Users/j/Desktop/MIT_Haystack_Observatory_logo.svg"
DEFAULT_UIT_LOGO = "/Users/j/Desktop/Screenshot 2026-06-05 at 22.28.53.png"
DEFAULT_IAP_LOGO = "/Users/j/Desktop/logo_en.svg"
NE_SHAPE_ROOT = Path.home() / ".local/share/cartopy/shapefiles/natural_earth"


def read_rows(path: Path, key: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get(key, [])
    if not rows:
        raise SystemExit(f"No {key!r} rows found in {path}")
    return rows


def data_uri(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def finite_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def simplify_polyline(points: list[list[float]], stride: int = 2) -> list[list[float]]:
    if len(points) <= 2:
        return points
    keep = points[::stride]
    if keep[-1] != points[-1]:
        keep.append(points[-1])
    return keep


def read_shp_lines(path: Path, bbox: tuple[float, float, float, float]) -> list[list[list[float]]]:
    if not path.exists():
        return []
    min_lon, min_lat, max_lon, max_lat = bbox
    out: list[list[list[float]]] = []
    data = path.read_bytes()
    offset = 100
    while offset + 8 <= len(data):
        _, content_words = struct.unpack(">2i", data[offset : offset + 8])
        content_len = content_words * 2
        start = offset + 8
        end = start + content_len
        offset = end
        if end > len(data) or content_len < 44:
            continue
        shape_type = struct.unpack("<i", data[start : start + 4])[0]
        if shape_type not in (3, 5):
            continue
        sxmin, symin, sxmax, symax = struct.unpack("<4d", data[start + 4 : start + 36])
        if sxmax < min_lon or sxmin > max_lon or symax < min_lat or symin > max_lat:
            continue
        num_parts, num_points = struct.unpack("<2i", data[start + 36 : start + 44])
        if num_parts <= 0 or num_points <= 0:
            continue
        parts0 = start + 44
        points0 = parts0 + 4 * num_parts
        if points0 + 16 * num_points > end:
            continue
        parts = list(struct.unpack(f"<{num_parts}i", data[parts0:points0]))
        parts.append(num_points)
        coords = [
            struct.unpack("<2d", data[points0 + 16 * i : points0 + 16 * i + 16])
            for i in range(num_points)
        ]
        for a, b in zip(parts[:-1], parts[1:]):
            segment = []
            for lon, lat in coords[a:b]:
                if min_lon - 0.5 <= lon <= max_lon + 0.5 and min_lat - 0.5 <= lat <= max_lat + 0.5:
                    segment.append([round(lon, 5), round(lat, 5)])
                elif segment:
                    if len(segment) > 1:
                        out.append(simplify_polyline(segment))
                    segment = []
            if len(segment) > 1:
                out.append(simplify_polyline(segment))
    return out


def load_map_lines(bbox: tuple[float, float, float, float]) -> dict[str, list[list[list[float]]]]:
    physical = NE_SHAPE_ROOT / "physical"
    cultural = NE_SHAPE_ROOT / "cultural"
    state_lines = read_shp_lines(cultural / "ne_10m_admin_1_states_provinces_lines.shp", bbox)
    if not state_lines:
        state_lines = [
            [[-73.50, 42.04], [-72.82, 42.04], [-72.13, 42.03], [-71.80, 42.01], [-71.50, 42.01], [-71.38, 41.89]],
            [[-71.38, 41.89], [-71.38, 41.64], [-71.79, 41.33]],
            [[-71.50, 42.01], [-71.20, 42.02], [-70.82, 42.03], [-70.66, 42.05]],
            [[-72.55, 42.73], [-71.80, 42.72], [-71.19, 42.74], [-70.82, 42.88]],
            [[-70.82, 42.88], [-70.73, 43.08], [-70.75, 43.35], [-70.82, 43.56]],
        ]
    return {
        "coast": read_shp_lines(physical / "ne_10m_coastline.shp", bbox),
        "states": state_lines,
        "rivers": read_shp_lines(physical / "ne_10m_rivers_lake_centerlines.shp", bbox),
    }


def project_rows(rows: list[dict], kind: str) -> list[dict]:
    out = []
    for row in rows:
        lat = finite_float(row, "lat_deg", math.nan)
        lon = finite_float(row, "lon_deg", math.nan)
        alt_m = finite_float(row, "alt_m", math.nan)
        unix = finite_float(row, "unix_time", math.nan)
        if not all(math.isfinite(v) for v in (lat, lon, alt_m, unix)):
            continue
        out.append(
            {
                "kind": kind,
                "station": row.get("station", ""),
                "t": unix,
                "lat": lat,
                "lon": lon,
                "altKm": alt_m / 1000.0,
                "snr": finite_float(row, "snr_db", 0.0),
                "doppler": finite_float(row, "doppler_hz", 0.0),
                "rangeKm": finite_float(row, "range_km", finite_float(row, "bistatic_range_m", 0.0) / 1000.0),
                "sigmaKm": finite_float(row.get("uncertainty", {}), "position_sigma_3d_m", 0.0) / 1000.0,
            }
        )
    return out


def local_xyz_km(lon: float, lat: float, alt_km: float, center: dict[str, float]) -> tuple[float, float, float]:
    earth_km = 6371.0
    lat0 = math.radians(center["lat"])
    lon0 = math.radians(center["lon"])
    x = earth_km * math.cos(lat0) * (math.radians(lon) - lon0)
    z = -earth_km * (math.radians(lat) - lat0)
    curve = -(x * x + z * z) / (2.0 * earth_km)
    return (x, alt_km + curve, z)


def load_ballistic(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model", {})
    if not model.get("time_unix"):
        raise SystemExit(f"No ballistic model time series found in {path}")
    return payload


def ballistic_trajectory_rows(ballistic: dict, t0_unix: float) -> list[dict]:
    model = ballistic["model"]
    rows = []
    for t, lat, lon, alt_m, speed in zip(
        model["time_unix"],
        model["lat_deg"],
        model["lon_deg"],
        model["alt_m"],
        model.get("speed_mps", [math.nan] * len(model["time_unix"])),
    ):
        rows.append(
            {
                "t": float(t) - t0_unix,
                "t_unix": float(t),
                "lat": float(lat),
                "lon": float(lon),
                "altKm": float(alt_m) / 1000.0,
                "speedKms": float(speed) / 1000.0 if math.isfinite(float(speed)) else math.nan,
            }
        )
    return rows


def annotate_by_ballistic_path(
    points: list[dict],
    trajectory: list[dict],
    center: dict[str, float],
    max_residual_km: float,
) -> tuple[list[dict], list[dict]]:
    traj_xyz = [local_xyz_km(row["lon"], row["lat"], row["altKm"], center) for row in trajectory]
    kept = []
    rejected = []
    for point in points:
        px, py, pz = local_xyz_km(point["lon"], point["lat"], point["altKm"], center)
        best = None
        for i in range(len(traj_xyz) - 1):
            ax, ay, az = traj_xyz[i]
            bx, by, bz = traj_xyz[i + 1]
            vx, vy, vz = bx - ax, by - ay, bz - az
            wx, wy, wz = px - ax, py - ay, pz - az
            vv = vx * vx + vy * vy + vz * vz
            u = 0.0 if vv <= 0.0 else max(0.0, min(1.0, (wx * vx + wy * vy + wz * vz) / vv))
            qx, qy, qz = ax + u * vx, ay + u * vy, az + u * vz
            residual = math.sqrt((px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2)
            t_path = trajectory[i]["t"] + u * (trajectory[i + 1]["t"] - trajectory[i]["t"])
            candidate = (residual, t_path, i, u)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            rejected.append(point)
            continue
        residual, t_path, seg_i, seg_u = best
        enriched = dict(point)
        enriched["t_measure"] = point["t"]
        enriched["t"] = t_path
        enriched["t_path"] = t_path
        enriched["pathResidualKm"] = residual
        enriched["pathSegment"] = seg_i
        enriched["pathSegmentU"] = seg_u
        if residual <= max_residual_km:
            kept.append(enriched)
        else:
            rejected.append(enriched)
    return kept, rejected


def strewn_field_rows(ballistic: dict, t0_unix: float) -> dict:
    impact = ballistic.get("fragment_impact", {})
    rows = []
    for sample in impact.get("impact_samples", []):
        llh = sample.get("impact_llh", [])
        if len(llh) < 3:
            continue
        rows.append(
            {
                "lat": float(llh[0]),
                "lon": float(llh[1]),
                "altKm": float(llh[2]) / 1000.0,
                "diameterM": finite_float(sample, "fragment_diameter_m", 0.0),
                "impactSpeedMps": finite_float(sample, "impact_speed_mps", 0.0),
                "fallDurationS": finite_float(sample, "t_impact_unix", t0_unix) - finite_float(sample, "t_break_unix", t0_unix),
                "tBreak": finite_float(sample, "t_break_unix", t0_unix) - t0_unix,
                "tImpact": finite_float(sample, "t_impact_unix", t0_unix) - t0_unix,
            }
        )
    nominal = impact.get("nominal", {})
    nominal_llh = nominal.get("impact_llh", [])
    nominal_row = None
    if len(nominal_llh) >= 3:
        nominal_row = {
            "lat": float(nominal_llh[0]),
            "lon": float(nominal_llh[1]),
            "altKm": float(nominal_llh[2]) / 1000.0,
            "diameterM": finite_float(nominal, "fragment_diameter_m", 0.0),
            "impactSpeedMps": finite_float(nominal, "impact_speed_mps", 0.0),
            "fallDurationS": finite_float(nominal, "t_impact_unix", t0_unix) - finite_float(nominal, "t_break_unix", t0_unix),
            "tBreak": finite_float(nominal, "t_break_unix", t0_unix) - t0_unix,
            "tImpact": finite_float(nominal, "t_impact_unix", t0_unix) - t0_unix,
        }
    frag_path = []
    for t, llh in zip(nominal.get("fragment_time_unix", []), nominal.get("fragment_llh", [])):
        if len(llh) >= 3:
            frag_path.append({"t": float(t) - t0_unix, "lat": float(llh[0]), "lon": float(llh[1]), "altKm": float(llh[2]) / 1000.0})
    return {
        "samples": rows,
        "nominal": nominal_row,
        "nominalPath": frag_path,
        "breakup": frag_path[0] if frag_path else None,
        "summary": impact.get("summary", {}),
    }


def station_marker_rows(data_dir: Path | str) -> list[dict]:
    configs = load_all_station_configs(data_dir)
    tx = transmitter_config(data_dir)
    rows = [
        {
            "kind": "tx",
            "id": "haystack",
            "label": "TX Haystack",
            "lat": float(tx.gps_tx[0]),
            "lon": float(tx.gps_tx[1]),
            "altKm": float(tx.gps_tx[2]) / 1000.0,
        }
    ]
    for station in STATION_ORDER:
        cfg = configs[station]
        rows.append(
            {
                "kind": "rx",
                "id": station,
                "label": f"RX {short_station_name(station)}",
                "lat": float(cfg.gps_rx[0]),
                "lon": float(cfg.gps_rx[1]),
                "altKm": float(cfg.gps_rx[2]) / 1000.0,
            }
        )
    return rows


def build_html(asset: dict) -> str:
    asset_json = json.dumps(asset, separators=(",", ":"), allow_nan=False)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cape Cod Bay Bolide May 30 2026</title>
<style>
html,body{{margin:0;height:100%;overflow:hidden;background:#05070b;color:#eaf5ff;font:13px/1.4 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}}
canvas{{display:block;width:100vw;height:100vh;background:radial-gradient(circle at 50% 38%,#15202d 0,#05070b 58%,#020307 100%)}}
.hud{{position:fixed;left:16px;right:16px;top:14px;display:flex;gap:12px;align-items:flex-start;justify-content:space-between;pointer-events:none}}
.panel{{pointer-events:auto;background:rgba(4,8,14,.72);border:1px solid rgba(150,210,255,.22);box-shadow:0 18px 55px rgba(0,0,0,.35);backdrop-filter:blur(10px);border-radius:8px;padding:12px 14px}}
.titlepanel{{display:grid;grid-template-columns:minmax(280px,560px) auto;gap:16px;align-items:center}}
h1{{font-size:15px;line-height:1.15;margin:0 0 7px 0;font-weight:700;letter-spacing:0}}
.meta{{color:#aac1d3;font-size:12px;max-width:560px}}
.logos{{display:flex;gap:10px;align-items:center;justify-content:flex-end}}
.logo{{height:46px;max-width:160px;object-fit:contain;background:rgba(255,255,255,.94);border:1px solid rgba(255,255,255,.28);border-radius:6px;padding:7px 9px;box-shadow:0 10px 26px rgba(0,0,0,.22)}}
.logo.uit{{max-width:70px}}
.logo.iap{{max-width:170px}}
.controls{{display:grid;grid-template-columns:auto 220px 190px;gap:9px;align-items:center;min-width:535px}}
button{{height:30px;border-radius:6px;border:1px solid rgba(175,225,255,.3);background:#0f1b25;color:#eaf5ff;cursor:pointer}}
button.active{{background:#1a5c7a;border-color:#57d0ff}}
input[type=range]{{width:220px;accent-color:#58d7ff}}
.readout{{font-variant-numeric:tabular-nums;color:#d6f4ff;min-width:184px;text-align:right}}
.legend{{position:fixed;left:16px;bottom:16px;display:flex;gap:10px;align-items:center;pointer-events:none;background:rgba(4,8,14,.65);border:1px solid rgba(150,210,255,.16);border-radius:8px;padding:9px 11px;color:#bed3df}}
.dot{{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px;box-shadow:0 0 13px currentColor}}
.help{{position:fixed;right:16px;bottom:16px;color:#9fb2c0;background:rgba(4,8,14,.58);border:1px solid rgba(150,210,255,.14);border-radius:8px;padding:8px 10px;pointer-events:none}}
.altitude{{position:fixed;right:16px;bottom:54px;color:#d6f4ff;background:rgba(4,8,14,.66);border:1px solid rgba(150,210,255,.16);border-radius:8px;padding:8px 10px;pointer-events:none;font-variant-numeric:tabular-nums}}
.label-layer{{position:fixed;inset:0;pointer-events:none;overflow:hidden}}
.station-label{{position:absolute;transform:translate(-50%,-140%);padding:3px 6px;border-radius:5px;background:rgba(3,7,12,.72);border:1px solid rgba(210,235,255,.22);color:#e8f8ff;font-size:11px;white-space:nowrap;text-shadow:0 1px 4px #000}}
.station-label.tx{{color:#ffe88a;border-color:rgba(255,232,138,.45)}}
.station-label.rx{{color:#9df7d0;border-color:rgba(157,247,208,.32)}}
.colorbar{{position:fixed;left:50%;bottom:16px;transform:translateX(-50%);width:min(460px,calc(100vw - 360px));min-width:260px;pointer-events:none;background:rgba(4,8,14,.66);border:1px solid rgba(150,210,255,.16);border-radius:8px;padding:9px 11px;color:#bed3df}}
.barlabel{{display:flex;justify-content:space-between;font-size:11px;color:#aebfcb;margin-bottom:5px}}
.bar{{position:relative;height:12px;border-radius:6px;background:linear-gradient(90deg,#30123b,#4662d7,#1bcfd4,#70e75f,#f9c63d,#e1442f);box-shadow:inset 0 0 0 1px rgba(255,255,255,.22),0 0 22px rgba(88,215,255,.12)}}
.tick{{position:absolute;top:-4px;width:2px;height:20px;border-radius:2px;background:white;box-shadow:0 0 10px rgba(255,255,255,.8)}}
</style>
</head>
<body>
<canvas id="gl"></canvas>
<div class="hud">
  <div class="panel titlepanel">
    <div>
      <h1>Cape Cod Bay Bolide May 30 2026</h1>
      <div class="meta">Actual measurements of bolide-created plasma using the MIT Haystack Observatory multi-static radar system, painted onto the fitted ballistic path over a curved New England/Cape Cod map.</div>
    </div>
    <div class="logos">
      {f'<img class="logo haystack" src="{asset.get("logos", {}).get("haystack", "")}" alt="MIT Haystack Observatory">' if asset.get("logos", {}).get("haystack") else ""}
      {f'<img class="logo uit" src="{asset.get("logos", {}).get("uit", "")}" alt="UiT The Arctic University of Norway">' if asset.get("logos", {}).get("uit") else ""}
      {f'<img class="logo iap" src="{asset.get("logos", {}).get("iap", "")}" alt="Leibniz Institute of Atmospheric Physics">' if asset.get("logos", {}).get("iap") else ""}
    </div>
  </div>
  <div class="panel controls">
    <button id="play" class="active">Pause</button>
    <input id="time" type="range" min="0" max="1000" value="0">
    <div id="readout" class="readout">2026-05-30T18:06:19.150Z</div>
    <button id="density">Density</button>
    <input id="window" type="range" min="8" max="80" value="26">
    <div id="winread" class="readout">26 s loop</div>
  </div>
</div>
<div class="legend"><span style="color:#d8f5ff"><span class="dot"></span>plasma blobs</span><span style="color:#8b7460"><span class="dot"></span>meteoroid rock</span><span style="color:#ffffff"><span class="dot"></span>painted tail</span><span style="color:#c08a5a"><span class="dot"></span>falling/landed fragments</span></div>
<div class="colorbar">
  <div class="barlabel"><span id="barmin">-1.11 s</span><span>eased UTC playback</span><span id="barmax">3.29 s</span></div>
  <div class="bar"><div id="bartick" class="tick"></div></div>
</div>
<div id="altitude" class="altitude">viewer altitude 96.8 km</div>
<div class="help">W/A/S/D fly · Q/E vertical · left drag look · right drag orbit · wheel zoom</div>
<div id="labels" class="label-layer"></div>
<script id="asset" type="application/json">{asset_json}</script>
<script>
const ASSET = JSON.parse(document.getElementById('asset').textContent);
const canvas = document.getElementById('gl');
const gl = canvas.getContext('webgl', {{antialias:true, alpha:false}});
if (!gl) alert('WebGL is not available in this browser.');

const R = 6371.0, DEG = Math.PI / 180;
const lat0 = ASSET.center.lat * DEG, lon0 = ASSET.center.lon * DEG, cos0 = Math.cos(lat0);
function llh(lon, lat, altKm=0) {{
  const x = R * cos0 * (lon * DEG - lon0);
  const y = R * (lat * DEG - lat0);
  const curve = -(x*x + y*y) / (2.0 * R);
  return [x, altKm + curve, -y];
}}
function shader(type, src) {{
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}}
function program(vs, fs) {{
  const p = gl.createProgram(); gl.attachShader(p, shader(gl.VERTEX_SHADER, vs)); gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}}
const pointProg = program(`
attribute vec3 a_pos; attribute vec4 a_col; attribute float a_t; attribute float a_size;
uniform mat4 u_mvp; uniform float u_time; uniform float u_span; uniform float u_window; uniform float u_density;
varying vec4 v_col; varying float v_alpha;
void main(){{
  float age = u_time - a_t;
  float fade = smoothstep(-0.04, 0.06, age);
  float dens = mix(1.0, 2.65, u_density);
  vec4 clip = u_mvp * vec4(a_pos, 1.0);
  gl_Position = clip;
  gl_PointSize = (a_size * dens) * (260.0 / max(32.0, clip.w));
  v_col = a_col; v_alpha = fade * mix(0.68, 0.95, u_density);
}}`, `
precision mediump float; varying vec4 v_col; varying float v_alpha;
void main(){{
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r = dot(p,p);
  float core = smoothstep(1.0, 0.0, r);
  float halo = smoothstep(1.0, 0.0, r) * smoothstep(0.04, 0.65, r);
  gl_FragColor = vec4(v_col.rgb * (1.0 + 0.75 * halo), v_col.a * v_alpha * core);
}}`);
const lineProg = program(`
attribute vec3 a_pos; attribute vec4 a_col; uniform mat4 u_mvp; varying vec4 v_col;
void main(){{gl_Position=u_mvp*vec4(a_pos,1.0); v_col=a_col;}}`, `
precision mediump float; varying vec4 v_col; void main(){{gl_FragColor=v_col;}}`);
const surfProg = program(`
attribute vec3 a_pos; attribute vec4 a_col; uniform mat4 u_mvp; varying vec4 v_col;
void main(){{gl_Position=u_mvp*vec4(a_pos,1.0); v_col=a_col;}}`, `
precision mediump float; varying vec4 v_col; void main(){{gl_FragColor=v_col;}}`);

function buffer(data) {{ const b=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,b); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.STATIC_DRAW); return b; }}
function timeColor(x) {{
  const stops = [
    [0.00, 0.19, 0.07, 0.23],
    [0.20, 0.27, 0.38, 0.84],
    [0.42, 0.10, 0.81, 0.83],
    [0.62, 0.44, 0.91, 0.37],
    [0.80, 0.98, 0.78, 0.24],
    [1.00, 0.88, 0.27, 0.18],
  ];
  x = Math.max(0, Math.min(1, x));
  for (let i=1; i<stops.length; i++) {{
    if (x <= stops[i][0]) {{
      const a = stops[i-1], b = stops[i], u = (x - a[0]) / (b[0] - a[0]);
      return [a[1] + (b[1]-a[1])*u, a[2] + (b[2]-a[2])*u, a[3] + (b[3]-a[3])*u, 1.0];
    }}
  }}
  return [0.88,0.27,0.18,1.0];
}}
function formatUtc(tRel) {{
  const d = new Date((ASSET.time.t0_unix + tRel) * 1000);
  return d.toISOString().replace(/\\.\\d{{3}}Z$/, 'Z');
}}
function viewerAltitudeKm() {{
  const surface = -(cam[0]*cam[0] + cam[2]*cam[2]) / (2.0 * R);
  return cam[1] - surface;
}}
const points = [];
let tMin=ASSET.paint_time.min, tMax=ASSET.paint_time.max;
const eventSpan = Math.max(0.001, tMax - tMin);
for (const p of ASSET.points) {{
  const pos = llh(p.lon,p.lat,p.altKm);
  const head = p.kind === 'head';
  const c = timeColor((p.t - tMin) / eventSpan);
  c[3] = head ? 1.0 : 0.88;
  const size = (head ? 14 : 11) + Math.max(0, Math.min(16, p.snr - 5)) * (head ? 1.4 : 1.0);
  points.push(...pos, ...c, p.t, size);
}}
const pointBuf = buffer(points);
const pointCount = points.length / 9;

const impactPoints = [];
const strewnLumps = [
  [0.00,0.00,0.00,1.00,0.44,0.35,0.27],
  [0.52,0.06,-0.18,0.72,0.31,0.27,0.24],
  [-0.44,0.08,0.20,0.66,0.55,0.45,0.32],
  [0.12,0.34,0.30,0.55,0.24,0.22,0.20],
  [-0.12,-0.27,-0.26,0.50,0.62,0.50,0.34],
  [0.36,-0.19,0.34,0.43,0.36,0.30,0.25],
  [-0.55,-0.12,-0.28,0.38,0.22,0.20,0.19],
];
for (let i=0; i<ASSET.strewn.samples.length; i++) {{
  const p = ASSET.strewn.samples[i];
  const d = Math.max(0.01, Math.min(1.0, p.diameterM || 0.05));
  const base = llh(p.lon, p.lat, Math.max(0.12,p.altKm + 0.12));
  const scale = 0.035 + 0.13 * Math.pow(d, 0.42);
  const baseSize = 5.0 + 24.0 * Math.pow(d, 0.55);
  for (const l of strewnLumps) {{
    impactPoints.push(
      base[0] + l[0] * scale,
      base[1] + l[1] * scale,
      base[2] + l[2] * scale,
      l[4], l[5], l[6], 0.72,
      p.tImpact,
      baseSize * l[3]
    );
  }}
}}
if (ASSET.strewn.nominal) {{
  const p = ASSET.strewn.nominal;
  const base = llh(p.lon,p.lat,Math.max(0.45,p.altKm + 0.45));
  for (const l of strewnLumps) {{
    impactPoints.push(base[0]+l[0]*0.18, base[1]+l[1]*0.18, base[2]+l[2]*0.18, 1.0,0.78,0.32,1.0, p.tImpact, 34.0*l[3]);
  }}
}}
const impactBuf = buffer(impactPoints);
const impactCount = impactPoints.length / 9;

const stationPoints = [];
for (const s of ASSET.stations) {{
  const pos = llh(s.lon, s.lat, Math.max(0.55, s.altKm + 0.55));
  const tx = s.kind === 'tx';
  const col = tx ? [1.0,0.86,0.18,1.0] : [0.32,1.0,0.72,0.95];
  stationPoints.push(...pos, ...col, -1.0, tx ? 27.0 : 21.0);
}}
const stationBuf = buffer(stationPoints);
const stationCount = stationPoints.length / 9;
const labelLayer = document.getElementById('labels');
const stationLabels = ASSET.stations.map(s => {{
  const el = document.createElement('div');
  el.className = 'station-label ' + s.kind;
  el.textContent = s.label;
  labelLayer.appendChild(el);
  return el;
}});

const rockPath = [...ASSET.trajectory, ...ASSET.strewn.nominalPath].sort((a,b)=>a.t-b.t);
const rockLumps = [
  [0.00,0.00,0.00,17,0.38,0.32,0.27],
  [0.33,0.06,-0.10,12,0.27,0.24,0.22],
  [-0.28,0.10,0.12,11,0.48,0.40,0.31],
  [0.08,0.27,0.19,10,0.31,0.29,0.27],
  [-0.07,-0.24,-0.16,9,0.20,0.19,0.18],
  [0.22,-0.16,0.24,8,0.55,0.45,0.32],
  [-0.34,-0.10,-0.20,8,0.25,0.23,0.21],
];
const rockBuf = gl.createBuffer();
const tailBuf = gl.createBuffer();
const fragmentBuf = gl.createBuffer();
let tailCount = 0;
let fragmentCount = 0;
let rockCount = 0;
function pathPoint(row) {{ return llh(row.lon,row.lat,row.altKm); }}
function interpPath(t) {{
  if (!rockPath.length) return [0,0,0];
  if (t <= rockPath[0].t) return pathPoint(rockPath[0]);
  for (let i=1;i<rockPath.length;i++) {{
    const a=rockPath[i-1], b=rockPath[i];
    if (t <= b.t) {{
      const u=(t-a.t)/Math.max(1e-6,b.t-a.t);
      const pa=pathPoint(a), pb=pathPoint(b);
      return [pa[0]+(pb[0]-pa[0])*u, pa[1]+(pb[1]-pa[1])*u, pa[2]+(pb[2]-pa[2])*u];
    }}
  }}
  return pathPoint(rockPath[rockPath.length-1]);
}}
function updateRock(t) {{
  const split = ASSET.strewn.breakup ? ASSET.strewn.breakup.t : meteorEnd;
  if (t > split) {{
    gl.bindBuffer(gl.ARRAY_BUFFER,rockBuf);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([]),gl.DYNAMIC_DRAW);
    rockCount = 0;
    return;
  }}
  const p=interpPath(t);
  const data=[];
  for (const l of rockLumps) {{
    data.push(p[0]+l[0],p[1]+l[1],p[2]+l[2], l[4],l[5],l[6],1.0, -1.0, l[3]);
  }}
  rockCount = data.length / 9;
  gl.bindBuffer(gl.ARRAY_BUFFER,rockBuf);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.DYNAMIC_DRAW);
}}
function fragmentPosition(sample, t) {{
  const b = ASSET.strewn.breakup || ASSET.trajectory[ASSET.trajectory.length-1];
  const u = Math.max(0, Math.min(1, (t - sample.tBreak) / Math.max(1e-6, sample.tImpact - sample.tBreak)));
  const ease = 1.0 - Math.pow(1.0 - u, 1.85);
  const lat = b.lat + (sample.lat - b.lat) * ease;
  const lon = b.lon + (sample.lon - b.lon) * ease;
  const alt = Math.max(0.12, b.altKm * Math.pow(1.0 - u, 1.65) + sample.altKm * u);
  return llh(lon, lat, alt);
}}
function updateFragments(t) {{
  const data=[];
  for (let i=0; i<ASSET.strewn.samples.length; i++) {{
    const sample=ASSET.strewn.samples[i];
    if (t < sample.tBreak || t >= sample.tImpact) continue;
    const d = Math.max(0.01, Math.min(1.0, sample.diameterM || 0.05));
    const pos = fragmentPosition(sample, t);
    const scale = 0.04 + 0.16 * Math.pow(d, 0.42);
    const baseSize = 5.0 + 23.0 * Math.pow(d, 0.55);
    for (const l of strewnLumps) {{
      data.push(
        pos[0] + l[0] * scale,
        pos[1] + l[1] * scale,
        pos[2] + l[2] * scale,
        l[4], l[5], l[6], 0.9,
        -1.0,
        baseSize * l[3]
      );
    }}
  }}
  fragmentCount = data.length / 9;
  gl.bindBuffer(gl.ARRAY_BUFFER,fragmentBuf);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.DYNAMIC_DRAW);
}}
function updateTail(t) {{
  const data=[];
  if (rockPath.length > 1 && t > rockPath[0].t) {{
    let prev = pathPoint(rockPath[0]);
    for (let i=1;i<rockPath.length;i++) {{
      const row = rockPath[i];
      const next = row.t <= t ? pathPoint(row) : interpPath(t);
      data.push(...prev,1.0,1.0,1.0,0.82, ...next,1.0,1.0,1.0,0.82);
      if (row.t > t) break;
      prev = next;
    }}
  }}
  tailCount = data.length / 7;
  gl.bindBuffer(gl.ARRAY_BUFFER,tailBuf);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.DYNAMIC_DRAW);
}}

function pushLine(out, seg, col, alt=0.08) {{
  for (let i=1;i<seg.length;i++) {{
    out.push(...llh(seg[i-1][0],seg[i-1][1],alt),...col);
    out.push(...llh(seg[i][0],seg[i][1],alt),...col);
  }}
}}
const lines = [];
for (const s of ASSET.map.coast) pushLine(lines,s,[0.52,0.95,0.82,0.7],0.12);
for (const s of ASSET.map.states) pushLine(lines,s,[0.42,0.66,0.95,0.45],0.15);
for (const s of ASSET.map.rivers) pushLine(lines,s,[0.25,0.58,0.92,0.38],0.18);
const lineBuf = buffer(lines), lineCount = lines.length / 7;

const surf = [];
const b = ASSET.bbox, nx=72, ny=54;
for (let iy=0; iy<ny; iy++) for (let ix=0; ix<nx; ix++) {{
  const lonA=b[0]+(b[2]-b[0])*ix/nx, lonB=b[0]+(b[2]-b[0])*(ix+1)/nx;
  const latA=b[1]+(b[3]-b[1])*iy/ny, latB=b[1]+(b[3]-b[1])*(iy+1)/ny;
  const col=[0.025,0.075+0.035*iy/ny,0.105+0.055*ix/nx,1.0];
  for (const q of [[lonA,latA],[lonB,latA],[lonB,latB],[lonA,latA],[lonB,latB],[lonA,latB]]) surf.push(...llh(q[0],q[1],-0.08),...col);
}}
const surfBuf = buffer(surf), surfCount = surf.length / 7;

function matMul(a,b) {{ const o=new Float32Array(16); for(let r=0;r<4;r++) for(let c=0;c<4;c++) o[c*4+r]=a[0*4+r]*b[c*4+0]+a[1*4+r]*b[c*4+1]+a[2*4+r]*b[c*4+2]+a[3*4+r]*b[c*4+3]; return o; }}
function perspective(fovy, asp, n, f) {{ const t=1/Math.tan(fovy/2), nf=1/(n-f); return new Float32Array([t/asp,0,0,0, 0,t,0,0, 0,0,(f+n)*nf,-1, 0,0,2*f*n*nf,0]); }}
function lookAt(e,c,u) {{ let zx=e[0]-c[0],zy=e[1]-c[1],zz=e[2]-c[2]; let zl=1/Math.hypot(zx,zy,zz); zx*=zl;zy*=zl;zz*=zl; let xx=u[1]*zz-u[2]*zy,xy=u[2]*zx-u[0]*zz,xz=u[0]*zy-u[1]*zx; let xl=1/Math.hypot(xx,xy,xz); xx*=xl;xy*=xl;xz*=xl; let yx=zy*xz-zz*xy,yy=zz*xx-zx*xz,yz=zx*xy-zy*xx; return new Float32Array([xx,yx,zx,0, xy,yy,zy,0, xz,yz,zz,0, -(xx*e[0]+xy*e[1]+xz*e[2]),-(yx*e[0]+yy*e[1]+yz*e[2]),-(zx*e[0]+zy*e[1]+zz*e[2]),1]); }}
function projectPoint(pos, mvp, w, h) {{
  const x=pos[0], y=pos[1], z=pos[2];
  const cx=mvp[0]*x+mvp[4]*y+mvp[8]*z+mvp[12];
  const cy=mvp[1]*x+mvp[5]*y+mvp[9]*z+mvp[13];
  const cz=mvp[2]*x+mvp[6]*y+mvp[10]*z+mvp[14];
  const cw=mvp[3]*x+mvp[7]*y+mvp[11]*z+mvp[15];
  if (cw <= 0.02) return null;
  const nx=cx/cw, ny=cy/cw, nz=cz/cw;
  if (nx < -1.2 || nx > 1.2 || ny < -1.2 || ny > 1.2 || nz < -1.2 || nz > 1.2) return null;
  return [(nx*0.5+0.5)*w, (0.5-ny*0.5)*h];
}}
function updateStationLabels(mvp, w, h) {{
  for (let i=0; i<ASSET.stations.length; i++) {{
    const s = ASSET.stations[i];
    const pos = llh(s.lon, s.lat, Math.max(0.55, s.altKm + 0.55));
    const p = projectPoint(pos, mvp, w, h);
    const el = stationLabels[i];
    if (!p) {{
      el.style.display = 'none';
    }} else {{
      el.style.display = 'block';
      el.style.left = p[0].toFixed(1) + 'px';
      el.style.top = p[1].toFixed(1) + 'px';
    }}
  }}
}}

let cam=[-32,72,96], yaw=0.34, pitch=-0.55, speed=35, playing=true, density=false, simT=tMin;
const keys = new Set(), span=Math.max(0.001,tMax-tMin), slider=document.getElementById('time'), readout=document.getElementById('readout'), win=document.getElementById('window'), winread=document.getElementById('winread'), bartick=document.getElementById('bartick'), barmin=document.getElementById('barmin'), barmax=document.getElementById('barmax'), altitude=document.getElementById('altitude');
const meteorEnd = Math.max(...ASSET.trajectory.map(p=>p.t));
const meteorSplit = 0.55;
let playProgress = 0.0;
function progressToTime(p) {{
  p = ((p % 1) + 1) % 1;
  if (p <= meteorSplit) {{
    return tMin + (p / meteorSplit) * (meteorEnd - tMin);
  }}
  return meteorEnd + ((p - meteorSplit) / (1.0 - meteorSplit)) * (tMax - meteorEnd);
}}
function timeToProgress(t) {{
  if (t <= meteorEnd) {{
    return meteorSplit * (t - tMin) / Math.max(1e-6, meteorEnd - tMin);
  }}
  return meteorSplit + (1.0 - meteorSplit) * (t - meteorEnd) / Math.max(1e-6, tMax - meteorEnd);
}}
function aimAt(target) {{
  const fx=target[0]-cam[0], fy=target[1]-cam[1], fz=target[2]-cam[2];
  const r=Math.max(1e-6, Math.hypot(fx,fy,fz));
  yaw = Math.atan2(fx, -fz);
  pitch = Math.asin(Math.max(-0.98, Math.min(0.98, fy/r)));
}}
function orbitAround(target, dx, dy) {{
  const ox=cam[0]-target[0], oy=cam[1]-target[1], oz=cam[2]-target[2];
  const radius=Math.max(1.0, Math.hypot(ox,oy,oz));
  let az=Math.atan2(ox, oz);
  let el=Math.asin(Math.max(-0.96, Math.min(0.96, oy/radius)));
  az -= dx * 0.005;
  el = Math.max(-1.35, Math.min(1.35, el + dy * 0.003));
  const c=Math.cos(el);
  cam[0]=target[0] + radius*Math.sin(az)*c;
  cam[1]=target[1] + radius*Math.sin(el);
  cam[2]=target[2] + radius*Math.cos(az)*c;
  aimAt(target);
}}
document.getElementById('play').onclick=()=>{{playing=!playing; play.textContent=playing?'Pause':'Play'; play.classList.toggle('active',playing);}};
document.getElementById('density').onclick=()=>{{density=!density; document.getElementById('density').classList.toggle('active',density);}};
slider.oninput=()=>{{playProgress=+slider.value/1000; simT=progressToTime(playProgress); playing=false; play.textContent='Play'; play.classList.remove('active');}};
win.oninput=()=>{{winread.textContent=(+win.value).toFixed(0)+' s loop';}};
barmin.textContent = tMin.toFixed(2) + ' s';
barmax.textContent = tMax.toFixed(2) + ' s';
addEventListener('keydown',e=>keys.add(e.key.toLowerCase())); addEventListener('keyup',e=>keys.delete(e.key.toLowerCase()));
let drag=false, orbitDrag=false, orbitFollow=false, lastX=0,lastY=0;
canvas.oncontextmenu=e=>e.preventDefault();
canvas.onmousedown=e=>{{drag=true; orbitDrag=e.button===2; orbitFollow=orbitDrag; lastX=e.clientX; lastY=e.clientY; e.preventDefault();}};
addEventListener('mouseup',()=>{{drag=false; orbitDrag=false;}});
addEventListener('mousemove',e=>{{
  if(!drag)return;
  const dx=e.clientX-lastX, dy=e.clientY-lastY;
  if(orbitDrag) {{
    orbitAround(interpPath(simT), dx, dy);
  }} else {{
    orbitFollow=false;
    yaw-=dx*0.004;
    pitch-=dy*0.003;
    pitch=Math.max(-1.45,Math.min(1.2,pitch));
  }}
  lastX=e.clientX;lastY=e.clientY;
}});
canvas.onwheel=e=>{{
  e.preventDefault();
  const forward=[Math.sin(yaw)*Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw)*Math.cos(pitch)];
  const dolly = -e.deltaY * 0.035;
  for(let i=0;i<3;i++) cam[i]+=forward[i]*dolly;
}};
let last=performance.now();
function frame(now) {{
  const dt=Math.min(0.05,(now-last)/1000); last=now;
  if (playing) {{
    playProgress = (playProgress + dt / Math.max(1.0, +win.value)) % 1.0;
    simT = progressToTime(playProgress);
  }}
  const forward=[Math.sin(yaw)*Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw)*Math.cos(pitch)];
  const right=[Math.cos(yaw),0,Math.sin(yaw)];
  const step=speed*dt;
  if(keys.has('w')) for(let i=0;i<3;i++) cam[i]+=forward[i]*step;
  if(keys.has('s')) for(let i=0;i<3;i++) cam[i]-=forward[i]*step;
  if(keys.has('d')) for(let i=0;i<3;i++) cam[i]+=right[i]*step;
  if(keys.has('a')) for(let i=0;i<3;i++) cam[i]-=right[i]*step;
  if(keys.has('e')) cam[1]+=step; if(keys.has('q')) cam[1]-=step;
  if(orbitFollow) aimAt(interpPath(simT));
  const w=canvas.clientWidth,h=canvas.clientHeight; if(canvas.width!==w||canvas.height!==h){{canvas.width=w;canvas.height=h;gl.viewport(0,0,w,h);}}
  gl.enable(gl.DEPTH_TEST); gl.clearColor(0.015,0.02,0.032,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  const view=lookAt(cam,[cam[0]+forward[0],cam[1]+forward[1],cam[2]+forward[2]],[0,1,0]);
  const mvp=matMul(perspective(58*DEG,w/h,0.1,800),view);
  updateStationLabels(mvp,w,h);
  function bind7(prog, buf){{ gl.useProgram(prog); gl.uniformMatrix4fv(gl.getUniformLocation(prog,'u_mvp'),false,mvp); gl.bindBuffer(gl.ARRAY_BUFFER,buf); const p=gl.getAttribLocation(prog,'a_pos'), c=gl.getAttribLocation(prog,'a_col'); gl.enableVertexAttribArray(p); gl.vertexAttribPointer(p,3,gl.FLOAT,false,28,0); gl.enableVertexAttribArray(c); gl.vertexAttribPointer(c,4,gl.FLOAT,false,28,12); }}
  updateRock(simT);
  updateFragments(simT);
  updateTail(simT);
  bind7(surfProg,surfBuf); gl.disable(gl.BLEND); gl.drawArrays(gl.TRIANGLES,0,surfCount);
  bind7(lineProg,lineBuf); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE); gl.drawArrays(gl.LINES,0,lineCount);
  bind7(lineProg,tailBuf); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE); gl.drawArrays(gl.LINES,0,tailCount);
  gl.useProgram(pointProg); gl.uniformMatrix4fv(gl.getUniformLocation(pointProg,'u_mvp'),false,mvp); gl.uniform1f(gl.getUniformLocation(pointProg,'u_time'),simT); gl.uniform1f(gl.getUniformLocation(pointProg,'u_span'),span); gl.uniform1f(gl.getUniformLocation(pointProg,'u_window'),+win.value/1000); gl.uniform1f(gl.getUniformLocation(pointProg,'u_density'),density?1:0);
  gl.bindBuffer(gl.ARRAY_BUFFER,pointBuf);
  const ap=gl.getAttribLocation(pointProg,'a_pos'), ac=gl.getAttribLocation(pointProg,'a_col'), at=gl.getAttribLocation(pointProg,'a_t'), as=gl.getAttribLocation(pointProg,'a_size');
  gl.enableVertexAttribArray(ap); gl.vertexAttribPointer(ap,3,gl.FLOAT,false,36,0);
  gl.enableVertexAttribArray(ac); gl.vertexAttribPointer(ac,4,gl.FLOAT,false,36,12);
  gl.enableVertexAttribArray(at); gl.vertexAttribPointer(at,1,gl.FLOAT,false,36,28);
  gl.enableVertexAttribArray(as); gl.vertexAttribPointer(as,1,gl.FLOAT,false,36,32);
  gl.depthMask(false); gl.blendFunc(gl.SRC_ALPHA,gl.ONE); gl.drawArrays(gl.POINTS,0,pointCount); gl.depthMask(true);
  gl.uniform1f(gl.getUniformLocation(pointProg,'u_time'),0.0);
  gl.bindBuffer(gl.ARRAY_BUFFER,fragmentBuf);
  gl.vertexAttribPointer(ap,3,gl.FLOAT,false,36,0);
  gl.vertexAttribPointer(ac,4,gl.FLOAT,false,36,12);
  gl.vertexAttribPointer(at,1,gl.FLOAT,false,36,28);
  gl.vertexAttribPointer(as,1,gl.FLOAT,false,36,32);
  gl.depthMask(false); gl.blendFunc(gl.SRC_ALPHA,gl.ONE); gl.drawArrays(gl.POINTS,0,fragmentCount); gl.depthMask(true);
  gl.uniform1f(gl.getUniformLocation(pointProg,'u_time'),simT); gl.uniform1f(gl.getUniformLocation(pointProg,'u_span'),span); gl.uniform1f(gl.getUniformLocation(pointProg,'u_window'),2.0);
  gl.bindBuffer(gl.ARRAY_BUFFER,impactBuf);
  gl.vertexAttribPointer(ap,3,gl.FLOAT,false,36,0);
  gl.vertexAttribPointer(ac,4,gl.FLOAT,false,36,12);
  gl.vertexAttribPointer(at,1,gl.FLOAT,false,36,28);
  gl.vertexAttribPointer(as,1,gl.FLOAT,false,36,32);
  gl.depthMask(false); gl.blendFunc(gl.SRC_ALPHA,gl.ONE); gl.drawArrays(gl.POINTS,0,impactCount); gl.depthMask(true);
  gl.uniform1f(gl.getUniformLocation(pointProg,'u_time'),0.0);
  gl.bindBuffer(gl.ARRAY_BUFFER,rockBuf);
  gl.vertexAttribPointer(ap,3,gl.FLOAT,false,36,0);
  gl.vertexAttribPointer(ac,4,gl.FLOAT,false,36,12);
  gl.vertexAttribPointer(at,1,gl.FLOAT,false,36,28);
  gl.vertexAttribPointer(as,1,gl.FLOAT,false,36,32);
  gl.depthMask(false); gl.blendFunc(gl.SRC_ALPHA,gl.ONE); gl.drawArrays(gl.POINTS,0,rockCount); gl.depthMask(true);
  gl.uniform1f(gl.getUniformLocation(pointProg,'u_time'),0.0);
  gl.bindBuffer(gl.ARRAY_BUFFER,stationBuf);
  gl.vertexAttribPointer(ap,3,gl.FLOAT,false,36,0);
  gl.vertexAttribPointer(ac,4,gl.FLOAT,false,36,12);
  gl.vertexAttribPointer(at,1,gl.FLOAT,false,36,28);
  gl.vertexAttribPointer(as,1,gl.FLOAT,false,36,32);
  gl.depthMask(false); gl.blendFunc(gl.SRC_ALPHA,gl.ONE); gl.drawArrays(gl.POINTS,0,stationCount); gl.depthMask(true);
  const timeFrac = timeToProgress(simT);
  slider.value = Math.round(1000*timeFrac); readout.textContent = formatUtc(simT);
  altitude.textContent = 'viewer altitude ' + viewerAltitudeKm().toFixed(1) + ' km';
  bartick.style.left = `calc(${{Math.max(0, Math.min(1, timeFrac)) * 100}}% - 1px)`;
  requestAnimationFrame(frame);
}}
win.oninput(); requestAnimationFrame(frame);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a self-contained WebGL bolide volume viewer.")
    parser.add_argument("--head", default=DEFAULT_HEAD)
    parser.add_argument("--trail", default=DEFAULT_TRAIL)
    parser.add_argument("--ballistic", default=DEFAULT_BALLISTIC)
    parser.add_argument("--path-residual-max-km", type=float, default=12.0)
    parser.add_argument("--haystack-logo", default=DEFAULT_HAYSTACK_LOGO)
    parser.add_argument("--uit-logo", default=DEFAULT_UIT_LOGO)
    parser.add_argument("--iap-logo", default=DEFAULT_IAP_LOGO)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    head = project_rows(read_rows(Path(args.head), "positions"), "head")
    trail = project_rows(read_rows(Path(args.trail), "positions"), "trail")
    raw_points = sorted(head + trail, key=lambda r: (r["t"], r["kind"]))
    ballistic = load_ballistic(Path(args.ballistic))
    t0_unix = min(p["t"] for p in raw_points)
    trajectory = ballistic_trajectory_rows(ballistic, t0_unix)
    for point in raw_points:
        point["t_unix"] = point["t"]
        point["t"] = point["t"] - t0_unix
    center_points = raw_points + trajectory
    center = {
        "lat": sum(p["lat"] for p in center_points) / len(center_points),
        "lon": sum(p["lon"] for p in center_points) / len(center_points),
    }
    points, rejected_points = annotate_by_ballistic_path(raw_points, trajectory, center, args.path_residual_max_km)
    strewn = strewn_field_rows(ballistic, t0_unix)
    stations = station_marker_rows("2026_05_30_bolide")
    t_max_rel = max(
        max(p["t"] for p in raw_points),
        max(row["t"] for row in trajectory),
        max((row["t"] for row in strewn["nominalPath"]), default=0.0),
        max((row["tImpact"] for row in strewn["samples"]), default=0.0),
        strewn["nominal"]["tImpact"] if strewn["nominal"] else 0.0,
    )
    t1_unix = t0_unix + t_max_rel
    extra_geo = []
    if strewn["nominal"]:
        extra_geo.append(strewn["nominal"])
    extra_geo.extend(strewn["samples"])
    extra_geo.extend(stations)
    lat_vals = [p["lat"] for p in center_points + extra_geo]
    lon_vals = [p["lon"] for p in center_points + extra_geo]
    bbox = (
        min(min(lon_vals) - 1.0, -73.2),
        min(min(lat_vals) - 0.7, 40.6),
        max(max(lon_vals) + 1.1, -68.8),
        max(max(lat_vals) + 0.9, 43.6),
    )
    asset = {
        "schema": "new_england_bolide.webgl_viewer.v1",
        "center": center,
        "bbox": bbox,
        "counts": {
            "head": sum(1 for p in points if p["kind"] == "head"),
            "trail": sum(1 for p in points if p["kind"] == "trail"),
            "rejected": len(rejected_points),
            "strewn": len(strewn["samples"]),
            "stations": len(stations),
        },
        "time": {"min": 0.0, "max": t1_unix - t0_unix, "t0_unix": t0_unix, "t1_unix": t1_unix},
        "paint_time": {"min": min(row["t"] for row in trajectory), "max": t_max_rel},
        "path_residual_max_km": args.path_residual_max_km,
        "points": points,
        "rejected_points": rejected_points,
        "trajectory": trajectory,
        "strewn": strewn,
        "stations": stations,
        "logos": {
            "haystack": data_uri(Path(args.haystack_logo)) if args.haystack_logo else None,
            "uit": data_uri(Path(args.uit_logo)) if args.uit_logo else None,
            "iap": data_uri(Path(args.iap_logo)) if args.iap_logo else None,
        },
        "map": load_map_lines(bbox),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(asset), encoding="utf-8")
    print(out)
    print(f"embedded {asset['counts']['head']} head points, {asset['counts']['trail']} trail points")
    print(f"filtered {len(rejected_points)} points with path residual > {args.path_residual_max_km:.1f} km")
    print(f"embedded {len(strewn['samples'])} strewn-field samples")
    print(
        "map segments:",
        {k: len(v) for k, v in asset["map"].items()},
    )


if __name__ == "__main__":
    main()
