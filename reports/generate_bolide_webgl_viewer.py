#!/usr/bin/env python3
"""Generate the bolide viewer or refresh its orbit from the current article.

Orbit-only refresh (preserves embedded observations and image assets)::

    python reports/generate_bolide_webgl_viewer.py \
        --existing-viewer reports/figures/bolide_webgl_volume_viewer.html \
        --article-orbit-table ../cape_cod_bay/reports/orbit_parameter_table.tex

The article table supplies the published elements, uncertainties, and epoch;
the generated asset records its fit version and SHA-256 for provenance.
"""
from __future__ import annotations

import argparse
import base64
import json
import hashlib
import re
from datetime import datetime, timezone
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
DEFAULT_BALLISTIC = "products/ballistic_head_trail_correlated_fit_v0p2p7_zp16_neighbor0p5/ballistic_head_trail_correlated_fit.json"
DEFAULT_OUT = "reports/figures/bolide_webgl_volume_viewer.html"
DEFAULT_HAYSTACK_LOGO = "reports/assets/logos/mit_haystack_logo.svg"
DEFAULT_UIT_LOGO = "reports/assets/logos/uit_space_physics_logo.png"
DEFAULT_IAP_LOGO = "reports/assets/logos/iap_logo.svg"
DEFAULT_ORBIT = "products/orbit_determination/best_fit_orbit.h5"
DEFAULT_TYCHO = str(Path.home() / "src/meteor_viz/web/data/tycho2_mag8_embedded.js")
DEFAULT_EARTH_TEXTURE = "reports/figures/world.jpg"
NE_SHAPE_ROOT = Path.home() / ".local/share/cartopy/shapefiles/natural_earth"
CSS_VERSION = "2026.06.12.1"
JS_VERSION = "2026.10.30.1"
AU_M = 149_597_870_700.0


SOLAR_APPROACH_SCENE_JS = r"""
const SolarApproachScene = (() => {
  const MU = 0.00029592115654562346;
  const AU_KM = 149597870.7;
  const EARTH_RADIUS_KM = 6371.0;
  const J2000_UNIX_SECOND = 946728000;
  const J2000_OBLIQUITY = 23.4392911 * DEG;
  const PLANETS = [
    { name: "Mercury", a: 0.38709893, e: 0.20563069, i: 7.00487, L: 252.25084, peri: 77.45645, node: 48.33167, color: [0.78,0.72,0.62] },
    { name: "Venus", a: 0.72333199, e: 0.00677323, i: 3.39471, L: 181.97973, peri: 131.53298, node: 76.68069, color: [1.0,0.78,0.42] },
    { name: "Earth", a: 1.00000011, e: 0.01671022, i: 0.00005, L: 100.46435, peri: 102.94719, node: -11.26064, color: [0.30,0.66,1.0] },
    { name: "Mars", a: 1.52366231, e: 0.09341233, i: 1.85061, L: -4.56813, peri: -23.94363, node: 49.57854, color: [1.0,0.36,0.22] },
    { name: "Jupiter", a: 5.20336301, e: 0.04839266, i: 1.3053, L: 34.40438, peri: 14.75385, node: 100.55615, color: [0.95,0.72,0.52] },
    { name: "Saturn", a: 9.53707032, e: 0.0541506, i: 2.48446, L: 49.94432, peri: 92.43194, node: 113.71504, color: [0.95,0.84,0.55] },
  ];
  const solar = ASSET.solar;
  const tycho = ASSET.tycho;
  if (!solar) return null;

  function norm(v) {
    const n = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / n, v[1] / n, v[2] / n];
  }
  function sub(a,b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
  function add(a,b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
  function mul(a,s) { return [a[0]*s, a[1]*s, a[2]*s]; }
  function mix(a,b,u) { return [a[0]+(b[0]-a[0])*u, a[1]+(b[1]-a[1])*u, a[2]+(b[2]-a[2])*u]; }
  function cross(a,b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
  function smoother(x) { x = Math.max(0, Math.min(1, x)); return x*x*x*(x*(x*6-15)+10); }
  function solarVec(v) { return [v[0], v[2], -v[1]]; }
  function planetElements(planet) {
    return {
      a: planet.a,
      e: planet.e,
      inc: planet.i * DEG,
      argp: ((planet.peri - planet.node) % 360) * DEG,
      node: planet.node * DEG,
      mean0: ((planet.L - planet.peri) % 360) * DEG,
    };
  }
  function positionFromElements(elements, meanAnomaly) {
    const a = elements.a, e = elements.e, inc = elements.inc, argp = elements.argp, node = elements.node;
    let M = ((meanAnomaly + Math.PI) % (2 * Math.PI)) - Math.PI;
    let E = M;
    for (let k = 0; k < 8; k++) E -= (E - e * Math.sin(E) - M) / Math.max(1 - e * Math.cos(E), 1e-6);
    const root = Math.sqrt(Math.max(1 - e * e, 0));
    const xpf = a * (Math.cos(E) - e);
    const ypf = a * root * Math.sin(E);
    const co = Math.cos(node), so = Math.sin(node), ci = Math.cos(inc), si = Math.sin(inc), cw = Math.cos(argp), sw = Math.sin(argp);
    return [
      (co*cw - so*sw*ci) * xpf + (-co*sw - so*cw*ci) * ypf,
      (so*cw + co*sw*ci) * xpf + (-so*sw + co*cw*ci) * ypf,
      sw*si*xpf + cw*si*ypf,
    ];
  }
  function trueToMeanAnomaly(nu, e) {
    const E = 2 * Math.atan2(Math.sqrt(Math.max(1 - e, 0)) * Math.sin(nu / 2), Math.sqrt(1 + e) * Math.cos(nu / 2));
    return E - e * Math.sin(E);
  }
  function planetPosition(planet, tDay) {
    const elements = planetElements(planet);
    const meanMotion = Math.sqrt(MU / (elements.a * elements.a * elements.a));
    return solarVec(positionFromElements(elements, elements.mean0 + meanMotion * tDay));
  }
  function starDirection(raHours, decDeg) {
    const ra = raHours * 15 * DEG, dec = decDeg * DEG, cd = Math.cos(dec);
    const xEq = cd * Math.cos(ra), yEq = cd * Math.sin(ra), zEq = Math.sin(dec);
    const ce = Math.cos(J2000_OBLIQUITY), se = Math.sin(J2000_OBLIQUITY);
    return [xEq, ce * yEq + se * zEq, -se * yEq + ce * zEq];
  }
  function bytesFromBase64(text) {
    const binary = atob(text), bytes = new Uint8Array(binary.length);
    for (let i=0; i<binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }
  function buffer7(data) { return { buf: buffer(data), count: data.length / 7 }; }
  function buffer9(data) { return { buf: buffer(data), count: data.length / 9 }; }
  function makeOrbitLine(elements, col, turns, startMean, endMean, stepsPerTurn=260) {
    const out = [];
    const steps = Math.max(40, Math.round(stepsPerTurn * turns));
    let prev = solarVec(positionFromElements(elements, startMean));
    for (let i=1; i<=steps; i++) {
      const m = startMean + (endMean - startMean) * i / steps;
      const p = solarVec(positionFromElements(elements, m));
      out.push(...prev, ...col, ...p, ...col);
      prev = p;
    }
    return out;
  }
  const meteorElements = {
    a: solar.orbit.a_au,
    e: solar.orbit.e,
    inc: solar.orbit.i_deg * DEG,
    argp: solar.orbit.omega_deg * DEG,
    node: solar.orbit.Omega_deg * DEG,
  };
  const meteorN = Math.sqrt(MU / Math.pow(meteorElements.a, 3));
  const meteorMeanAtEntry = trueToMeanAnomaly(solar.orbit.nu_deg * DEG, meteorElements.e)
    + meteorN * (solar.entry_unix - (solar.orbit_epoch_unix ?? solar.entry_unix)) / 86400;
  const periodDays = 2 * Math.PI / meteorN;
  const entryTDay = (solar.entry_unix - J2000_UNIX_SECOND) / 86400.0;
  const CLOSE_START_U = 0.542;
  const CLOSE_EASE_POWER = 0.36;
  const CLOSE_START_ALTITUDE_KM = 300000.0;
  const FREEZE_ALTITUDE_KM = 40000.0;
  const FINAL_APPROACH_DISTANCE_KM = 40000.0;
  const FINAL_APPROACH_STEP_KM = 100.0;
  const ENTRY_SPEED_KM_S = solar.entry_speed_km_s ?? 25.8;
  const FREEZE_CAMERA_START_U = 0.62;
  const FREEZE_CAMERA_END_U = 0.84;
  function meteorPositionAtUnix(unix) {
    const dtDay = (unix - solar.entry_unix) / 86400.0;
    return solarVec(positionFromElements(meteorElements, meteorMeanAtEntry + meteorN * dtDay));
  }
  function closeUForAltitude(altitudeKm) {
    const denom = Math.max(1e-6, CLOSE_START_ALTITUDE_KM - solar.entry_altitude_km);
    const fraction = Math.max(0.0, Math.min(1.0, (CLOSE_START_ALTITUDE_KM - altitudeKm) / denom));
    const q = Math.pow(fraction, 1.0 / CLOSE_EASE_POWER);
    return CLOSE_START_U + (1.0 - CLOSE_START_U) * q;
  }
  const FREEZE_MOTION_U = closeUForAltitude(FREEZE_ALTITUDE_KM);
  function motionUForCameraU(u) {
    u = Math.max(0, Math.min(1, u));
    if (u < FREEZE_CAMERA_START_U) {
      return FREEZE_MOTION_U * u / FREEZE_CAMERA_START_U;
    }
    if (u < FREEZE_CAMERA_END_U) {
      return FREEZE_MOTION_U;
    }
    return FREEZE_MOTION_U + (1.0 - FREEZE_MOTION_U) * (u - FREEZE_CAMERA_END_U) / Math.max(1e-6, 1.0 - FREEZE_CAMERA_END_U);
  }
  function orbitalTimeUnix(u) {
    const introDays = 2.0 * periodDays;
    u = Math.max(0, Math.min(1, u));
    const closeStart = CLOSE_START_U;
    const closeApproachSeconds = (CLOSE_START_ALTITUDE_KM - solar.entry_altitude_km) / ENTRY_SPEED_KM_S;
    const closeStartUnix = solar.entry_unix - closeApproachSeconds;
    if (u < closeStart) {
      const q = u / closeStart;
      const cruiseWarp = Math.pow(q, 0.58);
      return solar.entry_unix - introDays * 86400 * (1.0 - cruiseWarp) - closeApproachSeconds * cruiseWarp;
    } else {
      const q = (u - closeStart) / (1.0 - closeStart);
      const eased = Math.pow(q, CLOSE_EASE_POWER);
      return closeStartUnix + closeApproachSeconds * eased;
    }
  }
  function timeUnix(u) {
    return orbitalTimeUnix(motionUForCameraU(u));
  }
  const finalApproach = (() => {
    const samples = [];
    const n = Math.max(2, Math.round(FINAL_APPROACH_DISTANCE_KM / FINAL_APPROACH_STEP_KM));
    for (let i=0; i<=n; i++) {
      const rangeKm = FINAL_APPROACH_DISTANCE_KM * (1.0 - i / n);
      const unix = solar.entry_unix - rangeKm / ENTRY_SPEED_KM_S;
      samples.push({ rangeKm, unix, pos: meteorPositionAtUnix(unix) });
    }
    return samples;
  })();
  function finalApproachPosition(fraction) {
    fraction = Math.max(0, Math.min(1, fraction));
    const x = fraction * (finalApproach.length - 1);
    const i = Math.min(finalApproach.length - 2, Math.max(0, Math.floor(x)));
    return mix(finalApproach[i].pos, finalApproach[i + 1].pos, x - i);
  }
  function finalApproachFractionForCameraU(u) {
    if (u < FREEZE_CAMERA_END_U) return 0.0;
    return (u - FREEZE_CAMERA_END_U) / Math.max(1e-6, 1.0 - FREEZE_CAMERA_END_U);
  }
  const finalApproachSet = { buf: gl.createBuffer(), count: 0 };
  const finalApproachLowResDiffAu = (() => {
    let maxDiff = 0.0;
    for (const sample of finalApproach) {
      const p = meteorPositionAtUnix(sample.unix);
      const d = Math.hypot(p[0] - sample.pos[0], p[1] - sample.pos[1], p[2] - sample.pos[2]);
      maxDiff = Math.max(maxDiff, d);
    }
    return maxDiff;
  })();
  console.debug('final approach sanity: samples=', finalApproach.length, 'max orbit/path diff AU=', finalApproachLowResDiffAu);
  function earthRotationAngle(unix) {
    const days = (unix - J2000_UNIX_SECOND) / 86400.0;
    return 2.0 * Math.PI * (0.7790572732640 + 1.00273781191135448 * days);
  }

  const stars = [];
  if (tycho && tycho.base64Float32 && tycho.metadata) {
    const table = new Float32Array(bytesFromBase64(tycho.base64Float32).buffer);
    const stride = tycho.metadata.recordFloat32Count || 3;
    const count = Math.min(tycho.metadata.count || 0, Math.floor(table.length / stride));
    for (let i=0; i<count; i++) {
      const row = i * stride, mag = table[row + 2];
      if (mag > 7.4) continue;
      const d = starDirection(table[row], table[row + 1]);
      const alpha = Math.max(0.20, Math.min(0.88, (7.8 - mag) / 6.8));
      const size = Math.max(1.7, 6.8 - mag * 0.55);
      stars.push(d[0]*72, d[2]*72, -d[1]*72, 0.82,0.90,1.0,alpha, -1.0, size);
    }
  }
  const starSet = buffer9(stars);
  const planetOrbitLines = [];
  for (const planet of PLANETS) {
    const el = planetElements(planet);
    planetOrbitLines.push(...makeOrbitLine(el, [...planet.color, 0.30], 1, 0, 2*Math.PI, 220));
  }
  const planetOrbitSet = buffer7(planetOrbitLines);
  const planetPointSet = { buf: gl.createBuffer(), count: 0 };
  const meteorOrbitSet = buffer7(makeOrbitLine(meteorElements, [1.0,0.55,0.26,0.72], 2, meteorMeanAtEntry - 4*Math.PI, meteorMeanAtEntry, 280));
  const earthSphere = [];
  function pushSphere(center, radius, rows=28, cols=44) {
    for (let iy=0; iy<rows; iy++) for (let ix=0; ix<cols; ix++) {
      const latA = -Math.PI/2 + Math.PI*iy/rows, latB = -Math.PI/2 + Math.PI*(iy+1)/rows;
      const lonA = 2*Math.PI*ix/cols, lonB = 2*Math.PI*(ix+1)/cols;
      const qs = [[latA,lonA],[latB,lonA],[latB,lonB],[latA,lonA],[latB,lonB],[latA,lonB]];
      for (const q of qs) {
        const lat = q[0], lon = q[1], cl = Math.cos(lat);
        const land = Math.sin(3.1*lon + 1.7*Math.sin(2.0*lat)) + 0.55*Math.sin(5.7*lon - 2.4*lat);
        const polar = Math.abs(lat) > 1.16;
        const col = polar ? [0.92,0.96,1.0,1.0] : land > 0.45 ? [0.16,0.44,0.22,1.0] : land > 0.16 ? [0.42,0.34,0.20,1.0] : [0.05,0.25,0.58,1.0];
        earthSphere.push(center[0] + radius*cl*Math.cos(lon), center[1] + radius*Math.sin(lat), center[2] + radius*cl*Math.sin(lon), ...col);
      }
    }
  }
  pushSphere([0,0,0], 1.0);
  const earthSphereSet = { buf: buffer(earthSphere), count: earthSphere.length / 7 };
  const earthTextureMesh = [];
  function pushTexturedSphere(rows=96, cols=192) {
    function vertex(lat, lon) {
      const cl = Math.cos(lat);
      const u = 1.0 - (lon + Math.PI) / (2 * Math.PI);
      const v = 0.5 - lat / Math.PI;
      earthTextureMesh.push(cl*Math.cos(lon), Math.sin(lat), cl*Math.sin(lon), u, v);
    }
    for (let iy=0; iy<rows; iy++) for (let ix=0; ix<cols; ix++) {
      const latA = -Math.PI/2 + Math.PI*iy/rows, latB = -Math.PI/2 + Math.PI*(iy+1)/rows;
      const lonA = -Math.PI + 2*Math.PI*ix/cols, lonB = -Math.PI + 2*Math.PI*(ix+1)/cols;
      vertex(latA,lonA); vertex(latB,lonA); vertex(latB,lonB);
      vertex(latA,lonA); vertex(latB,lonB); vertex(latA,lonB);
    }
  }
  pushTexturedSphere();
  const earthTextureSet = { buf: buffer(earthTextureMesh), count: earthTextureMesh.length / 5 };
  const earthTexture = gl.createTexture();
  let earthTextureReady = false;
  gl.bindTexture(gl.TEXTURE_2D, earthTexture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([48,96,160,255]));
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  if (ASSET.earth_texture) {
    const earthImage = new Image();
    earthImage.onload = () => {
      gl.bindTexture(gl.TEXTURE_2D, earthTexture);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, earthImage);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      earthTextureReady = true;
    };
    earthImage.src = ASSET.earth_texture;
  }
  const meteorPointSet = { buf: gl.createBuffer(), count: 0 };
  const meteorTrailSet = { buf: gl.createBuffer(), count: 0 };
  const entryBurstSet = { buf: gl.createBuffer(), count: 0 };
  const flash = document.getElementById('flash');
  let orbitYaw = 0.0;

  function bind7Global(prog, buf, mvp) {
    gl.useProgram(prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog,'u_mvp'),false,mvp);
    gl.bindBuffer(gl.ARRAY_BUFFER,buf);
    const p=gl.getAttribLocation(prog,'a_pos'), c=gl.getAttribLocation(prog,'a_col');
    gl.enableVertexAttribArray(p); gl.vertexAttribPointer(p,3,gl.FLOAT,false,28,0);
    gl.enableVertexAttribArray(c); gl.vertexAttribPointer(c,4,gl.FLOAT,false,28,12);
  }
  function bind9Points(buf, mvp, time=0.0, windowS=2.0, densityScale=0.0) {
    gl.useProgram(pointProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(pointProg,'u_mvp'),false,mvp);
    gl.uniform1f(gl.getUniformLocation(pointProg,'u_time'),time);
    gl.uniform1f(gl.getUniformLocation(pointProg,'u_span'),1.0);
    gl.uniform1f(gl.getUniformLocation(pointProg,'u_window'),windowS);
    gl.uniform1f(gl.getUniformLocation(pointProg,'u_density'),densityScale);
    gl.bindBuffer(gl.ARRAY_BUFFER,buf);
    const ap=gl.getAttribLocation(pointProg,'a_pos'), ac=gl.getAttribLocation(pointProg,'a_col'), at=gl.getAttribLocation(pointProg,'a_t'), as=gl.getAttribLocation(pointProg,'a_size');
    gl.enableVertexAttribArray(ap); gl.vertexAttribPointer(ap,3,gl.FLOAT,false,36,0);
    gl.enableVertexAttribArray(ac); gl.vertexAttribPointer(ac,4,gl.FLOAT,false,36,12);
    gl.enableVertexAttribArray(at); gl.vertexAttribPointer(at,1,gl.FLOAT,false,36,28);
    gl.enableVertexAttribArray(as); gl.vertexAttribPointer(as,1,gl.FLOAT,false,36,32);
  }
  function bind9PlanetPoints(buf, mvp) {
    gl.useProgram(planetProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(planetProg,'u_mvp'),false,mvp);
    gl.bindBuffer(gl.ARRAY_BUFFER,buf);
    const ap=gl.getAttribLocation(planetProg,'a_pos'), ac=gl.getAttribLocation(planetProg,'a_col'), as=gl.getAttribLocation(planetProg,'a_size');
    gl.enableVertexAttribArray(ap); gl.vertexAttribPointer(ap,3,gl.FLOAT,false,36,0);
    gl.enableVertexAttribArray(ac); gl.vertexAttribPointer(ac,4,gl.FLOAT,false,36,12);
    gl.enableVertexAttribArray(as); gl.vertexAttribPointer(as,1,gl.FLOAT,false,36,32);
  }
  function bindEarthTexture(mvp, earthRot, sunDirWorld) {
    gl.useProgram(earthProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(earthProg,'u_mvp'),false,mvp);
    gl.uniform1i(gl.getUniformLocation(earthProg,'u_tex'),0);
    gl.uniformMatrix3fv(gl.getUniformLocation(earthProg,'u_rot'),false,new Float32Array(earthRot));
    gl.uniform3fv(gl.getUniformLocation(earthProg,'u_sun_dir'),new Float32Array(sunDirWorld));
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, earthTexture);
    gl.bindBuffer(gl.ARRAY_BUFFER,earthTextureSet.buf);
    const ap=gl.getAttribLocation(earthProg,'a_pos'), au=gl.getAttribLocation(earthProg,'a_uv');
    gl.enableVertexAttribArray(ap); gl.vertexAttribPointer(ap,3,gl.FLOAT,false,20,0);
    gl.enableVertexAttribArray(au); gl.vertexAttribPointer(au,2,gl.FLOAT,false,20,12);
  }
  function updatePlanetPoints(tDay) {
    const data = [];
    for (const planet of PLANETS) {
      const p = planetPosition(planet, tDay);
      const size = planet.name === "Jupiter" ? 17 : planet.name === "Saturn" ? 15 : planet.name === "Earth" ? 18 : 10;
      data.push(...p, ...planet.color, 1.0, -1.0, size);
    }
    data.push(0,0,0, 1.0,0.90,0.36,1.0, -1.0, 24.0);
    gl.bindBuffer(gl.ARRAY_BUFFER, planetPointSet.buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.DYNAMIC_DRAW);
    planetPointSet.count = data.length / 9;
  }
  function earthGeoPoint(earthPos, earthScale, earthTheta, latDeg, lonDeg, altKm=0.0) {
    const lat = latDeg * DEG, lon = -lonDeg * DEG;
    const radius = earthScale * (1.0 + altKm / 6371.0);
    const cl = Math.cos(lat);
    const x = radius * cl * Math.cos(lon);
    const y = radius * Math.sin(lat);
    const z = radius * cl * Math.sin(lon);
    const ec = Math.cos(earthTheta), es = Math.sin(earthTheta);
    return [earthPos[0] + ec*x + es*z, earthPos[1] + y, earthPos[2] - es*x + ec*z];
  }
  function fittedEntryBackDirection(earthPos, earthScale, earthTheta, entryLat, entryLon) {
    let ref = null;
    for (const row of ASSET.trajectory) {
      if (row.altKm > solar.entry_altitude_km + 1.0 && (!ref || row.altKm > ref.altKm)) ref = row;
    }
    if (!ref) return null;
    const high = earthGeoPoint(earthPos, earthScale, earthTheta, ref.lat, ref.lon, ref.altKm);
    const entry = earthGeoPoint(earthPos, earthScale, earthTheta, entryLat, entryLon, solar.entry_altitude_km);
    return norm(sub(high, entry));
  }
  function finalApproachDisplayPosition(fraction, entryPos, earthScale, backDirection) {
    fraction = Math.max(0, Math.min(1, fraction));
    const distanceKm = FINAL_APPROACH_DISTANCE_KM * (1.0 - fraction);
    return add(entryPos, mul(backDirection, earthScale * distanceKm / EARTH_RADIUS_KM));
  }
  function updateFinalApproachLine(entryPos, earthScale, backDirection) {
    const line = [];
    const n = Math.max(1, finalApproach.length - 1);
    let prev = finalApproachDisplayPosition(0.0, entryPos, earthScale, backDirection);
    for (let i=1; i<finalApproach.length; i++) {
      const q0 = (i - 1) / n;
      const q1 = i / n;
      const p = finalApproachDisplayPosition(q1, entryPos, earthScale, backDirection);
      line.push(...prev, 1.0,0.55,0.26,0.30 + 0.42*q0, ...p, 1.0,0.70,0.30,0.30 + 0.42*q1);
      prev = p;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, finalApproachSet.buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(line), gl.DYNAMIC_DRAW);
    finalApproachSet.count = line.length / 7;
  }
  function updateMeteorBuffers(u, unixNow, meteorPos, earthPos, visualMeteorPos, markerAlpha) {
    const trail = [];
    const steps = 120, startUnix = solar.entry_unix - 0.55 * periodDays * 86400;
    let prev = meteorPositionAtUnix(startUnix);
    for (let i=1; i<=steps; i++) {
      const t = startUnix + (unixNow - startUnix) * i / steps;
      const p = meteorPositionAtUnix(t);
      const localU = i / steps;
      trail.push(...prev, 1.0,0.62,0.25,0.10 + 0.58*localU, ...p, 1.0,0.76,0.32,0.12 + 0.66*localU);
      prev = p;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, meteorTrailSet.buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(trail), gl.DYNAMIC_DRAW);
    meteorTrailSet.count = trail.length / 7;
    const close = smoother((u - 0.54) / 0.34);
    const size = 11 + 43 * close;
    const data = [...visualMeteorPos, 1.0,0.62,0.20,markerAlpha, -1.0, size];
    gl.bindBuffer(gl.ARRAY_BUFFER, meteorPointSet.buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.DYNAMIC_DRAW);
    meteorPointSet.count = data.length / 9;
    const burst = smoother((u - 0.965) / 0.035);
    const burstData = burst > 0 ? [
      ...visualMeteorPos, 1.0,0.96,0.62,0.85, -1.0, 140.0 * burst,
      ...visualMeteorPos, 0.55,0.82,1.0,0.60, -1.0, 220.0 * burst,
    ] : [];
    gl.bindBuffer(gl.ARRAY_BUFFER, entryBurstSet.buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(burstData), gl.DYNAMIC_DRAW);
    entryBurstSet.count = burstData.length / 9;
  }
  function draw(u, w, h, flashAlpha=null) {
    const motionU = motionUForCameraU(u);
    const unixNow = timeUnix(u);
    const tDay = (unixNow - J2000_UNIX_SECOND) / 86400.0;
    const meteorPos = meteorPositionAtUnix(unixNow);
    const earthPos = planetPosition(PLANETS[2], tDay);
    const approach = smoother((u - 0.42) / 0.42);
    const close = smoother((u - 0.58) / 0.32);
    const chase = smoother((u - 0.28) / 0.42);
    const earthScale = 0.020 + 0.072 * close;
    const earthTheta = earthRotationAngle(unixNow);
    const entryLat = Number.isFinite(solar.entry_lat_deg) ? solar.entry_lat_deg : 41.98;
    const entryLon = Number.isFinite(solar.entry_lon_deg) ? solar.entry_lon_deg : -70.67;
    const entryPos = earthGeoPoint(earthPos, earthScale, earthTheta, entryLat, entryLon, solar.entry_altitude_km);
    const entryNormal = norm(sub(entryPos, earthPos));
    const entryBackDirection = fittedEntryBackDirection(earthPos, earthScale, earthTheta, entryLat, entryLon) || entryNormal;
    let entryRight = norm(cross([0,1,0], entryNormal));
    if (!Number.isFinite(entryRight[0])) entryRight = [1,0,0];
    const entryUp = norm(cross(entryNormal, entryRight));
    const offscreenPos = add(entryPos, add(mul(entryRight, earthScale * 7.2), add(mul(entryUp, earthScale * 3.2), mul(entryNormal, earthScale * 0.9))));
    const finalFraction = finalApproachFractionForCameraU(u);
    const visualMeteorPos = u >= FREEZE_CAMERA_START_U ? finalApproachDisplayPosition(finalFraction, entryPos, earthScale, entryBackDirection) : meteorPos;
    const markerAlpha = 1.0;
    updateFinalApproachLine(entryPos, earthScale, entryBackDirection);
    const solarEye = [10*Math.cos(0.78 + orbitYaw), 5.6, 13*Math.sin(0.78 + orbitYaw)];
    const fortyThousandKmEarthRadii = (6371.0 + 40000.0) / 6371.0;
    const chaseDistance = 0.22 + close * (earthScale * fortyThousandKmEarthRadii - 0.22);
    const chaseEye = add(entryPos, add(mul(entryNormal, chaseDistance), [0.0, 0.10 + 0.07*Math.sin(5*u), 0.0]));
    const target = mix([0,0,0], mix(entryPos, earthPos, close * 0.48), chase);
    const eye = mix(solarEye, chaseEye, approach);
    const view = lookAt(eye, target, [0,1,0]);
    const mvp = matMul(perspective((46 - 22*close)*DEG, w/h, 0.003, 180), view);
    updatePlanetPoints(tDay);
    updateMeteorBuffers(motionU, unixNow, meteorPos, earthPos, visualMeteorPos, markerAlpha);

    gl.disable(gl.DEPTH_TEST);
    bind9Points(starSet.buf, mvp, 0.0, 2.0, 0.0);
    gl.depthMask(false); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE); gl.drawArrays(gl.POINTS,0,starSet.count); gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);

    bind7Global(lineProg, planetOrbitSet.buf, mvp); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE); gl.drawArrays(gl.LINES,0,planetOrbitSet.count);
    if (u < FREEZE_CAMERA_START_U) {
      bind7Global(lineProg, meteorOrbitSet.buf, mvp); gl.drawArrays(gl.LINES,0,meteorOrbitSet.count);
      bind7Global(lineProg, meteorTrailSet.buf, mvp); gl.drawArrays(gl.LINES,0,meteorTrailSet.count);
    }
    bind7Global(lineProg, finalApproachSet.buf, mvp); gl.drawArrays(gl.LINES,0,finalApproachSet.count);

    const ec = Math.cos(earthTheta), es = Math.sin(earthTheta);
    const earthModel = new Float32Array([
      earthScale*ec,0,-earthScale*es,0,
      0,earthScale,0,0,
      earthScale*es,0,earthScale*ec,0,
      earthPos[0],earthPos[1],earthPos[2],1
    ]);
    const earthMvp = matMul(mvp, earthModel);
    const earthRot = [
      ec,0,-es,
      0,1,0,
      es,0,ec,
    ];
    const sunDirGlobal = norm(mul(earthPos, -1.0));
    if (earthTextureReady) {
      bindEarthTexture(earthMvp, earthRot, sunDirGlobal);
      gl.disable(gl.BLEND); gl.drawArrays(gl.TRIANGLES,0,earthTextureSet.count);
    } else {
      bind7Global(surfProg, earthSphereSet.buf, earthMvp);
      gl.disable(gl.BLEND); gl.drawArrays(gl.TRIANGLES,0,earthSphereSet.count);
    }

    bind9PlanetPoints(planetPointSet.buf, mvp);
    gl.depthMask(false); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA); gl.drawArrays(gl.POINTS,0,planetPointSet.count);
    bind9PlanetPoints(meteorPointSet.buf, mvp);
    gl.disable(gl.DEPTH_TEST);
    gl.blendFunc(gl.SRC_ALPHA,gl.ONE);
    gl.drawArrays(gl.POINTS,0,meteorPointSet.count);
    bind9PlanetPoints(entryBurstSet.buf, mvp);
    gl.drawArrays(gl.POINTS,0,entryBurstSet.count);
    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);

    flash.style.opacity = '0';
  }
  return {
    draw,
    timeUnix,
    orbitBy(dx) { orbitYaw += dx; },
    status(u) {
      const daysToEntry = Math.max(0, (solar.entry_unix - timeUnix(u)) / 86400.0);
      return daysToEntry > 3 ? `solar-system approach · ${daysToEntry.toFixed(0)} d before 100 km` : `Earth approach · ${daysToEntry.toFixed(2)} d before 100 km`;
    },
  };
})();
"""


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


def optional_data_uri(path: Path | None) -> str | None:
    try:
        return data_uri(path)
    except OSError as exc:
        print(f"warning: could not read optional asset {path}: {exc}", file=sys.stderr)
        return None


def load_tycho_catalog(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    prefix = "window.TYCHO_STAR_CATALOG = "
    start = text.find(prefix)
    if start < 0:
        return None
    payload = text[start + len(prefix) :].strip()
    if payload.endswith(";"):
        payload = payload[:-1]
    return json.loads(payload)


def true_anomaly_to_mean_deg(nu_deg: float, eccentricity: float) -> float:
    nu = math.radians(nu_deg)
    e = eccentricity
    eccentric_anomaly = 2.0 * math.atan2(
        math.sqrt(max(0.0, 1.0 - e)) * math.sin(nu / 2.0),
        math.sqrt(1.0 + e) * math.cos(nu / 2.0),
    )
    mean_anomaly = eccentric_anomaly - e * math.sin(eccentric_anomaly)
    return math.degrees(mean_anomaly)


def trajectory_crossing_unix(trajectory: list[dict], altitude_km: float) -> float | None:
    if not trajectory:
        return None
    rows = sorted(trajectory, key=lambda row: row["t_unix"])
    for a, b in zip(rows[:-1], rows[1:]):
        da = a["altKm"] - altitude_km
        db = b["altKm"] - altitude_km
        if da == 0.0:
            return a["t_unix"]
        if da * db <= 0.0:
            u = abs(da) / max(1e-9, abs(da) + abs(db))
            return a["t_unix"] + u * (b["t_unix"] - a["t_unix"])
    return min(rows, key=lambda row: abs(row["altKm"] - altitude_km))["t_unix"]


def trajectory_crossing_llh(trajectory: list[dict], altitude_km: float) -> dict[str, float] | None:
    if not trajectory:
        return None
    rows = sorted(trajectory, key=lambda row: row["t_unix"])
    for a, b in zip(rows[:-1], rows[1:]):
        da = a["altKm"] - altitude_km
        db = b["altKm"] - altitude_km
        if da == 0.0:
            return {"lat": a["lat"], "lon": a["lon"], "altKm": a["altKm"], "t_unix": a["t_unix"]}
        if da * db <= 0.0:
            u = abs(da) / max(1e-9, abs(da) + abs(db))
            return {
                "lat": a["lat"] + u * (b["lat"] - a["lat"]),
                "lon": a["lon"] + u * (b["lon"] - a["lon"]),
                "altKm": a["altKm"] + u * (b["altKm"] - a["altKm"]),
                "t_unix": a["t_unix"] + u * (b["t_unix"] - a["t_unix"]),
            }
    row = min(rows, key=lambda r: abs(r["altKm"] - altitude_km))
    return {"lat": row["lat"], "lon": row["lon"], "altKm": row["altKm"], "t_unix": row["t_unix"]}


def load_solar_orbit(path: Path | None, trajectory: list[dict]) -> dict | None:
    if path is None or not path.exists():
        return None
    import h5py

    with h5py.File(path, "r") as handle:
        elements = handle["kepler_HeliocentricMeanEcliptic_final_mean"][()]
        std = handle["kepler_HeliocentricMeanEcliptic_final_std"][()]
        t_ref_unix = float(handle.attrs.get("t_ref_unix", math.nan))
        t_ref_utc = str(handle.attrs.get("t_ref_utc", ""))
    a_au = float(elements[0]) / AU_M
    e = float(elements[1])
    i_deg = float(elements[2])
    omega_deg = float(elements[3])
    omega_node_deg = float(elements[4])
    nu_deg = float(elements[5])
    entry = trajectory_crossing_llh(trajectory, 100.0)
    entry_unix = entry["t_unix"] if entry is not None else None
    if entry_unix is None or not math.isfinite(entry_unix):
        entry_unix = t_ref_unix
    if entry is None:
        entry = {"lat": math.nan, "lon": math.nan, "altKm": 100.0, "t_unix": entry_unix}
    return {
        "source": str(path),
        "t_ref_unix": t_ref_unix,
        "t_ref_utc": t_ref_utc,
        "entry_unix": float(entry_unix),
        "entry_altitude_km": 100.0,
        "entry_lat_deg": float(entry["lat"]),
        "entry_lon_deg": float(entry["lon"]),
        "orbit": {
            "a_au": a_au,
            "e": e,
            "i_deg": i_deg,
            "omega_deg": omega_deg,
            "Omega_deg": omega_node_deg,
            "nu_deg": nu_deg,
            "M_deg": true_anomaly_to_mean_deg(nu_deg, e),
        },
        "orbit_std": {
            "a_au": float(std[0]) / AU_M,
            "e": float(std[1]),
            "i_deg": float(std[2]),
            "omega_deg": float(std[3]),
            "Omega_deg": float(std[4]),
            "nu_deg": float(std[5]),
        },
    }


def load_article_orbit(table: Path, solar: dict) -> dict:
    """Use the published table rather than potentially stale orbit HDF5 files.

    Keep the existing radar-scene entry anchor: these data are an orbital
    update, not a reprocessing of the atmospheric observations/dark flight.
    The table's orbital epoch is distinct from that visual entry time.
    """
    text = table.read_text(encoding="utf-8")
    rows = {}
    for line in text.splitlines():
        if line.lstrip().startswith("%"):
            continue
        cells = [cell.strip() for cell in line.split("&")]
        if len(cells) == 4:
            rows[(cells[0], cells[2])] = cells[3].rstrip("\\ ")

    def numbers(label: str, unit: str) -> list[float]:
        return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", rows[(label, unit)])]

    fields = {
        "a_au": ("Semimajor axis", "AU"),
        "e": ("Eccentricity", ""),
        "i_deg": ("Inclination", "deg"),
        "omega_deg": ("Argument of perihelion", "deg"),
        "Omega_deg": ("Longitude of ascending node", "deg"),
        "nu_deg": ("True anomaly at epoch", "deg"),
    }
    elements, std = {}, {}
    for key, (label, unit) in fields.items():
        values = numbers(label, unit)
        if len(values) != 2:
            raise ValueError(f"Expected estimate and uncertainty for {label}")
        elements[key], std[key] = values
    if not (elements["a_au"] > 0 and 0 <= elements["e"] < 1):
        raise ValueError("The viewer requires a bound elliptic orbit")
    epoch = datetime.fromisoformat(rows[("Epoch", "UTC")].removesuffix(" UTC")).replace(tzinfo=timezone.utc)
    version = re.search(r"^% trajectory_fit_version: (.+)$", text, re.M)
    if version is None:
        raise ValueError("Article table is missing trajectory-fit provenance")
    elements["M_deg"] = true_anomaly_to_mean_deg(elements["nu_deg"], elements["e"])
    result = dict(solar)
    result.update({
        "source": "article/reports/orbit_parameter_table.tex",
        "source_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "trajectory_fit_version": version[1],
        "orbit_epoch_unix": epoch.timestamp(),
        "orbit_epoch_utc": epoch.isoformat(),
        "orbit": elements,
        "orbit_std": std,
        "entry_speed_km_s": numbers("Ground-relative speed at 100 km", r"km s$^{-1}$")[0],
    })
    return result


def update_existing_viewer(source: Path, table: Path, out: Path) -> None:
    text = source.read_text(encoding="utf-8")
    match = re.search(r'<script id="asset" type="application/json">(.*?)</script>', text, re.S)
    if match is None:
        raise ValueError(f"No embedded viewer asset in {source}")
    asset = json.loads(match[1])
    asset["solar"] = load_article_orbit(table, asset["solar"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(asset), encoding="utf-8")
    print(f"Updated orbit from article fit {asset['solar']['trajectory_fit_version']}: {out}")


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
    logos = asset.get("logos", {})

    def logo_link(key: str, label: str, url: str, img_class: str) -> str:
        src = logos.get(key)
        if src:
            content = f'<img class="logo {img_class}" src="{src}" alt="{label}">'
        else:
            content = f'<span class="logo logo-text {img_class}">{label}</span>'
        return f'<a class="logo-link" href="{url}" target="_blank" rel="noopener">{content}</a>'

    logo_html = "\n  ".join(
        [
            logo_link("haystack", "MIT Haystack", "https://www.haystack.mit.edu/", "haystack"),
            logo_link("iap", "IAP", "https://www.iap-kborn.de/", "iap"),
            logo_link("uit", "UiT Space Physics", "https://en.uit.no/forskning/forskningsgrupper/gruppe?p_document_id=397987", "uit"),
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="bolide-css-version" content="{CSS_VERSION}">
<meta name="bolide-js-version" content="{JS_VERSION}">
<title>Cape Cod Bay Bolide May 30 2026</title>
<style data-version="{CSS_VERSION}">
html,body{{margin:0;height:100%;overflow:hidden;background:#05070b;color:#eaf5ff;font:13px/1.4 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}}
*,*:before,*:after{{box-sizing:border-box}}
canvas{{display:block;width:100vw;height:100vh;touch-action:none;background:radial-gradient(circle at 50% 38%,#15202d 0,#05070b 58%,#020307 100%)}}
.hud{{position:fixed;right:16px;top:14px;width:min(520px,calc(100vw - 32px));pointer-events:none}}
.panel{{pointer-events:auto;background:rgba(4,8,14,.72);border:1px solid rgba(150,210,255,.22);box-shadow:0 18px 55px rgba(0,0,0,.35);backdrop-filter:blur(10px);border-radius:8px;padding:12px 14px}}
.titlepanel{{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:start}}
h1{{font-size:15px;line-height:1.15;margin:0 0 7px 0;font-weight:700;letter-spacing:0}}
.hud-time{{font-variant-numeric:tabular-nums;color:#d6f4ff;font-size:12px;line-height:1.15;margin:0 0 7px 0;white-space:nowrap}}
.meta{{color:#aac1d3;font-size:12px;max-width:560px}}
.drawer-body{{grid-column:1 / -1;display:grid;gap:10px;margin-top:10px}}
.logo-links{{position:fixed;left:16px;bottom:16px;display:flex;gap:10px;align-items:center;z-index:5}}
.logo-link{{display:block;pointer-events:auto}}
.logo{{height:46px;max-width:160px;object-fit:contain;background:rgba(255,255,255,.94);border:1px solid rgba(255,255,255,.28);border-radius:6px;padding:7px 9px;box-shadow:0 10px 26px rgba(0,0,0,.22)}}
.logo-text{{display:inline-flex;align-items:center;justify-content:center;color:#102034;font-weight:700;text-decoration:none;min-width:64px}}
.logo-text.uit{{max-width:150px}}
.logo-text.haystack{{max-width:150px}}
.logo.uit{{max-width:70px}}
.logo.iap{{max-width:170px}}
.controls{{display:grid;grid-template-columns:auto auto minmax(180px,1fr);gap:9px;align-items:center}}
button{{height:30px;border-radius:6px;border:1px solid rgba(175,225,255,.3);background:#0f1b25;color:#eaf5ff;cursor:pointer}}
button{{min-width:0}}
button.active{{background:#1a5c7a;border-color:#57d0ff}}
.menubtn{{width:34px;padding:0;font-size:18px;line-height:1;justify-self:end}}
.hud.collapsed{{width:auto}}
.hud.collapsed .drawer-body,.hud.collapsed .meta{{display:none}}
.hud.collapsed .titlepanel{{grid-template-columns:auto auto;min-width:0}}
.hud.collapsed .titlepanel h1{{margin:0 0 4px 0}}
.hud.collapsed .hud-time{{margin:0}}
input[type=range]{{width:100%;accent-color:#58d7ff}}
.readout{{font-variant-numeric:tabular-nums;color:#d6f4ff;text-align:right}}
.legend{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;color:#bed3df}}
.dot{{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px;box-shadow:0 0 13px currentColor}}
.help{{color:#9fb2c0;border-top:1px solid rgba(150,210,255,.12);padding-top:8px}}
.altitude{{color:#d6f4ff;font-variant-numeric:tabular-nums}}
.scene-flash{{position:fixed;inset:0;background:white;opacity:0;pointer-events:none;z-index:100;transition:opacity 80ms linear}}
.label-layer{{position:fixed;inset:0;pointer-events:none;overflow:hidden}}
.station-label{{position:absolute;transform:translate(-50%,-140%);padding:3px 6px;border-radius:5px;background:rgba(3,7,12,.72);border:1px solid rgba(210,235,255,.22);color:#e8f8ff;font-size:11px;white-space:nowrap;text-shadow:0 1px 4px #000}}
.station-label.tx{{color:#ffe88a;border-color:rgba(255,232,138,.45)}}
.station-label.rx{{color:#9df7d0;border-color:rgba(157,247,208,.32)}}
.colorbar{{width:100%;color:#bed3df}}
.barlabel{{display:flex;justify-content:space-between;font-size:11px;color:#aebfcb;margin-bottom:5px}}
.bar{{position:relative;height:12px;border-radius:6px;background:linear-gradient(90deg,#30123b,#4662d7,#1bcfd4,#70e75f,#f9c63d,#e1442f);box-shadow:inset 0 0 0 1px rgba(255,255,255,.22),0 0 22px rgba(88,215,255,.12)}}
.tick{{position:absolute;top:-4px;width:2px;height:20px;border-radius:2px;background:white;box-shadow:0 0 10px rgba(255,255,255,.8)}}
@media (max-width: 820px){{
  .hud{{left:10px;right:10px;top:10px;width:auto;max-height:calc(100vh - 20px);overflow:auto}}
  .panel{{padding:9px 10px;border-radius:8px}}
  .titlepanel{{width:calc(100vw - 20px);grid-template-columns:minmax(0,1fr) auto;gap:8px}}
  .titlepanel .menubtn{{grid-column:2;grid-row:1}}
  h1{{font-size:14px;margin-bottom:5px}}
  .hud-time{{font-size:11px;max-width:calc(100vw - 94px);overflow:hidden;text-overflow:ellipsis}}
  .meta{{font-size:11px;max-width:none}}
  .logo-links{{left:10px;bottom:10px;gap:7px;flex-wrap:wrap;max-width:calc(100vw - 20px)}}
  .logo{{height:31px;max-width:112px;padding:5px 6px}}
  .logo.uit{{max-width:48px}}
  .logo.iap{{max-width:120px}}
  .controls{{width:100%;min-width:0;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:8px}}
  .controls button{{width:100%;padding:0 8px}}
  .controls input[type=range]{{width:100%;min-width:0}}
  .controls .readout{{text-align:left;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
  .controls input#time,.controls input#window{{grid-column:1 / -1}}
  .legend{{gap:7px;font-size:11px}}
  .help,.altitude{{font-size:11px}}
  .station-label{{font-size:10px;padding:2px 5px}}
  .hud.collapsed{{right:auto;max-height:none;overflow:visible}}
  .hud.collapsed .titlepanel{{width:auto;grid-template-columns:auto auto}}
}}
</style>
</head>
<body>
<canvas id="gl"></canvas>
<div class="hud collapsed">
  <div class="panel titlepanel">
    <div>
      <h1>Cape Cod Bay Bolide May 30 2026</h1>
      <div id="readout" class="hud-time">2026-05-30T18:06:19Z</div>
      <div class="meta">Actual measurements of bolide-created plasma using the MIT Haystack Observatory multi-static radar system, developed in collaboration with IAP Kühlungsborn, Germany and the University of Tromsø, Norway, painted onto the fitted ballistic path over a curved New England/Cape Cod map.</div>
    </div>
    <button id="menu" class="menubtn" aria-label="Expand menu">≡</button>
    <div class="drawer-body">
      <div class="controls">
        <button id="play" class="active">Pause</button>
        <button id="radar" class="active">Radar</button>
        <input id="time" type="range" min="0" max="1000" value="0">
        <button id="density">Density</button>
        <div id="winread" class="readout">26 s loop</div>
        <input id="window" type="range" min="8" max="80" value="26">
      </div>
      <div class="legend"><span style="color:#d8f5ff"><span class="dot"></span>plasma blobs</span><span style="color:#f6d56a"><span class="dot"></span>radar legs</span><span style="color:#8b7460"><span class="dot"></span>meteoroid rock</span><span style="color:#ffffff"><span class="dot"></span>painted tail</span><span style="color:#c08a5a"><span class="dot"></span>falling/landed fragments</span></div>
      <div class="colorbar">
        <div class="barlabel"><span id="barmin">-1.11 s</span><span>eased UTC playback</span><span id="barmax">3.29 s</span></div>
        <div class="bar"><div id="bartick" class="tick"></div></div>
      </div>
      <div id="altitude" class="altitude">viewer altitude 96.8 km</div>
      <div class="help">W/A/S/D fly · Q/E vertical · ←/→ orbit · left drag look · right drag orbit · wheel zoom · touch orbit · two-finger swipe/pinch zoom</div>
    </div>
  </div>
</div>
<div class="logo-links">
  {logo_html}
</div>
<div id="labels" class="label-layer"></div>
<div id="flash" class="scene-flash"></div>
<script id="asset" type="application/json">{asset_json}</script>
<script data-version="{JS_VERSION}">
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
const planetProg = program(`
attribute vec3 a_pos; attribute vec4 a_col; attribute float a_t; attribute float a_size;
uniform mat4 u_mvp; varying vec4 v_col;
void main(){{
  gl_Position = u_mvp * vec4(a_pos, 1.0);
  gl_PointSize = a_size;
  v_col = a_col;
}}`, `
precision mediump float; varying vec4 v_col;
void main(){{
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r = length(p);
  if (r > 1.0) discard;
  float edge = smoothstep(1.0, 0.94, r);
  vec2 light = normalize(vec2(-0.48, 0.74));
  float limb = sqrt(max(0.0, 1.0 - r*r));
  float shade = 0.50 + 0.42 * limb + 0.18 * max(0.0, dot(normalize(vec2(p.x + 0.0001, p.y + 0.0001)), light));
  vec3 color = v_col.rgb * shade;
  gl_FragColor = vec4(color, v_col.a * edge);
}}`);
const lineProg = program(`
attribute vec3 a_pos; attribute vec4 a_col; uniform mat4 u_mvp; varying vec4 v_col;
void main(){{gl_Position=u_mvp*vec4(a_pos,1.0); v_col=a_col;}}`, `
precision mediump float; varying vec4 v_col; void main(){{gl_FragColor=v_col;}}`);
const surfProg = program(`
attribute vec3 a_pos; attribute vec4 a_col; uniform mat4 u_mvp; varying vec4 v_col;
void main(){{gl_Position=u_mvp*vec4(a_pos,1.0); v_col=a_col;}}`, `
precision mediump float; varying vec4 v_col; void main(){{gl_FragColor=v_col;}}`);
const earthProg = program(`
attribute vec3 a_pos; attribute vec2 a_uv; uniform mat4 u_mvp; uniform mat3 u_rot; varying vec2 v_uv; varying vec3 v_norm_world;
void main(){{
  gl_Position = u_mvp * vec4(a_pos, 1.0);
  v_uv = a_uv;
  v_norm_world = normalize(u_rot * normalize(a_pos));
}}`, `
precision mediump float; uniform sampler2D u_tex; uniform vec3 u_sun_dir; varying vec2 v_uv; varying vec3 v_norm_world;
void main(){{
  vec3 tex = texture2D(u_tex, v_uv).rgb;
  float sun = dot(normalize(v_norm_world), normalize(u_sun_dir));
  float day = smoothstep(-0.035, 0.075, sun);
  vec3 night = tex * 0.055 + vec3(0.005, 0.012, 0.030);
  vec3 daylight = tex * (0.44 + 0.72 * max(sun, 0.0));
  vec3 terminator = vec3(0.12, 0.28, 0.55) * (1.0 - abs(day - 0.5) * 2.0) * 0.10;
  gl_FragColor = vec4(mix(night, daylight, day) + terminator, 1.0);
}}`);

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
function formatUnix(tUnix) {{
  const d = new Date(tUnix * 1000);
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
const txStation = ASSET.stations.find(s => s.kind === 'tx');
const rxStations = ASSET.stations.filter(s => s.kind === 'rx');
const txPos = txStation ? llh(txStation.lon, txStation.lat, Math.max(0.55, txStation.altKm + 0.55)) : null;
const rxPositions = rxStations.map(s => llh(s.lon, s.lat, Math.max(0.55, s.altKm + 0.55)));
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
const radarLegBuf = gl.createBuffer();
const fragmentBuf = gl.createBuffer();
let tailCount = 0;
let radarLegCount = 0;
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
function updateRadarLegs(t) {{
  const data=[];
  if (txPos && rxPositions.length) {{
    for (const p of ASSET.points) {{
      if (p.t > t) continue;
      const target = llh(p.lon, p.lat, p.altKm);
      const head = p.kind === 'head';
      const txAlpha = head ? 0.045 : 0.030;
      const rxAlpha = head ? 0.036 : 0.024;
      data.push(...txPos, 1.0,0.82,0.22,txAlpha, ...target, 1.0,0.74,0.18,txAlpha);
      for (const rx of rxPositions) {{
        data.push(...target, 0.30,0.86,1.0,rxAlpha, ...rx, 0.23,0.70,1.0,rxAlpha);
      }}
    }}
  }}
  radarLegCount = data.length / 7;
  gl.bindBuffer(gl.ARRAY_BUFFER,radarLegBuf);
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

{SOLAR_APPROACH_SCENE_JS}

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

let cam=[-46,95,138], yaw=0.34, pitch=-0.55, speed=35, playing=true, density=false, radarVisible=true, simT=tMin;
const keys = new Set(), span=Math.max(0.001,tMax-tMin), slider=document.getElementById('time'), readout=document.getElementById('readout'), win=document.getElementById('window'), winread=document.getElementById('winread'), bartick=document.getElementById('bartick'), barmin=document.getElementById('barmin'), barmax=document.getElementById('barmax'), altitude=document.getElementById('altitude');
const meteorEnd = Math.max(...ASSET.trajectory.map(p=>p.t));
const meteorSplit = 0.55;
const solarIntroSplit = SolarApproachScene ? 0.42 : 0.0;
let playProgress = 0.0;
function atmosphereProgress(globalProgress) {{
  if (!SolarApproachScene) return globalProgress;
  return Math.max(0.0, Math.min(1.0, (globalProgress - solarIntroSplit) / Math.max(1e-6, 1.0 - solarIntroSplit)));
}}
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
document.getElementById('radar').onclick=()=>{{radarVisible=!radarVisible; document.getElementById('radar').classList.toggle('active',radarVisible);}};
document.getElementById('menu').onclick=()=>{{
  const hud=document.querySelector('.hud');
  hud.classList.toggle('collapsed');
  document.getElementById('menu').setAttribute('aria-label', hud.classList.contains('collapsed') ? 'Expand menu' : 'Collapse menu');
}};
if (window.matchMedia('(max-width: 820px)').matches) {{
  const hud=document.querySelector('.hud');
  hud.classList.add('collapsed');
  document.getElementById('menu').setAttribute('aria-label', 'Expand menu');
}}
slider.oninput=()=>{{playProgress=+slider.value/1000; simT=progressToTime(atmosphereProgress(playProgress)); playing=false; play.textContent='Play'; play.classList.remove('active');}};
win.oninput=()=>{{winread.textContent=(+win.value).toFixed(0)+' s loop';}};
barmin.textContent = tMin.toFixed(2) + ' s';
barmax.textContent = tMax.toFixed(2) + ' s';
function keyName(e) {{
  if (e.key === 'ArrowLeft') return 'arrowleft';
  if (e.key === 'ArrowRight') return 'arrowright';
  if (e.key === 'ArrowUp') return 'arrowup';
  if (e.key === 'ArrowDown') return 'arrowdown';
  return e.key.toLowerCase();
}}
addEventListener('keydown',e=>{{
  const key = keyName(e);
  keys.add(key);
  if (key.startsWith('arrow')) e.preventDefault();
}});
addEventListener('keyup',e=>{{
  const key = keyName(e);
  keys.delete(key);
  if (key.startsWith('arrow')) e.preventDefault();
}});
let drag=false, orbitDrag=false, orbitFollow=true, lastX=0,lastY=0;
aimAt(interpPath(simT));
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
let touchMode='', touchX=0, touchY=0, touchDist=0, touchMidY=0;
function touchMid(touches) {{
  return [(touches[0].clientX + touches[1].clientX) * 0.5, (touches[0].clientY + touches[1].clientY) * 0.5];
}}
function touchDistance(touches) {{
  return Math.hypot(touches[0].clientX - touches[1].clientX, touches[0].clientY - touches[1].clientY);
}}
canvas.addEventListener('touchstart', e=>{{
  e.preventDefault();
  orbitFollow=true;
  if (e.touches.length === 1) {{
    touchMode='orbit';
    touchX=e.touches[0].clientX;
    touchY=e.touches[0].clientY;
  }} else if (e.touches.length >= 2) {{
    touchMode='panzoom';
    touchDist=touchDistance(e.touches);
    touchMidY=touchMid(e.touches)[1];
  }}
}}, {{passive:false}});
canvas.addEventListener('touchmove', e=>{{
  e.preventDefault();
  if (touchMode === 'orbit' && e.touches.length === 1) {{
    const x=e.touches[0].clientX, y=e.touches[0].clientY;
    orbitAround(interpPath(simT), x-touchX, y-touchY);
    touchX=x; touchY=y;
  }} else if (e.touches.length >= 2) {{
    const dist=touchDistance(e.touches);
    const mid=touchMid(e.touches);
    const forward=[Math.sin(yaw)*Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw)*Math.cos(pitch)];
    const dolly=(dist-touchDist)*0.045 + (touchMidY-mid[1])*0.070;
    for(let i=0;i<3;i++) cam[i]+=forward[i]*dolly;
    touchDist=dist;
    touchMidY=mid[1];
    aimAt(interpPath(simT));
  }}
}}, {{passive:false}});
canvas.addEventListener('touchend', e=>{{
  if (e.touches.length === 0) {{
    touchMode='';
  }} else if (e.touches.length === 1) {{
    touchMode='orbit';
    touchX=e.touches[0].clientX;
    touchY=e.touches[0].clientY;
  }}
}}, {{passive:false}});
let last=performance.now();
function frame(now) {{
  const dt=Math.min(0.05,(now-last)/1000); last=now;
  if (playing) {{
    playProgress = (playProgress + dt / Math.max(1.0, +win.value)) % 1.0;
    simT = progressToTime(atmosphereProgress(playProgress));
  }}
  const w=canvas.clientWidth,h=canvas.clientHeight; if(canvas.width!==w||canvas.height!==h){{canvas.width=w;canvas.height=h;gl.viewport(0,0,w,h);}}
  gl.enable(gl.DEPTH_TEST); gl.clearColor(0.015,0.02,0.032,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  if (SolarApproachScene && playProgress < solarIntroSplit) {{
    const solarU = Math.max(0.0, Math.min(1.0, playProgress / solarIntroSplit));
    const holdU = Math.min(0.26, Math.max(0.08, 2.0 / (Math.max(1.0, +win.value) * solarIntroSplit)));
    const impactU = 1.0 - holdU;
    const motionU = solarU < impactU ? solarU / impactU : 1.0;
    const flashQ = Math.max(0.0, Math.min(1.0, (solarU - impactU) / Math.max(1e-6, holdU)));
    const flashAlpha = 0.0;
    if(keys.has('arrowleft')) SolarApproachScene.orbitBy(-0.85 * dt);
    if(keys.has('arrowright')) SolarApproachScene.orbitBy(0.85 * dt);
    for (const el of stationLabels) el.style.display = 'none';
    SolarApproachScene.draw(motionU, w, h, flashAlpha);
    slider.value = Math.round(1000 * playProgress);
    readout.textContent = formatUnix(SolarApproachScene.timeUnix(motionU));
    altitude.textContent = flashQ > 0 ? '100 km entry flash' : SolarApproachScene.status(motionU);
    bartick.style.left = `calc(${{Math.max(0, Math.min(1, playProgress)) * 100}}% - 1px)`;
    requestAnimationFrame(frame);
    return;
  }}
  document.getElementById('flash').style.opacity = '0';
  simT = progressToTime(atmosphereProgress(playProgress));
  const forward=[Math.sin(yaw)*Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw)*Math.cos(pitch)];
  const right=[Math.cos(yaw),0,Math.sin(yaw)];
  const step=speed*dt;
  if(keys.has('w')) for(let i=0;i<3;i++) cam[i]+=forward[i]*step;
  if(keys.has('s')) for(let i=0;i<3;i++) cam[i]-=forward[i]*step;
  if(keys.has('d')) for(let i=0;i<3;i++) cam[i]+=right[i]*step;
  if(keys.has('a')) for(let i=0;i<3;i++) cam[i]-=right[i]*step;
  if(keys.has('e')) cam[1]+=step; if(keys.has('q')) cam[1]-=step;
  if(keys.has('arrowleft')) {{ orbitFollow=true; orbitAround(interpPath(simT), -180.0 * dt, 0.0); }}
  if(keys.has('arrowright')) {{ orbitFollow=true; orbitAround(interpPath(simT), 180.0 * dt, 0.0); }}
  if(orbitFollow) aimAt(interpPath(simT));
  const view=lookAt(cam,[cam[0]+forward[0],cam[1]+forward[1],cam[2]+forward[2]],[0,1,0]);
  const mvp=matMul(perspective(58*DEG,w/h,0.1,800),view);
  updateStationLabels(mvp,w,h);
  function bind7(prog, buf){{ gl.useProgram(prog); gl.uniformMatrix4fv(gl.getUniformLocation(prog,'u_mvp'),false,mvp); gl.bindBuffer(gl.ARRAY_BUFFER,buf); const p=gl.getAttribLocation(prog,'a_pos'), c=gl.getAttribLocation(prog,'a_col'); gl.enableVertexAttribArray(p); gl.vertexAttribPointer(p,3,gl.FLOAT,false,28,0); gl.enableVertexAttribArray(c); gl.vertexAttribPointer(c,4,gl.FLOAT,false,28,12); }}
  updateRock(simT);
  updateFragments(simT);
  updateTail(simT);
  updateRadarLegs(simT);
  bind7(surfProg,surfBuf); gl.disable(gl.BLEND); gl.drawArrays(gl.TRIANGLES,0,surfCount);
  bind7(lineProg,lineBuf); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE); gl.drawArrays(gl.LINES,0,lineCount);
  if (radarVisible) {{
    bind7(lineProg,radarLegBuf); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE); gl.depthMask(false); gl.drawArrays(gl.LINES,0,radarLegCount); gl.depthMask(true);
  }}
  bind7(lineProg,tailBuf); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE); gl.drawArrays(gl.LINES,0,tailCount);
  gl.useProgram(pointProg); gl.uniformMatrix4fv(gl.getUniformLocation(pointProg,'u_mvp'),false,mvp); gl.uniform1f(gl.getUniformLocation(pointProg,'u_time'),simT); gl.uniform1f(gl.getUniformLocation(pointProg,'u_span'),span); gl.uniform1f(gl.getUniformLocation(pointProg,'u_window'),+win.value/1000); gl.uniform1f(gl.getUniformLocation(pointProg,'u_density'),density?1:0);
  gl.bindBuffer(gl.ARRAY_BUFFER,pointBuf);
  const ap=gl.getAttribLocation(pointProg,'a_pos'), ac=gl.getAttribLocation(pointProg,'a_col'), at=gl.getAttribLocation(pointProg,'a_t'), as=gl.getAttribLocation(pointProg,'a_size');
  gl.enableVertexAttribArray(ap); gl.vertexAttribPointer(ap,3,gl.FLOAT,false,36,0);
  gl.enableVertexAttribArray(ac); gl.vertexAttribPointer(ac,4,gl.FLOAT,false,36,12);
  gl.enableVertexAttribArray(at); gl.vertexAttribPointer(at,1,gl.FLOAT,false,36,28);
  gl.enableVertexAttribArray(as); gl.vertexAttribPointer(as,1,gl.FLOAT,false,36,32);
  gl.depthMask(false); gl.blendFunc(gl.SRC_ALPHA,gl.ONE); gl.drawArrays(gl.POINTS,0,pointCount); gl.depthMask(true);
  gl.disable(gl.DEPTH_TEST);
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
  gl.enable(gl.DEPTH_TEST);
  gl.uniform1f(gl.getUniformLocation(pointProg,'u_time'),0.0);
  gl.bindBuffer(gl.ARRAY_BUFFER,stationBuf);
  gl.vertexAttribPointer(ap,3,gl.FLOAT,false,36,0);
  gl.vertexAttribPointer(ac,4,gl.FLOAT,false,36,12);
  gl.vertexAttribPointer(at,1,gl.FLOAT,false,36,28);
  gl.vertexAttribPointer(as,1,gl.FLOAT,false,36,32);
  gl.depthMask(false); gl.blendFunc(gl.SRC_ALPHA,gl.ONE); gl.drawArrays(gl.POINTS,0,stationCount); gl.depthMask(true);
  const timeFrac = SolarApproachScene ? solarIntroSplit + (1.0 - solarIntroSplit) * timeToProgress(simT) : timeToProgress(simT);
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
    parser.add_argument("--orbit", default=DEFAULT_ORBIT)
    parser.add_argument("--article-orbit-table", type=Path, help="Authoritative article orbital-element table (including epoch)")
    parser.add_argument("--existing-viewer", type=Path, help="Update only the orbit in an existing viewer, preserving embedded radar data")
    parser.add_argument("--tycho", default=DEFAULT_TYCHO)
    parser.add_argument("--earth-texture", default=DEFAULT_EARTH_TEXTURE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.existing_viewer:
        if not args.article_orbit_table:
            parser.error("--existing-viewer requires --article-orbit-table")
        update_existing_viewer(args.existing_viewer, args.article_orbit_table, Path(args.out))
        return

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
    fit_center = {
        "lat": sum(p["lat"] for p in center_points) / len(center_points),
        "lon": sum(p["lon"] for p in center_points) / len(center_points),
    }
    points, rejected_points = annotate_by_ballistic_path(raw_points, trajectory, fit_center, args.path_residual_max_km)
    strewn = strewn_field_rows(ballistic, t0_unix)
    stations = station_marker_rows("2026_05_30_bolide")
    solar = load_solar_orbit(Path(args.orbit) if args.orbit else None, trajectory)
    if args.article_orbit_table:
        if solar is None:
            parser.error("--article-orbit-table requires an orbit file for the scene entry anchor")
        solar = load_article_orbit(args.article_orbit_table, solar)
    tycho = load_tycho_catalog(Path(args.tycho) if args.tycho else None)
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
    map_geo = center_points + strewn["nominalPath"] + extra_geo
    if strewn["nominal"]:
        map_geo.append(strewn["nominal"])
    center_source = strewn["nominal"] if strewn["nominal"] else fit_center
    center = {"lat": center_source["lat"], "lon": center_source["lon"]}
    lat_vals = [p["lat"] for p in map_geo]
    lon_vals = [p["lon"] for p in map_geo]
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
        "solar": solar,
        "tycho": tycho,
        "earth_texture": optional_data_uri(Path(args.earth_texture)) if args.earth_texture else None,
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
