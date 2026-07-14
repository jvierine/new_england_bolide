#!/usr/bin/env python3
"""Scrape meteoriteorbits.info and compare known meteorite orbits to Cape Cod Bay."""

from __future__ import annotations

import argparse
import html
import math
import re
import ssl
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE_URL = "https://www.meteoriteorbits.info/"
DEFAULT_CSV = Path("reports/meteorite_orbits_info_orbits.csv")
DEFAULT_ORBIT_OUT = Path("reports/figures/meteorite_orbits_xy.png")
DEFAULT_AE_OUT = Path("reports/figures/meteorite_orbits_ae.png")
DEFAULT_COMBINED_OUT = Path("reports/figures/meteorite_orbits_comparison.png")


def default_macros_path() -> Path:
    for candidate in (
        Path("../cape_cod_bay_paper/reports/orbit_analysis_macros.tex"),
        Path("../cape_cod_bay/reports/orbit_analysis_macros.tex"),
        Path("reports/orbit_analysis_macros.tex"),
    ):
        if candidate.exists():
            return candidate
    return Path("reports/orbit_analysis_macros.tex")


@dataclass
class CapeCodOrbit:
    a_au: float
    a_sigma_au: float
    e: float
    e_sigma: float


class TablePressParser(HTMLParser):
    def __init__(self, table_id: str) -> None:
        super().__init__(convert_charrefs=False)
        self.table_id = table_id
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_cell: list[str] = []
        self.current_row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "table" and attrs_dict.get("id") == self.table_id:
            self.in_table = True
        if not self.in_table:
            return
        if tag == "tr":
            self.in_row = True
            self.current_row = []
        elif tag in {"th", "td"} and self.in_row:
            self.in_cell = True
            self.current_cell = []
        elif tag == "br" and self.in_cell:
            self.current_cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if not self.in_table:
            return
        if tag in {"th", "td"} and self.in_cell:
            text = html.unescape("".join(self.current_cell))
            text = re.sub(r"\s+", " ", text).strip()
            self.current_row.append(text)
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self.current_row:
                self.rows.append(self.current_row)
            self.in_row = False
        elif tag == "table":
            self.in_table = False

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell.append(data)

    def handle_entityref(self, name: str) -> None:
        if self.in_cell:
            self.current_cell.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self.in_cell:
            self.current_cell.append(f"&#{name};")


def read_source_html(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = ssl.create_default_context()
    with urlopen(request, timeout=30, context=context) as response:
        return response.read().decode("utf-8", errors="replace")


def first_float(value: object) -> float:
    if pd.isna(value):
        return math.nan
    text = str(value)
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not match:
        return math.nan
    return float(match.group(0))


def read_meteorite_orbits(url: str) -> pd.DataFrame:
    html = read_source_html(url)
    parser = TablePressParser("tablepress-1")
    parser.feed(html)
    if len(parser.rows) < 2:
        raise RuntimeError(f"No TablePress orbit rows found at {url}")
    df = pd.DataFrame(parser.rows[1:], columns=parser.rows[0])
    expected = {"Name", "a (AU)", "e", "i (\u00b0)", "Arg. Peri. (\u00b0)", "Asc. node (\u00b0)"}
    missing = expected.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing expected columns from {url}: {sorted(missing)}")

    clean = pd.DataFrame(
        {
            "abbrev": df["Ab."].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
            "name": df["Name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
            "type": df["Type"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
            "location": df["Loc."].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
            "date_utc": df["Date (UTC)"].astype(str).str.strip(),
            "a_au": df["a (AU)"].map(first_float),
            "e": df["e"].map(first_float),
            "i_deg": df["i (\u00b0)"].map(first_float),
            "arg_peri_deg": df["Arg. Peri. (\u00b0)"].map(first_float),
            "node_deg": df["Asc. node (\u00b0)"].map(first_float),
            "reference": df["Ref."].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
        }
    )
    good_ae = (
        np.isfinite(clean["a_au"])
        & np.isfinite(clean["e"])
        & (clean["a_au"] > 0.0)
        & (clean["e"] >= 0.0)
        & (clean["e"] < 1.0)
    )
    clean = clean.loc[good_ae].reset_index(drop=True)
    clean["has_full_orbit"] = (
        np.isfinite(clean["i_deg"])
        & np.isfinite(clean["arg_peri_deg"])
        & np.isfinite(clean["node_deg"])
    )
    return clean


def read_macro_value(path: Path, name: str) -> float:
    pattern = re.compile(rf"\\newcommand\{{\\{re.escape(name)}\}}\{{([^}}]+)\}}")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    raise KeyError(f"Macro \\{name} not found in {path}")


def read_cape_cod_orbit(path: Path) -> CapeCodOrbit:
    return CapeCodOrbit(
        a_au=read_macro_value(path, "OrbitSemiMajorMeanAU"),
        a_sigma_au=read_macro_value(path, "OrbitSemiMajorSigmaAU"),
        e=read_macro_value(path, "OrbitEccentricityMean"),
        e_sigma=read_macro_value(path, "OrbitEccentricitySigma"),
    )


def orbit_xy(a_au: float, e: float, i_deg: float, arg_peri_deg: float, node_deg: float, n: int = 720) -> tuple[np.ndarray, np.ndarray]:
    nu = np.linspace(0.0, 2.0 * np.pi, n)
    r = a_au * (1.0 - e * e) / (1.0 + e * np.cos(nu))
    x_p = r * np.cos(nu)
    y_p = r * np.sin(nu)

    inc = np.deg2rad(i_deg)
    arg = np.deg2rad(arg_peri_deg)
    node = np.deg2rad(node_deg)

    cos_o, sin_o = np.cos(node), np.sin(node)
    cos_i, sin_i = np.cos(inc), np.sin(inc)
    cos_w, sin_w = np.cos(arg), np.sin(arg)

    x = (cos_o * cos_w - sin_o * sin_w * cos_i) * x_p + (-cos_o * sin_w - sin_o * cos_w * cos_i) * y_p
    y = (sin_o * cos_w + cos_o * sin_w * cos_i) * x_p + (-sin_o * sin_w + cos_o * cos_w * cos_i) * y_p
    return x, y


def configure_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="0.88", linewidth=0.7, zorder=0)


def draw_orbit_panel(ax, df: pd.DataFrame, cape_cod: CapeCodOrbit) -> None:
    full = df.loc[df["has_full_orbit"]].copy()
    for _, row in full.iterrows():
        x, y = orbit_xy(row.a_au, row.e, row.i_deg, row.arg_peri_deg, row.node_deg)
        ax.plot(x, y, color="0.45", alpha=0.28, linewidth=0.8, zorder=1)

    earth = plt.Circle((0.0, 0.0), 1.0, fill=False, color="#2d74b2", linewidth=1.1, alpha=0.75, zorder=2)
    mars = plt.Circle((0.0, 0.0), 1.524, fill=False, color="#b95f43", linewidth=0.9, alpha=0.55, zorder=2)
    jupiter = plt.Circle((0.0, 0.0), 5.203, fill=False, color="#8b6f4e", linewidth=0.9, alpha=0.35, zorder=2)
    ax.add_patch(earth)
    ax.add_patch(mars)
    ax.add_patch(jupiter)
    ax.scatter([0.0], [0.0], s=70, marker="*", color="#d08b18", edgecolor="white", linewidth=0.5, zorder=5, label="Sun")

    # Use the manuscript mean inclination/angular elements when available from the table-equivalent macros.
    # For this comparison, the highlighted orbit shape is set by the manuscript a/e; orientation is illustrative.
    x, y = orbit_xy(cape_cod.a_au, cape_cod.e, 19.21, 93.4, 68.99209)
    ax.plot(x, y, color="#b51f1f", linewidth=2.4, zorder=6, label="Cape Cod Bay")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-5.4, 5.4)
    ax.set_ylim(-5.4, 5.4)
    ax.set_xlabel("Heliocentric ecliptic x (AU)")
    ax.set_ylabel("Heliocentric ecliptic y (AU)")
    ax.set_title(f"Known meteorite-fall orbits ({len(full)} complete tracks)")
    configure_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=9)


def draw_ae_panel(ax, df: pd.DataFrame, cape_cod: CapeCodOrbit) -> None:
    types = df["type"].fillna("").astype(str)
    carbonaceous = types.str.contains(r"\bC|CI|CM|CO|CV|CR|CK|C1|C2", case=False, regex=True)
    iron = types.str.contains("Iron", case=False, regex=False)
    ordinary = ~(carbonaceous | iron)

    ax.scatter(df.loc[ordinary, "a_au"], df.loc[ordinary, "e"], s=32, color="0.45", alpha=0.72, label="Other meteorite falls", zorder=3)
    ax.scatter(df.loc[carbonaceous, "a_au"], df.loc[carbonaceous, "e"], s=44, color="#26734d", alpha=0.86, label="Carbonaceous/ungrouped C", zorder=4)
    ax.scatter(df.loc[iron, "a_au"], df.loc[iron, "e"], s=58, marker="s", color="#6f5aa8", alpha=0.90, label="Iron", zorder=5)

    ax.errorbar(
        [cape_cod.a_au],
        [cape_cod.e],
        xerr=[cape_cod.a_sigma_au],
        yerr=[cape_cod.e_sigma],
        fmt="*",
        markersize=16,
        color="#b51f1f",
        markeredgecolor="white",
        markeredgewidth=0.8,
        elinewidth=1.5,
        capsize=3.0,
        label="Cape Cod Bay bolide",
        zorder=8,
    )

    a_grid = np.linspace(0.55, 5.25, 600)
    earth_cross = np.abs(1.0 - 1.0 / a_grid)
    ax.plot(a_grid, earth_cross, color="#2d74b2", linewidth=1.2, linestyle="--", label="Earth-crossing boundary", zorder=2)
    ax.fill_between(a_grid, earth_cross, 1.0, color="#2d74b2", alpha=0.05, zorder=1)

    ax.set_xlim(0.55, 5.25)
    ax.set_ylim(0.0, 0.95)
    ax.set_xlabel("Semimajor axis, $a$ (AU)")
    ax.set_ylabel("Eccentricity, $e$")
    ax.set_title(f"$a$-$e$ distribution ({len(df)} meteorites)")
    configure_axes(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)


def save_figures(df: pd.DataFrame, cape_cod: CapeCodOrbit, orbit_out: Path, ae_out: Path, combined_out: Path) -> None:
    orbit_out.parent.mkdir(parents=True, exist_ok=True)
    ae_out.parent.mkdir(parents=True, exist_ok=True)
    combined_out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    draw_orbit_panel(ax, df, cape_cod)
    fig.savefig(orbit_out, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.3), constrained_layout=True)
    draw_ae_panel(ax, df, cape_cod)
    fig.savefig(ae_out, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), constrained_layout=True)
    draw_orbit_panel(axes[0], df, cape_cod)
    draw_ae_panel(axes[1], df, cape_cod)
    fig.savefig(combined_out, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--macros", type=Path, default=default_macros_path())
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--orbit-out", type=Path, default=DEFAULT_ORBIT_OUT)
    parser.add_argument("--ae-out", type=Path, default=DEFAULT_AE_OUT)
    parser.add_argument("--combined-out", type=Path, default=DEFAULT_COMBINED_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_meteorite_orbits(args.source_url)
    cape_cod = read_cape_cod_orbit(args.macros)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    save_figures(df, cape_cod, args.orbit_out, args.ae_out, args.combined_out)
    print(f"Wrote {len(df)} meteorite orbits to {args.csv}")
    print(f"Wrote {args.orbit_out}")
    print(f"Wrote {args.ae_out}")
    print(f"Wrote {args.combined_out}")


if __name__ == "__main__":
    main()
