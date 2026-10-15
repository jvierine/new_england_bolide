//! Archived observing geometry rendered with rustplotlib.
//! Equal local geographic scales at 42.225 N; labels and markers sized in mm.
use anyhow::Result;
use rustplotlib::latex::Glyph;
use rustplotlib::*;
use std::{collections::BTreeMap, path::PathBuf};
const WEST: f64 = -74.35;
const EAST: f64 = -69.95;
const SOUTH: f64 = 40.55;
const NORTH: f64 = 43.9;
const X: f64 = 16.;
const Y: f64 = 5.;
const W: f64 = 109.;
const H: f64 = 111.856551724;
const RX: &str = "#252c31";
const TX: &str = "#a37623";
const WEATHER: &str = "#367a86";
const CAMERA: &str = "#b06c2f";
const PATH: &str = "#65706e";
const OBS: &str = "#285fa0";
const DET: &str = "#b84742";
fn points(h: &hdf5::File, key: &str) -> Result<Vec<[f64; 2]>> {
    Ok(h.dataset(key)?
        .read_raw::<f64>()?
        .chunks_exact(2)
        .map(|v| [v[0], v[1]])
        .collect())
}
fn line(p: &mut Panel, pts: Vec<[f64; 2]>, color: &str, width: f64, dash: bool) {
    let mut l = Line::new(pts, color);
    l.width_pt = width;
    if dash {
        l.dash = Some("1.6 1.1".into());
    }
    p.push(Layer::Line(l));
}
fn xy(a: [f64; 2]) -> [f64; 2] {
    [
        X + (a[0] - WEST) / (EAST - WEST) * W,
        Y + (NORTH - a[1]) / (NORTH - SOUTH) * H,
    ]
}
fn data(a: [f64; 2]) -> [f64; 2] {
    [
        WEST + (a[0] - X) / W * (EAST - WEST),
        NORTH - (a[1] - Y) / H * (NORTH - SOUTH),
    ]
}
fn scale(f: &mut Figure, c: &str) -> Result<usize> {
    let rgb = |i| u8::from_str_radix(&c[i..i + 2], 16).unwrap() as f64 / 255.;
    let v = [rgb(1), rgb(3), rgb(5)];
    Ok(f.scale(Scale::new(
        [0., 1.],
        Normalization::Linear,
        ColorMap::new(vec![v, v])?,
    )?))
}
fn marker(
    p: &mut Panel,
    at: [f64; 2],
    r: f64,
    shape: &str,
    fill: usize,
    color: &str,
) -> Result<()> {
    let a = xy(at);
    let offsets: Vec<[f64; 2]> = match shape {
        "triangle" => vec![[0., -r], [-r * 0.86, r * 0.5], [r * 0.86, r * 0.5]],
        "square" => vec![[-r, -r], [r, -r], [r, r], [-r, r]],
        "diamond" => vec![[0., -r], [r, 0.], [0., r], [-r, 0.]],
        "star" => (0..10)
            .map(|i| {
                let t = -std::f64::consts::FRAC_PI_2 + i as f64 * std::f64::consts::PI / 5.;
                let r = if i % 2 == 0 { r } else { r * 0.43 };
                [r * t.cos(), r * t.sin()]
            })
            .collect(),
        _ => (0..32)
            .map(|i| {
                let t = i as f64 * std::f64::consts::TAU / 32.;
                [r * t.cos(), r * t.sin()]
            })
            .collect(),
    };
    let pts: Vec<_> = offsets
        .iter()
        .map(|v| data([a[0] + v[0], a[1] + v[1]]))
        .collect();
    p.push(Layer::PolygonMesh(PolygonMesh::new(
        vec![pts.clone()],
        vec![0.5],
        fill,
    )?));
    let mut border = pts;
    border.push(border[0]);
    line(p, border, color, 0.45, false);
    Ok(())
}
fn label(
    f: &mut Figure,
    g: &BTreeMap<String, Glyph>,
    s: &str,
    x: f64,
    y: f64,
    c: &str,
    left: bool,
) {
    let x = if left {
        x + g[s].width_pt * 25.4 / 72. / 2.
    } else {
        x
    };
    let mut l = Label::new(s, x, y);
    l.color = c.into();
    f.labels.push(l);
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let h = hdf5::File::open(&args[1])?;
    let out = PathBuf::from(&args[2]);
    let mut f = Figure::new(IntendedUse::Custom {
        width_mm: 128.4,
        font_pt: 10.,
    });
    f.height_mm = 125.;
    f.tight_export = false;
    f.raster_dpi = 450.;
    f.latex.cache = out.parent().unwrap().join(".geometry-latex-cache");
    f.latex.preamble="\\usepackage[T1]{fontenc}\n\\usepackage{amsmath,amssymb}\n\\usepackage{newtxtext,newtxmath}".into();
    f.latex.font_pt = 10.;
    let specs = [
        ("Pomfret", RX, "triangle", 2.3, -3.),
        ("Siena", RX, "triangle", 2.6, 1.6),
        ("Amherst", RX, "triangle", 2.5, -2.5),
        ("Clinton", RX, "triangle", 2.5, 2.1),
        ("Beverly", RX, "triangle", 2.5, -2.2),
        ("TX", TX, "star", 3., 0.),
        ("Mechanicville", CAMERA, "square", 2.6, -2.3),
        ("KENX", WEATHER, "diamond", 2.7, 0.),
        ("KOKX", WEATHER, "diamond", 2.7, 0.),
        ("TBOS", WEATHER, "diamond", 2.7, -1.),
        ("KBOX", WEATHER, "diamond", -8., 4.7),
    ];
    let entries = [
        ("Meteor-radar receiver", RX, "triangle"),
        ("Meteor-radar transmitter", TX, "star"),
        ("Weather radar", WEATHER, "diamond"),
        ("Camera", CAMERA, "square"),
        ("Inferred trajectory", PATH, "line"),
        ("Meteor-radar observations", OBS, "line"),
        ("Weather-radar measurements", DET, "circle"),
    ];
    let mut texts: Vec<String> = specs
        .iter()
        .map(|v| v.0.to_string())
        .chain(entries.iter().map(|v| v.0.to_string()))
        .collect();
    let xt: Vec<String> = (70..=74)
        .rev()
        .map(|v| format!(r"${v}^\circ\mathrm{{W}}$"))
        .collect();
    let yt: Vec<String> = [41., 42., 43.]
        .iter()
        .map(|v| format!(r"${v}^\circ\mathrm{{N}}$"))
        .collect();
    texts.extend(xt.clone());
    texts.extend(yt.clone());
    let glyphs = f.latex.typeset(texts)?;
    let land = scale(&mut f, "#f1f2ee")?;
    let mut p = Panel::new(
        Rect::new(X, Y, W, H),
        Axes::new([WEST, EAST], [SOUTH, NORTH], "", ""),
    );
    p.axes_visible = false;
    for name in h.group("land")?.member_names()? {
        p.push(Layer::PolygonMesh(PolygonMesh::new(
            vec![points(&h, &format!("land/{name}"))?],
            vec![0.5],
            land,
        )?));
    }
    for name in h.group("states")?.member_names()? {
        line(
            &mut p,
            points(&h, &format!("states/{name}"))?,
            "#c2c7c3",
            0.35,
            false,
        );
    }
    for lon in [-74., -73., -72., -71., -70.] {
        line(
            &mut p,
            vec![[lon, SOUTH], [lon, NORTH]],
            "#dce0dc",
            0.3,
            false,
        );
    }
    for lat in [41., 42., 43.] {
        line(
            &mut p,
            vec![[WEST, lat], [EAST, lat]],
            "#dce0dc",
            0.3,
            false,
        );
    }
    for name in h.group("coast")?.member_names()? {
        line(
            &mut p,
            points(&h, &format!("coast/{name}"))?,
            "#8c9690",
            0.45,
            false,
        );
    }
    line(&mut p, points(&h, "trajectory")?, PATH, 1.25, true);
    line(&mut p, points(&h, "observed_trajectory")?, OBS, 2., false);
    for (name, c, shape, dx, dy) in specs {
        let a = h.dataset(&format!("stations/{name}"))?.read_raw::<f64>()?;
        let pt = [a[0], a[1]];
        let fill = scale(&mut f, c)?;
        marker(
            &mut p,
            pt,
            if shape == "star" { 1.65 } else { 1.1 },
            shape,
            fill,
            c,
        )?;
        let a = xy(pt);
        label(&mut f, &glyphs, name, a[0] + dx, a[1] + dy, c, true);
    }
    let fill = scale(&mut f, DET)?;
    for pt in points(&h, "weather_detections")? {
        marker(&mut p, pt, 0.8, "circle", fill, "#ffffff")?;
    }
    // Legend lies in the open inland region requested by the author.
    let (lx, ly) = (X + 5., Y + 59.);
    // A land-coloured knockout prevents borders and graticules crossing text.
    p.push(Layer::PolygonMesh(PolygonMesh::new(
        vec![vec![
            data([lx - 2.5, ly - 3.]),
            data([lx + 51., ly - 3.]),
            data([lx + 51., ly + 36.]),
            data([lx - 2.5, ly + 36.]),
        ]],
        vec![0.5],
        land,
    )?));
    for (i, (name, c, shape)) in entries.iter().enumerate() {
        let y = ly + i as f64 * 5.5;
        if *shape == "line" {
            line(
                &mut p,
                vec![data([lx - 1.8, y]), data([lx + 1.8, y])],
                c,
                if *c == OBS { 2. } else { 1.25 },
                *c == PATH,
            );
        } else {
            let fill = scale(&mut f, c)?;
            marker(
                &mut p,
                data([lx, y]),
                if *shape == "star" { 1.65 } else { 1.05 },
                shape,
                fill,
                c,
            )?;
        }
        label(&mut f, &glyphs, name, lx + 4., y, "#252c31", true);
    }
    line(
        &mut p,
        vec![
            [WEST, SOUTH],
            [EAST, SOUTH],
            [EAST, NORTH],
            [WEST, NORTH],
            [WEST, SOUTH],
        ],
        "#78817d",
        0.5,
        false,
    );
    for (i, lon) in [-74., -73., -72., -71., -70.].iter().enumerate() {
        let a = xy([*lon, SOUTH]);
        line(
            &mut p,
            vec![data(a), data([a[0], a[1] - 1.1])],
            "#78817d",
            0.5,
            false,
        );
        label(&mut f, &glyphs, &xt[i], a[0], Y + H + 4., "#3f4845", false);
    }
    for (i, lat) in [41., 42., 43.].iter().enumerate() {
        let a = xy([WEST, *lat]);
        line(
            &mut p,
            vec![data(a), data([a[0] + 1.1, a[1]])],
            "#78817d",
            0.5,
            false,
        );
        label(&mut f, &glyphs, &yt[i], X - 6.5, a[1], "#3f4845", false);
    }
    f.panels.push(p);
    f.save(out.with_extension("pdf"))?;
    f.save(out.with_extension("png"))?;
    println!("Rendered {}", out.display());
    Ok(())
}
