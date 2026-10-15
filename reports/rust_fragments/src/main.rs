use anyhow::Result;
use image::{Rgba, RgbaImage};
use rustplotlib::*;
use std::path::PathBuf;

const GREEN: &str = "#18765b";
const LIGHT_GREEN: &str = "#6ca98d";
const INK: &str = "#343b43";

fn points(g: &hdf5::Group, name: &str) -> Result<Vec<[f64; 2]>> {
    Ok(g.dataset(name)?
        .read_raw::<f64>()?
        .chunks_exact(2)
        .map(|v| [v[0], v[1]])
        .collect())
}

fn marker(f: &mut Figure, p: &mut Panel, xy: [f64; 2], kind: i64) -> Result<()> {
    let a = p.axes.as_ref().unwrap();
    let radius = if kind == 1 { 0.88 } else { 0.65 };
    let n = if kind == 2 { 24 } else { 4 };
    let phase = if kind == 0 {
        std::f64::consts::FRAC_PI_4
    } else {
        0.
    };
    let polygon: Vec<_> = (0..n)
        .map(|j| {
            let t = j as f64 / n as f64 * std::f64::consts::TAU + phase;
            [
                xy[0] + radius * t.cos() * (a.xlim[1] - a.xlim[0]) / p.rect.width,
                xy[1] + radius * t.sin() * (a.ylim[1] - a.ylim[0]) / p.rect.height,
            ]
        })
        .collect();
    let rgb = match kind {
        0 => [0.094, 0.463, 0.357],
        1 => [0.698, 0.231, 0.208],
        _ => [0.137, 0.400, 0.549],
    };
    let scale = f.scale(Scale::new(
        [0., 1.],
        Normalization::Linear,
        ColorMap::new(vec![rgb, rgb])?,
    )?);
    p.push(Layer::PolygonMesh(PolygonMesh::new(
        vec![polygon.clone()],
        vec![0.5],
        scale,
    )?));
    let mut outline = polygon;
    outline.push(outline[0]);
    let mut line = Line::new(outline, "#ffffff");
    line.width_pt = 0.3;
    p.push(Layer::Line(line));
    Ok(())
}

fn blob_image(g: &hdf5::Group, xlim: [f64; 2], ylim: [f64; 2]) -> Result<RgbaImage> {
    let (w, h) = (1000u32, 1900u32);
    let mut image = RgbaImage::new(w, h);
    for e in g.dataset("ellipses")?.read_raw::<f64>()?.chunks_exact(5) {
        let cx = (e[0] - xlim[0]) / (xlim[1] - xlim[0]) * w as f64;
        let cy = (ylim[1] - e[1]) / (ylim[1] - ylim[0]) * h as f64;
        let rx = e[2] / (xlim[1] - xlim[0]) * w as f64 / 2.;
        let ry = e[3] / (ylim[1] - ylim[0]) * h as f64 / 2.;
        for y in (cy - ry).max(0.) as u32..((cy + ry).ceil() as u32).min(h) {
            for x in (cx - rx).max(0.) as u32..((cx + rx).ceil() as u32).min(w) {
                if ((x as f64 + 0.5 - cx) / rx).powi(2) + ((y as f64 + 0.5 - cy) / ry).powi(2) <= 1.
                {
                    let old = image.get_pixel(x, y)[3] as f64 / 255.;
                    let alpha = e[4] + old * (1. - e[4]);
                    image.put_pixel(x, y, Rgba([35, 102, 140, (255. * alpha).round() as u8]));
                }
            }
        }
    }
    Ok(image)
}

fn main() -> Result<()> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("directory containing scene.h5"),
    );
    let data = hdf5::File::open(dir.join("scene.h5"))?;
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 164.6;
    f.height_mm = 124.;
    f.font_pt = 10.;
    f.latex.font_pt = 10.;
    for i in 0..3 {
        let g = data.group(&format!("panel_{i}"))?;
        let limits = g.dataset("xlim")?.read_raw::<f64>()?;
        let xlim = if i == 2 {
            [limits[0].log10(), limits[1].log10()]
        } else {
            [limits[0], limits[1]]
        };
        let ylim = [0., 100.];
        let xlabel = match i {
            0 => r"Latitude ($^\circ$N)",
            1 => r"Longitude ($^\circ$W)",
            _ => r"$\rho_a v_{\rm air}^{2}$ (MPa)",
        };
        let mut axes = Axes::new(xlim, ylim, xlabel, if i == 0 { "Height (km)" } else { "" });
        axes.xticks = match i {
            0 => vec![41.85, 41.90, 41.95, 42.00],
            1 => vec![-70.7, -70.6, -70.5, -70.4],
            _ => vec![-2., -1., 0., 1.],
        };
        if i == 1 {
            axes.xticklabels = Some(vec![
                "70.7".into(),
                "70.6".into(),
                "70.5".into(),
                "70.4".into(),
            ]);
        }
        if i == 2 {
            axes.xticklabels = Some(vec![
                "$10^{-2}$".into(),
                "$10^{-1}$".into(),
                "$1$".into(),
                "$10$".into(),
            ]);
        }
        axes.yticks = vec![0., 20., 40., 60., 80., 100.];
        if i > 0 {
            axes.yticks.clear();
        }
        let rect = Rect::new(16. + i as f64 * 50., 22., 44., 85.);
        let mut p = Panel::new(rect, axes);
        if i < 2 {
            p.push(Layer::Image(blob_image(&g, xlim, ylim)?));
        }
        for y in [20., 40., 60., 80., 100.] {
            let mut grid = Line::new(vec![[xlim[0], y], [xlim[1], y]], "#e3e7ea");
            grid.width_pt = 0.35;
            p.push(Layer::Line(grid));
        }
        let lines = g.group("lines")?;
        for name in lines.member_names()? {
            let line = lines.group(&name)?;
            let kind = line.attr("kind")?.read_scalar::<i64>()?;
            let mut xy = points(&line, "xy")?;
            if i == 2 {
                for v in &mut xy {
                    v[0] = v[0].log10();
                }
            }
            let mut l = Line::new(
                xy,
                match kind {
                    0 => INK,
                    1 => LIGHT_GREEN,
                    _ => GREEN,
                },
            );
            l.width_pt = if kind == 1 { 0.7 } else { 1.15 };
            if kind == 1 {
                l.dash = Some("1.4 0.9".into());
            }
            p.push(Layer::Line(l));
        }
        let markers = g.group("markers")?;
        for name in markers.member_names()? {
            let item = markers.group(&name)?;
            let kind = item.attr("kind")?.read_scalar::<i64>()?;
            for xy in points(&item, "xy")? {
                marker(&mut f, &mut p, xy, kind)?;
            }
        }
        f.labels.push(Label::new(
            format!("({})", ['a', 'b', 'c'][i]),
            rect.x + 1.,
            18.,
        ));
        if i == 2 {
            f.labels
                .push(Label::new("Fitted trajectory", rect.x + 25., 30.));
        }
        f.panels.push(p);
    }
    // A shared key above the panels leaves the observed features unobscured.
    let mut key = Panel::new(
        Rect::new(0., 0., 164.6, 13.),
        Axes::new([0., 164.6], [0., 13.], "", ""),
    );
    key.axes_visible = false;
    for (kind, x, y, text) in [
        (2, 17., 4., "VHF plasma trail"),
        (1, 94., 4., "Weather-radar detections"),
    ] {
        marker(&mut f, &mut key, [x, 13. - y], kind)?;
        let glyphs = f.latex.typeset([text.to_string()])?;
        f.labels.push(Label::new(text, x + 4. + glyphs[text].width_pt * 25.4 / 72. / 2., y));
    }
    for (x, y, dashed, text) in [
        (14., 10., true, "J2 generic mass hypotheses"),
        (91., 10., false, "J2 radar-constrained paths"),
    ] {
        let mut l = Line::new(
            vec![[x, 13. - y], [x + 6., 13. - y]],
            if dashed { LIGHT_GREEN } else { GREEN },
        );
        if dashed {
            l.dash = Some("1.4 0.9".into());
        }
        l.width_pt = 1.;
        key.push(Layer::Line(l));
        let glyphs = f.latex.typeset([text.to_string()])?;
        f.labels.push(Label::new(text, x + 9. + glyphs[text].width_pt * 25.4 / 72. / 2., y));
    }
    f.panels.push(key);
    f.save(dir.join("fragment_size_hypotheses.pdf"))?;
    f.save(dir.join("fragment_size_hypotheses.png"))?;
    println!("Rendered unchanged HDF5 coordinates to PDF and PNG");
    Ok(())
}
