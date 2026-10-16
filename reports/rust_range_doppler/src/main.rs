use anyhow::Result;
use hdf5::types::VarLenUnicode;
use rustplotlib::*;
use std::path::PathBuf;

const STATION_COLORS: [[u8; 3]; 3] = [[82, 120, 140], [173, 120, 97], [112, 134, 117]];

fn muted_rgb(rgb: &[f64]) -> [f64; 3] {
    let original = [[31, 119, 180], [255, 127, 14], [44, 160, 44]];
    for (source, target) in original.iter().zip(STATION_COLORS) {
        if (0..3).all(|i| (rgb[i] * 255. - source[i] as f64).abs() < 0.5) {
            return target.map(|v| v as f64 / 255.);
        }
    }
    [rgb[0], rgb[1], rgb[2]]
}

fn points(g: &hdf5::Group) -> Result<Vec<[f64; 2]>> {
    Ok(g.dataset("xy")?
        .read_raw::<f64>()?
        .chunks_exact(2)
        .map(|v| [v[0], v[1]])
        .collect())
}
fn color(rgb: &[f64]) -> String {
    format!(
        "#{:02x}{:02x}{:02x}",
        (rgb[0] * 255.).round() as u8,
        (rgb[1] * 255.).round() as u8,
        (rgb[2] * 255.).round() as u8
    )
}
fn text(g: &hdf5::Group, key: &str) -> Result<String> {
    Ok(g.attr(key)?
        .read_scalar::<VarLenUnicode>()?
        .as_str()
        .to_string())
}
fn main() -> Result<()> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("directory containing scene.h5"),
    );
    let h = hdf5::File::open(dir.join("scene.h5"))?;
    let article = ArticleLayout::new(164.6, 10.)?;
    let mut f = article.figure(1., 146.)?;
    for i in 0..2 {
        let g = h.group(&format!("panel_{i}"))?;
        if i == 1 {
            anyhow::ensure!(
                text(&g, "xlabel")? == "Seconds relative to 2026-05-30T18:06:16.260000Z",
                "Update the time-axis label for this reference epoch"
            );
        }
        let x = g.dataset("xlim")?.read_raw::<f64>()?;
        let y = g.dataset("ylim")?.read_raw::<f64>()?;
        let mut a = Axes::new(
            [x[0], x[1]],
            [y[0], y[1]],
            if i == 1 {
                "Seconds after 18:06:16.26 UTC on 30 May 2026"
            } else {
                ""
            },
            if i == 0 {
                "Bistatic range (km)"
            } else {
                "Doppler shift (kHz)"
            },
        );
        a.xticks = if i == 1 {
            vec![4., 4.2, 4.4, 4.6, 4.8, 5.]
        } else {
            vec![]
        };
        a.yticks = if i == 0 {
            vec![250., 300., 350., 400.]
        } else {
            vec![-2.5, -2., -1.5, -1., -0.5]
        };
        a.yticks.retain(|v| *v >= y[0] && *v <= y[1]);
        let rect = Rect::new(21., 8. + i as f64 * 67., 139., 53.);
        let mut p = Panel::new(rect, a.clone());
        for tick in &a.yticks {
            let mut l = Line::new(vec![[x[0], *tick], [x[1], *tick]], "#e3e7ea");
            l.width_pt = 0.35;
            p.push(Layer::Line(l));
        }
        let bands = g.group("bands")?;
        for name in bands.member_names()? {
            let b = bands.group(&name)?;
            let rgba = b.dataset("rgba")?.read_raw::<f64>()?;
            let muted = muted_rgb(&rgba);
            let rgb = [0, 1, 2].map(|j| muted[j] * rgba[3] + 1. - rgba[3]);
            let scale = f.scale(Scale::new(
                [0., 1.],
                Normalization::Linear,
                ColorMap::new(vec![rgb, rgb])?,
            )?);
            p.push(Layer::PolygonMesh(PolygonMesh::new(
                vec![points(&b)?],
                vec![0.5],
                scale,
            )?));
        }
        let protected_start = p.layers.len();
        let errors = g.group("errors")?;
        for name in errors.member_names()? {
            let e = errors.group(&name)?;
            let rgb = muted_rgb(&e.dataset("rgba")?.read_raw::<f64>()?);
            let mut l = Line::new(points(&e)?, &color(&rgb));
            l.width_pt = 0.6;
            p.push(Layer::Line(l));
        }
        let lines = g.group("lines")?;
        for name in lines.member_names()? {
            let l = lines.group(&name)?;
            let rgb = muted_rgb(&l.dataset("rgb")?.read_raw::<f64>()?);
            let xy = points(&l)?;
            if l.attr("draw_line")?.read_scalar::<bool>()? {
                let mut line = Line::new(xy.clone(), &color(&rgb));
                line.width_pt = 1.2;
                p.push(Layer::Line(line));
            }
            let shape = match text(&l, "marker")?.as_str() {
                "o" => Some(MarkerShape::Circle),
                "_" => Some(MarkerShape::HorizontalCap),
                "|" => Some(MarkerShape::VerticalCap),
                _ => None,
            };
            if let Some(shape) = shape {
                p.markers(
                    &mut f,
                    &xy,
                    &MarkerStyle {
                        shape,
                        size_pt: l.attr("size")?.read_scalar::<f64>()?,
                        color: [rgb[0], rgb[1], rgb[2]],
                    },
                )?;
            }
        }
        if i == 0 {
            let mut legend = Legend::new(vec![
                LegendEntry::line(
                    "Beverly",
                    &color(&STATION_COLORS[0].map(|v| v as f64 / 255.)),
                ),
                LegendEntry::line(
                    "Clinton",
                    &color(&STATION_COLORS[1].map(|v| v as f64 / 255.)),
                ),
                LegendEntry::line(
                    "Pomfret",
                    &color(&STATION_COLORS[2].map(|v| v as f64 / 255.)),
                ),
            ]);
            legend.avoid_layers = (protected_start..p.layers.len()).collect();
            p.legend = Some(legend);
        } else {
            let xy = g.dataset("annotation_xy")?.read_raw::<f64>()?;
            f.labels.push(Label::new(
                text(&g, "annotation")?,
                rect.x + (xy[0] - x[0]) / (x[1] - x[0]) * rect.width + 8.,
                rect.y + (y[1] - xy[1]) / (y[1] - y[0]) * rect.height - 2.,
            ));
        }
        f.labels.push(Label::new(
            if i == 0 { "(a)" } else { "(b)" },
            rect.x + 1.,
            rect.y - 4.,
        ));
        f.panels.push(p);
    }
    f.save_for_article(dir.join("ballistic_head_trail_correlated_range_doppler_fit.pdf"), "reports/figures/appendix_correlated_fit/ballistic_head_trail_correlated_range_doppler_fit.pdf")?;
    f.save(dir.join("ballistic_head_trail_correlated_range_doppler_fit.png"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn station_palette_is_consistent_and_leaves_other_colors_unchanged() {
        for (source, target) in [[31, 119, 180], [255, 127, 14], [44, 160, 44]]
            .iter()
            .zip(STATION_COLORS)
        {
            let source = source.map(|v| v as f64 / 255.);
            assert_eq!(muted_rgb(&source), target.map(|v| v as f64 / 255.));
            assert_eq!(
                muted_rgb(&[source[0], source[1], source[2], 0.15]),
                muted_rgb(&source)
            );
        }
        assert_eq!(muted_rgb(&[0., 0., 0.]), [0., 0., 0.]);
    }
}
