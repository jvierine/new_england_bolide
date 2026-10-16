use anyhow::Result;
use rustplotlib::*;
use std::path::PathBuf;

const METEORITES: &str = "#547f85";
const BOLIDE: &str = "#424a50";

fn line(p: &mut Panel, points: Vec<[f64; 2]>, color: &str, width: f64, opacity: f64) {
    let mut l = Line::new(points, color);
    l.width_pt = width;
    l.opacity = opacity;
    p.push(Layer::Line(l));
}
fn points(values: &[f64]) -> Vec<[f64; 2]> {
    values.chunks_exact(2).map(|p| [p[0], p[1]]).collect()
}
fn rgb(hex: &str) -> [f64; 3] {
    [1, 3, 5].map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap() as f64 / 255.)
}
fn main() -> Result<()> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("directory containing scene.h5"),
    );
    let h = hdf5::File::open(dir.join("scene.h5"))?;
    let mut f = ArticleLayout::new(164.6, 10.)?.figure(0.7, 117.)?;
    let mut axes = Axes::new(
        [-8., 8.],
        [-8., 8.],
        r"Ecliptic $x$ (AU)",
        r"Ecliptic $y$ (AU)",
    );
    axes.xticks = vec![-8., -4., 0., 4., 8.];
    axes.yticks = axes.xticks.clone();
    // A square plotting area keeps the two physical distance scales equal.
    let mut panel = Panel::new(Rect::new(17., 5., 94., 94.), axes);
    for xy in [vec![[-8., 0.], [8., 0.]], vec![[0., -8.], [0., 8.]]] {
        line(&mut panel, xy, "#e5e8e8", 0.3, 1.);
    }
    let comparisons = h.dataset("meteorites_xy")?.read_raw::<f64>()?;
    let comparison_start = panel.layers.len();
    for xy in comparisons.chunks_exact(1000) {
        line(&mut panel, points(xy), METEORITES, 0.35, 0.18);
    }
    let comparison_end = panel.layers.len();
    let planets = [
        ("Mercury", "#929a9e"),
        ("Venus", "#b19063"),
        ("Earth", "#3c79a0"),
        ("Mars", "#b86f54"),
        ("Jupiter", "#9b8d76"),
        ("Saturn", "#c3b69a"),
    ];
    for (name, color) in planets {
        let g = h.group(&format!("planets/{name}"))?;
        line(
            &mut panel,
            points(&g.dataset("xy")?.read_raw::<f64>()?),
            color,
            0.55,
            0.8,
        );
    }
    let sample_start = panel.layers.len();
    let samples = h.dataset("samples_xy")?.read_raw::<f64>()?;
    for xy in samples.chunks_exact(1000) {
        line(&mut panel, points(xy), BOLIDE, 0.25, 0.018);
    }
    let sample_end = panel.layers.len();
    panel.rasterize_lines(sample_start..sample_end, 450.)?;
    let nominal_layer = panel.layers.len();
    line(
        &mut panel,
        points(&h.dataset("nominal_xy")?.read_raw::<f64>()?),
        BOLIDE,
        1.3,
        1.,
    );
    let aphelion = h.dataset("nominal_aphelion_xy")?.read_raw::<f64>()?;
    line(&mut panel, vec![[aphelion[0], aphelion[1]], [aphelion[0], -5.5], [3.2, -5.5]], BOLIDE, 0.55, 1.);
    panel.label_at("Cape Cod Bay Bolide", [3.2, -6.1]).color = BOLIDE.into();
    for (name, color) in planets {
        let v = h
            .dataset(&format!("planets/{name}/position_xy"))?
            .read_raw::<f64>()?;
        if v.iter().all(|x| x.abs() < 8.) {
            panel.markers(
                &mut f,
                &[[v[0], v[1]]],
                &MarkerStyle {
                    shape: MarkerShape::Circle,
                    size_pt: 3.7,
                    color: rgb(color),
                },
            )?;
        }
    }
    panel.markers(
        &mut f,
        &[[0., 0.]],
        &MarkerStyle {
            shape: MarkerShape::Circle,
            size_pt: 5.5,
            color: rgb("#c69e35"),
        },
    )?;
    let mut entries: Vec<_> = planets
        .iter()
        .map(|(name, color)| LegendEntry::line(*name, *color))
        .collect();
    entries.extend([
        LegendEntry {
            label: "Bolide orbit".into(),
            symbol: LegendSymbol::Line {
                color: BOLIDE.into(),
                width_pt: 1.3,
                dash: None,
            },
        },
        LegendEntry::line("57 meteorites", METEORITES),
        LegendEntry::circle("Sun", "#c69e35"),
    ]);
    let mut legend = Legend::new(entries);
    legend.columns = 3;
    legend.position = LegendPosition::UpperRight;
    legend.padding_mm = 1.;
    legend.background_opacity = 0.96;
    legend.avoid_layers = (comparison_start..comparison_end)
        .chain([sample_start, nominal_layer])
        .collect();
    panel.legend = Some(legend);
    f.panels.push(panel);
    f.save_for_article(
        dir.join("orbit_xy_bolide.pdf"),
        "reports/figures/orbit_xy_bolide.pdf",
    )?;
    f.save(dir.join("orbit_xy_bolide.png"))?;
    Ok(())
}
