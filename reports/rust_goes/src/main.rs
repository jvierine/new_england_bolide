use anyhow::Result;
use rustplotlib::*;
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Deserialize)]
struct Scene {
    bounds: [f64;4],
    polygons: Vec<Vec<[f64;2]>>,
    values: Vec<f64>,
    coast: Vec<Vec<[f64;2]>>,
    maximum_flashes_per_minute: u32,
}

fn main() -> Result<()> {
    let data = PathBuf::from(std::env::args().nth(1).expect("data directory"));
    let scene: Scene = serde_json::from_slice(&fs::read(data.join("rustplotlib_scene.json"))?)?;
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 180.;
    fig.height_mm = 170.;
    let maximum = scene.maximum_flashes_per_minute.max(2);
    let scale = fig.scale(Scale::new([1.,maximum as f64], Normalization::Linear,
        ColorMap::new(vec![[1.,0.88,0.08],[1.,0.48,0.02],[0.85,0.02,0.01]])?)?);
    let [west,east,south,north] = scene.bounds;
    let mut axes = Axes::new([west,east],[south,north],"Longitude","Latitude");
    axes.xticks = vec![-73.,-72.,-71.,-70.,-69.];
    axes.yticks = vec![41.,42.,43.,44.];
    axes.xticklabels = Some((69..=73).rev().map(|v|format!(r"${v}^\circ\mathrm{{W}}$")).collect());
    axes.yticklabels = Some((41..=44).map(|v|format!(r"${v}^\circ\mathrm{{N}}$")).collect());
    let mut panel = Panel::new(Rect::new(20.,15.,137.,142.), axes);
    panel.push(Layer::Image(image::open(data.join("cloud_background.png"))?.into_rgba8()));
    for points in scene.coast {
        let mut line = Line::new(points,"#42c5df");
        line.width_pt = 0.4;
        panel.push(Layer::Line(line));
    }
    fig.panels.push(panel);
    fig.labels.push(Label::new("GOES-19 ABI + GLM, 30 May 2026, 18:06 UTC",90.,5.));
    fig.save(data.join("goes19_visible_clouds.pdf"))?;
    fig.save(data.join("goes19_visible_clouds.png"))?;
    fig.panels[0].layers.insert(1,Layer::PolygonMesh(PolygonMesh::new(scene.polygons,scene.values,scale)?));
    fig.colorbars.push(Colorbar::new(Rect::new(160.,15.,3.,142.),scale,
        r"GLM flash count (flashes min$^{-1}$)", (1..=maximum).map(|v|v as f64).collect()));
    fig.save(data.join("goes19_visible_clouds_glm_fed.pdf"))?;
    fig.save(data.join("goes19_visible_clouds_glm_fed.png"))?;
    fig.save(data.join("goes19_visible_clouds_glm.pdf"))?;
    fig.save(data.join("goes19_visible_clouds_glm.png"))?;
    Ok(())
}
