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
    energy_polygons: Vec<Vec<[f64;2]>>,
    energy_fj: Vec<f64>,
    light_curve: Vec<[f64;2]>,
    altitude_ticks: Vec<[f64;2]>,
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
    let energy_scale = fig.scale(Scale::new([0.001,100.],Normalization::Log,
        ColorMap::new(vec![[1.,0.92,0.05],[1.,0.45,0.01],[0.75,0.,0.]])?)?);
    fig.panels[0].layers[1] = Layer::PolygonMesh(PolygonMesh::new(
        scene.energy_polygons,scene.energy_fj.iter().map(|v|v/1000.).collect(),energy_scale)?);
    fig.colorbars.clear();
    fig.colorbars.push(Colorbar::new(Rect::new(160.,15.,3.,142.),energy_scale,
        r"Accumulated GLM event energy (pJ pixel$^{-1}$)",vec![0.001,0.01,0.1,1.,10.,100.]));
    fig.labels.clear();
    fig.labels.push(Label::new("GOES-19, 30 May 2026, 18:06:20--24 UTC",90.,5.));
    fig.save(data.join("goes19_visible_clouds_glm_energy.pdf"))?;
    fig.save(data.join("goes19_visible_clouds_glm_energy.png"))?;

    let mut curve = Figure::new(IntendedUse::ArticleWide);
    curve.width_mm = 180.;
    curve.height_mm = 100.;
    let mut axes = Axes::new([1.5,3.4],[0.,3.5],
        "Seconds after 18:06:20 UTC on 30 May 2026",
        "GLM event energy (pJ per 10 ms bin)");
    axes.xticks = vec![1.5,2.,2.5,3.];
    axes.yticks = vec![0.,1.,2.,3.];
    axes.secondary_x = Some(SecondaryAxis {
        label: "Modeled altitude (km)".into(),
        ticks: scene.altitude_ticks.iter().map(|[time,altitude]| (*time,*altitude)).collect(),
    });
    let mut p = Panel::new(Rect::new(22.,30.,150.,52.),axes);
    let steps = scene.light_curve.iter().flat_map(|[t,e]| [[t-0.005,e/1000.],[t+0.005,e/1000.]]).collect();
    p.push(Layer::Line(Line::new(steps,"#a52c20")));
    curve.panels.push(p);
    curve.labels.push(Label::new("GOES-19 GLM bolide light curve",95.,4.));
    curve.save(data.join("goes19_glm_event_energy_lightcurve.pdf"))?;
    curve.save(data.join("goes19_glm_event_energy_lightcurve.png"))?;
    Ok(())
}
