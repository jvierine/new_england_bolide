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
    figure1_energy_fj: Vec<f64>,
    light_curve: Vec<[f64;2]>,
    altitude_ticks: Vec<[f64;2]>,
    borders: Vec<Vec<[f64;2]>>,
    apparent_trajectory: Vec<[f64;2]>,
    ground_trajectory: Vec<[f64;2]>,
    mapping_heights_km: Vec<f64>,
    mapping_ground: Vec<[f64;2]>,
    mapping_expected: Vec<[f64;2]>,
    apparent_terminus: [f64;2],
    terminus_time: f64,
}

fn main() -> Result<()> {
    let data = PathBuf::from(std::env::args().nth(1).expect("data directory"));
    let scene: Scene = serde_json::from_slice(&fs::read(data.join("rustplotlib_scene.json"))?)?;
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 180.;
    fig.font_pt = 11.5;
    fig.height_mm = 105.;
    let maximum = scene.maximum_flashes_per_minute.max(2);
    let scale = fig.scale(Scale::new([1.,maximum as f64], Normalization::Linear,
        ColorMap::new(vec![[1.,0.88,0.08],[1.,0.48,0.02],[0.85,0.02,0.01]])?)?);
    let [west,east,_,_] = scene.bounds;
    let (south,north) = (41.,43.);
    let mut axes = Axes::new([west,east],[south,north],"Longitude","Latitude");
    axes.xticks = vec![-73.,-72.,-71.,-70.,-69.];
    axes.yticks = vec![41.,41.5,42.,42.5,43.];
    axes.xticklabels = Some((69..=73).rev().map(|v|format!(r"${v}^\circ\mathrm{{W}}$")).collect());
    axes.yticklabels = Some([41.,41.5,42.,42.5,43.].iter().map(|v|format!(r"${v}^\circ\mathrm{{N}}$")).collect());
    let mut panel = Panel::new(Rect::new(20.,11.,137.,75.), axes);
    let cloud = image::open(data.join("cloud_background.png"))?.into_rgba8();
    let top = ((scene.bounds[3]-north)/(scene.bounds[3]-scene.bounds[2])*cloud.height() as f64).round() as u32;
    let bottom = ((scene.bounds[3]-south)/(scene.bounds[3]-scene.bounds[2])*cloud.height() as f64).round() as u32;
    panel.push(Layer::Image(image::imageops::crop_imm(&cloud,0,top,cloud.width(),bottom-top).to_image()));
    for points in scene.coast {
        let mut line = Line::new(points,"#42c5df");
        line.width_pt = 0.65;
        panel.push(Layer::Line(line));
    }
    for points in scene.borders {
        let mut halo = Line::new(points.clone(),"#ffffff");halo.width_pt = 1.1;
        panel.push(Layer::Line(halo));
        let mut line = Line::new(points,"#16889e");line.width_pt = 0.65;
        panel.push(Layer::Line(line));
    }
    fig.panels.push(panel);
    fig.labels.push(Label::new("GOES-19 ABI + GLM, 30 May 2026, 18:06 UTC",90.,5.));
    fig.save(data.join("goes19_visible_clouds.pdf"))?;
    fig.save(data.join("goes19_visible_clouds.png"))?;
    fig.panels[0].layers.insert(1,Layer::PolygonMesh(PolygonMesh::new(scene.polygons,scene.values,scale)?));
    fig.colorbars.push(Colorbar::new(Rect::new(160.,11.,3.,75.),scale,
        r"GLM flash count (flashes min$^{-1}$)", (1..=maximum).map(|v|v as f64).collect()));
    fig.save(data.join("goes19_visible_clouds_glm_fed.pdf"))?;
    fig.save(data.join("goes19_visible_clouds_glm_fed.png"))?;
    fig.save(data.join("goes19_visible_clouds_glm.pdf"))?;
    fig.save(data.join("goes19_visible_clouds_glm.png"))?;
    let energy_scale = fig.scale(Scale::new([0.001,100.],Normalization::Log,
        ColorMap::new(vec![[1.,0.92,0.05],[1.,0.45,0.01],[0.75,0.,0.]])?)?);
    fig.panels[0].layers[1] = Layer::PolygonMesh(PolygonMesh::new(
        scene.energy_polygons,scene.figure1_energy_fj.iter().map(|v|v/1000.).collect(),energy_scale)?);
    fig.colorbars.clear();
    fig.colorbars.push(Colorbar::new(Rect::new(160.,11.,3.,75.),energy_scale,
        r"Accumulated GLM event energy (pJ pixel$^{-1}$)",vec![0.001,0.01,0.1,1.,10.,100.]));
    fig.labels.clear();
    fig.labels.push(Label::new("GOES-19, 30 May 2026",90.,5.));
    let first_overlay = fig.panels[0].layers.len();
    for (i,(&ground,&expected)) in scene.mapping_ground.iter().zip(&scene.mapping_expected).enumerate() {
        let points=vec![ground,expected];
        let mut halo=Line::new(points.clone(),"#ffffff");halo.width_pt=1.3;
        fig.panels[0].push(Layer::Line(halo));
        let mut mapping=Line::new(points,"#555b60");mapping.width_pt=0.7;mapping.dash=Some("1.1 0.9".into());
        fig.panels[0].push(Layer::Line(mapping));
        let x=20.+(expected[0]-west)/(east-west)*137.;
        let y=11.+(north-expected[1])/(north-south)*75.;
        let mut label=Label::new(format!("{} km",scene.mapping_heights_km[i]),x+15.,y-0.5);
        label.color="#003c9e".into();
        fig.labels.push(label);
    }
    let mut halo=Line::new(scene.ground_trajectory.clone(),"#ffffff");halo.width_pt=3.;
    fig.panels[0].push(Layer::Line(halo));
    let mut ground=Line::new(scene.ground_trajectory,"#007a52");ground.width_pt=1.5;
    fig.panels[0].push(Layer::Line(ground));
    let mut halo = Line::new(scene.apparent_trajectory.clone(),"#ffffff");halo.width_pt=3.;
    fig.panels[0].push(Layer::Line(halo));
    let mut trajectory = Line::new(scene.apparent_trajectory,"#003c9e");trajectory.width_pt=1.5;
    fig.panels[0].push(Layer::Line(trajectory));
    let [lon,lat] = scene.apparent_terminus;
    let marker = (0..32).map(|i| {
        let angle = i as f64/32.*std::f64::consts::TAU;
        [lon+0.025*angle.cos(),lat+0.025*(137./75.)*(north-south)/(east-west)*angle.sin()]
    }).collect();
    let marker_scale=fig.scale(Scale::new([0.,1.],Normalization::Linear,
        ColorMap::new(vec![[0.,0.48,0.32],[0.,0.48,0.32]])?)?);
    fig.panels[0].push(Layer::PolygonMesh(PolygonMesh::new(vec![marker],vec![0.5],marker_scale)?));
    let mut legend = Legend::new(vec![
        LegendEntry::line("Expected GLM trajectory", "#003c9e"),
        LegendEntry::line("True ground track", "#007a52"),
        LegendEntry::line("Equal-height mapping", "#555b60"),
        LegendEntry::circle("Radar terminus", "#007a52"),
    ]);
    if let LegendSymbol::Line {dash,..}=&mut legend.entries[2].symbol {*dash=Some("1.1 0.9".into());}
    legend.position = LegendPosition::UpperLeft;
    legend.avoid_layers = std::iter::once(1).chain(first_overlay..fig.panels[0].layers.len()).collect();
    fig.panels[0].legend = Some(legend);
    fig.save(data.join("goes19_visible_clouds_glm_energy.pdf"))?;
    fig.save(data.join("goes19_visible_clouds_glm_energy.png"))?;
    let mut paper = fig.clone();
    let offset=75.;
    paper.width_mm+=offset;
    for panel in &mut paper.panels {
        panel.rect.x+=offset;
        for label in &mut panel.labels {label.x+=offset;}
    }
    for bar in &mut paper.colorbars {bar.rect.x+=offset;}
    for label in &mut paper.labels {label.x+=offset;}
    let photo_path=PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../figures/optical_mechanicville_cropped.png");
    let photo=image::open(photo_path)?.into_rgba8();
    let width=75.*photo.width() as f64/photo.height() as f64;
    let mut optical=Panel::new(Rect::new(2.,11.,width,75.),Axes::new([0.,1.],[0.,1.],"",""));
    optical.axes_visible=false;
    optical.push(Layer::Image(photo));
    paper.panels.insert(0,optical);
    paper.labels.push(Label::new("(a)",5.,5.));
    paper.labels.push(Label::new("(b)",20.+offset+3.,5.));
    paper.save(data.join("goes19_figure1_panels.pdf"))?;
    paper.save(data.join("goes19_figure1_panels.png"))?;

    let mut curve = Figure::new(IntendedUse::ArticleWide);
    curve.width_mm = 180.;
    curve.font_pt = 11.5;
    curve.height_mm = 85.;
    let mut axes = Axes::new([1.5,3.4],[0.,3.5],
        "Seconds after 18:06:20 UTC on 30 May 2026",
        "GLM event energy (pJ per 10 ms bin)");
    axes.xticks = vec![1.5,2.,2.5,3.];
    axes.yticks = vec![0.,1.,2.,3.];
    axes.secondary_x = Some(SecondaryAxis {
        label: "Modeled altitude (km)".into(),
        ticks: scene.altitude_ticks.iter().map(|[time,altitude]| (*time,*altitude)).collect(),
    });
    let mut p = Panel::new(Rect::new(22.,17.,150.,52.),axes);
    let steps = scene.light_curve.iter().flat_map(|[t,e]| [[t-0.005,e/1000.],[t+0.005,e/1000.]]).collect();
    p.push(Layer::Line(Line::new(steps,"#a52c20")));
    let mut terminus = Line::new(vec![[scene.terminus_time,0.],[scene.terminus_time,3.5]],"#007a52");
    terminus.width_pt = 1.1;
    terminus.dash = Some("1.5 1".into());
    p.push(Layer::Line(terminus));
    curve.panels.push(p);
    curve.save(data.join("goes19_glm_event_energy_lightcurve.pdf"))?;
    curve.save(data.join("goes19_glm_event_energy_lightcurve.png"))?;
    Ok(())
}
