use anyhow::Result;
use rustplotlib::*;
use serde::Deserialize;
use std::{fs,path::PathBuf};

#[derive(Deserialize)]
struct Scene {
    ground:Vec<[f64;2]>, expected:Vec<[f64;2]>, extension:Vec<[f64;2]>,
    ground_points:Vec<[f64;2]>, expected_points:Vec<[f64;2]>,
    heights_km:Vec<f64>, coast:Vec<Vec<[f64;2]>>,
}
fn main()->Result<()> {
    let dir=PathBuf::from(std::env::args().nth(1).expect("data directory"));
    let scene:Scene=serde_json::from_slice(&fs::read(dir.join("glm_projection_explanation_scene.json"))?)?;
    let mut f=Figure::new(IntendedUse::ArticleWide);
    f.width_mm=180.;f.height_mm=190.;
    let xlim=[-71.6,-70.1];let ylim=[41.7,43.];
    let rect=Rect::new(25.,15.,130.,150.);
    let xy=|p:[f64;2]| [rect.x+(p[0]-xlim[0])/(xlim[1]-xlim[0])*rect.width,
                          rect.y+(ylim[1]-p[1])/(ylim[1]-ylim[0])*rect.height];
    let inverse=|p:[f64;2]| [xlim[0]+(p[0]-rect.x)/rect.width*(xlim[1]-xlim[0]),
                                ylim[1]-(p[1]-rect.y)/rect.height*(ylim[1]-ylim[0])];
    let mut axes=Axes::new(xlim,ylim,"Longitude","Latitude");
    axes.xticks=vec![-71.5,-71.,-70.5];axes.yticks=vec![41.75,42.,42.25,42.5,42.75,43.];
    let mut p=Panel::new(rect,axes);
    for points in scene.coast {
        let mut line=Line::new(points,"#b0b6ba");line.width_pt=0.5;
        p.push(Layer::Line(line));
    }
    let extension_id=p.layers.len();
    let mut extension=Line::new(scene.extension,"#228636");extension.width_pt=1.;extension.dash=Some("1.4 1".into());
    p.push(Layer::Line(extension));
    p.push(Layer::Line(Line::new(scene.ground,"#228636")));
    p.push(Layer::Line(Line::new(scene.expected,"#003c9e")));
    let scale=f.scale(Scale::new([0.,1.],Normalization::Linear,
        ColorMap::new(vec![[0.13,0.53,0.21],[0.,0.235,0.62]])?)?);
    let mut polygons=vec![];let mut values=vec![];
    for (i,(&a,&b)) in scene.ground_points.iter().zip(&scene.expected_points).enumerate() {
        let mut pair=Line::new(vec![a,b],"#8d9398");pair.width_pt=0.7;pair.dash=Some("1 1".into());
        p.push(Layer::Line(pair));
        for (point,value) in [(a,0.),(b,1.)] {
            let center=xy(point);
            polygons.push((0..24).map(|j| {
                let angle=j as f64/24.*std::f64::consts::TAU;
                inverse([center[0]+0.9*angle.cos(),center[1]+0.9*angle.sin()])
            }).collect());values.push(value);
        }
        let [x,y]=xy(b);
        f.labels.push(Label::new(format!("{} km",scene.heights_km[i]),x+9.,y-1.5));
    }
    p.push(Layer::PolygonMesh(PolygonMesh::new(polygons,values,scale)?));
    let mut legend=Legend::new(vec![
        LegendEntry::line("True ground track", "#228636"),
        LegendEntry::line("Expected GLM position", "#003c9e"),
        LegendEntry::line("Equal-height pairs", "#8d9398"),
    ]);
    if let LegendSymbol::Line {dash,..}=&mut legend.entries[2].symbol {*dash=Some("1 1".into());}
    legend.position=LegendPosition::UpperLeft;
    legend.avoid_layers=(extension_id..p.layers.len()).collect();
    p.legend=Some(legend);
    f.panels.push(p);
    f.labels.push(Label::new("Same 3-D trajectory, different reported positions",90.,5.));
    f.labels.push(Label::new("Dashed green: backward extension to 300 km",90.,183.));
    f.save(dir.join("glm_projection_explanation.pdf"))?;
    f.save(dir.join("glm_projection_explanation.png"))?;
    Ok(())
}
