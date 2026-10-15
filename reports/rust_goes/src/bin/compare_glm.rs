use anyhow::Result;
use rustplotlib::*;
use serde::Deserialize;
use std::{fs,path::PathBuf};

#[derive(Deserialize)]
struct AngularPanel {polygons:Vec<Vec<[f64;2]>>,values:Vec<f64>,track:Vec<[f64;2]>}
#[derive(Deserialize)]
struct Scene {panels:Vec<AngularPanel>,binned:Vec<[f64;4]>,terminus:f64,lightcurve:Vec<[f64;2]>}

fn main()->Result<()> {
    let dir=PathBuf::from(std::env::args().nth(1).expect("data directory"));
    let s:Scene=serde_json::from_slice(&fs::read(dir.join("glm_trajectory_comparison_scene.json"))?)?;
    let mut f=Figure::new(IntendedUse::ArticleWide);
    f.width_mm=185.; f.height_mm=218.;
    let scale=f.scale(Scale::new([0.001,100.],Normalization::Log,
        ColorMap::new(vec![[1.,0.94,0.5],[1.,0.45,0.],[0.7,0.,0.]])?)?);
    let names=["(a) Compact early","(b) Bright pre-terminus","(c) Post-terminus"];
    for (i,p) in s.panels.into_iter().enumerate() {
        let x=20.+i as f64*55.;
        let mut ax=Axes::new([-1500.,750.],[-1500.,750.],r"$\Delta\beta$ ($\mu$rad)",
            if i==0 {r"$\Delta\alpha$ ($\mu$rad)"}else{""});
        ax.xticks=vec![-1000.,0.,500.];
        ax.yticks=if i==0 {vec![-1000.,0.,500.]}else{vec![]};
        let mut panel=Panel::new(Rect::new(x,16.,45.,45.),ax);
        panel.push(Layer::PolygonMesh(PolygonMesh::new(p.polygons,p.values,scale)?));
        let mut model=Line::new(p.track,"#006ba4"); model.width_pt=1.8;
        panel.push(Layer::Line(model));f.panels.push(panel);
        f.labels.push(Label::new(names[i],x+22.5,7.));
    }
    let mut cb=Colorbar::new(Rect::new(20.,78.,155.,2.),scale,
        "Received event energy (pJ per native pixel)",vec![0.001,0.01,0.1,1.,10.,100.]);
    cb.horizontal=true;f.colorbars.push(cb);
    f.labels.push(Label::new("Blue curves: time-matched radar trajectory in satellite viewing angles",97.,99.));
    let mut ax=Axes::new([1.5,3.1],[0.,3.5],"","Energy (pJ per 10 ms)");
    ax.xticks=vec![1.5,2.,2.5,3.];ax.yticks=vec![0.,1.,2.,3.];
    let mut p=Panel::new(Rect::new(20.,114.,155.,32.),ax);
    p.push(Layer::Line(Line::new(s.lightcurve,"#a52c20")));
    p.push(Layer::Line(Line::new(vec![[s.terminus,0.],[s.terminus,3.5]],"#171717")));
    f.panels.push(p);
    f.labels.push(Label::new("Black line: nominal 33.61 km radar terminus",97.,157.));
    let mut ax=Axes::new([1.5,3.1],[0.,1.05],"Seconds after 18:06:20 UTC","Energy fraction in model pixel");
    ax.xticks=vec![1.5,2.,2.5,3.];ax.yticks=vec![0.,0.5,1.];
    let mut p=Panel::new(Rect::new(20.,174.,155.,30.),ax);
    // Keep separate segments across intervals with no retained detections.
    let mut segment=Vec::new(); let mut previous=-100.;
    for row in s.binned {
        if row[0]-previous>0.015 && !segment.is_empty() {
            p.push(Layer::Line(Line::new(std::mem::take(&mut segment),"#006ba4")));
        }
        segment.push([row[0],row[3]]);previous=row[0];
    }
    if !segment.is_empty(){p.push(Layer::Line(Line::new(segment,"#006ba4")));}
    p.push(Layer::Line(Line::new(vec![[s.terminus,0.],[s.terminus,1.05]],"#171717")));
    f.panels.push(p);
    f.save(dir.join("glm_trajectory_consistency.pdf"))?;
    f.save(dir.join("glm_trajectory_consistency.png"))?;
    Ok(())
}
