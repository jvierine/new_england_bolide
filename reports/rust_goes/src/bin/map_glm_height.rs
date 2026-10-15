use anyhow::Result;
use rustplotlib::*;
use serde::Deserialize;
use std::{fs,path::PathBuf};

#[derive(Deserialize)]
struct Scene {
    points:Vec<[f64;3]>, early_footprints:Vec<Vec<[f64;2]>>,
    track:Vec<[f64;2]>,terminus_lonlat:[f64;2],coast:Vec<Vec<[f64;2]>>,
}

fn main()->Result<()> {
    let dir=PathBuf::from(std::env::args().nth(1).expect("data directory"));
    let s:Scene=serde_json::from_slice(&fs::read(dir.join("glm_height_corrected_map_scene.json"))?)?;
    let cloud=image::open(dir.join("cloud_background.png"))?.into_rgba8();
    let mut f=Figure::new(IntendedUse::ArticleWide);
    f.width_mm=190.;f.height_mm=120.;
    let scale=f.scale(Scale::new([1.6,3.1],Normalization::Linear,ColorMap::estuary())?);
    let bounds=[[-70.65,-70.25,41.78,42.09],[-71.,-69.9,41.3,42.2]];
    for (i,b) in bounds.into_iter().enumerate() {
        let [west,east,south,north]=b;
        let x=20.+i as f64*85.;
        let mut axes=Axes::new([west,east],[south,north],"Longitude",if i==0 {"Latitude"}else{""});
        axes.xticks=if i==0 {vec![-70.6,-70.4]}else{vec![-70.8,-70.4,-70.]};
        axes.yticks=if i==0 {vec![41.8,41.9,42.]}else{vec![41.4,41.8,42.2]};
        let mut p=Panel::new(Rect::new(x,14.,72.,76.),axes);
        let px0=((west+73.2)/4.9*cloud.width() as f64).round() as u32;
        let px1=((east+73.2)/4.9*cloud.width() as f64).round() as u32;
        let py0=((44.2-north)/3.9*cloud.height() as f64).round() as u32;
        let py1=((44.2-south)/3.9*cloud.height() as f64).round() as u32;
        p.push(Layer::Image(image::imageops::crop_imm(&cloud,px0,py0,px1-px0,py1-py0).to_image()));
        for points in &s.coast {
            let mut line=Line::new(points.clone(),"#42c5df");line.width_pt=0.4;
            p.push(Layer::Line(line));
        }
        let mut polygons=Vec::new();let mut values=Vec::new();
        let radius=(east-west)*0.003;
        for [lon,lat,time] in &s.points {
            if (i==0 && *time>2.05) || *lon<west || *lon>east || *lat<south || *lat>north {continue;}
            let circle=(0..12).map(|j|{
                let a=j as f64/12.*std::f64::consts::TAU;
                [lon+radius*a.cos(),lat+radius*a.sin()]
            }).collect();
            polygons.push(circle);values.push(*time);
        }
        p.push(Layer::PolygonMesh(PolygonMesh::new(polygons,values,scale)?));
        if i==0 {
            for outline in &s.early_footprints {
                let mut closed=outline.clone();closed.push(closed[0]);
                let mut line=Line::new(closed,"#19466a");line.width_pt=0.55;
                p.push(Layer::Line(line));
            }
        }
        let mut line=Line::new(s.track.clone(),"#003b70");line.width_pt=1.4;
        p.push(Layer::Line(line));
        let [lon,lat]=s.terminus_lonlat;
        let marker=(0..16).map(|j|{let a=j as f64/16.*std::f64::consts::TAU;[lon+radius*1.8*a.cos(),lat+radius*1.8*a.sin()]}).collect();
        let marker_scale=f.scale(Scale::new([0.,1.],Normalization::Linear,ColorMap::new(vec![[1.,0.45,0.],[1.,0.45,0.]])?)?);
        p.push(Layer::PolygonMesh(PolygonMesh::new(vec![marker],vec![0.5],marker_scale)?));
        f.panels.push(p);
        f.labels.push(Label::new(if i==0 {"(a) Compact early emission"}else{"(b) All mapped event centers"},x+36.,6.));
    }
    let mut cb=Colorbar::new(Rect::new(20.,106.,157.,2.),scale,"Seconds after 18:06:20 UTC",vec![1.6,2.,2.5,3.1]);
    cb.horizontal=true;f.colorbars.push(cb);
    f.save(dir.join("glm_height_corrected_map.pdf"))?;
    f.save(dir.join("glm_height_corrected_map.png"))?;
    Ok(())
}
