use anyhow::{Context, Result, ensure};
use rustplotlib::*;
use serde::Deserialize;
use std::{fs, io::Write, path::PathBuf, process::{Command,Stdio}};

#[derive(Deserialize)]
struct Scene {
    bounds: [f64;4],
    coast: Vec<Vec<[f64;2]>>,
    energy_polygons: Vec<Vec<[f64;2]>>,
    light_curve: Vec<[f64;2]>,
    altitude_ticks: Vec<[f64;2]>,
    animation_times: Vec<f64>,
    animation_energy_pj: Vec<Vec<f64>>,
}

fn main() -> Result<()> {
    let data = PathBuf::from(std::env::args().nth(1).context("data directory")?);
    let scene: Scene = serde_json::from_slice(&fs::read(data.join("rustplotlib_scene.json"))?)?;
    let limit = std::env::var("GLM_ANIMATION_FRAME_LIMIT").ok().map(|s|s.parse::<usize>()).transpose()?;
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 320.;
    fig.height_mm = 180.;
    fig.raster_dpi = 152.4;
    fig.tight_export = false;
    let maximum = scene.animation_energy_pj.iter().flatten().copied().fold(0.,f64::max);
    let upper = 10_f64.powf(maximum.log10().ceil());
    let scale = fig.scale(Scale::new([0.001,upper],Normalization::Log,
        ColorMap::new(vec![[1.,0.92,0.05],[1.,0.45,0.01],[0.75,0.,0.]])?)?);
    let [west,east,south,north] = scene.bounds;
    let mut axes = Axes::new([west,east],[south,north],"Longitude","Latitude");
    axes.xticks = vec![-73.,-72.,-71.,-70.,-69.];
    axes.yticks = vec![41.,42.,43.,44.];
    axes.xticklabels = Some((69..=73).rev().map(|v|format!(r"${v}^\circ\mathrm{{W}}$")).collect());
    axes.yticklabels = Some((41..=44).map(|v|format!(r"${v}^\circ\mathrm{{N}}$")).collect());
    let mut map = Panel::new(Rect::new(16.,22.,125.,132.),axes);
    map.push(Layer::Image(image::open(data.join("cloud_background.png"))?.into_rgba8()));
    map.push(Layer::PolygonMesh(PolygonMesh::new(scene.energy_polygons.clone(),vec![f64::NAN;scene.energy_polygons.len()],scale)?));
    for points in scene.coast {
        let mut line = Line::new(points,"#42c5df");
        line.width_pt = 0.4;
        map.push(Layer::Line(line));
    }
    fig.panels.push(map);
    let mut ticks = vec![0.001,0.01,0.1];
    if upper >= 1. {ticks.push(1.);}
    if upper >= 10. {ticks.push(10.);}
    fig.colorbars.push(Colorbar::new(Rect::new(144.,22.,3.,132.),scale,
        r"GLM event energy (pJ pixel$^{-1}$ per 2 ms)",ticks));
    let mut axes = Axes::new([1.5,3.4],[0.,3.5],
        "Seconds after 18:06:20 UTC","Energy (pJ per 10 ms bin)");
    axes.xticks = vec![1.5,2.,2.5,3.];
    axes.yticks = vec![0.,1.,2.,3.];
    axes.secondary_x = Some(SecondaryAxis {
        label: "Modeled altitude (km)".into(),
        ticks: scene.altitude_ticks.iter().map(|[t,h]|(*t,*h)).collect(),
    });
    let mut curve = Panel::new(Rect::new(183.,52.,122.,75.),axes);
    let steps = scene.light_curve.iter().flat_map(|[t,e]|[[t-0.005,e/1000.],[t+0.005,e/1000.]]).collect();
    curve.push(Layer::Line(Line::new(steps,"#a52c20")));
    curve.push(Layer::Line(Line::new(vec![[1.5,0.],[1.5,3.5]],"#116b91")));
    fig.panels.push(curve);
    fig.labels.push(Label::new("GOES-19 ABI + GLM, 30 May 2026",160.,7.));
    fig.labels.push(Label::new("2 ms energy bins; playback 16.7 times slower",244.,154.));
    fig.labels.push(Label::new("Reported GLM coordinates; no altitude parallax correction",160.,171.));
    let output = data.join("goes19_glm_energy_2ms_animation.mp4");
    let working_output = data.join("goes19_glm_energy_2ms_animation.rendering.mp4");
    let mut ffmpeg = Command::new("ffmpeg").args([
        "-hide_banner","-loglevel","warning","-y","-f","image2pipe","-framerate","30",
        "-vcodec","png","-i","pipe:0","-an","-c:v","libx264","-preset","medium",
        "-crf","17","-pix_fmt","yuv420p","-movflags","+faststart",
    ]).arg(&working_output).stdin(Stdio::piped()).spawn()?;
    let mut input = ffmpeg.stdin.take().context("ffmpeg stdin")?;
    let frame_path = data.join("animation_work_frame.png");
    let count = limit.unwrap_or(scene.animation_times.len()).min(scene.animation_times.len());
    for i in 0..count {
        if let Layer::PolygonMesh(mesh) = &mut fig.panels[0].layers[1] {
            mesh.values = scene.animation_energy_pj[i].iter().map(|v| if *v>0. {*v} else {f64::NAN}).collect();
        }
        let time = scene.animation_times[i];
        if let Layer::Line(line) = &mut fig.panels[1].layers[1] {
            line.points = vec![[time,0.],[time,3.5]];
        }
        fig.save(&frame_path)?;
        input.write_all(&fs::read(&frame_path)?)?;
        if i%100==0 {println!("Frame {}/{}",i+1,count);}
        if [0,100,250,450,650,949].contains(&i) {
            fs::copy(&frame_path,data.join(format!("animation_check_{i:04}.png")))?;
        }
    }
    drop(input);
    ensure!(ffmpeg.wait()?.success(),"ffmpeg encoding failed");
    fs::rename(&working_output,&output)?;
    fs::write(data.join("goes19_glm_energy_2ms_animation.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "frames":count,"fps":30,"duration_s":count as f64/30.,
        "measurement_bin_s":0.002,"playback_slowdown":1./(30.*0.002),
        "time_origin_utc":"2026-05-30T18:06:20Z",
        "bin_centers_s":&scene.animation_times[..count],
        "background":"Static GOES-19 ABI C02 18:06 UTC image",
        "quantity":"Level-2 received event energy in the GLM passband per pixel per 2 ms bin",
        "spatial_geometry":"Reconstructed native-pixel footprints at reported GLM coordinates; no bolide-altitude parallax correction"
    }))?)?;
    println!("{}",output.display());
    Ok(())
}
