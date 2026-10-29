use std::path::Path;

use cardscan_core::{accept_box, inverse_tenths, letterbox};
use ort::session::Session;
use ort::value::Tensor;

use crate::imageops::{self, Rgb};

#[derive(Debug, Clone, Copy)]
pub struct Detection {
    pub xyxy: [f64; 4],
    pub conf: f32,
    pub conf_units: i32,
}

pub struct Engine {
    yolo: Session,
    milo: Session,
    yolo_input: String,
    milo_input: String,
}

impl Engine {
    pub fn load(yolo_path: &Path, milo_path: &Path, threads: usize) -> Result<Self, String> {
        let yolo = Session::builder()
            .map_err(|e| format!("onnxruntime: {e}"))?
            .with_intra_threads(threads)
            .map_err(|e| e.to_string())?
            .with_inter_threads(1)
            .map_err(|e| e.to_string())?
            .commit_from_file(yolo_path)
            .map_err(|e| format!("{}: {e}", yolo_path.display()))?;
        let milo = Session::builder()
            .map_err(|e| format!("onnxruntime: {e}"))?
            .with_intra_threads(threads)
            .map_err(|e| e.to_string())?
            .with_inter_threads(1)
            .map_err(|e| e.to_string())?
            .commit_from_file(milo_path)
            .map_err(|e| format!("{}: {e}", milo_path.display()))?;
        let yolo_input = yolo
            .inputs
            .first()
            .map(|i| i.name.clone())
            .ok_or_else(|| "detector has no inputs".to_string())?;
        let milo_input = milo
            .inputs
            .first()
            .map(|i| i.name.clone())
            .ok_or_else(|| "milo has no inputs".to_string())?;
        // Warm both graphs so the first request is not the one paying init.
        Ok(Self {
            yolo,
            milo,
            yolo_input,
            milo_input,
        })
    }

    pub fn detect(&mut self, rgb: &Rgb) -> Result<Vec<Detection>, String> {
        let lb = letterbox(rgb.w, rgb.h);
        let canvas = imageops::letterbox_canvas(rgb, lb.nw, lb.nh, lb.left, lb.top);
        let input = imageops::nchw_640(&canvas);
        let tensor =
            Tensor::from_array(([1usize, 3, 640, 640], input)).map_err(|e| e.to_string())?;
        let outputs = self
            .yolo
            .run(ort::inputs![self.yolo_input.clone() => tensor])
            .map_err(|e| e.to_string())?;
        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let mut boxes = Vec::new();
        for row in data.as_chunks::<6>().0 {
            let conf = row[4];
            if conf < 0.25 {
                continue;
            }
            let raw: [f64; 4] = std::array::from_fn(|i| {
                let pad = if i % 2 == 0 { lb.left } else { lb.top };
                let size = if i % 2 == 0 { rgb.w } else { rgb.h };
                ((row[i] - pad as f32) / lb.scale as f32).clamp(0.0, size as f32) as f64
            });
            if !accept_box(raw, conf, rgb.w, rgb.h) {
                continue;
            }
            let xyxy: [f64; 4] = std::array::from_fn(|i| {
                let size = if i % 2 == 0 { rgb.w } else { rgb.h };
                inverse_tenths(raw[i], size) as f64 / 10.0
            });
            boxes.push(Detection {
                xyxy,
                conf,
                conf_units: cardscan_core::score_units(conf as f64),
            });
        }
        boxes.sort_by_key(|b| std::cmp::Reverse(b.conf_units));
        Ok(boxes)
    }

    pub fn embed(&mut self, rgb: &Rgb) -> Result<Vec<f32>, String> {
        let input = imageops::milo_input(rgb);
        let tensor =
            Tensor::from_array(([1usize, 3, 448, 448], input)).map_err(|e| e.to_string())?;
        let outputs = self
            .milo
            .run(ort::inputs![self.milo_input.clone() => tensor])
            .map_err(|e| e.to_string())?;
        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let mut vec = data.to_vec();
        let norm = vec
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            .sqrt();
        let scale = 1.0 / norm.max(1e-8);
        for v in vec.iter_mut() {
            *v = (*v as f64 * scale) as f32;
        }
        Ok(vec)
    }
}
