use std::time::Instant;

use cardscan_core::{crop_rect, early_exit, immediate, orientation_plan, replace_best};

use crate::engine::{Detection, Engine};
use crate::imageops::{self, Rgb};
use crate::index::{Hit, Index};

pub struct IdentifyResult {
    pub detect_ms: f64,
    pub identify_ms: f64,
    pub boxes: Vec<Detection>,
    pub hits: Vec<Hit>,
    pub orientations: usize,
    pub catalog: &'static str,
    pub unknown_catalog: bool,
}

pub fn run_identify(
    engine: &mut Engine,
    index: &Index,
    rgb: &Rgb,
    top_k: usize,
    catalog: Option<&str>,
) -> Result<IdentifyResult, String> {
    let (label, unknown, rows) = index.route(catalog);
    let t0 = Instant::now();
    let boxes = engine.detect(rgb)?;
    let detect_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut hits = Vec::new();
    let mut orientations = 0;
    let mut identify_ms = 0.0;
    if let Some(first) = boxes.first() {
        let rect = crop_rect(first.xyxy, rgb.w, rgb.h);
        if !rect.is_empty() {
            let crop = rgb.crop(rect.x1, rect.y1, rect.x2, rect.y2);
            let t1 = Instant::now();
            let (best, tried) = embed_best(engine, index, &crop, top_k, rows)?;
            identify_ms = t1.elapsed().as_secs_f64() * 1000.0;
            hits = best;
            orientations = tried;
        }
    }
    Ok(IdentifyResult {
        detect_ms,
        identify_ms,
        boxes,
        hits,
        orientations,
        catalog: label,
        unknown_catalog: unknown,
    })
}

fn embed_best(
    engine: &mut Engine,
    index: &Index,
    crop: &Rgb,
    top_k: usize,
    rows: &[usize],
) -> Result<(Vec<Hit>, usize), String> {
    let plan = orientation_plan(crop.w, crop.h);
    let mut best: Vec<Hit> = Vec::new();
    let mut best_units: i32 = -10000;
    let mut tried = 0;
    for rot in plan {
        tried += 1;
        let oriented = imageops::rotate(crop, *rot);
        let query = engine.embed(&oriented)?;
        let hits = index.search(&query, top_k, rows);
        let units = hits.first().map(|h| h.score_units).unwrap_or(-10000);
        if replace_best(best_units, units) {
            best_units = units;
            best = hits;
        }
        if early_exit(best_units) {
            break;
        }
    }
    Ok((best, tried))
}

pub fn is_immediate(result: &IdentifyResult) -> bool {
    let top = result.hits.first().map(|h| h.score_units);
    immediate(top, result.boxes.len())
}
