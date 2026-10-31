use std::collections::BTreeMap;
use std::path::Path;

use cardscan_core::{catalog_id, route_rows, score_key, top_k, CATALOGS};
use serde::Serialize;

use crate::npy;

#[derive(Debug, Clone, Serialize)]
pub struct Card {
    pub id: String,
    pub ct_id: String,
    pub public_id: String,
    pub name: String,
    pub collector_number: Option<String>,
    pub set: Option<String>,
    pub pokoin_url: String,
    pub catalogs: Vec<String>,
}

#[derive(Debug)]
pub struct Hit {
    pub card: Card,
    pub score_units: i32,
}

pub struct Index {
    pub cards: Vec<Card>,
    rows: Vec<f32>,
    dims: usize,
    all: Vec<usize>,
    routes: BTreeMap<&'static str, Vec<usize>>,
}

impl Index {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let npy = npy::read_f32_2d(&dir.join("embeddings.npy"))?;
        if npy.cols != 128 {
            return Err(format!(
                "cdn_milo embeddings shape ({}, {})",
                npy.rows, npy.cols
            ));
        }
        let text = std::fs::read_to_string(dir.join("metadata.jsonl"))
            .map_err(|e| format!("{}: {e}", dir.join("metadata.jsonl").display()))?;
        let mut cards = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let obj: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| format!("{}: {e}", dir.join("metadata.jsonl").display()))?;
            let ct = string_field(&obj, "ct_id");
            if !is_digits(&ct) {
                continue;
            }
            let mut public = string_field(&obj, "public_id");
            if !is_digits(&public) {
                public = (ct.parse::<u64>().unwrap_or(0) * 2).to_string();
            }
            let catalogs = obj
                .get("catalogs")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            cards.push(Card {
                id: ct.clone(),
                ct_id: ct,
                public_id: public.clone(),
                name: obj
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                collector_number: optional_string(obj.get("n")),
                set: optional_string(obj.get("set")),
                pokoin_url: format!("https://pokoin.com/{public}"),
                catalogs,
            });
        }
        if cards.len() != npy.rows {
            return Err(format!(
                "metadata {} vs embeddings {}",
                cards.len(),
                npy.rows
            ));
        }
        Self::from_parts(cards, npy.data, npy.cols)
    }

    pub fn from_parts(cards: Vec<Card>, raw: Vec<f32>, dims: usize) -> Result<Self, String> {
        if dims == 0 || raw.len() != cards.len() * dims {
            return Err("embedding data does not match card rows".to_string());
        }
        let mut rows = raw;
        let mut usable = Vec::with_capacity(cards.len());
        for row in rows.chunks_exact_mut(dims) {
            let norm = row
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>()
                .sqrt();
            usable.push(norm > 0.0);
            let scale = 1.0 / norm.max(1e-8);
            for v in row.iter_mut() {
                *v = (*v as f64 * scale) as f32;
            }
        }
        let all = (0..cards.len()).filter(|&i| usable[i]).collect();
        let mut routes = BTreeMap::new();
        for id in CATALOGS {
            let member: Vec<bool> = cards
                .iter()
                .map(|c| c.catalogs.iter().any(|x| x == id))
                .collect();
            let uncatalogued: Vec<bool> = cards.iter().map(|c| c.catalogs.is_empty()).collect();
            routes.insert(
                id,
                route_rows(&member, &uncatalogued, false)
                    .into_iter()
                    .filter(|&i| usable[i])
                    .collect(),
            );
        }
        Ok(Self {
            cards,
            rows,
            dims,
            all,
            routes,
        })
    }

    pub fn n(&self) -> usize {
        self.cards.len()
    }

    pub fn no_image(&self) -> usize {
        self.cards.len() - self.all.len()
    }

    pub fn catalog_counts(&self) -> BTreeMap<&'static str, usize> {
        self.routes.iter().map(|(k, v)| (*k, v.len())).collect()
    }

    /// `(label, unknown, row indices)`; unknown ids fall back to all rows.
    pub fn route(&self, request: Option<&str>) -> (&'static str, bool, &[usize]) {
        let (label, unknown) = catalog_id(request);
        if unknown || label == "all" {
            ("all", unknown, &self.all)
        } else {
            (label, false, &self.routes[label])
        }
    }

    pub fn search(&self, query: &[f32], k: usize, rows: &[usize]) -> Vec<Hit> {
        let mut keys = Vec::with_capacity(rows.len());
        let mut scores = Vec::with_capacity(rows.len());
        for &row in rows {
            let base = row * self.dims;
            let v = &self.rows[base..base + self.dims];
            let mut dot = 0f32;
            for i in 0..self.dims {
                dot += v[i] * query[i];
            }
            scores.push(dot);
            keys.push(score_key(dot));
        }
        let k = k.min(keys.len());
        let picked = top_k(&keys, k);
        picked
            .into_iter()
            .map(|i| {
                let row = rows[i];
                Hit {
                    card: self.cards[row].clone(),
                    score_units: cardscan_core::score_units(scores[i] as f64),
                }
            })
            .collect()
    }
}

fn string_field(obj: &serde_json::Value, key: &str) -> String {
    match obj.get(key) {
        Some(serde_json::Value::String(s)) => s.trim().to_string(),
        Some(serde_json::Value::Number(n)) => n.to_string(),
        _ => String::new(),
    }
}

fn optional_string(v: Option<&serde_json::Value>) -> Option<String> {
    match v {
        Some(serde_json::Value::String(s)) if !s.trim().is_empty() => Some(s.trim().to_string()),
        Some(serde_json::Value::Number(n)) if n.as_i64() != Some(0) => Some(n.to_string()),
        _ => None,
    }
}

fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}
