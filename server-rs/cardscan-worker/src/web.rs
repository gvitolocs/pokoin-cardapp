use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::extract::multipart::{Multipart, MultipartRejection};
use axum::extract::{DefaultBodyLimit, Query, State};
use axum::http::{header, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use cardscan_core::{clamp_top_k, rounded};
use serde_json::{json, Value};
use tokio::sync::Semaphore;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

use crate::config::Config;
use crate::engine::Engine;
use crate::imageops;
use crate::index::Index;
use crate::pipeline;

const MAX_BYTES: usize = 4 * 1024 * 1024;

pub struct AppState {
    pub cfg: Config,
    pub index: Index,
    pub engine: Option<Mutex<Engine>>,
    pub gate: Arc<Semaphore>,
}

impl AppState {
    pub fn new(cfg: Config, index: Index, engine: Option<Engine>) -> Arc<Self> {
        Arc::new(Self {
            cfg,
            index,
            engine: engine.map(Mutex::new),
            gate: Arc::new(Semaphore::new(1)),
        })
    }
}

pub fn app(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _| allowed_origin(origin)))
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);
    Router::new()
        .route("/", get(index_page))
        .route("/health", get(health))
        .route("/identify", post(identify))
        .layer(DefaultBodyLimit::max(MAX_BYTES + 1024 * 1024))
        .layer(cors)
        .with_state(state)
}

fn allowed_origin(origin: &HeaderValue) -> bool {
    let Ok(origin) = origin.to_str() else {
        return false;
    };
    matches!(
        origin,
        "https://pokoin.com" | "https://www.pokoin.com" | "https://cardscan.pokoin.com"
    ) || (origin.starts_with("https://") && origin.ends_with(".vercel.app"))
}

async fn index_page(State(state): State<Arc<AppState>>) -> Response {
    let page: PathBuf = state.cfg.web.join("index.html");
    match tokio::fs::read(&page).await {
        Ok(body) => ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], body).into_response(),
        Err(_) => error(StatusCode::NOT_FOUND, "scanner page missing"),
    }
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    let index = state
        .cfg
        .index
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    Json(json!({
        "ok": true,
        "host": "nezopt",
        "n": state.index.n(),
        "no_image": state.index.no_image(),
        "identity": "ct_id",
        "detect": "yolo-onnx-cpu",
        "identify": "cdn-milo-128",
        "orient": "90-270",
        "index": index,
        "engine": "rust",
        "catalogs": state.index.catalog_counts(),
    }))
    .into_response()
}

async fn identify(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HashMap<String, String>>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Response {
    let top_k = match query.get("top_k") {
        None => 5,
        Some(raw) => match raw.parse::<i64>() {
            Ok(v) => v,
            Err(_) => {
                return error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "top_k: Input should be a valid integer",
                )
            }
        },
    };
    let live = match query.get("live") {
        None => false,
        Some(raw) => match parse_bool(raw) {
            Some(v) => v,
            None => {
                return error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "live: Input should be a valid boolean",
                )
            }
        },
    };
    let catalog = query.get("catalog").cloned();
    let top_k = clamp_top_k(top_k, live);

    let mut multipart = match multipart {
        Ok(m) => m,
        Err(_) => return error(StatusCode::UNPROCESSABLE_ENTITY, "file: field required"),
    };
    let mut blob: Option<Vec<u8>> = None;
    loop {
        match multipart.next_field().await {
            Ok(Some(field)) => {
                if field.name() == Some("file") {
                    match field.bytes().await {
                        Ok(bytes) => blob = Some(bytes.to_vec()),
                        Err(_) => {
                            return error(
                                StatusCode::UNPROCESSABLE_ENTITY,
                                "file: could not read upload",
                            )
                        }
                    }
                }
            }
            Ok(None) => break,
            Err(_) => return error(StatusCode::UNPROCESSABLE_ENTITY, "file: field required"),
        }
    }
    let Some(blob) = blob else {
        return error(StatusCode::UNPROCESSABLE_ENTITY, "file: field required");
    };
    if blob.is_empty() || blob.len() > MAX_BYTES {
        return error(
            StatusCode::BAD_REQUEST,
            &format!("image must be 1\u{2013}{MAX_BYTES} bytes"),
        );
    }
    let Some(mut rgb) = imageops::decode(&blob) else {
        return error(StatusCode::BAD_REQUEST, "could not decode image");
    };
    let (nw, nh) = cardscan_core::long_edge_size(rgb.w, rgb.h);
    if nw != rgb.w || nh != rgb.h {
        rgb = imageops::resize_bilinear(&rgb, nw, nh);
    }
    let (img_w, img_h) = (rgb.w, rgb.h);

    let permit = if live {
        match state.gate.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                return Json(json!({
                    "ok": true,
                    "busy": true,
                    "identity": "ct_id",
                    "host": "nezopt",
                }))
                .into_response()
            }
        }
    } else {
        state
            .gate
            .clone()
            .acquire_owned()
            .await
            .expect("semaphore is never closed")
    };

    let task_state = state.clone();
    let catalog_for_task = catalog.clone();
    let joined = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let Some(engine) = task_state.engine.as_ref() else {
            return Err("models not loaded".to_string());
        };
        let mut engine = engine.lock().expect("model lock poisoned");
        pipeline::run_identify(
            &mut engine,
            &task_state.index,
            &rgb,
            top_k,
            catalog_for_task.as_deref(),
        )
    })
    .await;
    let result = match joined {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => return error(StatusCode::INTERNAL_SERVER_ERROR, &e),
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };

    if result.unknown_catalog {
        eprintln!("warning: unknown catalog, searched all rows");
    }

    let hits: Vec<Value> = result
        .hits
        .iter()
        .map(|hit| {
            let mut v = serde_json::to_value(&hit.card).expect("card serializes");
            v["score"] = json!(hit.score_units as f64 / 10000.0);
            v
        })
        .collect();
    let top1 = hits.first().cloned().unwrap_or(Value::Null);
    let boxes: Vec<Value> = result
        .boxes
        .iter()
        .map(|b| {
            json!({
                "xyxy": b.xyxy,
                "conf": b.conf_units as f64 / 10000.0,
            })
        })
        .collect();
    let (top_ct, top_score, top_url) = result
        .hits
        .first()
        .map(|h| {
            (
                h.card.ct_id.clone(),
                h.score_units as f64 / 10000.0,
                h.card.pokoin_url.clone(),
            )
        })
        .unwrap_or_else(|| ("-".to_string(), 0.0, String::new()));
    println!(
        "identify live={} detect={:.0} milo={:.0} orient={} ct={} score={} pokoin={}",
        live as u8,
        result.detect_ms,
        result.identify_ms,
        result.orientations,
        top_ct,
        top_score,
        top_url
    );

    Json(json!({
        "ok": true,
        "host": "nezopt",
        "identity": "ct_id",
        "detect_ms": rounded(result.detect_ms, 1),
        "identify_ms": rounded(result.identify_ms, 1),
        "img_w": img_w,
        "img_h": img_h,
        "orientations": result.orientations,
        "immediate": pipeline::is_immediate(&result),
        "boxes": boxes,
        "hits": hits,
        "top1": top1,
        "catalog": result.catalog,
    }))
    .into_response()
}

fn parse_bool(raw: &str) -> Option<bool> {
    match raw.to_ascii_lowercase().as_str() {
        "true" | "1" | "on" | "yes" | "y" | "t" => Some(true),
        "false" | "0" | "off" | "no" | "n" | "f" => Some(false),
        _ => None,
    }
}

fn error(status: StatusCode, detail: &str) -> Response {
    (status, Json(json!({"detail": detail}))).into_response()
}
