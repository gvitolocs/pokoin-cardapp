use cardscan_worker::config::Config;
use cardscan_worker::engine::Engine;
use cardscan_worker::index::Index;
use cardscan_worker::web::{app, AppState};

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("cardscan-worker: {e}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let cfg = Config::from_env();
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        eprintln!(
            "note: ORT_DYLIB_PATH is not set; ort (load-dynamic) needs a libonnxruntime.so \
             (>= 1.22), e.g. BattleScan/.venv/.../onnxruntime/capi/libonnxruntime.so.1.29.0"
        );
    }
    let index = Index::load(&cfg.index)?;
    let engine = Engine::load(&cfg.yolo, &cfg.milo, cfg.threads)?;
    println!(
        "cardscan-worker: n={} catalogs={:?} yolo={} milo={} index={} threads={}",
        index.n(),
        index.catalog_counts(),
        cfg.yolo.display(),
        cfg.milo.display(),
        cfg.index.display(),
        cfg.threads
    );
    let state = AppState::new(cfg.clone(), index, Some(engine));
    let listener = tokio::net::TcpListener::bind((cfg.bind, cfg.port))
        .await
        .map_err(|e| format!("bind {}:{}: {e}", cfg.bind, cfg.port))?;
    println!(
        "cardscan-worker: listening on http://{}:{}",
        cfg.bind, cfg.port
    );
    axum::serve(listener, app(state))
        .await
        .map_err(|e| e.to_string())
}
