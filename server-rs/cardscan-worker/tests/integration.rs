use std::path::PathBuf;

use cardscan_worker::config::Config;
use cardscan_worker::engine::Engine;
use cardscan_worker::imageops;
use cardscan_worker::index::Index;
use cardscan_worker::pipeline::run_identify;

/// Loads the real detector, Milo and index from the environment and checks the
/// Numel photo end to end. Run with:
///   ORT_DYLIB_PATH=... CDN_MILO=... cargo test -p cardscan-worker -- --ignored
#[test]
#[ignore]
fn real_models_identify_numel() {
    let cfg = Config::from_env();
    let index = Index::load(&cfg.index).expect("index");
    let mut engine = Engine::load(&cfg.yolo, &cfg.milo, cfg.threads).expect("models");
    let path = std::env::var("CARDSCAN_TEST_IMAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/home/nez/data/pokoin-leftovers/objects/370861_numel.jpg")
        });
    let blob = std::fs::read(&path).expect("test image");
    let mut rgb = imageops::decode(&blob).expect("decode");
    let (w, h) = cardscan_core::long_edge_size(rgb.w, rgb.h);
    if (w, h) != (rgb.w, rgb.h) {
        rgb = imageops::resize_bilinear(&rgb, w, h);
    }
    let result =
        run_identify(&mut engine, &index, &rgb, 3, Some("pokemon_western")).expect("identify");
    let top = result.hits.first().expect("a hit");
    assert_eq!(top.card.ct_id, "370861");
    assert_eq!(top.card.name, "Numel");
    assert!(result.detect_ms > 0.0);
}
