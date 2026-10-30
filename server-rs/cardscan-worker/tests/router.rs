use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use cardscan_worker::config::Config;
use cardscan_worker::index::{Card, Index};
use cardscan_worker::web::{app, AppState};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

fn card(ct: &str, catalogs: &[&str]) -> Card {
    Card {
        id: ct.to_string(),
        ct_id: ct.to_string(),
        public_id: (ct.parse::<u64>().unwrap() * 2).to_string(),
        name: format!("card {ct}"),
        collector_number: None,
        set: None,
        pokoin_url: format!("https://pokoin.com/{}", ct.parse::<u64>().unwrap() * 2),
        catalogs: catalogs.iter().map(|s| s.to_string()).collect(),
    }
}

fn fake_index() -> Index {
    let cards = vec![
        card("1", &["pokemon_western"]),
        card("2", &["pokemon_japanese"]),
        card("3", &[]),
    ];
    let mut raw = vec![0f32; 3 * 128];
    raw[0] = 1.0;
    raw[128] = 1.0;
    raw[256] = 1.0;
    Index::from_parts(cards, raw, 128).unwrap()
}

fn state(web: PathBuf) -> Arc<AppState> {
    let cfg = Config {
        bind: IpAddr::from([127, 0, 0, 1]),
        port: 0,
        threads: 1,
        web,
        index: PathBuf::from("cdn_milo_test"),
        milo: PathBuf::from("milo.onnx"),
        yolo: PathBuf::from("yolo.onnx"),
    };
    AppState::new(cfg, fake_index(), None)
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cardscan-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

async fn body_json(response: axum::response::Response) -> Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn health_shape_and_catalog_counts() {
    let dir = temp_dir("health");
    let response = app(state(dir.clone()))
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let v = body_json(response).await;
    assert_eq!(v["ok"], true);
    assert_eq!(v["host"], "nezopt");
    assert_eq!(v["n"], 3);
    assert_eq!(v["identity"], "ct_id");
    assert_eq!(v["detect"], "yolo-onnx-cpu");
    assert_eq!(v["identify"], "cdn-milo-128");
    assert_eq!(v["orient"], "90-270");
    assert_eq!(v["index"], "cdn_milo_test");
    assert_eq!(v["engine"], "rust");
    assert_eq!(v["catalogs"]["pokemon_western"], 2);
    assert_eq!(v["catalogs"]["pokemon_japanese"], 2);
    assert_eq!(v["catalogs"]["pokemon_chinese"], 1);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn index_page_missing_and_present() {
    let dir = temp_dir("page");
    let response = app(state(dir.clone()))
        .oneshot(Request::get("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(body_json(response).await["detail"], "scanner page missing");

    std::fs::write(dir.join("index.html"), b"<html>scanner</html>").unwrap();
    let response = app(state(dir.clone()))
        .oneshot(Request::get("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/html; charset=utf-8"
    );
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&bytes[..], b"<html>scanner</html>");
    std::fs::remove_dir_all(&dir).unwrap();
}

fn multipart(field: Option<&[u8]>) -> (String, Vec<u8>) {
    let boundary = "cardscan-boundary";
    let mut body = Vec::new();
    if let Some(data) = field {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.jpg\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(data);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

async fn post_identify(field: Option<&[u8]>, query: &str) -> axum::response::Response {
    let (content_type, body) = multipart(field);
    let dir = temp_dir("identify");
    let response = app(state(dir.clone()))
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(format!("/identify{query}"))
                .header(header::CONTENT_TYPE, content_type)
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    response
}

#[tokio::test]
async fn identify_errors_match_fastapi() {
    let response = post_identify(None, "").await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body_json(response).await["detail"], "file: field required");

    let response = post_identify(Some(b""), "").await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(response).await["detail"],
        "image must be 1\u{2013}4194304 bytes"
    );

    let response = post_identify(Some(b"not an image"), "").await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(response).await["detail"],
        "could not decode image"
    );

    let response = post_identify(Some(b"not an image"), "?top_k=abc").await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[test]
fn catalog_routing_keeps_uncatalogued_rows() {
    let index = fake_index();
    assert_eq!(index.route(None).0, "all");
    assert_eq!(index.route(None).2, &[0, 1, 2]);
    assert_eq!(index.route(Some("")).0, "all");
    assert_eq!(index.route(Some("pokemon")).0, "all");
    assert_eq!(index.route(Some("pokemon_western")).2, &[0, 2]);
    assert_eq!(index.route(Some("pokemon_japanese")).2, &[1, 2]);
    assert_eq!(index.route(Some("pokemon_chinese")).2, &[2]);
    let (label, unknown, rows) = index.route(Some("typo"));
    assert_eq!((label, unknown, rows), ("all", true, &[0, 1, 2][..]));
}

#[tokio::test]
async fn cors_allows_pokoin_and_vercel_only() {
    for origin in [
        "https://pokoin.com",
        "https://www.pokoin.com",
        "https://cardscan.pokoin.com",
        "https://preview.vercel.app",
    ] {
        let response = app(state(temp_dir("cors")))
            .oneshot(
                Request::builder()
                    .method(Method::OPTIONS)
                    .uri("/identify")
                    .header(header::ORIGIN, origin)
                    .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN],
            origin
        );
    }
    let response = app(state(temp_dir("cors-evil")))
        .oneshot(
            Request::builder()
                .method(Method::OPTIONS)
                .uri("/identify")
                .header(header::ORIGIN, "https://evil.example")
                .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(response
        .headers()
        .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
        .is_none());
}
