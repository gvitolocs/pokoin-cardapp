use std::net::IpAddr;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    pub bind: IpAddr,
    pub port: u16,
    pub threads: usize,
    pub web: PathBuf,
    pub index: PathBuf,
    pub milo: PathBuf,
    pub yolo: PathBuf,
}

fn path_var(key: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| default.to_string()))
}

impl Config {
    pub fn from_env() -> Self {
        let bind = std::env::var("CARDSCAN_BIND")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(IpAddr::from([127, 0, 0, 1]));
        let port = std::env::var("CARDSCAN_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8096);
        let threads = std::env::var("CARDSCAN_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4)
            .max(1);
        Self {
            bind,
            port,
            threads,
            web: path_var("CARDSCAN_WEB", "web"),
            index: path_var(
                "CDN_MILO",
                "/home/nez/Projects/Startup/Pokoin/PokoinTest/index/cdn_milo",
            ),
            milo: path_var(
                "MILO_ONNX",
                "/home/nez/Projects/Startup/Pokoin/BattleScan/clones/CollectorVision/collector_vision/weights/milo.onnx",
            ),
            yolo: path_var(
                "YOLO_ONNX",
                "/home/nez/Projects/Startup/Pokoin/BattleScan/clones/TCG/backend/models/card_detector.onnx",
            ),
        }
    }
}
