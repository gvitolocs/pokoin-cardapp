//! Posts the same images to two `/identify` servers (A then B, interleaved, one at a time)
//! and reports top-1 agreement, score deltas and client-side latency percentiles.
//!
//! scan-compare --a URL --b URL --images LIST [--reps N] [--query 'top_k=3'] [--out FILE.json]
//! LIST has one image path per line. Uses curl so timings match the earlier httpbench runs.

use std::collections::HashMap;
use std::process::Command;

use serde_json::{json, Value};

struct Shot {
    ct: String,
    name: String,
    score: f64,
    ms: f64,
    detect_ms: f64,
    identify_ms: f64,
}

fn post(base: &str, query: &str, image: &str) -> Result<Shot, String> {
    let url = format!("{}/identify?{}", base.trim_end_matches('/'), query);
    let out = Command::new("curl")
        .args(["-s", "-m", "30", "-w", "\n%{time_total}", "-F", &format!("file=@{image}"), &url])
        .output()
        .map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&out.stdout);
    let (body, secs) = text.rsplit_once('\n').ok_or("no curl timing")?;
    let v: Value = serde_json::from_str(body).map_err(|e| format!("{url}: {e}: {body:.200}"))?;
    let top = &v["top1"];
    Ok(Shot {
        ct: top["ct_id"].as_str().unwrap_or("-").to_string(),
        name: top["name"].as_str().unwrap_or("").to_string(),
        score: top["score"].as_f64().unwrap_or(0.0),
        ms: secs.trim().parse::<f64>().map_err(|e| e.to_string())? * 1000.0,
        detect_ms: v["detect_ms"].as_f64().unwrap_or(0.0),
        identify_ms: v["identify_ms"].as_f64().unwrap_or(0.0),
    })
}

fn pct(xs: &mut [f64], p: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[((xs.len() - 1) as f64 * p).round() as usize]
}

fn stats(xs: &[f64]) -> Value {
    let mut v = xs.to_vec();
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    json!({"n": v.len(), "mean": mean.round(), "p50": pct(&mut v, 0.5).round(),
           "p90": pct(&mut v, 0.9).round(), "p99": pct(&mut v, 0.99).round(), "max": pct(&mut v, 1.0).round()})
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut flags: HashMap<&str, &str> = HashMap::new();
    for pair in args.chunks(2) {
        if let [k, v] = pair {
            flags.insert(k.trim_start_matches("--"), v);
        }
    }
    let (Some(a), Some(b), Some(list)) = (flags.get("a"), flags.get("b"), flags.get("images")) else {
        eprintln!("usage: scan-compare --a URL --b URL --images LIST [--reps N] [--query Q] [--out FILE]");
        std::process::exit(2);
    };
    let reps: usize = flags.get("reps").and_then(|r| r.parse().ok()).unwrap_or(1);
    let query = flags.get("query").copied().unwrap_or("top_k=3");
    let images: Vec<String> = std::fs::read_to_string(list)
        .expect("image list")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();

    // Warm-up, not measured.
    for base in [a, b] {
        let _ = post(base, query, &images[0]);
    }
    let (mut lat_a, mut lat_b, mut det_a, mut det_b, mut id_a, mut id_b) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    let mut rows = Vec::new();
    let (mut same, mut errors) = (0usize, 0usize);
    for rep in 0..reps {
        for img in &images {
            let (ra, rb) = (post(a, query, img), post(b, query, img));
            let (sa, sb) = match (ra, rb) {
                (Ok(x), Ok(y)) => (x, y),
                (ea, eb) => {
                    errors += 1;
                    eprintln!("{img}: a={:?} b={:?}", ea.err(), eb.err());
                    continue;
                }
            };
            lat_a.push(sa.ms);
            lat_b.push(sb.ms);
            det_a.push(sa.detect_ms);
            det_b.push(sb.detect_ms);
            id_a.push(sa.identify_ms);
            id_b.push(sb.identify_ms);
            if rep == 0 {
                let agree = sa.ct == sb.ct;
                same += usize::from(agree);
                rows.push(json!({"image": img, "agree": agree,
                    "a": {"ct_id": sa.ct, "name": sa.name, "score": sa.score},
                    "b": {"ct_id": sb.ct, "name": sb.name, "score": sb.score}}));
            }
        }
    }
    let diffs: Vec<&Value> = rows.iter().filter(|r| r["agree"] == false).collect();
    let report = json!({
        "a": a, "b": b, "images": images.len(), "reps": reps, "errors": errors,
        "top1_same": same, "top1_diff": diffs.len(), "diffs": diffs,
        "latency_ms": {"a": stats(&lat_a), "b": stats(&lat_b)},
        "detect_ms": {"a": stats(&det_a), "b": stats(&det_b)},
        "identify_ms": {"a": stats(&id_a), "b": stats(&id_b)},
        "rows": rows,
    });
    let text = serde_json::to_string_pretty(&report).unwrap();
    if let Some(out) = flags.get("out") {
        std::fs::write(out, &text).expect("write report");
    }
    let summary = json!({"top1_same": same, "top1_diff": report["top1_diff"], "errors": errors,
        "latency_ms": report["latency_ms"], "diffs": report["diffs"]});
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
}
