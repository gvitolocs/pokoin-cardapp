//! Keeps the nezopt scan server's Milo index (`cdn_milo`) in step with the catalogs.
//!
//! `plan`   reads the current index, the on-device catalogs and CardTrader's daily blueprint
//!          export, and stages the faces of Pokemon singles the index lacks as symlinks named
//!          `{ct_id}_{slug}.{ext}` (the layout `embed_cdn_milo.py --cdn-root` reads), with an
//!          allowlist, a report and `cardmeta.jsonl` (name, number, set, catalogs per ct_id).
//! `enrich` rewrites an index's `metadata.jsonl` with those proper names, numbers, sets and
//!          catalogs. Row order, ids and filenames stay; the old filename name moves to `slug`.
//! `scan-objects` lists every face in the CDN objects tree that is a placeholder picture.
//! `purge`  writes a copy of an index without placeholder rows: rows whose face is a placeholder,
//!          and rows whose embedding is within `--cos` of a placeholder face's embedding.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

mod npy;

use card_placeholder::{Detector, Verdict};

const POKEMON_GAME: u64 = 5;
const SINGLES_CATEGORY: u64 = 73;
const LANG_CATALOGS: [&str; 3] = ["pokemon_western", "pokemon_japanese", "pokemon_chinese"];
const IMAGE_EXTS: [&str; 4] = ["jpg", "jpeg", "png", "webp"];
// CardTrader serves its card back (186x260, 7920 bytes) when a blueprint has no picture.
const CT_BACK_BYTES: u64 = 7920;
/// Cosine to a placeholder face's embedding above which an index row is that placeholder too.
const PLACEHOLDER_COS: f32 = 0.995;

type Res<T> = Result<T, String>;

#[derive(Debug, Default, Clone, Serialize)]
struct CardMeta {
    ct_id: u64,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    set: Option<String>,
    catalogs: BTreeSet<String>,
    /// Basename of the catalog's cdn.pokoin.com image, when a catalog lists the card.
    #[serde(skip)]
    image: Option<String>,
    source: &'static str,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("plan") => plan(&Flags::parse(&args[1..])),
        Some("enrich") => enrich(&Flags::parse(&args[1..])),
        Some("scan-objects") => scan_objects(&Flags::parse(&args[1..])),
        Some("purge") => purge(&Flags::parse(&args[1..])),
        _ => Err("usage: index-refresh plan --index DIR --catalogs DIR --blueprints DIR \
                  --objects DIR --placeholders FILE --out DIR\n       \
                  index-refresh enrich --index DIR --cardmeta FILE\n       \
                  index-refresh scan-objects --objects DIR --out FILE [--threads N]\n       \
                  index-refresh purge --index DIR --objects DIR --out DIR [--cos F] [--threads N]"
            .into()),
    };
    if let Err(e) = result {
        eprintln!("index-refresh: {e}");
        std::process::exit(1);
    }
}

struct Flags(HashMap<String, String>);

impl Flags {
    fn parse(args: &[String]) -> Self {
        let mut map = HashMap::new();
        for pair in args.chunks(2) {
            if let [k, v] = pair {
                map.insert(k.trim_start_matches("--").to_string(), v.clone());
            }
        }
        Flags(map)
    }
    fn path(&self, key: &str) -> Res<PathBuf> {
        self.0.get(key).map(PathBuf::from).ok_or_else(|| format!("missing --{key}"))
    }
    fn num<T: std::str::FromStr>(&self, key: &str, default: T) -> Res<T> {
        self.0.get(key).map_or(Ok(default), |v| v.parse().map_err(|_| format!("bad --{key} {v}")))
    }
}

fn read_json(path: &Path) -> Res<Value> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn str_of(v: &Value) -> Option<String> {
    match v {
        Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// ct_ids already in the index, in row order.
fn index_ct_ids(index: &Path) -> Res<Vec<u64>> {
    let file = fs::File::open(index.join("metadata.jsonl")).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        let ct = row["ct_id"].as_str().and_then(|s| s.parse().ok()).ok_or("row without ct_id")?;
        out.push(ct);
    }
    Ok(out)
}

/// Language catalogs from `current/index.json`: public id = 2 * ct_id.
fn load_catalogs(root: &Path, meta: &mut BTreeMap<u64, CardMeta>) -> Res<BTreeMap<String, usize>> {
    let index = read_json(&root.join("index.json"))?;
    let mut counts = BTreeMap::new();
    for cat in index["catalogs"].as_array().ok_or("index.json without catalogs")? {
        let id = cat["id"].as_str().unwrap_or_default();
        if !LANG_CATALOGS.contains(&id) {
            continue;
        }
        let cards = read_json(&root.join(cat["cards"]["path"].as_str().ok_or("catalog without cards.path")?))?;
        let fields: Vec<&str> =
            cards["fields"].as_array().ok_or("cards without fields")?.iter().filter_map(Value::as_str).collect();
        let col = |name: &str| fields.iter().position(|f| *f == name).ok_or(format!("{id}: no field {name}"));
        let (c_id, c_name, c_num, c_set, c_img) = (col("id")?, col("name")?, col("number")?, col("set")?, col("image")?);
        let rows = cards["rows"].as_array().ok_or("cards without rows")?;
        for row in rows {
            let public: u64 = str_of(&row[c_id]).and_then(|s| s.parse().ok()).ok_or(format!("{id}: bad id"))?;
            if !public.is_multiple_of(2) {
                return Err(format!("{id}: public id {public} is odd"));
            }
            let entry = meta.entry(public / 2).or_insert_with(|| CardMeta {
                ct_id: public / 2,
                name: str_of(&row[c_name]).unwrap_or_default(),
                n: str_of(&row[c_num]),
                set: str_of(&row[c_set]),
                image: str_of(&row[c_img]).and_then(|u| u.rsplit('/').next().map(str::to_string)),
                source: "catalog",
                ..Default::default()
            });
            entry.catalogs.insert(id.to_string());
        }
        counts.insert(id.to_string(), rows.len());
    }
    Ok(counts)
}

struct Blueprint {
    ct_id: u64,
    name: String,
    number: Option<String>,
    expansion: u64,
    languages: Vec<String>,
}

fn load_blueprints(dir: &Path) -> Res<(Vec<Blueprint>, HashMap<u64, String>)> {
    let expansions: HashMap<u64, String> = read_json(&dir.join("_expansions-all.json"))?
        .as_array()
        .ok_or("expansions not a list")?
        .iter()
        .filter_map(|e| Some((e["id"].as_u64()?, str_of(&e["name"])?)))
        .collect();
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        if !name.starts_with("_blueprints_export-") || !name.ends_with(".json") {
            continue;
        }
        for b in read_json(&path)?.as_array().into_iter().flatten() {
            if b["game_id"].as_u64() != Some(POKEMON_GAME) || b["category_id"].as_u64() != Some(SINGLES_CATEGORY) {
                continue;
            }
            let languages = b["editable_properties"]
                .as_array()
                .into_iter()
                .flatten()
                .find(|p| p["name"] == "pokemon_language")
                .and_then(|p| p["possible_values"].as_array())
                .map(|vs| vs.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            out.push(Blueprint {
                ct_id: b["id"].as_u64().ok_or("blueprint without id")?,
                name: str_of(&b["name"]).unwrap_or_default(),
                number: str_of(&b["fixed_properties"]["collector_number"]).or_else(|| str_of(&b["version"])),
                expansion: b["expansion_id"].as_u64().unwrap_or(0),
                languages,
            });
        }
    }
    Ok((out, expansions))
}

/// Catalog for a card no catalog lists: the catalog most of its expansion's cards are in,
/// else CardTrader's language list (English printings are western).
fn route(b: &Blueprint, votes: &HashMap<u64, BTreeMap<String, usize>>) -> Option<String> {
    if let Some(best) = votes.get(&b.expansion).and_then(|v| v.iter().max_by_key(|(_, n)| **n)) {
        return Some(best.0.clone());
    }
    let langs: Vec<&str> = b.languages.iter().map(String::as_str).collect();
    if langs.contains(&"en") {
        Some("pokemon_western".into())
    } else if langs.first() == Some(&"jp") {
        Some("pokemon_japanese".into())
    } else if matches!(langs.first(), Some(&"zh-CN") | Some(&"zh-TW")) {
        Some("pokemon_chinese".into())
    } else {
        None
    }
}

/// ct_id -> face files in the leftovers store (no `_homepage` thumbnails).
fn list_objects(dir: &Path) -> Res<HashMap<u64, Vec<String>>> {
    let mut map: HashMap<u64, Vec<String>> = HashMap::new();
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let name = entry.map_err(|e| e.to_string())?.file_name().to_string_lossy().into_owned();
        let lower = name.to_ascii_lowercase();
        let ext_ok = lower.rsplit_once('.').is_some_and(|(_, ext)| IMAGE_EXTS.contains(&ext));
        if !ext_ok || lower.contains("_homepage") {
            continue;
        }
        if let Some(ct) = name.split_once('_').and_then(|(p, _)| p.parse::<u64>().ok()) {
            map.entry(ct).or_default().push(name);
        }
    }
    for files in map.values_mut() {
        // Prefer JPEG, then PNG, then WebP, then name order.
        files.sort_by_key(|f| {
            let ext = f.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()).unwrap_or_default();
            (IMAGE_EXTS.iter().position(|e| *e == ext).unwrap_or(9), f.clone())
        });
    }
    Ok(map)
}

fn placeholder_ids(path: &Path) -> Res<BTreeSet<u64>> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text.split(|c: char| !c.is_ascii_digit()).filter_map(|s| s.parse().ok()).collect())
}

#[derive(Serialize)]
struct Skip {
    ct_id: u64,
    reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    file: Option<String>,
}

fn plan(flags: &Flags) -> Res<()> {
    let (index, catalogs, blueprints, objects, out) = (
        flags.path("index")?,
        flags.path("catalogs")?,
        flags.path("blueprints")?,
        flags.path("objects")?,
        flags.path("out")?,
    );
    let placeholders = placeholder_ids(&flags.path("placeholders")?)?;
    let indexed: BTreeSet<u64> = index_ct_ids(&index)?.into_iter().collect();
    let mut meta = BTreeMap::new();
    let catalog_counts = load_catalogs(&catalogs, &mut meta)?;
    let (bps, expansions) = load_blueprints(&blueprints)?;

    let mut votes: HashMap<u64, BTreeMap<String, usize>> = HashMap::new();
    for b in &bps {
        if let Some(m) = meta.get(&b.ct_id) {
            for c in &m.catalogs {
                *votes.entry(b.expansion).or_default().entry(c.clone()).or_default() += 1;
            }
        }
    }
    let mut unrouted = 0usize;
    for b in &bps {
        let set = expansions.get(&b.expansion).cloned();
        match meta.get_mut(&b.ct_id) {
            Some(m) => {
                m.n = m.n.take().or_else(|| b.number.clone());
                m.set = m.set.take().or(set);
            }
            None => {
                let catalogs: BTreeSet<String> = route(b, &votes).into_iter().collect();
                unrouted += usize::from(catalogs.is_empty());
                meta.insert(
                    b.ct_id,
                    CardMeta {
                        ct_id: b.ct_id,
                        name: b.name.clone(),
                        n: b.number.clone(),
                        set,
                        catalogs,
                        source: "cardtrader",
                        ..Default::default()
                    },
                );
            }
        }
    }

    let faces = list_objects(&objects)?;
    let detector = Detector::builtin();
    let staged = out.join("faces");
    fs::create_dir_all(&staged).map_err(|e| e.to_string())?;
    let mut picked: Vec<(u64, String, String)> = Vec::new(); // ct, file, sha256
    let mut skips = Vec::new();
    for m in meta.values().filter(|m| !indexed.contains(&m.ct_id)) {
        if placeholders.contains(&m.ct_id) {
            skips.push(Skip { ct_id: m.ct_id, reason: "pokoin_placeholder_list", file: None });
            continue;
        }
        let candidates = faces.get(&m.ct_id).cloned().unwrap_or_default();
        let file = m
            .image
            .as_ref()
            .filter(|img| candidates.contains(img))
            .cloned()
            .or_else(|| candidates.first().cloned());
        let Some(file) = file else {
            skips.push(Skip { ct_id: m.ct_id, reason: "no_image", file: None });
            continue;
        };
        let path = objects.join(&file);
        let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let dims = image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()
            .ok()
            .and_then(|r| r.into_dimensions().ok());
        let reason = match dims {
            None => Some("undecodable"),
            Some(_) if bytes.len() as u64 == CT_BACK_BYTES => Some("cardtrader_back"),
            Some((w, h)) if w < 120 || h < 120 => Some("too_small"),
            _ => match detector.classify(&bytes) {
                Verdict::Real => None,
                Verdict::Undecodable => Some("undecodable"),
                Verdict::Placeholder { .. } => Some("placeholder_face"),
            },
        };
        if let Some(reason) = reason {
            skips.push(Skip { ct_id: m.ct_id, reason, file: Some(file) });
            continue;
        }
        picked.push((m.ct_id, file, format!("{:x}", Sha256::digest(&bytes))));
    }

    // The same bytes under several ct_ids is a stamped "missing card" picture, not a face.
    let mut by_hash: HashMap<&str, usize> = HashMap::new();
    for (_, _, h) in &picked {
        *by_hash.entry(h.as_str()).or_default() += 1;
    }
    let dup: BTreeSet<String> = by_hash.iter().filter(|(_, n)| **n > 1).map(|(h, _)| h.to_string()).collect();
    let mut allow = Vec::new();
    for (ct, file, hash) in &picked {
        if dup.contains(hash) {
            skips.push(Skip { ct_id: *ct, reason: "duplicate_face", file: Some(file.clone()) });
            continue;
        }
        let link = staged.join(file);
        if !link.exists() {
            std::os::unix::fs::symlink(objects.join(file), &link).map_err(|e| e.to_string())?;
        }
        allow.push(*ct);
    }

    let mut w = BufWriter::new(fs::File::create(out.join("allow.txt")).map_err(|e| e.to_string())?);
    for ct in &allow {
        writeln!(w, "{ct}").map_err(|e| e.to_string())?;
    }
    let mut w = BufWriter::new(fs::File::create(out.join("cardmeta.jsonl")).map_err(|e| e.to_string())?);
    for m in meta.values() {
        writeln!(w, "{}", serde_json::to_string(m).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;

    let mut missing_by: BTreeMap<String, usize> = BTreeMap::new();
    let mut staged_by: BTreeMap<String, usize> = BTreeMap::new();
    let allow_set: BTreeSet<u64> = allow.iter().copied().collect();
    for m in meta.values().filter(|m| !indexed.contains(&m.ct_id)) {
        let key = format!("{}:{}", m.source, m.catalogs.iter().cloned().collect::<Vec<_>>().join("+"));
        *missing_by.entry(key.clone()).or_default() += 1;
        if allow_set.contains(&m.ct_id) {
            *staged_by.entry(key).or_default() += 1;
        }
    }
    let mut skip_by: BTreeMap<&str, usize> = BTreeMap::new();
    for s in &skips {
        *skip_by.entry(s.reason).or_default() += 1;
    }
    let report = json!({
        "index_rows": indexed.len(),
        "catalog_rows": catalog_counts,
        "cardtrader_singles": bps.len(),
        "known_cards": meta.len(),
        "unrouted_cardtrader_only": unrouted,
        "missing_by_source_catalog": missing_by,
        "staged_by_source_catalog": staged_by,
        "staged": allow.len(),
        "skipped_by_reason": skip_by,
        "skipped": skips,
    });
    fs::write(out.join("report.json"), serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    println!(
        "index={} known={} missing={} staged={} skipped={:?}",
        indexed.len(),
        meta.len(),
        missing_by.values().sum::<usize>(),
        allow.len(),
        skip_by
    );
    Ok(())
}

fn enrich(flags: &Flags) -> Res<()> {
    let index = flags.path("index")?;
    let mut meta: HashMap<String, Value> = HashMap::new();
    let file = fs::File::open(flags.path("cardmeta")?).map_err(|e| e.to_string())?;
    for line in BufReader::new(file).lines() {
        let v: Value = serde_json::from_str(&line.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        meta.insert(v["ct_id"].to_string(), v);
    }
    let src = index.join("metadata.jsonl");
    let tmp = index.join("metadata.jsonl.tmp");
    let text = fs::read_to_string(&src).map_err(|e| e.to_string())?;
    let mut w = BufWriter::new(fs::File::create(&tmp).map_err(|e| e.to_string())?);
    let (mut rows, mut named) = (0usize, 0usize);
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let mut row: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let ct = row["ct_id"].as_str().ok_or("row without ct_id")?.to_string();
        let obj = row.as_object_mut().ok_or("row is not an object")?;
        if !obj.contains_key("slug") {
            let slug = obj.get("name").cloned().unwrap_or(Value::Null);
            obj.insert("slug".into(), slug);
        }
        match meta.get(&ct) {
            Some(m) if !m["name"].as_str().unwrap_or_default().is_empty() => {
                obj.insert("name".into(), m["name"].clone());
                for key in ["n", "set"] {
                    if let Some(v) = m.get(key) {
                        obj.insert(key.into(), v.clone());
                    }
                }
                obj.insert("catalogs".into(), m["catalogs"].clone());
                named += 1;
            }
            _ => {
                obj.entry("catalogs").or_insert(json!([]));
            }
        }
        writeln!(w, "{}", serde_json::to_string(&row).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        rows += 1;
    }
    w.flush().map_err(|e| e.to_string())?;
    drop(w);
    fs::rename(&tmp, &src).map_err(|e| e.to_string())?;
    println!("rows={rows} named={named} slug_only={}", rows - named);
    Ok(())
}

/// Classifies `items` on `threads` workers, keeping input order.
fn classify_files(detector: &Detector, paths: &[PathBuf], threads: usize) -> Vec<Verdict> {
    let chunk = paths.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = paths
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    part.iter()
                        .map(|p| fs::read(p).map_or(Verdict::Undecodable, |b| detector.classify(&b)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("classifier thread")).collect()
    })
}

#[derive(Serialize)]
struct Flagged {
    #[serde(skip_serializing_if = "Option::is_none")]
    ct_id: Option<u64>,
    file: String,
    reason: String,
    reference: String,
}

fn scan_objects(flags: &Flags) -> Res<()> {
    let (objects, out) = (flags.path("objects")?, flags.path("out")?);
    let threads = flags.num("threads", 4usize)?;
    let mut names: Vec<String> = fs::read_dir(&objects)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|n| n.rsplit_once('.').is_some_and(|(_, ext)| IMAGE_EXTS.contains(&ext.to_ascii_lowercase().as_str())))
        .collect();
    names.sort();
    let paths: Vec<PathBuf> = names.iter().map(|n| objects.join(n)).collect();
    let verdicts = classify_files(&Detector::builtin(), &paths, threads);
    let mut w = BufWriter::new(fs::File::create(&out).map_err(|e| e.to_string())?);
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (name, v) in names.iter().zip(&verdicts) {
        let (reason, reference) = match v {
            Verdict::Real => continue,
            Verdict::Undecodable => ("undecodable".to_string(), String::new()),
            Verdict::Placeholder { reason, reference } => (reason.to_string(), reference.clone()),
        };
        *counts.entry(format!("{reason}:{reference}")).or_default() += 1;
        let ct_id = name.split_once('_').and_then(|(p, _)| p.parse().ok());
        let row = Flagged { ct_id, file: name.clone(), reason, reference };
        writeln!(w, "{}", serde_json::to_string(&row).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    println!("files={} flagged={} {:?}", names.len(), counts.values().sum::<usize>(), counts);
    Ok(())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn purge(flags: &Flags) -> Res<()> {
    let (index, objects, out) = (flags.path("index")?, flags.path("objects")?, flags.path("out")?);
    let threads = flags.num("threads", 4usize)?;
    let min_cos = flags.num("cos", PLACEHOLDER_COS)?;
    if out.exists() {
        return Err(format!("{} already exists", out.display()));
    }
    let lines: Vec<String> = fs::read_to_string(index.join("metadata.jsonl"))
        .map_err(|e| e.to_string())?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(str::to_string)
        .collect();
    let emb = npy::read(&index.join("embeddings.npy"))?;
    if emb.rows != lines.len() {
        return Err(format!("{} embeddings for {} metadata rows", emb.rows, lines.len()));
    }
    let rows: Vec<Value> = lines.iter().map(|l| serde_json::from_str(l).map_err(|e| e.to_string())).collect::<Res<_>>()?;
    let paths: Vec<PathBuf> =
        rows.iter().map(|r| objects.join(r["filename"].as_str().unwrap_or_default())).collect();
    let verdicts = classify_files(&Detector::builtin(), &paths, threads);

    // One embedding per distinct placeholder picture: rows within `min_cos` of a kept one add nothing.
    let mut refs: Vec<usize> = Vec::new();
    for (i, v) in verdicts.iter().enumerate() {
        if v.is_placeholder() && !refs.iter().any(|&j| cosine(emb.row(i), emb.row(j)) >= min_cos) {
            refs.push(i);
        }
    }
    let (mut removed, mut undecodable) = (Vec::new(), Vec::new());
    let mut keep = Vec::with_capacity(rows.len());
    for (i, v) in verdicts.iter().enumerate() {
        let near = refs.iter().map(|&j| (j, cosine(emb.row(i), emb.row(j)))).find(|(_, c)| *c >= min_cos);
        let reason = match (v, near) {
            (Verdict::Placeholder { reference, .. }, _) => Some(format!("face:{reference}")),
            (_, Some((j, c))) => Some(format!("embedding:cos={c:.4}:like={}", rows[j]["ct_id"].as_str().unwrap_or("?"))),
            // AVIF and other formats this build cannot read: the embedder could, so the row stays.
            (Verdict::Undecodable, None) => {
                undecodable.push(rows[i]["filename"].clone());
                None
            }
            (Verdict::Real, None) => None,
        };
        match reason {
            Some(reason) => removed.push(json!({
                "ct_id": rows[i]["ct_id"], "filename": rows[i]["filename"], "reason": reason,
            })),
            None => keep.push(i),
        }
    }

    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut data = Vec::with_capacity(keep.len() * emb.cols);
    let mut meta = BufWriter::new(fs::File::create(out.join("metadata.jsonl")).map_err(|e| e.to_string())?);
    for &i in &keep {
        data.extend_from_slice(emb.row(i));
        writeln!(meta, "{}", lines[i]).map_err(|e| e.to_string())?;
    }
    meta.flush().map_err(|e| e.to_string())?;
    npy::write(&out.join("embeddings.npy"), &npy::Matrix { rows: keep.len(), cols: emb.cols, data })?;
    let mut by_reason: BTreeMap<String, usize> = BTreeMap::new();
    for r in &removed {
        let key = r["reason"].as_str().unwrap_or_default().split(":cos").next().unwrap_or_default().to_string();
        *by_reason.entry(key).or_default() += 1;
    }
    let progress = json!({
        "mode": "purge", "base": index.display().to_string(), "embedded": keep.len(),
        "removed": removed.len(), "min_cos": min_cos,
    });
    fs::write(out.join("progress.json"), serde_json::to_string_pretty(&progress).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let report = json!({ "rows_in": rows.len(), "rows_out": keep.len(), "removed_by_reason": by_reason,
                         "placeholder_vectors": refs.len(), "kept_undecodable": undecodable,
                         "removed": removed });
    fs::write(out.join("purge-report.json"), serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    println!("rows_in={} rows_out={} removed={} {:?}", rows.len(), keep.len(), removed.len(), by_reason);
    Ok(())
}
