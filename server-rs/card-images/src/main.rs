//! Audit CDN face images and recover verified replacements. See ../README.md.
use card_placeholder::{Detector, Verdict};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};

mod catalog_purge;
mod recover;
#[cfg(test)]
mod tests;

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
const EXTS: &[&str] = &["jpg", "jpeg", "png", "webp"];
const SKIP: &[&str] = &[
    "previews",
    "figure-masks",
    "figure-masks-clean",
    "tcgdex",
    "expansions",
    "artist-profiles",
    "poko-video",
];

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let result = (|| {
        let flags = Flags::parse(&args[1.min(args.len())..])?;
        match args.first().map(String::as_str) {
            Some("audit") => audit(&flags),
            Some("recover") => recover::run(&flags),
            Some("catalog-purge") => catalog_purge::run(&flags),
            _ => Err("usage: card-images catalog-purge --catalogs DIR --audit FILE --out DIR [--cos 0.995] [--only id,id,...]\n       card-images audit --objects DIR --out FILE [--threads N]\n       card-images recover --audit FILE --objects DIR --blueprints DIR --expansions FILE --tcgdex DIR --pokemontcg DIR --scryfall FILE --out DIR [--download] [--apply --backup DIR]".into()),
        }
    })();
    if let Err(e) = result {
        eprintln!("card-images: {e}");
        std::process::exit(1);
    }
}

struct Flags(HashMap<String, String>);
impl Flags {
    fn parse(args: &[String]) -> Res<Self> {
        let mut map = HashMap::new();
        let mut i = 0;
        while i < args.len() {
            let key = args[i].strip_prefix("--").ok_or("expected --flag")?;
            if matches!(key, "download" | "apply") {
                map.insert(key.into(), String::new());
                i += 1;
            } else {
                let value = args
                    .get(i + 1)
                    .filter(|s| !s.starts_with("--"))
                    .ok_or_else(|| format!("missing value for --{key}"))?;
                map.insert(key.into(), value.clone());
                i += 2;
            }
        }
        Ok(Self(map))
    }
    fn path(&self, key: &str) -> Res<PathBuf> {
        self.0
            .get(key)
            .map(PathBuf::from)
            .ok_or_else(|| format!("missing --{key}").into())
    }
    fn has(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read_json(path: &Path) -> Res<Value> {
    Ok(serde_json::from_reader(BufReader::new(fs::File::open(
        path,
    )?))?)
}
fn write_json(path: &Path, value: &impl Serialize) -> Res<()> {
    let mut out = BufWriter::new(fs::File::create(path)?);
    serde_json::to_writer_pretty(&mut out, value)?;
    out.flush()?;
    Ok(())
}
fn jsonl<T: Serialize>(out: &mut impl Write, value: &T) -> Res<()> {
    serde_json::to_writer(&mut *out, value)?;
    writeln!(out)?;
    Ok(())
}
fn lines(path: &Path, mut f: impl FnMut(Value) -> Res<()>) -> Res<()> {
    for line in BufReader::new(fs::File::open(path)?).lines() {
        let line = line?;
        if !line.trim().is_empty() {
            f(serde_json::from_str(&line)?)?;
        }
    }
    Ok(())
}
fn ct_id(path: &Path) -> Option<u64> {
    path.file_name()?.to_str()?.split_once('_')?.0.parse().ok()
}
fn face(path: &Path) -> bool {
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_ascii_lowercase();
    !name.ends_with("_homepage.webp")
        && path
            .extension()
            .is_some_and(|e| EXTS.contains(&e.to_string_lossy().to_ascii_lowercase().as_str()))
}
fn safe_key(key: &str) -> Res<&Path> {
    let p = Path::new(key);
    if p.as_os_str().is_empty() || p.components().any(|c| !matches!(c, Component::Normal(_))) {
        return Err(format!("unsafe object key: {key}").into());
    }
    Ok(p)
}
fn walk(dir: &Path, paths: &mut Vec<PathBuf>) -> Res<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            if !SKIP.contains(&entry.file_name().to_string_lossy().as_ref()) {
                walk(&entry.path(), paths)?;
            }
        } else if kind.is_file() && face(&entry.path()) {
            paths.push(entry.path());
        }
    }
    Ok(())
}
fn parallel<T: Sync, U: Send>(
    items: &[T],
    n: usize,
    f: impl Fn(&T) -> Res<U> + Sync,
) -> Res<Vec<U>> {
    std::thread::scope(|s| {
        let f = &f;
        let handles: Vec<_> = items
            .chunks(items.len().div_ceil(n).max(1))
            .map(|chunk| s.spawn(move || chunk.iter().map(f).collect::<Res<Vec<_>>>()))
            .collect();
        let mut out = Vec::new();
        for h in handles {
            out.extend(h.join().map_err(|_| "worker panicked")??);
        }
        Ok(out)
    })
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct AuditRow {
    key: String,
    game: String,
    ct_id: Option<u64>,
    reason: String,
    reference: String,
    sha256: String,
    w: u32,
    h: u32,
}
#[derive(Default, Serialize)]
struct AuditCount {
    files: usize,
    placeholders: usize,
    reasons: BTreeMap<String, usize>,
}
fn audit(flags: &Flags) -> Res<()> {
    let objects = flags.path("objects")?;
    let out = flags.path("out")?;
    let n = flags
        .0
        .get("threads")
        .map_or(Ok(4), |s| s.parse::<usize>())?;
    if n == 0 {
        return Err("--threads must be positive".into());
    }
    let mut paths = Vec::new();
    walk(&objects, &mut paths)?;
    paths.sort();
    let detector = Detector::builtin();
    let mut rows = parallel(&paths, n, |p| {
        let key = p.strip_prefix(&objects)?.to_string_lossy().into_owned();
        let game = if key.contains('/') {
            key.split('/').next().unwrap().to_string()
        } else {
            "pokemon".into()
        };
        let dims = image::ImageReader::open(p)?
            .with_guessed_format()?
            .into_dimensions()
            .ok();
        let (w, h) = dims.unwrap_or((0, 0));
        let bytes = fs::read(p)?;
        let verdict = if (w, h) == (186, 260) {
            Verdict::Placeholder {
                reason: "cardtrader_size",
                reference: "cardtrader-preview".into(),
            }
        } else if !key.contains('/') {
            detector.classify(&bytes)
        } else {
            Verdict::Real
        };
        let (reason, reference) = match verdict {
            Verdict::Placeholder { reason, reference } => (reason.into(), reference),
            _ => (String::new(), String::new()),
        };
        Ok(AuditRow {
            key,
            game,
            ct_id: ct_id(p),
            reason,
            reference,
            sha256: hash(&bytes),
            w,
            h,
        })
    })?;
    let mut groups: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    for (i, r) in rows.iter().enumerate().filter(|(_, r)| r.key.contains('/')) {
        groups
            .entry((r.game.clone(), r.sha256.clone()))
            .or_default()
            .push(i);
    }
    let groups: Vec<_> = groups
        .into_values()
        .filter(|g| {
            g.iter()
                .filter_map(|&i| rows[i].ct_id)
                .collect::<HashSet<_>>()
                .len()
                >= 3
        })
        .collect();
    let verdicts = parallel(&groups, n, |g| {
        Ok(detector.classify(&fs::read(objects.join(&rows[g[0]].key))?))
    })?;
    let mut suspects = Vec::new();
    for (g, verdict) in groups.iter().zip(verdicts) {
        if let Verdict::Placeholder { reference, .. } = verdict {
            for &i in g {
                rows[i].reason = format!("shared_bytes:{reference}");
                rows[i].reference = reference.clone();
            }
        } else if suspects.len() < 50 {
            suspects.push(json!({"game":rows[g[0]].game,"size":g.len(),"sha256":rows[g[0]].sha256,"sample_keys":g.iter().take(5).map(|&i| &rows[i].key).collect::<Vec<_>>() }));
        }
    }
    let mut counts: BTreeMap<String, AuditCount> = BTreeMap::new();
    let mut file = BufWriter::new(fs::File::create(&out)?);
    for row in &rows {
        let c = counts.entry(row.game.clone()).or_default();
        c.files += 1;
        if !row.reason.is_empty() {
            c.placeholders += 1;
            *c.reasons.entry(row.reason.clone()).or_default() += 1;
            jsonl(&mut file, row)?;
        }
    }
    file.flush()?;
    let report = json!({"games": counts, "suspect_groups": suspects});
    write_json(
        &PathBuf::from(format!("{}.report.json", out.display())),
        &report,
    )?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
