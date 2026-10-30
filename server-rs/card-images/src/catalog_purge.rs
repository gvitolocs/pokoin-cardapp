//! Immutable, hard-linked catalog exports with audited placeholder rows removed.
use super::*;

fn string<'a>(v: &'a Value, key: &str) -> Res<&'a str> {
    v[key]
        .as_str()
        .ok_or_else(|| format!("missing string {key}").into())
}

// IEEE 754 binary16, including subnormals, infinities and NaNs.
pub(super) fn f16(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 31;
    let fraction = (bits & 1023) as f32;
    sign * match exponent {
        0 => fraction * 2.0f32.powi(-24),
        31 if fraction == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + fraction / 1024.0) * 2.0f32.powi(exponent as i32 - 15),
    }
}

fn vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f16(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn clone_tree(source: &Path, out: &Path, root: &Path, skip: &HashSet<PathBuf>) -> Res<()> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        let relative = path.strip_prefix(root)?;
        if skip.contains(relative) {
            continue;
        }
        let dest = out.join(entry.file_name());
        let kind = entry.file_type()?;
        if kind.is_dir() {
            fs::create_dir_all(&dest)?;
            clone_tree(&path, &dest, root, skip)?;
        } else if kind.is_symlink() {
            #[cfg(unix)]
            std::os::unix::fs::symlink(fs::read_link(&path)?, dest)?;
            #[cfg(not(unix))]
            return Err("catalog symlinks require Unix".into());
        } else if kind.is_file() {
            if fs::hard_link(&path, &dest).is_err() {
                fs::copy(&path, &dest)?;
            }
        } else {
            return Err(format!("unsupported catalog entry: {}", path.display()).into());
        }
    }
    Ok(())
}

fn artifact(out: &Path, old: &str, kind: &str, ext: &str, bytes: &[u8]) -> Res<Value> {
    let sha = hash(bytes);
    let path = safe_key(old)?.with_file_name(format!("{kind}-{}.{ext}", &sha[..12]));
    let dest = out.join(&path);
    fs::create_dir_all(dest.parent().ok_or("missing artifact parent")?)?;
    // Exclusive creation also protects artifacts shared between catalogs.
    match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&dest)
    {
        Ok(mut f) => f.write_all(bytes)?,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            if fs::read(&dest)? != bytes {
                return Err("artifact hash-prefix collision".into());
            }
        }
        Err(e) => return Err(e.into()),
    }
    Ok(
        json!({"path": path.to_str().ok_or("non-UTF8 artifact path")?, "bytes": bytes.len(), "sha256": sha}),
    )
}

pub(super) fn run(flags: &Flags) -> Res<()> {
    let source = fs::canonicalize(flags.path("catalogs")?)?;
    let out = flags.path("out")?;
    let min_cos: f64 = flags.0.get("cos").map_or(Ok(0.995), |v| v.parse())?;
    if !min_cos.is_finite() || !(-1.0..=1.0).contains(&min_cos) {
        return Err("--cos must be finite and between -1 and 1".into());
    }
    if fs::symlink_metadata(&out).is_ok() {
        return Err("--out must not exist".into());
    }
    // Resolve the existing parent before creating anything, rejecting exports inside the source.
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let resolved_out = fs::canonicalize(parent)?.join(out.file_name().ok_or("invalid --out")?);
    if resolved_out.starts_with(&source) {
        return Err("--out must be outside --catalogs".into());
    }
    let audit = fs::read(flags.path("audit")?)?;
    let mut keys = HashSet::new();
    for line in audit
        .split(|b| *b == b'\n')
        .filter(|l| !l.iter().all(u8::is_ascii_whitespace))
    {
        let row: Value = serde_json::from_slice(line)?;
        keys.insert(string(&row, "key")?.to_owned());
    }
    let mut index = read_json(&source.join("index.json"))?;
    let catalogs = index["catalogs"]
        .as_array_mut()
        .ok_or("missing catalogs array")?;
    let only: Option<HashSet<&str>> = flags.0.get("only").map(|v| v.split(',').collect());
    if let Some(ids) = &only {
        for id in ids {
            if !catalogs.iter().any(|c| c["id"].as_str() == Some(id)) {
                return Err(format!("unknown catalog id: {id}").into());
            }
        }
    }
    fs::create_dir(&out)?;
    let mut skip = HashSet::from([
        PathBuf::from("index.json"),
        PathBuf::from("placeholder-purge-report.json"),
    ]);
    let mut obsolete = HashSet::new();
    let mut removed = serde_json::Map::new();
    let mut report = Vec::new();
    for catalog in catalogs.iter_mut() {
        let id = string(catalog, "id")?.to_owned();
        let count = catalog["count"].as_u64().ok_or("missing catalog count")? as usize;
        let mut details = Vec::new();
        let mut face_count = 0;
        let mut embedding_count = 0;
        if only.as_ref().is_none_or(|ids| ids.contains(id.as_str())) {
            let cards_path = string(&catalog["cards"], "path")?.to_owned();
            let embeddings_path = string(&catalog["embeddings"], "path")?.to_owned();
            let mut cards = read_json(&source.join(safe_key(&cards_path)?))?;
            let fields = cards["fields"].as_array().ok_or("missing fields")?.clone();
            let image_col = fields
                .iter()
                .position(|f| f == "image")
                .ok_or("missing image field")?;
            if !fields.iter().any(|f| f == "id") {
                return Err("missing id field".into());
            }
            let rows = cards["rows"].as_array_mut().ok_or("missing rows")?;
            if rows.len() != count
                || rows
                    .iter()
                    .any(|r| r.as_array().is_none_or(|r| r.len() != fields.len()))
            {
                return Err(format!("invalid rows in {id}").into());
            }
            let bytes = fs::read(source.join(safe_key(&embeddings_path)?))?;
            if (count == 0 && !bytes.is_empty())
                || (count > 0 && (bytes.is_empty() || bytes.len() % (count * 2) != 0))
            {
                return Err(format!("invalid embedding dimensions in {id}").into());
            }
            let stride = bytes.len().checked_div(count).unwrap_or(0);
            let faces: Vec<bool> = rows
                .iter()
                .map(|r| {
                    r[image_col]
                        .as_str()
                        .and_then(|s| s.strip_prefix("https://cdn.pokoin.com/"))
                        .is_some_and(|k| keys.contains(k))
                })
                .collect();
            // Identical audited embeddings need only one comparison; never seed from inferred rows.
            let seeds: HashSet<&[u8]> = faces
                .iter()
                .enumerate()
                .filter(|(_, face)| **face)
                .map(|(i, _)| &bytes[i * stride..(i + 1) * stride])
                .collect();
            let seeds: Vec<_> = seeds
                .into_iter()
                .map(|b| {
                    let v = vector(b);
                    let n = norm(&v);
                    (v, n)
                })
                .collect();
            let mut kept_rows = Vec::new();
            let mut kept_bytes = Vec::new();
            for (i, row) in rows.iter().enumerate() {
                let raw = &bytes[i * stride..(i + 1) * stride];
                let reason = if faces[i] {
                    Some("face")
                } else if !seeds.is_empty() {
                    let v = vector(raw);
                    let n = norm(&v);
                    seeds
                        .iter()
                        .any(|(seed, sn)| {
                            n > 0.0
                                && *sn > 0.0
                                && v.iter().zip(seed).map(|(a, b)| a * b).sum::<f32>() / (n * sn)
                                    >= min_cos as f32
                        })
                        .then_some("embedding")
                } else {
                    None
                };
                if let Some(reason) = reason {
                    if reason == "face" {
                        face_count += 1;
                    } else {
                        embedding_count += 1;
                    }
                    let mut detail = serde_json::Map::new();
                    for field in ["id", "name", "number", "set", "image"] {
                        let value = fields
                            .iter()
                            .position(|f| f == field)
                            .map_or(Value::Null, |col| row[col].clone());
                        detail.insert(field.into(), value);
                    }
                    detail.insert("reason".into(), reason.into());
                    details.push(Value::Object(detail));
                } else {
                    kept_rows.push(row.clone());
                    kept_bytes.extend_from_slice(raw);
                }
            }
            removed.insert(
                id.clone(),
                Value::Array(details.iter().map(|r| r["id"].clone()).collect()),
            );
            if !details.is_empty() {
                catalog["count"] = kept_rows.len().into();
                cards["rows"] = kept_rows.into();
                for (kind, old, ext, data) in [
                    ("cards", cards_path, "json", serde_json::to_vec(&cards)?),
                    ("embeddings", embeddings_path, "f16", kept_bytes),
                ] {
                    let updated = artifact(&out, &old, kind, ext, &data)?;
                    skip.insert(PathBuf::from(string(&updated, "path")?));
                    obsolete.insert(PathBuf::from(old));
                    for key in ["path", "bytes", "sha256"] {
                        catalog[kind][key] = updated[key].clone();
                    }
                }
            }
        } else {
            removed.insert(id.clone(), json!([]));
        }
        report.push(json!({"id": id, "rowsIn": count, "rowsOut": count - details.len(), "removedByReason": {"face": face_count, "embedding": embedding_count}, "removed": details}));
    }
    // Preserve old files if an untouched catalog still references them.
    for catalog in catalogs.iter() {
        for kind in ["cards", "embeddings"] {
            obsolete.remove(Path::new(string(&catalog[kind], "path")?));
        }
    }
    skip.extend(obsolete);
    clone_tree(&source, &out, &source, &skip)?;
    let revision = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".into());
    index["placeholderPurgeExport"] = json!({"tool": "card-images catalog-purge", "scannerRevision": revision.trim(), "sourceRevision": source.file_name().unwrap_or_default().to_string_lossy(), "auditSha256": hash(&audit), "minCos": min_cos, "removed": removed});
    write_json(&out.join("index.json"), &index)?;
    write_json(&out.join("placeholder-purge-report.json"), &report)?;
    for row in &report {
        println!(
            "{}: {} -> {} (face {}, embedding {})",
            row["id"].as_str().unwrap_or_default(),
            row["rowsIn"],
            row["rowsOut"],
            row["removedByReason"]["face"],
            row["removedByReason"]["embedding"]
        );
    }
    Ok(())
}
