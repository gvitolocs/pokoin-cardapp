//! Keep catalog positions stable while marking and recovering missing pictures.
use super::*;
use catalog_purge::{artifact, clone_tree, norm, string, vector};

fn collect_pins(dir: &Path, pins: &mut HashSet<String>) -> Res<()> {
    if !dir.exists() {
        return Ok(());
    }
    for e in fs::read_dir(dir)? {
        let e = e?;
        if e.file_type()?.is_dir() {
            collect_pins(&e.path(), pins)?;
        } else if e.file_name().to_string_lossy().contains("manifest")
            && e.path().extension().is_some_and(|s| s == "json")
        {
            fn visit(v: &Value, pins: &mut HashSet<String>) {
                match v {
                    Value::Object(m) => {
                        if let Some(s) = m.get("catalog_sha256").and_then(Value::as_str) {
                            pins.insert(s.into());
                        }
                        for v in m.values() {
                            visit(v, pins);
                        }
                    }
                    Value::Array(a) => {
                        for v in a {
                            visit(v, pins);
                        }
                    }
                    _ => {}
                }
            }
            visit(&read_json(&e.path())?, pins);
        }
    }
    Ok(())
}

pub(super) fn run(flags: &Flags) -> Res<()> {
    let source = fs::canonicalize(flags.path("catalogs")?)?;
    let out = flags.path("out")?;
    if fs::symlink_metadata(&out).is_ok() {
        return Err("--out must not exist".into());
    }
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if fs::canonicalize(parent)?
        .join(out.file_name().ok_or("invalid out")?)
        .starts_with(&source)
    {
        return Err("--out must be outside --catalogs".into());
    }
    let min_cos: f32 = flags.0.get("cos").map_or(Ok(0.995), |s| s.parse())?;
    if !min_cos.is_finite() || !(-1.0..=1.0).contains(&min_cos) {
        return Err("invalid --cos".into());
    }
    let audit = fs::read(flags.path("audit")?)?;
    let mut keys = HashSet::new();
    lines(&flags.path("audit")?, |r| {
        keys.insert(string(&r, "key")?.to_owned());
        Ok(())
    })?;
    let mut updates: HashMap<String, BTreeMap<String, Value>> = HashMap::new();
    if flags.has("embeddings") {
        lines(&flags.path("embeddings")?, |r| {
            let cat = string(&r, "catalog")?.to_owned();
            let id = string(&r, "id")?.to_owned();
            if !r["vector"].is_null() {
                let v = r["vector"].as_array().ok_or("invalid vector")?;
                if v.len() != 128
                    || v.iter().any(|x| {
                        x.as_f64()
                            .is_none_or(|f| !f.is_finite() || !half::f16::from_f64(f).is_finite())
                    })
                {
                    return Err("expected 128 finite f16 values".into());
                }
            } else if r["append"].is_null() {
                return Err("replacement requires vector".into());
            }
            if updates.entry(cat).or_default().insert(id, r).is_some() {
                return Err("duplicate embedding update".into());
            }
            Ok(())
        })?;
    }
    let mut index = read_json(&source.join("index.json"))?;
    let catalogs = index["catalogs"].as_array_mut().ok_or("missing catalogs")?;
    let only: Option<HashSet<_>> = flags.0.get("only").map(|s| s.split(',').collect());
    for id in updates
        .keys()
        .map(String::as_str)
        .chain(only.iter().flat_map(|s| s.iter().copied()))
    {
        if !catalogs.iter().any(|c| c["id"] == id) {
            return Err(format!("unknown catalog {id}").into());
        }
    }
    let mut pins = HashSet::new();
    collect_pins(&source.join("packs"), &mut pins)?;
    fs::create_dir(&out)?;
    let mut skip = HashSet::from([
        PathBuf::from("index.json"),
        PathBuf::from("placeholder-mark-report.json"),
    ]);
    let mut obsolete = HashSet::new();
    let mut reports = serde_json::Map::new();
    for cat in catalogs.iter_mut() {
        let id = string(cat, "id")?.to_owned();
        if pins.contains(string(&cat["cards"], "sha256")?) {
            reports.insert(id, json!({"pinned":true,"marked":0,"replaced":0,"appended":0,"noImage":cat["noImagePublicIDs"].as_array().map_or(0,Vec::len)}));
            continue;
        }
        if only.as_ref().is_some_and(|s| !s.contains(id.as_str())) {
            continue;
        }
        let cp = string(&cat["cards"], "path")?.to_owned();
        let ep = string(&cat["embeddings"], "path")?.to_owned();
        let mut cards = read_json(&source.join(safe_key(&cp)?))?;
        let fields = cards["fields"].as_array().ok_or("missing fields")?.clone();
        let col = |s: &str| {
            fields
                .iter()
                .position(|v| v == s)
                .ok_or_else(|| format!("missing {s}"))
        };
        let (ic, im) = (col("id")?, col("image")?);
        let rows = cards["rows"].as_array_mut().ok_or("missing rows")?;
        if rows.iter().any(|r| {
            r.as_array().is_none_or(|a| {
                a.len() != fields.len() || a.iter().any(|v| !v.is_null() && !v.is_string())
            })
        }) {
            return Err("cards must contain strings or null in existing fields".into());
        }
        let original = fs::read(source.join(safe_key(&ep)?))?;
        if original.len() != rows.len() * 256 || cat["count"].as_u64() != Some(rows.len() as u64) {
            return Err(format!("invalid dimensions/count in {id}").into());
        }
        let faces: Vec<_> = rows
            .iter()
            .map(|r| {
                r[im]
                    .as_str()
                    .and_then(|s| s.strip_prefix("https://cdn.pokoin.com/"))
                    .is_some_and(|k| keys.contains(k))
            })
            .collect();
        let seeds: HashSet<_> = faces
            .iter()
            .enumerate()
            .filter(|(_, f)| **f)
            .map(|(i, _)| &original[i * 256..(i + 1) * 256])
            .collect();
        let seeds: Vec<_> = seeds
            .into_iter()
            .map(|b| {
                let v = vector(b);
                let n = norm(&v);
                (v, n)
            })
            .filter(|(_, n)| *n > 0.0)
            .collect();
        let mut noimage: std::collections::BTreeSet<String> = cat["noImagePublicIDs"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
        let (mut marked, mut replaced, mut appended) = (0, 0, 0);
        let mut image_changed = false;
        let mut bytes = original.clone();
        let mut changes = updates.remove(&id).unwrap_or_default();
        for (i, row) in rows.iter_mut().enumerate() {
            let public = row[ic].as_str().ok_or("id must be string")?.to_owned();
            let raw = &original[i * 256..(i + 1) * 256];
            let v = vector(raw);
            let n = norm(&v);
            let bad = faces[i]
                || seeds.iter().any(|(s, sn)| {
                    n > 0.0
                        && v.iter().zip(s).map(|(a, b)| a * b).sum::<f32>() / (n * sn) >= min_cos
                });
            if bad || n == 0.0 || noimage.contains(&public) {
                if n > 0.0 {
                    marked += 1;
                }
                bytes[i * 256..(i + 1) * 256].fill(0);
                noimage.insert(public.clone());
            }
            if let Some(update) = changes.remove(&public) {
                // An append for an existing id is an idempotent no-op.
                if update["append"].is_null() {
                    let data = encode(&update["vector"])?;
                    if bytes[i * 256..(i + 1) * 256] != data {
                        replaced += 1;
                    }
                    bytes[i * 256..(i + 1) * 256].copy_from_slice(&data);
                    noimage.remove(&public);
                    if let Some(image) = update.get("image") {
                        if !image.is_string() {
                            return Err("replacement image must be string".into());
                        }
                        image_changed |= row[im] != *image;
                        row[im] = image.clone();
                    }
                }
            }
        }
        for (public, update) in changes {
            if update["append"].is_null() {
                return Err(format!("replacement id {public} absent in {id}").into());
            }
            if string(&update["append"], "id")? != public {
                return Err("append id mismatch".into());
            }
            let row: Vec<_> = fields
                .iter()
                .map(|f| {
                    update["append"]
                        .get(f.as_str().unwrap_or(""))
                        .cloned()
                        .unwrap_or(Value::Null)
                })
                .collect();
            if row.iter().any(|v| !v.is_null() && !v.is_string()) {
                return Err("append cells must be strings or null".into());
            }
            rows.push(row.into());
            appended += 1;
            if update["vector"].is_null() {
                bytes.extend([0u8; 256]);
                noimage.insert(public);
            } else {
                bytes.extend(encode(&update["vector"])?);
                noimage.remove(&public);
            }
        }
        cat["count"] = rows.len().into();
        cat["noImagePublicIDs"] = json!(noimage);
        if appended > 0 || replaced > 0 || image_changed || bytes != original {
            for (kind, old, ext, data) in [
                ("cards", cp, "json", serde_json::to_vec(&cards)?),
                ("embeddings", ep, "f16", bytes),
            ] {
                let a = artifact(&out, &old, kind, ext, &data)?;
                skip.insert(PathBuf::from(string(&a, "path")?));
                obsolete.insert(PathBuf::from(old));
                for k in ["path", "bytes", "sha256"] {
                    cat[kind][k] = a[k].clone();
                }
            }
        }
        reports.insert(id,json!({"marked":marked,"replaced":replaced,"appended":appended,"noImage":noimage.len()}));
    }
    for cat in catalogs.iter() {
        for kind in ["cards", "embeddings"] {
            obsolete.remove(Path::new(string(&cat[kind], "path")?));
        }
    }
    skip.extend(obsolete);
    clone_tree(&source, &out, &source, &skip)?;
    index["placeholderMarkExport"] = json!({"tool":"card-images catalog-mark","scannerRevision":env!("CARD_IMAGES_REVISION"),"sourceRevision":source.file_name().unwrap_or_default().to_string_lossy(),"auditSha256":hash(&audit),"minCos":min_cos,"catalogs":reports});
    write_json(&out.join("index.json"), &index)?;
    write_json(&out.join("placeholder-mark-report.json"), &reports)?;
    println!("{}", serde_json::to_string(&reports)?);
    Ok(())
}
fn encode(v: &Value) -> Res<Vec<u8>> {
    Ok(v.as_array()
        .ok_or("missing vector")?
        .iter()
        .flat_map(|v| half::f16::from_f64(v.as_f64().unwrap()).to_le_bytes())
        .collect())
}
