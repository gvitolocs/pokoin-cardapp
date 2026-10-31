use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "card-images-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn join(&self, p: &str) -> PathBuf {
        self.0.join(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn sample(name: &str) -> Vec<u8> {
    fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../card-placeholder/testdata")
            .join(name),
    )
    .unwrap()
}
fn put(path: &Path, bytes: &[u8]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}
fn flags(pairs: &[(&str, PathBuf)]) -> Flags {
    Flags(
        pairs
            .iter()
            .map(|(k, p)| (k.to_string(), p.to_string_lossy().into_owned()))
            .collect(),
    )
}
fn output_rows(path: &Path) -> Vec<Value> {
    let mut out = Vec::new();
    lines(path, |v| {
        out.push(v);
        Ok(())
    })
    .unwrap();
    out
}
#[test]
fn normalisation() {
    assert_eq!(recover::numbers("042/214"), ["042", "42"]);
    assert_eq!(recover::numbers("SM148"), ["SM148"]);
    assert_eq!(recover::numbers("TG05/TG30"), ["TG05"]);
    assert_eq!(recover::numbers("223"), ["223"]);
    assert_eq!(recover::numbers("1/10"), ["1", "001"]);
    assert_eq!(
        recover::fold("Pokémon Communication"),
        "pokemoncommunication"
    );
    assert_eq!(recover::fold("Mewtwo VSTAR (Full Art)"), "mewtwovstar");
    assert_eq!(recover::fold("Poke\u{301}mon"), "pokemon");
    for stamp in [
        "Finalist",
        "Semi-Finalist",
        "Champion",
        "Top Sixteen",
        "Top Eight",
        "Staff",
        "Prerelease",
        "Winner",
        "Stamp",
        "League",
        "Championship",
        "Top Thirty-Two",
        "Quarter-Finalist",
    ] {
        assert!(recover::stamped("Numel", stamp, ""), "{stamp}");
    }
    assert!(!recover::stamped("Numel", "Full Art", ""));
    assert!(!recover::stamped(
        "Champions Festival",
        "Holo Promo | SM148",
        "126934_champions-festival-holo-promo-sm148-sm-black-star-promos.png"
    ));
    assert!(recover::stamped(
        "Champions Festival",
        "Non-Holo Promo | BW95 | Finalist",
        ""
    ));
    assert!(recover::stamped("Pikachu (Staff)", "", ""));
    // CardTrader left the stamp out of `version`; only the key slug says it.
    assert!(recover::stamped(
        "Tropical Wind",
        "Non-Holo Promo | 26",
        "124009_tropical-wind-semi-finalist-non-holo-promo-26-nintendo-black-star-promos.jpg"
    ));
    assert!(!recover::stamped(
        "Spinning Top",
        "Common",
        "1_spinning-top-common.jpg"
    ));
    assert!(!recover::stamped(
        "Pokémon League Pass",
        "",
        "2_pokemon-league-pass.jpg"
    ));
    let bp = serde_json::json!({
        "image_url": "https://cardtrader.com/uploads/blueprints/image/7/preview_onix.webp",
        "image": {"url": "/uploads/blueprints/image/7/onix.webp"}
    });
    assert_eq!(
        recover::full_image_url(&bp),
        "https://www.cardtrader.com/uploads/blueprints/image/7/onix.webp"
    );
    let fallback = serde_json::json!({"image_url": "https://cardtrader.com/x/preview_a.jpg", "image": {"url": null}});
    assert_eq!(
        recover::full_image_url(&fallback),
        "https://cardtrader.com/x/preview_a.jpg"
    );
    for (a, b) in [
        ("me1", "me01"),
        ("me3", "me03"),
        ("sv7", "sv07"),
        ("sv8", "sv08"),
        ("sv8pt5", "sv08.5"),
        ("sv9", "sv09"),
        ("swsh45sv", "sv03.5"),
        ("xy9", "xy9"),
    ] {
        assert_eq!(recover::alias(a), b);
    }
}
#[test]
fn audit_tree() {
    let t = Temp::new();
    let root = t.join("objects");
    let placeholder = sample("placeholder-cardtrader.png");
    let real = sample("real-370861_numel.jpg");
    put(&root.join("1_missing.png"), &placeholder);
    put(&root.join("2_real.jpg"), &real);
    for i in 3..6 {
        put(&root.join(format!("magic/{i}_missing.png")), &placeholder);
    }
    put(&root.join("magic/6_real.jpg"), &real);
    // Three identical real faces become a suspect; a repeated ct_id alone does not qualify.
    for i in 7..10 {
        put(&root.join(format!("magic/{i}_real.jpg")), &real);
    }
    for dir in SKIP {
        put(
            &root.join(format!("magic/nested/{dir}/99_skip.png")),
            &placeholder,
        );
    }
    put(&root.join("previews/1_missing.webp"), &placeholder);
    put(&root.join("1_missing_homepage.webp"), &placeholder);
    put(
        &root.join("magic/10_single-mascot.webp"),
        &sample("placeholder-mascot-homepage.webp"),
    );
    let out = t.join("audit.jsonl");
    audit(&flags(&[
        ("objects", root),
        ("out", out.clone()),
        ("threads", PathBuf::from("3")),
    ]))
    .unwrap();
    let rows = output_rows(&out);
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0]["reason"], "cardtrader_size");
    for row in &rows[1..] {
        assert_eq!(row["game"], "magic");
        assert_eq!(row["reason"], "shared_bytes:cardtrader-preview");
    }
    let report = read_json(&t.join("audit.jsonl.report.json")).unwrap();
    assert_eq!(report["games"]["pokemon"]["files"], 2);
    assert_eq!(report["games"]["magic"]["files"], 8);
    assert_eq!(report["suspect_groups"][0]["size"], 4);
}
struct Fixture {
    temp: Temp,
    flags: Flags,
    old: Vec<u8>,
    real: Vec<u8>,
}
impl Fixture {
    fn new() -> Self {
        let temp = Temp::new();
        let objects = temp.join("objects");
        let old = sample("placeholder-cardtrader.png");
        let img = image::load_from_memory(&sample("real-370861_numel.jpg"))
            .unwrap()
            .resize_exact(420, 600, image::imageops::FilterType::Lanczos3);
        let mut real = Cursor::new(Vec::new());
        img.write_to(&mut real, image::ImageFormat::Png).unwrap();
        let real = real.into_inner();
        put(&objects.join("1_numel.jpg"), &old);
        let derived = image::load_from_memory(&old).unwrap().resize_exact(
            100,
            140,
            image::imageops::FilterType::Triangle,
        );
        for key in ["1_numel_homepage.webp", "previews/1_numel.webp"] {
            let mut b = Cursor::new(Vec::new());
            derived.write_to(&mut b, image::ImageFormat::WebP).unwrap();
            put(&objects.join(key), b.get_ref());
        }
        let bp = json!([{"id":1,"name":"Numel","game_id":5,"expansion_id":10,"fixed_properties":{"collector_number":"042/214"},"image_url":"https://example.invalid/fallback.png"}]);
        put(
            &temp.join("bp/_blueprints_export-5.json"),
            serde_json::to_string(&bp).unwrap().as_bytes(),
        );
        put(
            &temp.join("bp/_expansions-all.json"),
            b"[{\"id\":10,\"name\":\"Test Set\"}]",
        );
        put(
            &temp.join("expansions.tsv"),
            b"10\t5\tsv7\tsv7\tTest Set\twestern\n",
        );
        put(&temp.join("tcg/tcgdex/manifest/en.jsonl"),b"{\"id\":\"sv07-42\",\"localId\":\"42\",\"name\":\"Numel\",\"key\":\"tcgdex/en/test/42.webp\"}\n");
        put(&temp.join("tcg/tcgdex/en/test/42.webp"), &real);
        let audit_file = temp.join("audit.jsonl");
        audit(&flags(&[
            ("objects", objects.clone()),
            ("out", audit_file.clone()),
        ]))
        .unwrap();
        let flags = flags(&[
            ("audit", audit_file),
            ("objects", objects),
            ("blueprints", temp.join("bp")),
            ("expansions", temp.join("expansions.tsv")),
            ("tcgdex", temp.join("tcg")),
            ("pokemontcg", temp.join("pokemon")),
            ("scryfall", temp.join("unused.jsonl.gz")),
            ("out", temp.join("out")),
        ]);
        Self {
            temp,
            flags,
            old,
            real,
        }
    }
    fn blueprint(&self, mutate: impl FnOnce(&mut Value)) {
        let p = self.temp.join("bp/_blueprints_export-5.json");
        let mut v = read_json(&p).unwrap();
        mutate(&mut v[0]);
        write_json(&p, &v).unwrap();
    }
    fn reason(&self) -> String {
        output_rows(&self.temp.join("out/missing.jsonl"))[0]["reason"]
            .as_str()
            .unwrap()
            .into()
    }
}
use std::io::Cursor;
#[test]
fn recover_dry_apply_and_skip() {
    use std::os::unix::fs::PermissionsExt;
    let mut f = Fixture::new();
    let face = f.temp.join("objects/1_numel.jpg");
    fs::set_permissions(&face, fs::Permissions::from_mode(0o640)).unwrap();
    let homepage = fs::read(f.temp.join("objects/1_numel_homepage.webp")).unwrap();
    recover::run(&f.flags).unwrap();
    assert_eq!(fs::read(&face).unwrap(), f.old);
    assert_eq!(
        fs::read(f.temp.join("objects/1_numel_homepage.webp")).unwrap(),
        homepage
    );
    let recovered = output_rows(&f.temp.join("out/recovered.jsonl"));
    assert_eq!(recovered[0]["source"], "tcgdex");
    f.flags.0.insert("apply".into(), String::new());
    f.flags.0.insert(
        "backup".into(),
        f.temp.join("backup").to_string_lossy().into_owned(),
    );
    put(
        &f.temp.join("backup/previews/1_numel.webp"),
        b"existing backup",
    );
    recover::run(&f.flags).unwrap();
    let new = fs::read(&face).unwrap();
    assert_eq!(Detector::builtin().classify(&new), Verdict::Real);
    assert_eq!(hash(&new), recovered[0]["sha256_new"]);
    assert_eq!(
        fs::metadata(&face).unwrap().permissions().mode() & 0o777,
        0o640
    );
    assert_eq!(fs::read(f.temp.join("backup/1_numel.jpg")).unwrap(), f.old);
    assert_eq!(
        fs::read(f.temp.join("backup/1_numel_homepage.webp")).unwrap(),
        homepage
    );
    assert_eq!(
        fs::read(f.temp.join("backup/previews/1_numel.webp")).unwrap(),
        b"existing backup"
    );
    for key in ["1_numel_homepage.webp", "previews/1_numel.webp"] {
        let image = image::open(f.temp.join(&format!("objects/{key}"))).unwrap();
        assert_eq!(image.width(), 100);
        assert_eq!(image.height(), 143);
        assert_eq!(
            Detector::builtin()
                .classify(&fs::read(f.temp.join(&format!("objects/{key}"))).unwrap()),
            Verdict::Real
        );
    }
    recover::run(&f.flags).unwrap();
    assert_eq!(fs::read(&face).unwrap(), new);
    assert_eq!(fs::read(f.temp.join("backup/1_numel.jpg")).unwrap(), f.old);
    let report = read_json(&f.temp.join("out/report.json")).unwrap();
    assert_eq!(report["games"]["pokemon"]["skipped"], 1);
    assert!(output_rows(&f.temp.join("out/recovered.jsonl")).is_empty());
}
#[test]
fn recovery_filters_and_source_priority() {
    let f = Fixture::new();
    f.blueprint(|v| {
        v["name"] = json!("Wrong Card");
    });
    recover::run(&f.flags).unwrap();
    assert_eq!(f.reason(), "mismatch");
    f.blueprint(|v| {
        v["name"] = json!("Numel");
        v["version"] = json!("Staff");
    });
    recover::run(&f.flags).unwrap();
    assert_eq!(f.reason(), "stamped_variant");
    put(&f.temp.join("objects/1_other_homepage.jpg"), &f.real);
    recover::run(&f.flags).unwrap();
    assert_eq!(f.reason(), "stamped_variant");
    // Siblings are permitted even for stamps and outrank all other sources.
    put(&f.temp.join("objects/1_other.png"), &f.real);
    recover::run(&f.flags).unwrap();
    assert_eq!(
        output_rows(&f.temp.join("out/recovered.jsonl"))[0]["source"],
        "sibling"
    );
    fs::remove_file(f.temp.join("objects/1_other.png")).unwrap();
    f.blueprint(|v| {
        v["version"] = json!("");
        v["expansion_id"] = json!(99);
        v["image_url"] = json!("https://example.invalid/card.jpg");
    });
    recover::run(&f.flags).unwrap();
    assert_eq!(f.reason(), "download_disabled");
    f.blueprint(|v| {
        v["image_url"] = json!("fallback");
    });
    recover::run(&f.flags).unwrap();
    assert_eq!(f.reason(), "no_source");
    f.blueprint(|v| {
        v["id"] = json!(2);
    });
    recover::run(&f.flags).unwrap();
    assert_eq!(f.reason(), "no_blueprint");
}
#[test]
fn scryfall_local_mirror_and_number_mismatch() {
    let f = Fixture::new();
    f.blueprint(|v| {
        v["game_id"] = json!(1);
        v["scryfall_id"] = json!("test-uuid");
    });
    let card = json!({"id":"test-uuid","name":"Numel","collector_number":"42","image_uris":{"large":"https://example.invalid/card.jpg"}});
    put(&f.temp.join("unused.jsonl.gz"), &[]);
    let file = fs::File::create(f.temp.join("unused.jsonl.gz")).unwrap();
    let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    writeln!(gz, "{card}").unwrap();
    gz.finish().unwrap();
    put(
        &f.temp.join("tcg/scryfall/en/test/test-uuid-0.jpg"),
        &f.real,
    );
    recover::run(&f.flags).unwrap();
    assert_eq!(
        output_rows(&f.temp.join("out/recovered.jsonl"))[0]["source"],
        "scryfall"
    );
    f.blueprint(|v| {
        v["fixed_properties"]["collector_number"] = json!("43");
    });
    recover::run(&f.flags).unwrap();
    assert_eq!(f.reason(), "mismatch");
}
#[test]
fn apply_requires_backup_and_keys_are_relative() {
    let mut f = Fixture::new();
    f.flags.0.insert("apply".into(), String::new());
    assert!(recover::run(&f.flags)
        .unwrap_err()
        .to_string()
        .contains("backup"));
    for key in ["../escape.jpg", "/absolute.jpg", "magic/../1.jpg", ""] {
        assert!(safe_key(key).is_err());
    }
    f.flags.0.remove("apply");
    f.flags.0.insert(
        "out".into(),
        f.temp.join("objects/output").to_string_lossy().into_owned(),
    );
    assert!(recover::run(&f.flags)
        .unwrap_err()
        .to_string()
        .contains("separate"));
    assert!(!f.temp.join("objects/output").exists());
}

#[test]
fn catalog_purge_export() {
    let t = Temp::new();
    let source = t.join("revision");
    let out = t.join("export");
    let audit = t.join("audit.jsonl");
    put(&audit, b"{\"key\":\"missing.webp\"}\n");
    let cards = json!({"fields": ["id", "image", "name", "number", "set"], "rows": [
        ["face", "https://cdn.pokoin.com/missing.webp", "Missing", "1", "S"],
        ["near", "https://cdn.pokoin.com/near.webp", "Near", "2", "S"],
        ["real", "https://cdn.pokoin.com/real.webp", "Real", "3", "S"]], "extra": {"z": 1, "a": 2}});
    // Non-unit vectors ensure cosine uses both norms. Near is close, not identical.
    let bytes: Vec<u8> = [0x4000u16, 0, 0x3c00, 0x211f, 0, 0xbc00]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    let untouched = json!({"fields": ["id", "image"], "rows": [["other", "https://cdn.pokoin.com/other.webp"]]});
    let mut catalogs = Vec::new();
    for (id, data, embedding, count) in [
        ("changed", &cards, bytes.as_slice(), 3),
        ("untouched", &untouched, &bytes[8..], 1),
    ] {
        let c = format!(
            "{id}/cards-{}.json",
            &hash(&serde_json::to_vec(data).unwrap())[..12]
        );
        let e = format!("{id}/embeddings-{}.f16", &hash(embedding)[..12]);
        put(&source.join(&c), &serde_json::to_vec(data).unwrap());
        put(&source.join(&e), embedding);
        catalogs.push(json!({"id": id, "count": count, "cards": {"path": c, "bytes": 1, "sha256": "old", "extra": true}, "embeddings": {"path": e, "bytes": embedding.len(), "sha256": hash(embedding)}, "legacy": true}));
    }
    let mut index: Value =
        serde_json::from_str(r#"{"z":{"b":1,"a":2},"catalogs":[],"a":"keep"}"#).unwrap();
    index["catalogs"] = catalogs.clone().into();
    put(
        &source.join("index.json"),
        &serde_json::to_vec(&index).unwrap(),
    );
    put(&source.join("packs/nested/file"), b"preserved");
    let source_index = fs::read(source.join("index.json")).unwrap();
    #[cfg(unix)]
    let input = {
        let link = t.join("current");
        std::os::unix::fs::symlink(&source, &link).unwrap();
        link
    };
    #[cfg(not(unix))]
    let input = source.clone();
    let args = flags(&[
        ("catalogs", input),
        ("audit", audit.clone()),
        ("out", out.clone()),
    ]);
    catalog_purge::run(&args).unwrap();
    assert!(catalog_purge::run(&args)
        .unwrap_err()
        .to_string()
        .contains("must not exist"));
    let result = read_json(&out.join("index.json")).unwrap();
    assert_eq!(
        result.as_object().unwrap().keys().collect::<Vec<_>>(),
        ["z", "catalogs", "a", "placeholderPurgeExport"]
    );
    assert_eq!(result["z"], index["z"]);
    assert_eq!(result["a"], index["a"]);
    assert_eq!(result["catalogs"][1], catalogs[1]);
    let changed = &result["catalogs"][0];
    assert_eq!(changed["count"], 1);
    assert_eq!(changed["legacy"], true);
    assert_eq!(changed["cards"]["extra"], true);
    for kind in ["cards", "embeddings"] {
        let p = changed[kind]["path"].as_str().unwrap();
        let data = fs::read(out.join(p)).unwrap();
        let sha = hash(&data);
        assert!(p.contains(&sha[..12]));
        assert_eq!(changed[kind]["sha256"], sha);
        assert_eq!(changed[kind]["bytes"], data.len());
        assert!(!out
            .join(catalogs[0][kind]["path"].as_str().unwrap())
            .exists());
    }
    let kept = read_json(&out.join(changed["cards"]["path"].as_str().unwrap())).unwrap();
    assert_eq!(kept["rows"], json!([cards["rows"][2]]));
    assert_eq!(kept["fields"], cards["fields"]);
    assert_eq!(kept["extra"], cards["extra"]);
    assert_eq!(
        fs::read(out.join(changed["embeddings"]["path"].as_str().unwrap())).unwrap(),
        bytes[8..]
    );
    assert_eq!(fs::read(source.join("index.json")).unwrap(), source_index);
    let export = &result["placeholderPurgeExport"];
    assert_eq!(export["removed"]["changed"], json!(["face", "near"]));
    assert_eq!(export["auditSha256"], hash(&fs::read(audit).unwrap()));
    assert_eq!(export["sourceRevision"], "revision");
    let report = read_json(&out.join("placeholder-purge-report.json")).unwrap();
    assert_eq!(
        report[0]["removedByReason"],
        json!({"face":1,"embedding":1})
    );
    assert_eq!(
        report[0]["removed"][1],
        json!({"id":"near","name":"Near","number":"2","set":"S","image":"https://cdn.pokoin.com/near.webp","reason":"embedding"})
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        for p in [
            catalogs[1]["cards"]["path"].as_str().unwrap(),
            catalogs[1]["embeddings"]["path"].as_str().unwrap(),
            "packs/nested/file",
        ] {
            assert_eq!(
                fs::metadata(source.join(p)).unwrap().ino(),
                fs::metadata(out.join(p)).unwrap().ino()
            );
        }
    }
    let only_out = t.join("only-export");
    let only_args = flags(&[
        ("catalogs", source.clone()),
        ("audit", t.join("audit.jsonl")),
        ("out", only_out.clone()),
        ("only", "untouched".into()),
    ]);
    catalog_purge::run(&only_args).unwrap();
    assert_eq!(
        read_json(&only_out.join("index.json")).unwrap()["catalogs"],
        index["catalogs"]
    );
    let empty_out = t.join("empty-export");
    catalog_purge::run(&flags(&[
        ("catalogs", source.clone()),
        ("audit", t.join("audit.jsonl")),
        ("out", empty_out.clone()),
        ("cos", "-1".into()),
    ]))
    .unwrap();
    let empty_index = read_json(&empty_out.join("index.json")).unwrap();
    assert_eq!(empty_index["catalogs"][0]["count"], 0);
    assert!(fs::read(
        empty_out.join(
            empty_index["catalogs"][0]["embeddings"]["path"]
                .as_str()
                .unwrap()
        )
    )
    .unwrap()
    .is_empty());
    catalog_purge::run(&flags(&[
        ("catalogs", empty_out),
        ("audit", t.join("audit.jsonl")),
        ("out", t.join("empty-again")),
    ]))
    .unwrap();

    let invalid = flags(&[
        ("catalogs", source.clone()),
        ("audit", t.join("audit.jsonl")),
        ("out", source.join("bad")),
    ]);
    assert!(catalog_purge::run(&invalid).is_err());
    assert!(!source.join("bad").exists());
}

#[test]
fn catalog_half_conversion() {
    assert_eq!(catalog_purge::f16(1), 2.0f32.powi(-24));
    assert_eq!(catalog_purge::f16(0x3c00), 1.0);
    assert_eq!(catalog_purge::f16(0xbc00), -1.0);
    assert_eq!(catalog_purge::f16(0x7c00), f32::INFINITY);
    assert!(catalog_purge::f16(0x7e00).is_nan());
}

#[test]
fn incremental_audit_rechecks_changed_and_carries_placeholders() {
    let t = Temp::new();
    let root = t.join("objects");
    let placeholder = sample("placeholder-cardtrader.png");
    for id in 1..=4 {
        put(&root.join(format!("{id}_face.png")), &placeholder);
    }
    put(&root.join("5_real.jpg"), &sample("real-370861_numel.jpg"));
    let previous = t.join("previous.jsonl");
    audit(&flags(&[
        ("objects", root.clone()),
        ("out", previous.clone()),
    ]))
    .unwrap();
    let cutoff = fs::metadata(&previous).unwrap().modified().unwrap();
    let later = cutoff + std::time::Duration::from_secs(2);
    // Recovered, touched, deleted and new files; use deterministic mtimes without sleeps.
    put(&root.join("1_face.png"), &sample("real-370861_numel.jpg"));
    for name in ["1_face.png", "2_face.png"] {
        fs::File::options()
            .write(true)
            .open(root.join(name))
            .unwrap()
            .set_modified(later)
            .unwrap();
    }
    fs::remove_file(root.join("4_face.png")).unwrap();
    put(&root.join("6_new.png"), &placeholder);
    fs::File::options()
        .write(true)
        .open(root.join("6_new.png"))
        .unwrap()
        .set_modified(later)
        .unwrap();
    let out = t.join("incremental.jsonl");
    audit(&flags(&[
        ("objects", root),
        ("out", out.clone()),
        ("previous", previous.clone()),
    ]))
    .unwrap();
    let rows = output_rows(&out);
    assert_eq!(rows.len(), 3);
    assert!(!rows
        .iter()
        .any(|r| r["key"] == "1_face.png" || r["key"] == "4_face.png"));
    assert_eq!(
        rows.iter().find(|r| r["key"] == "3_face.png"),
        output_rows(&previous)
            .iter()
            .find(|r| r["key"] == "3_face.png")
    );
    let report = read_json(&t.join("incremental.jsonl.report.json")).unwrap();
    assert_eq!(report["examined"], 3);
    assert_eq!(report["carried_over"], 1);
}

#[test]
fn catalog_mark_roundtrip_and_pinned_hardlinks() {
    let t = Temp::new();
    let source = t.join("source");
    let audit = t.join("audit");
    put(&audit, b"{\"key\":\"missing.jpg\"}\n");
    let fields = json!(["name", "id", "image", "number", "set", "extra"]);
    let cards = json!({"fields":fields,"rows":[["Missing","2","https://cdn.pokoin.com/missing.jpg",null,null,null],["Near","4","https://cdn.pokoin.com/near.jpg",null,null,"keep"],["Real","6","https://cdn.pokoin.com/real.jpg",null,null,null],["Zero","8","",null,null,null]]});
    let mut bytes = vec![0u8; 4 * 256];
    for (i, d) in [(0, 0), (1, 0), (2, 1)] {
        bytes[i * 256 + d * 2..i * 256 + d * 2 + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
    }
    let mut cats = Vec::new();
    for id in ["pokemon_western", "pokemon_artwork"] {
        put(
            &source.join(format!("{id}/cards.json")),
            &serde_json::to_vec(&cards).unwrap(),
        );
        put(&source.join(format!("{id}/vectors.f16")), &bytes);
        cats.push(json!({"id":id,"count":4,"cards":{"path":format!("{id}/cards.json"),"sha256":if id=="pokemon_artwork" {"pinned"} else {"editable"}},"embeddings":{"path":format!("{id}/vectors.f16")},"keep":true}));
    }
    put(
        &source.join("packs/a/manifest-v1.json"),
        b"{\"nested\":{\"catalog_sha256\":\"pinned\"}}",
    );
    put(
        &source.join("index.json"),
        &serde_json::to_vec(&json!({"z":1,"catalogs":cats,"a":2})).unwrap(),
    );
    let marked = t.join("marked");
    catalog_mark::run(&flags(&[
        ("catalogs", source.clone()),
        ("audit", audit.clone()),
        ("out", marked.clone()),
    ]))
    .unwrap();
    let index = read_json(&marked.join("index.json")).unwrap();
    assert_eq!(
        index["catalogs"][0]["noImagePublicIDs"],
        json!(["2", "4", "8"])
    );
    assert_eq!(index["catalogs"][1], cats[1]);
    assert_eq!(
        index["placeholderMarkExport"]["catalogs"]["pokemon_artwork"]["pinned"],
        true
    );
    assert_eq!(
        index.as_object().unwrap().keys().collect::<Vec<_>>(),
        ["z", "catalogs", "a", "placeholderMarkExport"]
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        assert_eq!(
            fs::metadata(source.join("pokemon_artwork/vectors.f16"))
                .unwrap()
                .ino(),
            fs::metadata(marked.join("pokemon_artwork/vectors.f16"))
                .unwrap()
                .ino()
        );
    }
    let mut vector = vec![0.0; 128];
    vector[2] = 1.0;
    let updates = t.join("updates");
    put(&updates,format!("{}\n{}\n{}\n",json!({"catalog":"pokemon_western","id":"2","vector":vector}),json!({"catalog":"pokemon_western","id":"10","vector":null,"append":{"id":"10","name":"New","number":"5","set":"S","image":""}}),json!({"catalog":"pokemon_western","id":"12","vector":vector,"append":{"id":"12","name":"Real","number":"6","set":"S","image":"https://cdn.pokoin.com/new.jpg"}})).as_bytes());
    // Real recovery no longer belongs to the audit.
    put(&audit, b"");
    let out = t.join("recovered");
    catalog_mark::run(&flags(&[
        ("catalogs", marked),
        ("audit", audit.clone()),
        ("out", out.clone()),
        ("embeddings", updates.clone()),
    ]))
    .unwrap();
    let index = read_json(&out.join("index.json")).unwrap();
    let cat = &index["catalogs"][0];
    assert_eq!(cat["count"], 6);
    assert_eq!(cat["noImagePublicIDs"], json!(["10", "4", "8"]));
    let result = read_json(&out.join(cat["cards"]["path"].as_str().unwrap())).unwrap();
    assert_eq!(result["fields"], fields);
    assert_eq!(result["rows"][4], json!(["New", "10", "", "5", "S", null]));
    let vectors = fs::read(out.join(cat["embeddings"]["path"].as_str().unwrap())).unwrap();
    assert_eq!(&vectors[4..6], &0x3c00u16.to_le_bytes());
    let again = t.join("again");
    catalog_mark::run(&flags(&[
        ("catalogs", out),
        ("audit", audit),
        ("out", again.clone()),
        ("embeddings", updates),
    ]))
    .unwrap();
    assert_eq!(
        read_json(&again.join("index.json")).unwrap()["catalogs"][0]["count"],
        6
    );
    assert_eq!(
        fs::read(source.join("pokemon_western/vectors.f16")).unwrap(),
        bytes
    );
}

#[test]
fn incremental_shared_bytes_include_carried_placeholders() {
    let t = Temp::new();
    let root = t.join("objects");
    let placeholder = sample("placeholder-cardtrader.png");
    for id in 1..=3 {
        put(&root.join(format!("magic/{id}_face.png")), &placeholder);
    }
    let previous = t.join("previous.jsonl");
    audit(&flags(&[
        ("objects", root.clone()),
        ("out", previous.clone()),
    ]))
    .unwrap();
    let later =
        fs::metadata(&previous).unwrap().modified().unwrap() + std::time::Duration::from_secs(2);
    put(&root.join("magic/4_face.png"), &placeholder);
    fs::File::options()
        .write(true)
        .open(root.join("magic/4_face.png"))
        .unwrap()
        .set_modified(later)
        .unwrap();
    let out = t.join("new.jsonl");
    audit(&flags(&[
        ("objects", root),
        ("out", out.clone()),
        ("previous", previous),
    ]))
    .unwrap();
    let rows = output_rows(&out);
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[3]["reason"], "shared_bytes:cardtrader-preview");
    let report = read_json(&t.join("new.jsonl.report.json")).unwrap();
    assert_eq!(report["examined"], 1);
    assert_eq!(report["carried_over"], 3);
}
