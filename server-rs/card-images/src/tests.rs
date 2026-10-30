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
    ] {
        assert!(recover::stamped("Numel", stamp));
    }
    assert!(!recover::stamped("Numel", "Full Art"));
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
