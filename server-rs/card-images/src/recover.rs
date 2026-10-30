use super::*;
use image::{DynamicImage, ImageFormat};
use std::io::Cursor;
use std::time::Duration;

#[derive(Clone, Default)]
struct Blueprint {
    name: String,
    version: String,
    number: String,
    game: u64,
    expansion: u64,
    url: String,
    scryfall: String,
}
#[derive(Default)]
struct Expansion {
    official: String,
    name: String,
    nationality: String,
}
fn string(v: &Value) -> String {
    v.as_str()
        .map(str::to_string)
        .or_else(|| v.as_u64().map(|n| n.to_string()))
        .unwrap_or_default()
}

pub(super) fn numbers(s: &str) -> Vec<String> {
    let raw = s.split('/').next().unwrap_or_default().trim();
    if raw.is_empty() {
        return Vec::new();
    }
    let mut out = vec![raw.to_string()];
    if raw.bytes().all(|b| b.is_ascii_digit()) {
        let trimmed = raw.trim_start_matches('0');
        let trimmed = if trimmed.is_empty() { "0" } else { trimmed };
        for n in [trimmed.to_string(), format!("{trimmed:0>3}")] {
            if !out.contains(&n) {
                out.push(n);
            }
        }
    }
    out
}
pub(super) fn fold(s: &str) -> String {
    let s = s.trim();
    let s = if s.ends_with(')') {
        s.rfind('(').map_or(s, |i| s[..i].trim())
    } else {
        s
    };
    let mut out = String::new();
    for c in s.chars().flat_map(char::to_lowercase) {
        let mapped = match c {
            'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' | 'ǎ' | 'ǟ' | 'ǡ' | 'ǻ' | 'ȁ'
            | 'ȃ' | 'ȧ' | 'ḁ' | 'ạ' | 'ả' | 'ấ' | 'ầ' | 'ẩ' | 'ẫ' | 'ậ' | 'ắ' | 'ằ' | 'ẳ' | 'ẵ'
            | 'ặ' => "a",
            'æ' => "ae",
            'ç' | 'ć' | 'č' | 'ĉ' | 'ċ' => "c",
            'ď' | 'đ' | 'ð' => "d",
            'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ė' | 'ę' | 'ě' | 'ĕ' | 'ȅ' | 'ȇ' | 'ḕ' | 'ḗ' | 'ḙ'
            | 'ḛ' | 'ḝ' | 'ẹ' | 'ẻ' | 'ẽ' | 'ế' | 'ề' | 'ể' | 'ễ' | 'ệ' => "e",
            'ĝ' | 'ğ' | 'ġ' | 'ģ' => "g",
            'ĥ' | 'ħ' => "h",
            'ì' | 'í' | 'î' | 'ï' | 'ī' | 'į' | 'ı' | 'ĭ' | 'ǐ' | 'ȉ' | 'ȋ' | 'ḭ' | 'ḯ' | 'ỉ'
            | 'ị' => "i",
            'ĵ' => "j",
            'ķ' => "k",
            'ł' | 'ľ' | 'ĺ' | 'ļ' | 'ŀ' => "l",
            'ñ' | 'ń' | 'ň' | 'ņ' => "n",
            'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ő' | 'ŏ' | 'ǒ' | 'ǫ' | 'ǭ' | 'ǿ' | 'ȍ'
            | 'ȏ' | 'ȫ' | 'ȭ' | 'ȯ' | 'ȱ' | 'ṍ' | 'ṏ' | 'ṑ' | 'ṓ' | 'ọ' | 'ỏ' | 'ố' | 'ồ' | 'ổ'
            | 'ỗ' | 'ộ' | 'ớ' | 'ờ' | 'ở' | 'ỡ' | 'ợ' | 'ơ' => "o",
            'œ' => "oe",
            'ŕ' | 'ř' | 'ŗ' => "r",
            'ś' | 'š' | 'ş' | 'ŝ' | 'ș' => "s",
            'ß' => "ss",
            'ť' | 'ţ' | 'ț' | 'ŧ' => "t",
            'þ' => "th",
            'ù' | 'ú' | 'û' | 'ü' | 'ū' | 'ů' | 'ű' | 'ŭ' | 'ų' | 'ǔ' | 'ǖ' | 'ǘ' | 'ǚ' | 'ǜ'
            | 'ȕ' | 'ȗ' | 'ṳ' | 'ṵ' | 'ṷ' | 'ṹ' | 'ṻ' | 'ụ' | 'ủ' | 'ứ' | 'ừ' | 'ử' | 'ữ' | 'ự'
            | 'ư' => "u",
            'ŵ' => "w",
            'ý' | 'ÿ' | 'ŷ' => "y",
            'ź' | 'ż' | 'ž' => "z",
            _ => "",
        };
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else {
            out.push_str(mapped);
        }
    }
    out
}
fn names_match(a: &str, b: &str) -> bool {
    let (a, b) = (fold(a), fold(b));
    !a.is_empty() && !b.is_empty() && (a.contains(&b) || b.contains(&a))
}
fn number_match(a: &str, b: &str) -> bool {
    let b = numbers(b);
    numbers(a)
        .iter()
        .any(|n| b.iter().any(|v| n.eq_ignore_ascii_case(v)))
}
fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric()).map(fold).filter(|w| !w.is_empty()).collect()
}

/// CardTrader marks a stamped printing in `version` (`Non-Holo Promo | BW95 | Finalist`), in a `(...)` name suffix,
/// or only in the key's slug (`124009_tropical-wind-semi-finalist-non-holo-promo-26-...`, version `Non-Holo Promo | 26`).
/// The card name itself is not checked, so the unstamped `Champions Festival` SM148 still matches the datasets.
pub(super) fn stamped(name: &str, version: &str, key: &str) -> bool {
    const STAMPS: [&str; 10] = [
        "finalist", "champion", "staff", "prerelease", "winner", "stamp", "stamped", "league", "championship",
        "participant",
    ];
    const PLACES: [&str; 10] = ["two", "four", "eight", "sixteen", "thirty", "sixty", "4", "8", "16", "32"];
    let name = name.trim();
    let suffix = name.strip_suffix(')').and_then(|n| n.rfind('(').map(|i| &n[i..])).unwrap_or_default();
    let stem = Path::new(key).file_stem().and_then(|s| s.to_str()).unwrap_or_default();
    let mut slug = words(stem.split_once('_').map_or("", |(_, s)| s));
    let bare = words(name.split('(').next().unwrap_or_default());
    if !bare.is_empty() && slug.starts_with(&bare) {
        slug.drain(..bare.len());
    }
    let mut text = words(&format!("{suffix} {version}"));
    text.extend(slug);
    text.iter().any(|w| STAMPS.contains(&w.as_str()) || w.ends_with("finalist") || w.ends_with("championships"))
        || text.windows(2).any(|p| p[0] == "top" && PLACES.contains(&p[1].as_str()))
}
pub(super) fn alias(s: &str) -> &str {
    match s {
        "me1" => "me01",
        "me3" => "me03",
        "sv7" => "sv07",
        "sv8" => "sv08",
        "sv8pt5" => "sv08.5",
        "sv9" => "sv09",
        "swsh45sv" => "sv03.5",
        _ => s,
    }
}

struct Catalogs {
    blueprints: HashMap<u64, Blueprint>,
    expansions: HashMap<u64, Expansion>,
    tcgdex: HashMap<(String, String), Value>,
    pokemon: HashMap<String, Vec<Value>>,
    scryfall: HashMap<String, Value>,
    mirror: HashMap<String, PathBuf>,
}
impl Catalogs {
    fn load(flags: &Flags, ids: &HashSet<u64>) -> Res<Self> {
        let mut expansions = HashMap::new();
        for line in BufReader::new(fs::File::open(flags.path("expansions")?)?).lines() {
            let line = line?;
            let fields: Vec<_> = line.split('\t').collect();
            if fields.len() != 6 {
                return Err("expansions TSV needs six columns".into());
            }
            expansions.insert(
                fields[0].parse()?,
                Expansion {
                    official: fields[2].into(),
                    name: fields[4].into(),
                    nationality: fields[5].into(),
                },
            );
        }
        let bp_root = flags.path("blueprints")?;
        let sets = read_json(&bp_root.join("_expansions-all.json"))?;
        for set in sets.as_array().ok_or("expansions JSON must be an array")? {
            if let Some(id) = set["id"].as_u64() {
                expansions.entry(id).or_insert_with(Expansion::default).name = string(&set["name"]);
            }
        }
        let mut blueprints = HashMap::new();
        let mut paths: Vec<_> = fs::read_dir(bp_root)?
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<_, _>>()?;
        paths.sort();
        for path in paths {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if !name.starts_with("_blueprints_export-") || !name.ends_with(".json") {
                continue;
            }
            let data = read_json(&path)?;
            for v in data.as_array().ok_or("blueprints JSON must be an array")? {
                let Some(id) = v["id"].as_u64().filter(|id| ids.contains(id)) else {
                    continue;
                };
                blueprints.insert(
                    id,
                    Blueprint {
                        name: string(&v["name"]),
                        version: string(&v["version"]),
                        number: string(&v["fixed_properties"]["collector_number"]),
                        game: v["game_id"].as_u64().unwrap_or_default(),
                        expansion: v["expansion_id"].as_u64().unwrap_or_default(),
                        url: string(&v["image_url"]),
                        scryfall: string(&v["scryfall_id"]),
                    },
                );
            }
        }
        let mut tcgdex = HashMap::new();
        let mut pokemon = HashMap::new();
        let wanted: HashSet<_> = blueprints
            .values()
            .filter(|b| b.game == 5 && !stamped(&b.name, &b.version, ""))
            .filter_map(|b| expansions.get(&b.expansion))
            .filter(|e| !e.official.is_empty())
            .map(|e| {
                (
                    if e.nationality.eq_ignore_ascii_case("japanese") {
                        "ja"
                    } else {
                        "en"
                    }
                    .to_string(),
                    alias(&e.official).to_string(),
                )
            })
            .collect();
        for lang in ["en", "ja"] {
            let sets: HashSet<_> = wanted
                .iter()
                .filter(|(l, _)| l == lang)
                .map(|(_, s)| s.as_str())
                .collect();
            if sets.is_empty() {
                continue;
            }
            let manifest = flags
                .path("tcgdex")?
                .join(format!("tcgdex/manifest/{lang}.jsonl"));
            if manifest.exists() {
                lines(&manifest, |v| {
                    let id = string(&v["id"]);
                    if id.rsplit_once('-').is_some_and(|(s, _)| sets.contains(s)) {
                        tcgdex.insert((lang.into(), id), v);
                    }
                    Ok(())
                })?;
            }
        }
        for (_, set) in &wanted {
            let path = flags
                .path("pokemontcg")?
                .join("cards/en")
                .join(format!("{set}.json"));
            if path.exists() {
                let data = read_json(&path)?;
                pokemon.insert(
                    set.clone(),
                    data.as_array()
                        .ok_or("pokemon cards must be an array")?
                        .clone(),
                );
            }
        }
        let scry_ids: HashSet<_> = blueprints
            .values()
            .filter(|b| b.game == 1 && !b.scryfall.is_empty() && !stamped(&b.name, &b.version, ""))
            .map(|b| b.scryfall.clone())
            .collect();
        let mut scryfall = HashMap::new();
        let mut mirror = HashMap::new();
        if !scry_ids.is_empty() {
            let file = fs::File::open(flags.path("scryfall")?)?;
            let reader: Box<dyn Read> = if flags
                .path("scryfall")?
                .extension()
                .is_some_and(|s| s == "gz")
            {
                Box::new(flate2::read::GzDecoder::new(file))
            } else {
                Box::new(file)
            };
            for line in BufReader::new(reader).lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let v: Value = serde_json::from_str(&line)?;
                let id = string(&v["id"]);
                if scry_ids.contains(&id) {
                    scryfall.insert(id, v);
                }
            }
            let root = flags.path("tcgdex")?.join("scryfall");
            if root.exists() {
                mirror_files(&root, &scry_ids, &mut mirror)?;
            }
        }
        Ok(Self {
            blueprints,
            expansions,
            tcgdex,
            pokemon,
            scryfall,
            mirror,
        })
    }
}
fn mirror_files(
    dir: &Path,
    wanted: &HashSet<String>,
    out: &mut HashMap<String, PathBuf>,
) -> Res<()> {
    let mut entries: Vec<_> = fs::read_dir(dir)?.collect::<Result<_, _>>()?;
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        if entry.file_type()?.is_dir() {
            mirror_files(&entry.path(), wanted, out)?;
        } else if entry.file_type()?.is_file() {
            let path = entry.path();
            let stem = path.file_stem().unwrap_or_default().to_string_lossy();
            let id = stem.strip_suffix("-0").unwrap_or(&stem);
            if wanted.contains(id) && face(&path) {
                out.entry(id.to_string()).or_insert(path);
            }
        }
    }
    Ok(())
}

struct Downloader {
    agent: Option<ureq::Agent>,
    cache: PathBuf,
}
impl Downloader {
    fn new(flags: &Flags, out: &Path) -> Self {
        Self {
            agent: flags.has("download").then(|| {
                ureq::AgentBuilder::new()
                    .timeout(Duration::from_secs(20))
                    .redirects(0)
                    .try_proxy_from_env(false)
                    .user_agent("BattleScan-card-images/0.1 (placeholder card image recovery)")
                    .build()
            }),
            cache: out.join("cache"),
        }
    }
    // Recovery is sequential: at most one HTTP request is in flight, below the limit of four.
    fn get(&self, url: &str) -> Result<Vec<u8>, &'static str> {
        let agent = self.agent.as_ref().ok_or("download_disabled")?;
        if !(url.starts_with("https://") || url.starts_with("http://")) {
            return Err("download_failed");
        }
        let path = self.cache.join(hash(url.as_bytes()));
        if let Ok(bytes) = fs::read(&path) {
            return Ok(bytes);
        }
        fs::create_dir_all(&self.cache).map_err(|_| "download_failed")?;
        for _ in 0..3 {
            if let Ok(response) = Self::request(agent, url) {
                let mut bytes = Vec::new();
                if response
                    .into_reader()
                    .take(40 * 1024 * 1024)
                    .read_to_end(&mut bytes)
                    .is_ok()
                    && image::load_from_memory(&bytes).is_ok()
                {
                    atomic_write(&path, &bytes, None).map_err(|_| "download_failed")?;
                    return Ok(bytes);
                }
            }
        }
        Err("download_failed")
    }

    fn request(agent: &ureq::Agent, url: &str) -> Res<ureq::Response> {
        let parsed = agent.get(url).request_url()?;
        let mut url = parsed.as_url().clone();
        let start = std::time::Instant::now();
        // Check every redirect too: ureq otherwise turns URL userinfo into Basic auth.
        for _ in 0..10 {
            if !matches!(url.scheme(), "http" | "https")
                || !url.username().is_empty()
                || url.password().is_some()
            {
                return Err("image URL must be HTTP(S) without credentials".into());
            }
            let remaining = Duration::from_secs(20)
                .checked_sub(start.elapsed())
                .ok_or("HTTP timeout")?;
            let response = agent.get(url.as_str()).timeout(remaining).call()?;
            if matches!(response.status(), 301 | 302 | 303 | 307 | 308) {
                url = url.join(
                    response
                        .header("Location")
                        .ok_or("redirect without Location")?,
                )?;
            } else {
                return Ok(response);
            }
        }
        Err("too many image redirects".into())
    }
}
struct Candidate {
    image: DynamicImage,
    source: &'static str,
    reference: String,
    licence: &'static str,
}
fn verified(bytes: &[u8], detector: &Detector) -> Option<DynamicImage> {
    let image = image::load_from_memory(bytes).ok()?;
    let (short, long) = (
        image.width().min(image.height()),
        image.width().max(image.height()),
    );
    if short < 200
        || !(0.62..=0.80).contains(&(short as f64 / long as f64))
        || detector.classify(bytes) != Verdict::Real
    {
        return None;
    }
    Some(image)
}
#[derive(Default)]
struct Failures {
    mismatch: bool,
    failed: bool,
    disabled: bool,
}
impl Failures {
    fn download(&mut self, dl: &Downloader, url: &str) -> Option<Vec<u8>> {
        match dl.get(url) {
            Ok(b) => Some(b),
            Err("download_disabled") => {
                self.disabled = true;
                None
            }
            Err(_) => {
                self.failed = true;
                None
            }
        }
    }
    fn reason(&self) -> &'static str {
        if self.mismatch {
            "mismatch"
        } else if self.failed {
            "download_failed"
        } else if self.disabled {
            "download_disabled"
        } else {
            "no_source"
        }
    }
    fn check(&mut self, b: &Blueprint, name: &str, number: &str) -> bool {
        let good = names_match(&b.name, name) && number_match(&b.number, number);
        self.mismatch |= !good;
        good
    }
}
struct Search<'a> {
    flags: &'a Flags,
    catalogs: &'a Catalogs,
    detector: &'a Detector,
    dl: &'a Downloader,
    siblings: &'a HashMap<(PathBuf, u64), Vec<String>>,
}
impl Search<'_> {
    fn find(&self, row: &AuditRow, b: &Blueprint) -> Res<Result<Candidate, &'static str>> {
        let mut fail = Failures::default();
        let parent = Path::new(&row.key).parent().unwrap();
        let siblings = row
            .ct_id
            .and_then(|id| self.siblings.get(&(parent.to_path_buf(), id)));
        for key in siblings.into_iter().flatten() {
            let path = self.flags.path("objects")?.join(key);
            if let Some(image) = verified(&fs::read(&path)?, self.detector) {
                return Ok(Ok(Candidate {
                    image,
                    source: "sibling",
                    reference: key.clone(),
                    licence: "CardTrader blueprint upload (same ct_id)",
                }));
            }
        }
        if !b.url.is_empty() && !b.url.to_ascii_lowercase().contains("fallback") {
            if let Some(image) = fail
                .download(self.dl, &b.url)
                .and_then(|bytes| verified(&bytes, self.detector))
            {
                return Ok(Ok(Candidate {
                    image,
                    source: "cardtrader",
                    reference: b.url.clone(),
                    licence: "CardTrader blueprint image",
                }));
            }
        }
        if stamped(&b.name, &b.version, &row.key) {
            return Ok(Err("stamped_variant"));
        }
        if b.game == 5 {
            if let Some(e) = self
                .catalogs
                .expansions
                .get(&b.expansion)
                .filter(|e| !e.official.is_empty())
            {
                let set = alias(&e.official);
                let lang = if e.nationality.eq_ignore_ascii_case("japanese") {
                    "ja"
                } else {
                    "en"
                };
                for n in numbers(&b.number) {
                    if let Some(v) = self
                        .catalogs
                        .tcgdex
                        .get(&(lang.into(), format!("{set}-{n}")))
                    {
                        if !fail.check(b, &string(&v["name"]), &string(&v["localId"])) {
                            continue;
                        }
                        let key = string(&v["key"]);
                        let path = self.flags.path("tcgdex")?.join(safe_key(&key)?);
                        if let Ok(bytes) = fs::read(&path) {
                            if let Some(image) = verified(&bytes, self.detector) {
                                return Ok(Ok(Candidate { image, source: "tcgdex", reference: key, licence: "TCGdex scan; card image © Pokémon/Nintendo/Creatures/GAME FREAK" }));
                            }
                        }
                    }
                }
                if let Some(cards) = self.catalogs.pokemon.get(set) {
                    for v in cards
                        .iter()
                        .filter(|v| number_match(&b.number, &string(&v["number"])))
                    {
                        if !fail.check(b, &string(&v["name"]), &string(&v["number"])) {
                            continue;
                        }
                        let url = string(&v["images"]["large"]);
                        if !url.is_empty() {
                            if let Some(image) = fail
                                .download(self.dl, &url)
                                .and_then(|bytes| verified(&bytes, self.detector))
                            {
                                return Ok(Ok(Candidate {
                                    image,
                                    source: "pokemontcg",
                                    reference: url,
                                    licence: "pokemontcg.io image; © The Pokémon Company",
                                }));
                            }
                        }
                    }
                }
            }
        }
        if b.game == 1 {
            if let Some(v) = self.catalogs.scryfall.get(&b.scryfall) {
                if fail.check(b, &string(&v["name"]), &string(&v["collector_number"])) {
                    if let Some(path) = self.catalogs.mirror.get(&b.scryfall) {
                        if let Some(image) = verified(&fs::read(path)?, self.detector) {
                            return Ok(Ok(Candidate { image, source: "scryfall", reference: path.to_string_lossy().into_owned(), licence: "Scryfall image; © Wizards of the Coast (Scryfall image guidelines)" }));
                        }
                    }
                    let url = v["image_uris"]["large"]
                        .as_str()
                        .or_else(|| v["card_faces"][0]["image_uris"]["large"].as_str())
                        .unwrap_or_default();
                    if !url.is_empty() {
                        if let Some(image) = fail
                            .download(self.dl, url)
                            .and_then(|bytes| verified(&bytes, self.detector))
                        {
                            return Ok(Ok(Candidate { image, source: "scryfall", reference: url.into(), licence: "Scryfall image; © Wizards of the Coast (Scryfall image guidelines)" }));
                        }
                    }
                }
            }
        }
        Ok(Err(fail.reason()))
    }
}

fn encode(image: &DynamicImage, path: &Path) -> Res<Vec<u8>> {
    let mut bytes = Cursor::new(Vec::new());
    match path
        .extension()
        .unwrap_or_default()
        .to_string_lossy()
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 92)
            .encode_image(&image.to_rgb8())?,
        "png" => image.write_to(&mut bytes, ImageFormat::Png)?,
        "webp" => image.write_to(&mut bytes, ImageFormat::WebP)?,
        _ => return Err("unsupported output extension".into()),
    }
    Ok(bytes.into_inner())
}
fn atomic_write(path: &Path, bytes: &[u8], mode: Option<fs::Permissions>) -> Res<()> {
    let temp = path.with_file_name(format!(
        ".card-images-{}-{}.tmp",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        if let Some(mode) = mode {
            file.set_permissions(mode)?;
        }
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}
fn backup(path: &Path, to: &Path) -> Res<()> {
    fs::create_dir_all(to.parent().ok_or("backup needs parent")?)?;
    match fs::OpenOptions::new().create_new(true).write(true).open(to) {
        Ok(mut out) => {
            std::io::copy(&mut fs::File::open(path)?, &mut out)?;
            out.set_permissions(fs::metadata(path)?.permissions())?;
            out.sync_all()?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e.into()),
    }
    Ok(())
}
fn apply(
    objects: &Path,
    backups: &Path,
    key: &str,
    image: &DynamicImage,
    encoded: &[u8],
) -> Res<()> {
    let key = safe_key(key)?;
    let path = objects.join(key);
    let parent = key.parent().unwrap();
    let stem = key.file_stem().unwrap().to_string_lossy();
    // Finish derivatives first; the face remains a placeholder if a derivative fails.
    for derived in [
        parent.join(format!("{stem}_homepage.webp")),
        parent.join("previews").join(format!("{stem}.webp")),
    ] {
        let target = objects.join(&derived);
        if target.exists() {
            let (width, _) = image::ImageReader::open(&target)?
                .with_guessed_format()?
                .into_dimensions()?;
            let height = ((u64::from(width) * u64::from(image.height())
                + u64::from(image.width()) / 2)
                / u64::from(image.width()))
            .max(1) as u32;
            let resized = image.resize_exact(width, height, image::imageops::FilterType::Lanczos3);
            let bytes = encode(&resized, &target)?;
            backup(&target, &backups.join(&derived))?;
            atomic_write(&target, &bytes, Some(fs::metadata(&target)?.permissions()))?;
        }
    }
    backup(&path, &backups.join(key))?;
    atomic_write(&path, encoded, Some(fs::metadata(&path)?.permissions()))
}
#[derive(Default, Serialize)]
struct Counts {
    placeholders: usize,
    recovered: BTreeMap<String, usize>,
    missing: BTreeMap<String, usize>,
    skipped: usize,
}

// Index only relevant ids and visit each face directory once, even for thousands of rows.
fn sibling_index(objects: &Path, rows: &[AuditRow]) -> Res<HashMap<(PathBuf, u64), Vec<String>>> {
    let audited: HashSet<_> = rows.iter().map(|r| r.key.as_str()).collect();
    let mut parents: BTreeMap<PathBuf, HashSet<u64>> = BTreeMap::new();
    for row in rows {
        if let Some(id) = row.ct_id {
            parents
                .entry(Path::new(&row.key).parent().unwrap().to_path_buf())
                .or_default()
                .insert(id);
        }
    }
    let mut out: HashMap<_, Vec<String>> = HashMap::new();
    for (parent, ids) in parents {
        for entry in fs::read_dir(objects.join(&parent))? {
            let entry = entry?;
            let path = entry.path();
            let Some(id) = ct_id(&path).filter(|id| ids.contains(id)) else {
                continue;
            };
            if !entry.file_type()?.is_file()
                || !face(&path)
                || path.file_stem().is_some_and(|s| {
                    s.to_string_lossy()
                        .to_ascii_lowercase()
                        .ends_with("_homepage")
                })
            {
                continue;
            }
            let key = parent
                .join(entry.file_name())
                .to_string_lossy()
                .into_owned();
            if !audited.contains(key.as_str()) {
                out.entry((parent.clone(), id)).or_default().push(key);
            }
        }
    }
    for keys in out.values_mut() {
        keys.sort();
    }
    Ok(out)
}

fn resolve_path(path: &Path) -> Res<PathBuf> {
    if path.exists() {
        return Ok(fs::canonicalize(path)?);
    }
    let absolute = std::path::absolute(path)?;
    let parent = absolute.parent().ok_or("path needs a parent")?;
    Ok(resolve_path(parent)?.join(absolute.file_name().ok_or("path needs a filename")?))
}
fn outside_objects(objects: &Path, output: &Path) -> Res<()> {
    let objects = resolve_path(objects)?;
    let output = resolve_path(output)?;
    if output.starts_with(&objects) || objects.starts_with(&output) {
        return Err("output and backup directories must be separate from the objects tree".into());
    }
    Ok(())
}
pub(super) fn run(flags: &Flags) -> Res<()> {
    // Validate all required arguments before reading inputs or writing outputs.
    for key in [
        "audit",
        "objects",
        "blueprints",
        "expansions",
        "tcgdex",
        "pokemontcg",
        "scryfall",
        "out",
    ] {
        flags.path(key)?;
    }
    let backups = if flags.has("apply") {
        Some(flags.path("backup")?)
    } else {
        None
    };
    outside_objects(&flags.path("objects")?, &flags.path("out")?)?;
    if let Some(backups) = &backups {
        outside_objects(&flags.path("objects")?, backups)?;
    }
    let mut rows = Vec::new();
    lines(&flags.path("audit")?, |v| {
        rows.push(serde_json::from_value::<AuditRow>(v)?);
        Ok(())
    })?;
    for row in &rows {
        safe_key(&row.key)?;
        if !face(Path::new(&row.key))
            || Path::new(&row.key)
                .components()
                .any(|c| SKIP.contains(&c.as_os_str().to_string_lossy().as_ref()))
        {
            return Err("audit must contain only face keys".into());
        }
    }
    let ids = rows.iter().filter_map(|r| r.ct_id).collect();
    let catalogs = Catalogs::load(flags, &ids)?;
    let detector = Detector::builtin();
    let out = flags.path("out")?;
    fs::create_dir_all(&out)?;
    let dl = Downloader::new(flags, &out);
    let siblings = sibling_index(&flags.path("objects")?, &rows)?;
    let search = Search {
        flags,
        catalogs: &catalogs,
        detector: &detector,
        dl: &dl,
        siblings: &siblings,
    };
    let mut recovered = BufWriter::new(fs::File::create(out.join("recovered.jsonl"))?);
    let mut missing = BufWriter::new(fs::File::create(out.join("missing.jsonl"))?);
    let mut counts: BTreeMap<String, Counts> = BTreeMap::new();
    for row in rows {
        let count = counts.entry(row.game.clone()).or_default();
        count.placeholders += 1;
        let current = fs::read(flags.path("objects")?.join(&row.key))?;
        if !detector.classify(&current).is_placeholder() {
            count.skipped += 1;
            continue;
        }
        let b = row.ct_id.and_then(|id| catalogs.blueprints.get(&id));
        let mut meta = json!({"key":row.key,"game":row.game,"ct_id":row.ct_id,"name":b.map(|b| &b.name),"number":b.map(|b| &b.number),"set":b.and_then(|b| catalogs.expansions.get(&b.expansion)).map(|e| &e.name)});
        let found = match b {
            Some(b) => search.find(&row, b)?,
            None => Err("no_blueprint"),
        };
        match found {
            Ok(c) => {
                let encoded = encode(&c.image, Path::new(&row.key))?;
                if let Some(backups) = &backups {
                    apply(
                        &flags.path("objects")?,
                        backups,
                        &row.key,
                        &c.image,
                        &encoded,
                    )?;
                }
                let obj = meta.as_object_mut().unwrap();
                obj.insert("source".into(), json!(c.source));
                obj.insert("source_ref".into(), json!(c.reference));
                obj.insert("licence".into(), json!(c.licence));
                obj.insert("sha256_old".into(), json!(hash(&current)));
                obj.insert("sha256_new".into(), json!(hash(&encoded)));
                obj.insert("w".into(), json!(c.image.width()));
                obj.insert("h".into(), json!(c.image.height()));
                jsonl(&mut recovered, &meta)?;
                *count.recovered.entry(c.source.into()).or_default() += 1;
            }
            Err(reason) => {
                meta.as_object_mut()
                    .unwrap()
                    .insert("reason".into(), json!(reason));
                jsonl(&mut missing, &meta)?;
                *count.missing.entry(reason.into()).or_default() += 1;
            }
        }
    }
    recovered.flush()?;
    missing.flush()?;
    let report = json!({"games":counts});
    write_json(&out.join("report.json"), &report)?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn rejects_invalid_candidates_and_encodes_all_extensions() {
        let detector = Detector::builtin();
        let bytes = include_bytes!("../../card-placeholder/testdata/real-370861_numel.jpg");
        let real = image::load_from_memory(bytes).unwrap().resize_exact(
            420,
            600,
            image::imageops::FilterType::Triangle,
        );
        for (ext, format) in [
            ("jpg", ImageFormat::Jpeg),
            ("png", ImageFormat::Png),
            ("webp", ImageFormat::WebP),
        ] {
            let encoded = encode(&real, Path::new(&format!("face.{ext}"))).unwrap();
            assert_eq!(image::guess_format(&encoded).unwrap(), format);
            let decoded = verified(&encoded, &detector).unwrap();
            assert_eq!((decoded.width(), decoded.height()), (420, 600));
        }
        for (w, h) in [(100, 140), (400, 400), (200, 600)] {
            let invalid = real.resize_exact(w, h, image::imageops::FilterType::Triangle);
            assert!(
                verified(&encode(&invalid, Path::new("face.png")).unwrap(), &detector).is_none()
            );
        }
        let placeholder = image::load_from_memory(include_bytes!(
            "../../card-placeholder/testdata/placeholder-cardtrader.png"
        ))
        .unwrap()
        .resize_exact(372, 520, image::imageops::FilterType::Triangle);
        assert!(verified(
            &encode(&placeholder, Path::new("face.webp")).unwrap(),
            &detector
        )
        .is_none());
        assert!(verified(b"invalid image", &detector).is_none());
    }

    #[test]
    fn rejects_url_credentials_before_any_network_request() {
        let agent = ureq::AgentBuilder::new().redirects(0).build();
        assert!(
            Downloader::request(&agent, "https://user:secret@example.invalid/card.jpg")
                .unwrap_err()
                .to_string()
                .contains("credentials")
        );
        assert!(Downloader::request(&agent, "file:///tmp/card.jpg").is_err());
    }
}
