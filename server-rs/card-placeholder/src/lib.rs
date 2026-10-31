//! Placeholder faces: pictures that stand in for a card that has no image.
//!
//! Two kinds sit under real card keys in the CDN tree: CardTrader's grey card back
//! (`fallbacks/card_uploader/preview.png`, 186x260) and Pokoin's own missing-card art (the
//! coin mascot, 630x880, and `home/missing-card.webp`). Re-encodes (`_homepage.webp`) change
//! the bytes, so a face is matched against the references by a 64-bit difference hash and a
//! 32x32 grey thumbnail, not by its SHA-256 alone.

use image::imageops::FilterType;
use image::DynamicImage;

/// CardTrader's fallback size; pokoin-test `missing-card.js` treats it the same way.
pub const CT_PLACEHOLDER_SIZE: (u32, u32) = (186, 260);
/// Hamming distance on the difference hash still counted as the same picture.
const MAX_HASH_BITS: u32 = 8;
/// Mean absolute difference (0..255) of the 32x32 grey thumbnails still counted as the same picture.
const MAX_THUMB_DIFF: f32 = 14.0;

const BUILTIN_REFS: [(&str, &[u8]); 3] = [
    ("cardtrader-preview", include_bytes!("../refs/cardtrader-preview.png")),
    ("pokoin-mascot", include_bytes!("../refs/pokoin-mascot.jpg")),
    ("pokoin-missing-card", include_bytes!("../refs/pokoin-missing-card.webp")),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Real,
    /// `reason` is `cardtrader_size` or `matches_reference`; `reference` names the picture matched.
    Placeholder { reason: &'static str, reference: String },
    Undecodable,
}

impl Verdict {
    pub fn is_placeholder(&self) -> bool {
        matches!(self, Verdict::Placeholder { .. })
    }
}

struct Reference {
    name: String,
    hash: u64,
    thumb: Vec<u8>,
}

pub struct Detector {
    refs: Vec<Reference>,
}

/// 9x8 grey difference hash: bit set where a pixel is darker than its right neighbour.
pub fn dhash(img: &DynamicImage) -> u64 {
    let small = img.resize_exact(9, 8, FilterType::Triangle).to_luma8();
    let mut bits = 0u64;
    for y in 0..8 {
        for x in 0..8 {
            bits <<= 1;
            if small.get_pixel(x, y)[0] < small.get_pixel(x + 1, y)[0] {
                bits |= 1;
            }
        }
    }
    bits
}

fn thumb(img: &DynamicImage) -> Vec<u8> {
    img.resize_exact(32, 32, FilterType::Triangle).to_luma8().into_raw()
}

fn thumb_diff(a: &[u8], b: &[u8]) -> f32 {
    let sum: u32 = a.iter().zip(b).map(|(x, y)| u32::from(x.abs_diff(*y))).sum();
    sum as f32 / a.len() as f32
}

/// Flattens alpha onto white, as the site and the scanner see the picture.
fn decode(bytes: &[u8]) -> Option<DynamicImage> {
    let img = image::load_from_memory(bytes).ok()?;
    if !img.color().has_alpha() {
        return Some(img);
    }
    let mut rgba = img.to_rgba8();
    for p in rgba.pixels_mut() {
        let a = u16::from(p[3]);
        for c in 0..3 {
            p[c] = ((u16::from(p[c]) * a + 255 * (255 - a)) / 255) as u8;
        }
        p[3] = 255;
    }
    Some(DynamicImage::ImageRgba8(rgba))
}

impl Detector {
    pub fn builtin() -> Self {
        let mut d = Detector { refs: Vec::new() };
        for (name, bytes) in BUILTIN_REFS {
            d.add_reference(name, bytes).expect("built-in placeholder reference decodes");
        }
        d
    }

    pub fn add_reference(&mut self, name: &str, bytes: &[u8]) -> Result<(), String> {
        let img = decode(bytes).ok_or_else(|| format!("reference {name} does not decode"))?;
        self.refs.push(Reference { name: name.to_string(), hash: dhash(&img), thumb: thumb(&img) });
        Ok(())
    }

    pub fn classify(&self, bytes: &[u8]) -> Verdict {
        let Some(img) = decode(bytes) else { return Verdict::Undecodable };
        if (img.width(), img.height()) == CT_PLACEHOLDER_SIZE {
            return Verdict::Placeholder { reason: "cardtrader_size", reference: "cardtrader-preview".into() };
        }
        let (hash, th) = (dhash(&img), thumb(&img));
        for r in &self.refs {
            if (hash ^ r.hash).count_ones() <= MAX_HASH_BITS && thumb_diff(&th, &r.thumb) <= MAX_THUMB_DIFF {
                return Verdict::Placeholder { reason: "matches_reference", reference: r.name.clone() };
            }
        }
        Verdict::Real
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(name: &str) -> Vec<u8> {
        std::fs::read(format!("{}/testdata/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
    }

    #[test]
    fn cardtrader_back_is_a_placeholder() {
        let v = Detector::builtin().classify(&sample("placeholder-cardtrader.png"));
        assert!(v.is_placeholder(), "{v:?}");
    }

    #[test]
    fn reencoded_mascot_thumbnail_is_a_placeholder() {
        let v = Detector::builtin().classify(&sample("placeholder-mascot-homepage.webp"));
        assert_eq!(v, Verdict::Placeholder { reason: "matches_reference", reference: "pokoin-mascot".into() });
    }

    #[test]
    fn real_cards_are_not_placeholders() {
        let d = Detector::builtin();
        for name in ["real-370861_numel.jpg", "real-110398_numel-12-122-breakpoint.jpg"] {
            assert_eq!(d.classify(&sample(name)), Verdict::Real, "{name}");
        }
    }

    #[test]
    fn resized_cardtrader_back_still_matches() {
        let img = decode(&sample("placeholder-cardtrader.png")).unwrap().resize_exact(372, 520, FilterType::Triangle);
        let mut png = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
        assert!(Detector::builtin().classify(&png).is_placeholder());
    }

    #[test]
    fn garbage_is_undecodable() {
        assert_eq!(Detector::builtin().classify(b"not an image"), Verdict::Undecodable);
    }
}
