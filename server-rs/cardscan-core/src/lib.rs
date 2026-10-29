//! Decision rules shared by the worker and the verifier. Floating-point adapters
//! are deliberately outside `verus!`; the integer decisions are the shipped code.
//!
//! Every function inside `verus!` is also compiled by plain `cargo` with the
//! specs and proofs erased, so the verified code is the code the worker ships.
#![allow(
    clippy::manual_clamp,
    clippy::assign_op_pattern,
    clippy::needless_range_loop,
    clippy::ptr_arg
)]
use vstd::prelude::*;

verus! {
#[cfg(verus_only)]
use vstd::arithmetic::div_mod::{lemma_div_by_multiple, lemma_div_is_ordered};

pub fn clamp_top_k(top_k: i64, live: bool) -> (k: usize)
    ensures 1 <= k <= 8, live ==> k <= 3,
        k == if top_k < 1 { 1 } else if top_k > (if live {3int} else {8int}) { if live {3int} else {8int} } else {top_k as int},
{
    let cap: i64 = if live { 3 } else { 8 };
    if top_k < 1 { 1 } else if top_k > cap { cap as usize } else { top_k as usize }
}

/// 1: tiny, 4: square, 2: portrait, 3: landscape.
pub fn orientation_kind(w: u32, h: u32) -> (r: u8)
    ensures 1 <= r <= 4,
        r == if w < 8 || h < 8 { 1u8 } else if 85 * h <= 100 * w && 100 * w <= 118 * h { 4u8 } else if w > h { 3u8 } else { 2u8 },
{
    if w < 8 || h < 8 { 1 }
    else if 85 * (h as u64) <= 100 * (w as u64) && 100 * (w as u64) <= 118 * (h as u64) { 4 }
    else if w > h { 3 } else { 2 }
}

pub fn orientation_count(kind: u8) -> (n: usize)
    requires 1 <= kind <= 4,
    ensures n == 1 || n == 2 || n == 4,
        n == if kind == 1 {1int} else if kind == 4 {4int} else {2int},
{ if kind == 1 { 1 } else if kind == 4 { 4 } else { 2 } }

/// The plan length is exactly the count proved above (1/2/4 orientations).
pub fn orientation_plan_len(w: u32, h: u32) -> (n: usize)
    ensures n == 1 || n == 2 || n == 4,
{ orientation_count(orientation_kind(w, h)) }

pub fn early_exit(best: i32) -> (r: bool)
    ensures r == (best >= 8200),
{ best >= 8200 }

pub fn replace_best(best: i32, candidate: i32) -> (r: bool)
    ensures r == (candidate > best),
{ candidate > best }

pub fn immediate(top1: Option<i32>, n_boxes: usize) -> (r: bool)
    ensures r == (n_boxes > 0 && top1 is Some && top1.unwrap() >= 6000),
{ match top1 { Some(s) => n_boxes > 0 && s >= 6000, None => false } }

/// Positive tenths to nearest pixel, ties even, with Python slice clamping.
pub fn crop_axis(tenth: i64, limit: u32) -> (r: u32)
    ensures r <= limit,
        tenth <= 0 ==> r == 0,
{
    if tenth <= 0 { return 0; }
    let q = tenth / 10;
    let rem = tenth % 10;
    let rounded = if rem > 5 || (rem == 5 && q % 2 == 1) { q + 1 } else { q };
    if rounded > limit as i64 { limit } else { rounded as u32 }
}

/// Integer inverse-coordinate clamp. The floating inverse transform supplies
/// rounded tenths; every returned coordinate is in the image domain.
pub fn inverse_clamp(value: i64, size: u32) -> (r: i64)
    ensures 0 <= r <= size as int * 10,
        0 <= value <= size as int * 10 ==> r == value,
{
    let high = size as i64 * 10;
    if value < 0 { 0 } else if value > high { high } else { value }
}

/// Verified adapter the worker calls after the float inverse letterbox math:
/// round to a tenth (Python `round(v, 1)`) and clamp into `[0, size]`.
pub fn inverse_axis_tenths(value: i64, size: u32) -> (r: i64)
    ensures 0 <= r <= size as int * 10,
{ inverse_clamp(value, size) }

pub fn scaled_dimension(d: u32, long: u32) -> (r: u32)
    requires d > 0, d <= long, long > 1600,
    ensures 1 <= r <= 1600,
        r == if (d as int * 1600) / (long as int) < 1 {1int} else {(d as int * 1600) / (long as int)},
{
    let n = (d as u64 * 1600) / long as u64;
    proof {
        lemma_div_is_ordered(d as int * 1600, long as int * 1600, long as int);
        lemma_div_by_multiple(1600, long as int);
    }
    assert((d as int * 1600) / (long as int) <= 1600);
    assert(n as int == (d as int * 1600) / (long as int));
    if n < 1 { 1 } else { n as u32 }
}

/// Catalog membership is computed at load; unknown/alias ids route to all.
pub fn route_rows(member: &Vec<bool>, uncatalogued: &Vec<bool>, all: bool) -> (out: Vec<usize>)
    requires member.len() == uncatalogued.len(),
    ensures forall|j:int| 0 <= j < out.len() ==> out[j] < member.len(),
{
    let mut out: Vec<usize> = Vec::new();
    let mut i = 0usize;
    while i < member.len()
        invariant i <= member.len(), member.len() == uncatalogued.len(),
            forall|j:int| 0 <= j < out.len() ==> out[j] < i,
        decreases member.len() - i,
    {
        if all || member[i] || uncatalogued[i] { out.push(i); }
        i = i + 1;
    }
    out
}

pub open spec fn better(s: Seq<i64>, a: int, b: int) -> bool {
    s[a] > s[b] || (s[a] == s[b] && a < b)
}
pub fn ranks_before(scores: &Vec<i64>, a: usize, b: usize) -> (r: bool)
    requires a < scores.len(), b < scores.len(),
    ensures r == better(scores@, a as int, b as int),
{ scores[a] > scores[b] || (scores[a] == scores[b] && a < b) }

/// Repeated maximum selection, O(k*n), with a strict rank cutoff. Scores are
/// order-preserving integer keys of f32 values, not rounded score buckets.
pub fn top_k(scores: &Vec<i64>, k: usize) -> (out: Vec<usize>)
    requires k <= scores.len(),
    ensures out.len() <= k,
        forall|j:int| 0 <= j < out.len() ==> out[j] < scores.len(),
        forall|a:int,b:int| 0 <= a < b < out.len() ==> better(scores@, out[a] as int, out[b] as int),
{
    let mut out: Vec<usize> = Vec::new();
    let mut pass = 0usize;
    while pass < k
        invariant pass == out.len(), pass <= k <= scores.len(),
            forall|j:int| 0 <= j < out.len() ==> out[j] < scores.len(),
            forall|a:int,b:int| 0 <= a < b < out.len() ==> better(scores@, out[a] as int, out[b] as int),
        decreases k - pass,
    {
        let mut best: Option<usize> = None;
        let mut i = 0usize;
        while i < scores.len()
            invariant i <= scores.len(),
                forall|j:int| 0 <= j < out.len() ==> out[j] < scores.len(),
                forall|a:int,b:int| 0 <= a < b < out.len() ==> better(scores@, out[a] as int, out[b] as int),
                best is Some ==> best.unwrap() < scores.len(),
                best is Some ==> forall|j:int| 0 <= j < out.len() ==> better(scores@, out[j] as int, best.unwrap() as int),
            decreases scores.len() - i,
        {
            let mut eligible = true;
            let mut j = 0usize;
            while j < out.len()
                invariant j <= out.len(), i < scores.len(),
                    forall|t:int| 0 <= t < out.len() ==> out[t] < scores.len(),
                    eligible ==> forall|t:int| 0 <= t < j ==> better(scores@, out[t] as int, i as int),
                decreases out.len() - j,
            {
                if !ranks_before(scores, out[j], i) { eligible = false; }
                j = j + 1;
            }
            if eligible {
                match best {
                    None => { best = Some(i); },
                    Some(b) => { if ranks_before(scores, i, b) { best = Some(i); } },
                }
            }
            i = i + 1;
        }
        match best {
            None => { return out; },
            Some(b) => { out.push(b); },
        }
        pass = pass + 1;
    }
    out
}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rot {
    R0,
    R90,
    R180,
    R270,
}
pub fn orientation_plan(w: u32, h: u32) -> &'static [Rot] {
    use Rot::*;
    match orientation_kind(w, h) {
        1 => &[R0],
        4 => &[R0, R90, R180, R270],
        3 => &[R90, R270],
        _ => &[R0, R180],
    }
}

pub const CATALOGS: [&str; 3] = ["pokemon_western", "pokemon_japanese", "pokemon_chinese"];
pub fn catalog_id(request: Option<&str>) -> (&'static str, bool) {
    match request {
        None | Some("" | "all" | "pokemon") => ("all", false),
        Some(s) => CATALOGS
            .iter()
            .copied()
            .find(|c| *c == s)
            .map_or(("all", true), |c| (c, false)),
    }
}

/// Decimal rounding adapter, with ties-to-even for box tenths and score units.
pub fn rounded(v: f64, digits: i32) -> f64 {
    let scale = 10f64.powi(digits);
    (v * scale).round_ties_even() / scale
}
pub fn score_units(s: f64) -> i32 {
    (s * 10000.0).round_ties_even() as i32
}
pub fn score_key(s: f32) -> i64 {
    // All non-finite dot products sort last; signed zero ties by row index.
    if !s.is_finite() {
        return i64::MIN;
    }
    if s == 0.0 {
        return 0x8000_0000;
    }
    let b = s.to_bits();
    if b & 0x8000_0000 != 0 {
        (!b) as i64
    } else {
        (b ^ 0x8000_0000) as i64
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
}
impl Rect {
    pub fn is_empty(self) -> bool {
        self.x2 <= self.x1 || self.y2 <= self.y1
    }
}
pub fn crop_rect(xyxy: [f64; 4], w: u32, h: u32) -> Rect {
    let t = xyxy.map(|x| (rounded(x, 1) * 10.0).round_ties_even() as i64);
    Rect {
        x1: crop_axis(t[0], w),
        y1: crop_axis(t[1], h),
        x2: crop_axis(t[2], w),
        y2: crop_axis(t[3], h),
    }
}
pub fn long_edge_size(w: u32, h: u32) -> (u32, u32) {
    let long = w.max(h);
    if long <= 1600 || w == 0 || h == 0 {
        return (w, h);
    }
    (scaled_dimension(w, long), scaled_dimension(h, long))
}
#[derive(Debug, Clone, Copy)]
pub struct Letterbox {
    pub scale: f64,
    pub nw: u32,
    pub nh: u32,
    pub left: u32,
    pub top: u32,
}
pub fn letterbox(w: u32, h: u32) -> Letterbox {
    let scale = 640.0 / w.max(h).max(1) as f64;
    let nw = (w as f64 * scale).round_ties_even().max(1.0) as u32;
    let nh = (h as f64 * scale).round_ties_even().max(1.0) as u32;
    Letterbox {
        scale,
        nw,
        nh,
        left: (640 - nw) / 2,
        top: (640 - nh) / 2,
    }
}
/// Round a raw inverse coordinate (already divided by the letterbox scale) to
/// tenths and clamp it into the image; the clamp is the verified function.
pub fn inverse_tenths(value: f64, size: u32) -> i64 {
    if !value.is_finite() {
        return if value > 0.0 { size as i64 * 10 } else { 0 };
    }
    inverse_axis_tenths((value * 10.0).round_ties_even() as i64, size)
}
pub fn inverse_box(b: [f32; 4], lb: Letterbox, w: u32, h: u32) -> [f64; 4] {
    std::array::from_fn(|i| {
        let pad = if i % 2 == 0 { lb.left } else { lb.top };
        let size = if i % 2 == 0 { w } else { h };
        // numpy keeps this in f32: float32 scalar with weak Python scalars.
        let raw = ((b[i] - pad as f32) / lb.scale as f32) as f64;
        inverse_tenths(raw, size) as f64 / 10.0
    })
}
pub fn accept_box(b: [f64; 4], conf: f32, w: u32, h: u32) -> bool {
    conf >= 0.25
        && b.iter().all(|x| x.is_finite())
        && (b[2] - b[0]) * (b[3] - b[1]) >= 0.01 * w as f64 * h as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn limits() {
        for live in [false, true] {
            for n in [i64::MIN, 0, 1, 3, 8, i64::MAX] {
                assert_eq!(
                    clamp_top_k(n, live),
                    n.clamp(1, if live { 3 } else { 8 }) as usize
                );
            }
        }
    }
    #[test]
    fn orientations() {
        use Rot::*;
        assert_eq!(orientation_plan(7, 100), &[R0]);
        for w in [85, 118] {
            assert_eq!(orientation_plan(w, 100), &[R0, R90, R180, R270]);
        }
        assert_eq!(orientation_plan(84, 100), &[R0, R180]);
        assert_eq!(orientation_plan(119, 100), &[R90, R270]);
        for w in [1u32, 7, 8, 84, 85, 100, 118, 119, 200] {
            for h in [1u32, 7, 8, 100, 200] {
                assert_eq!(orientation_plan(w, h).len(), orientation_plan_len(w, h));
            }
        }
        for k in 1..=4 {
            assert!([1, 2, 4].contains(&orientation_count(k)));
        }
    }
    #[test]
    fn thresholds() {
        assert!(!early_exit(8199));
        assert!(early_exit(8200));
        assert!(!replace_best(8200, 8200));
        assert!(replace_best(1, 2));
        assert!(!immediate(None, 1));
        assert!(!immediate(Some(6000), 0));
        assert!(!immediate(Some(5999), 1));
        assert!(immediate(Some(6000), 1));
    }
    #[test]
    fn crops() {
        assert_eq!(crop_axis(125, 100), 12);
        assert_eq!(crop_axis(135, 100), 14);
        assert_eq!(
            crop_rect([-3.0, 12.5, 200.0, 13.5], 100, 100),
            Rect {
                x1: 0,
                y1: 12,
                x2: 100,
                y2: 14
            }
        );
        assert!(crop_rect([100., 0., 200., 1.], 100, 100).is_empty());
        assert!(crop_rect([20., 0., 10., 1.], 100, 100).is_empty());
        assert_eq!(crop_axis(i64::MAX, 100), 100);
        assert_eq!(inverse_clamp(-1, 10), 0);
        assert_eq!(inverse_clamp(101, 10), 100);
    }
    #[test]
    fn geometry() {
        assert_eq!(long_edge_size(2000, 1001), (1600, 800));
        assert_eq!(long_edge_size(1, 4000), (1, 1600));
        assert_eq!(long_edge_size(1600, 20), (1600, 20));
        assert_eq!(scaled_dimension(1001, 2000), 800);
        let lb = letterbox(1000, 500);
        assert_eq!((lb.nw, lb.nh, lb.left, lb.top), (640, 320, 0, 160));
        assert_eq!(
            inverse_box([-10., 0., 650., 640.], lb, 1000, 500),
            [0., 0., 1000., 500.]
        );
        assert_eq!(inverse_tenths(-1.0, 100), 0);
        assert_eq!(inverse_tenths(1000.04, 1000), 10000);
        assert!(accept_box([0., 0., 10., 10.], 0.25, 100, 100));
        assert!(!accept_box([0., 0., 9., 10.], 0.9, 100, 100));
        assert!(!accept_box([0., 0., 100., 100.], 0.249, 100, 100));
        assert_eq!(letterbox(256, 1).nh, 2); // 2.5 -> even
    }
    #[test]
    fn selection() {
        assert_eq!(top_k(&vec![1, 3, 3, -2], 3), vec![1, 2, 0]);
        assert!(top_k(&vec![], 0).is_empty());
        assert_eq!(top_k(&vec![4, 4, 4], 3), vec![0, 1, 2]);
        let floats = [-1., -0.2, -0., 0., 0.00001, 0.5, 1.];
        assert!(floats
            .windows(2)
            .all(|s| score_key(s[0]) <= score_key(s[1])));
        assert_eq!(score_units(0.82), 8200);
    }
    #[test]
    fn routing() {
        let m = vec![true, false, false];
        let u = vec![false, true, false];
        assert_eq!(route_rows(&m, &u, false), vec![0, 1]);
        assert_eq!(route_rows(&m, &u, true), vec![0, 1, 2]);
        for a in [None, Some(""), Some("all"), Some("pokemon")] {
            assert_eq!(catalog_id(a), ("all", false));
        }
        assert_eq!(
            catalog_id(Some("pokemon_western")),
            ("pokemon_western", false)
        );
        assert_eq!(catalog_id(Some("typo")), ("all", true));
    }
}
