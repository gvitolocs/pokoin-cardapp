use cardscan_core::Rot;
use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rgb {
    pub w: u32,
    pub h: u32,
    pub data: Vec<u8>,
}

impl Rgb {
    pub fn new(w: u32, h: u32, data: Vec<u8>) -> Self {
        Self { w, h, data }
    }

    pub fn crop(&self, x1: u32, y1: u32, x2: u32, y2: u32) -> Rgb {
        let cw = x2.saturating_sub(x1);
        let ch = y2.saturating_sub(y1);
        let mut data = Vec::with_capacity((cw * ch * 3) as usize);
        for y in y1..y2 {
            let start = ((y * self.w + x1) * 3) as usize;
            let end = start + (cw * 3) as usize;
            data.extend_from_slice(&self.data[start..end]);
        }
        Rgb::new(cw, ch, data)
    }
}

pub fn decode(blob: &[u8]) -> Option<Rgb> {
    let img = image::load_from_memory(blob).ok()?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Some(Rgb::new(w, h, rgb.into_raw()))
}

fn resize(src: &Rgb, w: u32, h: u32) -> Rgb {
    if src.w == w && src.h == h {
        return src.clone();
    }
    let src_img = Image::from_vec_u8(src.w, src.h, src.data.clone(), PixelType::U8x3)
        .expect("source buffer matches dimensions");
    let mut dst = Image::new(w, h, PixelType::U8x3);
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
    Resizer::new()
        .resize(&src_img, &mut dst, &options)
        .expect("bilinear resize");
    Rgb::new(w, h, dst.into_vec())
}

pub fn resize_bilinear(src: &Rgb, w: u32, h: u32) -> Rgb {
    resize(src, w, h)
}

/// Pillow `np.rot90(k=-1)` repeated: clockwise quarter turns.
pub fn rotate(src: &Rgb, rot: Rot) -> Rgb {
    match rot {
        Rot::R0 => src.clone(),
        Rot::R90 => rotate_cw(src),
        Rot::R180 => rotate_cw(&rotate_cw(src)),
        Rot::R270 => rotate_cw(&rotate_cw(&rotate_cw(src))),
    }
}

fn rotate_cw(src: &Rgb) -> Rgb {
    let (w, h) = (src.w, src.h);
    let mut data = vec![0u8; src.data.len()];
    for i in 0..w {
        for j in 0..h {
            let si = ((h - 1 - j) * w + i) as usize * 3;
            let di = (i * h + j) as usize * 3;
            data[di..di + 3].copy_from_slice(&src.data[si..si + 3]);
        }
    }
    Rgb::new(h, w, data)
}

/// 640x640 RGB canvas with the letterboxed image at (left, top), pad 114.
pub fn letterbox_canvas(src: &Rgb, nw: u32, nh: u32, left: u32, top: u32) -> Vec<u8> {
    let scaled = resize(src, nw, nh);
    let mut canvas = vec![114u8; 640 * 640 * 3];
    for y in 0..nh {
        let dst = (((top + y) * 640 + left) * 3) as usize;
        let src_off = (y * nw * 3) as usize;
        canvas[dst..dst + (nw * 3) as usize]
            .copy_from_slice(&scaled.data[src_off..src_off + (nw * 3) as usize]);
    }
    canvas
}

pub fn nchw_640(canvas: &[u8]) -> Vec<f32> {
    let plane = 640 * 640;
    let mut out = vec![0f32; plane * 3];
    for (i, px) in canvas.as_chunks::<3>().0.iter().enumerate() {
        out[i] = px[0] as f32 / 255.0;
        out[plane + i] = px[1] as f32 / 255.0;
        out[2 * plane + i] = px[2] as f32 / 255.0;
    }
    out
}

pub const MILO_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const MILO_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// 1x3x448x448, /255 then ImageNet mean/std, Pillow-compatible bilinear.
pub fn milo_input(src: &Rgb) -> Vec<f32> {
    let resized = resize(src, 448, 448);
    let plane = 448 * 448;
    let mut out = vec![0f32; plane * 3];
    for (i, px) in resized.data.as_chunks::<3>().0.iter().enumerate() {
        for c in 0..3 {
            out[c * plane + i] = (px[c] as f32 / 255.0 - MILO_MEAN[c]) / MILO_STD[c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Rgb {
        // 2x3 image: rows [1,2,3] and [4,5,6].
        Rgb::new(
            3,
            2,
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            ],
        )
    }

    #[test]
    fn clockwise_rotation_matches_numpy() {
        let r = rotate(&sample(), Rot::R90);
        assert_eq!((r.w, r.h), (2, 3));
        assert_eq!(r.data[..3], [10, 11, 12]);
        assert_eq!(r.data[3..6], [1, 2, 3]);
        let r = rotate(&sample(), Rot::R180);
        assert_eq!((r.w, r.h), (3, 2));
        assert_eq!(r.data[..3], [16, 17, 18]);
    }

    #[test]
    fn crop_uses_slice_semantics() {
        let c = sample().crop(1, 1, 3, 2);
        assert_eq!((c.w, c.h), (2, 1));
        assert_eq!(c.data, vec![13, 14, 15, 16, 17, 18]);
    }
}
