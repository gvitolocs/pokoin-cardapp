//! Little-endian f32 2-D `.npy` arrays, the layout of `cdn_milo/embeddings.npy`.

use std::fs;
use std::path::Path;

pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}

pub fn read(path: &Path) -> Result<Matrix, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{}: not a .npy file", path.display()));
    }
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
        v => return Err(format!("{}: npy version {v}", path.display())),
    };
    let header = std::str::from_utf8(&bytes[start..start + header_len]).map_err(|e| e.to_string())?;
    if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
        return Err(format!("{}: expected C-order <f4, got {header}", path.display()));
    }
    let shape = header.split("'shape':").nth(1).and_then(|s| s.split(')').next()).ok_or("npy without shape")?;
    let dims: Vec<usize> =
        shape.trim_start_matches([' ', '(']).split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let [rows, cols] = dims[..] else { return Err(format!("{}: shape {shape} is not 2-D", path.display())) };
    let body = &bytes[start + header_len..];
    if body.len() != rows * cols * 4 {
        return Err(format!("{}: {} data bytes for shape ({rows}, {cols})", path.display(), body.len()));
    }
    let data = body.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    Ok(Matrix { rows, cols, data })
}

pub fn write(path: &Path, m: &Matrix) -> Result<(), String> {
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}", m.rows, m.cols);
    // Version 1: magic(6) + version(2) + len(2) + header, padded with spaces to a multiple of 64, ending in \n.
    let pad = 64 - (10 + header.len() + 1) % 64;
    header.push_str(&" ".repeat(pad % 64));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + m.data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in &m.data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, out).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let dir = std::env::temp_dir().join(format!("npy-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.npy");
        let m = Matrix { rows: 3, cols: 2, data: vec![1.0, -2.0, 0.5, 0.25, 3.0, 4.0] };
        write(&path, &m).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!((10 + u16::from_le_bytes([bytes[8], bytes[9]]) as usize) % 64, 0);
        let back = read(&path).unwrap();
        assert_eq!((back.rows, back.cols, back.data), (3, 2, m.data));
        fs::remove_dir_all(dir).unwrap();
    }
}
