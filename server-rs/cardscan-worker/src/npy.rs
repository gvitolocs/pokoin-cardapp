//! Minimal reader for the little-endian float32 `.npy` file the index ships.
//! Handles version 1.0/2.0 headers, C order only, exactly what `np.save` writes.

use std::fs;
use std::path::Path;

pub struct NpyF32 {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

pub fn read_f32_2d(path: &Path) -> Result<NpyF32, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{}: not an .npy file", path.display()));
    }
    let major = bytes[6];
    let (header_start, header_len): (usize, usize) = match major {
        1 => (10, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(format!("{}: truncated .npy header", path.display()));
            }
            let n = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (12, n)
        }
        _ => {
            return Err(format!(
                "{}: unsupported .npy version {major}",
                path.display()
            ))
        }
    };
    let header_end = header_start
        .checked_add(header_len)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| format!("{}: truncated .npy header", path.display()))?;
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|_| format!("{}: invalid .npy header", path.display()))?;
    if !header.contains("'fortran_order': False") && !header.contains("\"fortran_order\": False") {
        return Err(format!(
            "{}: Fortran order is not supported",
            path.display()
        ));
    }
    let descr = quoted_after(header, "descr")
        .ok_or_else(|| format!("{}: missing descr", path.display()))?;
    if !matches!(descr.as_str(), "<f4" | "|f4" | "=f4") {
        return Err(format!("{}: expected float32, got {descr}", path.display()));
    }
    let shape = header
        .split_once("shape")
        .and_then(|(_, rest)| rest.split_once('('))
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(inner, _)| inner)
        .ok_or_else(|| format!("{}: missing shape", path.display()))?;
    let dims: Vec<usize> = shape
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    if dims.len() != 2 {
        return Err(format!(
            "{}: expected a 2-D array, got {dims:?}",
            path.display()
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| format!("{}: shape overflow", path.display()))?;
    let body = &bytes[header_end..];
    if body.len() != count * 4 {
        return Err(format!(
            "{}: {} bytes of data for {rows}x{cols} float32",
            path.display(),
            body.len()
        ));
    }
    let data = body
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(NpyF32 { rows, cols, data })
}

fn quoted_after(header: &str, key: &str) -> Option<String> {
    let (_, rest) = header.split_once(key)?;
    let (_, rest) = rest.split_once(':')?;
    let (_, rest) = rest.split_once('\'')?;
    let (value, _) = rest.split_once('\'')?;
    Some(value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn reads_a_numpy_header() {
        let dir = std::env::temp_dir().join(format!("cardscan-npy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.npy");
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2), }";
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(b"\x93NUMPY\x01\x00").unwrap();
        file.write_all(&(header.len() as u16).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(file);
        let npy = read_f32_2d(&path).unwrap();
        assert_eq!((npy.rows, npy.cols), (2, 2));
        assert_eq!(npy.data, vec![1.0, 2.0, 3.0, 4.0]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
