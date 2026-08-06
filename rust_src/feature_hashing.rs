// Feature hashing (the "hashing trick").
//
// Hashes each feature's UTF-8 token with MurmurHash3 (x86 32-bit, signed) — the
// same hash scikit-learn's `FeatureHasher` uses — and folds it into a bucket in
// `[0, n_features)`. With `alternate_sign` the hash's sign bit decides the sign
// of the contributed value, so collisions tend to cancel rather than accumulate
// (Weinberger et al., 2009). Doing the whole transform in one Rust call avoids a
// per-feature Python loop, an `isinstance` check, and a `Counter` build.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use pyo3::IntoPyObjectExt;

/// MurmurHash3 x86_32 — the canonical 32-bit variant (Austin Appleby, public
/// domain). Returns the raw `u32`; callers reinterpret it as `i32` when they
/// need the sign bit.
fn murmurhash3_x86_32(data: &[u8], seed: u32) -> u32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;
    let mut h1 = seed;

    let nblocks = data.len() / 4;
    for i in 0..nblocks {
        let j = i * 4;
        let mut k1 = u32::from_le_bytes([data[j], data[j + 1], data[j + 2], data[j + 3]]);
        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(C2);
        h1 ^= k1;
        h1 = h1.rotate_left(13);
        h1 = h1.wrapping_mul(5).wrapping_add(0xe654_6b64);
    }

    // Tail (the canonical fall-through switch on `len & 3`).
    let tail = &data[nblocks * 4..];
    let mut k1: u32 = 0;
    if tail.len() == 3 {
        k1 ^= (tail[2] as u32) << 16;
    }
    if tail.len() >= 2 {
        k1 ^= (tail[1] as u32) << 8;
    }
    if !tail.is_empty() {
        k1 ^= tail[0] as u32;
        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(C2);
        h1 ^= k1;
    }

    // Finalization.
    h1 ^= data.len() as u32;
    h1 ^= h1 >> 16;
    h1 = h1.wrapping_mul(0x85eb_ca6b);
    h1 ^= h1 >> 13;
    h1 = h1.wrapping_mul(0xc2b2_ae35);
    h1 ^= h1 >> 16;
    h1
}

/// Apply the hashing trick to a single example.
///
/// Mirrors the previous Python implementation's feature encoding: a string
/// value `v` under key `k` is hashed as the token `"k=v"` and contributes `1`;
/// any other value is hashed under the token `"k"` and contributes the value
/// itself. Contributions are summed per bucket. The accumulator preserves the
/// Python value types (int stays int, float stays float).
#[pyfunction]
pub fn feature_hash<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyDict>,
    n_features: i64,
    seed: u32,
    alternate_sign: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let res = PyDict::new(py);
    let neg_one = (-1i64).into_bound_py_any(py)?;

    for (key, value) in x.iter() {
        let is_str = value.is_instance_of::<PyString>();
        let token: String = if is_str {
            let v = value.cast::<PyString>()?;
            format!("{}={}", key.str()?.to_cow()?, v.to_cow()?)
        } else {
            key.str()?.to_cow()?.into_owned()
        };

        let h = murmurhash3_x86_32(token.as_bytes(), seed) as i32;
        let idx = (h as i64).rem_euclid(n_features);

        // String features contribute 1; numeric features contribute their value.
        let val: Bound<'py, PyAny> = if is_str {
            1i64.into_bound_py_any(py)?
        } else {
            value.clone()
        };
        // `alternate_sign` flips the contribution when the signed hash is negative.
        let val = if alternate_sign && h < 0 {
            val.mul(&neg_one)?
        } else {
            val
        };

        let key_obj = idx.into_bound_py_any(py)?;
        match res.get_item(&key_obj)? {
            Some(existing) => res.set_item(key_obj, existing.add(val)?)?,
            None => res.set_item(key_obj, val)?,
        }
    }

    Ok(res)
}
