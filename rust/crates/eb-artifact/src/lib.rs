use eb_core::sha256_hex;
use std::fs;
use std::path::{Path, PathBuf};

/// Store bytes at `cas/<ab>/<sha256>` and return the path.
pub fn put(root: &Path, bytes: &[u8]) -> PathBuf {
    let digest = sha256_hex(bytes);
    let dir = root.join("cas").join(&digest[..2]);
    let _ = fs::create_dir_all(&dir);
    let p = dir.join(&digest);
    if !p.exists() {
        let _ = fs::write(&p, bytes);
    }
    p
}

pub fn get(root: &Path, sha256: &str) -> Option<PathBuf> {
    if sha256.len() < 2 {
        return None;
    }
    let p = root.join("cas").join(&sha256[..2]).join(sha256);
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

pub fn put_file(root: &Path, src: &Path) -> std::io::Result<(String, PathBuf, u64)> {
    let bytes = fs::read(src)?;
    let digest = sha256_hex(&bytes);
    let dest = put(root, &bytes);
    Ok((digest, dest, bytes.len() as u64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn put_get_roundtrip() {
        let dir = env::temp_dir().join(format!("eb-cas-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let p = put(&dir, b"hello-cas");
        let digest = p.file_name().unwrap().to_str().unwrap();
        assert_eq!(digest.len(), 64);
        assert!(get(&dir, digest).unwrap().exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
