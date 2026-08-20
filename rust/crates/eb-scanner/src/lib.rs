use eb_core::sha256_hex;
use serde_json::{json, Value};
use sha2::Digest;
use std::fs;
use std::io::Read;
use std::path::Path;
use walkdir::WalkDir;

fn sniff_kind(path: &Path) -> &'static str {
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
    if ext == "mp4" || ext == "webm" || ext == "mkv" {
        "video"
    } else if ext == "parquet" {
        "parquet"
    } else if ext == "jsonl" || ext == "ndjson" || name.contains("ego") {
        "ego_jsonl"
    } else if ext == "json" {
        "json"
    } else {
        "file"
    }
}

fn file_sha256(path: &Path) -> std::io::Result<String> {
    let mut f = fs::File::open(path)?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = [0u8; 1024 * 64];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Directory merkle: sort (relpath, file_sha256) then hash the concatenation.
pub fn content_hash(path: &str) -> String {
    let root = Path::new(path);
    if !root.exists() {
        return sha256_hex(b"");
    }
    let mut recs: Vec<(String, String)> = Vec::new();
    for ent in WalkDir::new(root).into_iter().flatten() {
        if !ent.file_type().is_file() {
            continue;
        }
        let rel = ent
            .path()
            .strip_prefix(root)
            .unwrap_or(ent.path())
            .to_string_lossy()
            .to_string();
        let digest = file_sha256(ent.path()).unwrap_or_default();
        recs.push((rel, digest));
    }
    recs.sort();
    let mut joined = String::new();
    for (rel, digest) in recs {
        joined.push_str(&rel);
        joined.push('\n');
        joined.push_str(&digest);
        joined.push('\n');
    }
    sha256_hex(joined.as_bytes())
}

pub fn scan(path: &str) -> Value {
    let mut files = 0u64;
    let mut bytes = 0u64;
    let mut samples = vec![];
    let mut kinds = std::collections::BTreeMap::<&str, u64>::new();
    let mut rejected: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    let root = Path::new(path);
    if root.exists() {
        for ent in WalkDir::new(path).max_depth(12).into_iter().flatten() {
            if !ent.file_type().is_file() {
                continue;
            }
            files += 1;
            let sz = ent.metadata().map(|m| m.len()).unwrap_or(0);
            bytes += sz;
            let kind = sniff_kind(ent.path());
            *kinds.entry(kind).or_default() += 1;
            if sz == 0 {
                *rejected.entry("empty_file".into()).or_default() += 1;
            }
            if samples.len() < 40 {
                let digest = file_sha256(ent.path()).unwrap_or_default();
                samples.push(json!({
                    "path": ent.path().display().to_string(),
                    "size": sz,
                    "kind": kind,
                    "sha256": digest,
                }));
            }
        }
    }
    let hash = content_hash(path);
    json!({
        "path": path,
        "files": files,
        "bytes": bytes,
        "content_hash": hash,
        "kinds": kinds,
        "samples": samples,
        "funnel": {
            "source": {"n": files, "rejected": []},
            "raw": {"n": files, "rejected": rejected.iter().map(|(reason, count)| json!({"reason": reason, "count": count})).collect::<Vec<_>>()},
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn merkle_hash_stable() {
        let dir = env::temp_dir().join(format!("eb-scan-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("a")).unwrap();
        fs::write(dir.join("a/x.jsonl"), b"{\"t\":1}\n").unwrap();
        fs::write(dir.join("b.mp4"), b"fake").unwrap();
        let h1 = content_hash(dir.to_str().unwrap());
        let h2 = content_hash(dir.to_str().unwrap());
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
        let v = scan(dir.to_str().unwrap());
        assert_eq!(v["files"], 2);
        let _ = fs::remove_dir_all(&dir);
    }
}
