use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::Path;

// Explicit code-only exclusions; datasets/checkpoints must be versioned separately.
const EXCLUDED: &[&str] = &[
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    ".claude",
    ".agents",
    "CLAUDE.md",
    "node_modules",
    ".cache",
];
// Bare "venv" can also be an importable source package, e.g. rlinf.envs.venv.
const ROOT_OUTPUTS: &[&str] = &["target", "logs", "outputs", "checkpoints", "datasets", "venv"];
// Model payloads are separate versioned inputs, never part of a code snapshot.
const WEIGHT_EXTENSIONS: &[&str] = &[
    "safetensors",
    "pt",
    "pth",
    "ckpt",
    "onnx",
    "gguf",
    "msgpack",
    "h5",
];
const WEIGHT_BIN_PREFIXES: &[&str] = &["pytorch_model", "model", "adapter_model"];

#[derive(Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub path: String,
    pub content_hash: String,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Entry {
    path: String,
    sha256: Option<String>,
    executable: bool,
}

fn manifest_bytes(entries: &[Entry]) -> Result<Vec<u8>, String> {
    serde_json::to_vec(&serde_json::json!({
        "version": 2,
        "entries": entries,
        "exclusions": {
            "directories": EXCLUDED,
            "root_outputs": ROOT_OUTPUTS,
            "weight_extensions": WEIGHT_EXTENSIONS,
            "weight_bin_prefixes": WEIGHT_BIN_PREFIXES,
            "weight_extension_case": "ascii_case_insensitive",
        },
    }))
    .map_err(|e| e.to_string())
}

fn entries(
    root: &Path,
    dir: &Path,
    dest: Option<&Path>,
    result: &mut Vec<Entry>,
) -> Result<(), String> {
    let mut children = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    children.sort_by_key(|e| e.file_name());
    for entry in children {
        if EXCLUDED.iter().any(|name| entry.file_name() == *name) {
            continue;
        }
        if dir == root && ROOT_OUTPUTS.iter().any(|name| entry.file_name() == *name) {
            continue;
        }
        let path = entry.path();
        let rel = path.strip_prefix(root).map_err(|e| e.to_string())?;
        let name = rel.to_str().ok_or("non-UTF8 snapshot path")?.to_owned();
        let meta = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
        // ponytail: reject all symlinks; add explicit versioned asset imports when required.
        if meta.file_type().is_symlink() {
            return Err(format!("snapshot symlink forbidden: {name}"));
        }
        if meta.is_file()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| {
                    WEIGHT_EXTENSIONS
                        .iter()
                        .any(|excluded| ext.eq_ignore_ascii_case(excluded))
                        || (ext.eq_ignore_ascii_case("bin")
                            && path.file_name().and_then(|name| name.to_str()).is_some_and(
                                |name| {
                                    let name = name.to_ascii_lowercase();
                                    WEIGHT_BIN_PREFIXES
                                        .iter()
                                        .any(|prefix| name.starts_with(prefix))
                                },
                            ))
                })
        {
            continue;
        }
        if !fs::canonicalize(&path)
            .map_err(|e| e.to_string())?
            .starts_with(root)
        {
            return Err(format!("snapshot path escape: {name}"));
        }
        if meta.is_dir() {
            if let Some(out) = dest {
                fs::create_dir(out.join(rel)).map_err(|e| e.to_string())?;
            }
            result.push(Entry {
                path: name,
                sha256: None,
                executable: false,
            });
            entries(root, &path, dest, result)?;
        } else if meta.is_file() {
            let mut input = OpenOptions::new()
                .read(true)
                .custom_flags(0x20000)
                .open(&path)
                .map_err(|e| e.to_string())?; // Linux O_NOFOLLOW
            let opened = fs::canonicalize(format!("/proc/self/fd/{}", input.as_raw_fd()))
                .map_err(|e| e.to_string())?;
            if !opened.starts_with(root) {
                return Err("snapshot opened path escaped source".into());
            }
            let before = input.metadata().map_err(|e| e.to_string())?;
            let executable = before.permissions().mode() & 0o111 != 0;
            let mut output = dest
                .map(|out| File::create(out.join(rel)))
                .transpose()
                .map_err(|e| e.to_string())?;
            let mut hash = Sha256::new();
            let mut buf = [0u8; 65536];
            loop {
                let n = input.read(&mut buf).map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                hash.update(&buf[..n]);
                if let Some(out) = output.as_mut() {
                    out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
                }
            }
            let after = input.metadata().map_err(|e| e.to_string())?;
            if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
                return Err(format!("source changed during snapshot: {name}"));
            }
            if let Some(out) = output {
                out.set_permissions(fs::Permissions::from_mode(if executable {
                    0o555
                } else {
                    0o444
                }))
                .map_err(|e| e.to_string())?;
            }
            result.push(Entry {
                path: name,
                sha256: Some(format!("{:x}", hash.finalize())),
                executable,
            });
        } else {
            return Err(format!("special file forbidden: {name}"));
        }
    }
    Ok(())
}

pub fn prepare_snapshot(
    source: &Path,
    data_root: &Path,
    expected: Option<&str>,
) -> Result<Snapshot, String> {
    if let Some(value) = expected {
        if !value.starts_with("sha256:")
            || value.len() != 71
            || !value[7..].bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err("Git revision selection is unavailable; omit commit or supply sha256:<content hash>".into());
        }
    }
    let source = fs::canonicalize(source).map_err(|e| e.to_string())?;
    let snapshots = data_root.join("snapshots");
    fs::create_dir_all(&snapshots).map_err(|e| e.to_string())?;
    let snapshots = fs::canonicalize(snapshots).map_err(|e| e.to_string())?;
    if snapshots.starts_with(&source) {
        return Err("snapshot destination is inside source".into());
    }
    if let Some(hash) = expected {
        if source == snapshots.join(&hash[7..]).join("code") {
            let mut manifest = Vec::new();
            entries(&source, &source, None, &mut manifest)?;
            let bytes = manifest_bytes(&manifest)?;
            if format!("sha256:{:x}", Sha256::digest(&bytes)) != hash {
                return Err("owned snapshot content corrupted".into());
            }
            return Ok(Snapshot {
                path: source.to_string_lossy().into_owned(),
                content_hash: hash.into(),
            });
        }
    }
    let tmp = snapshots.join(format!(
        ".tmp-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos()
    ));
    fs::create_dir(&tmp).map_err(|e| e.to_string())?;
    let result = (|| {
        fs::create_dir(tmp.join("code")).map_err(|e| e.to_string())?;
        let mut manifest = Vec::new();
        entries(&source, &source, Some(&tmp.join("code")), &mut manifest)?;
        let mut verify = Vec::new();
        entries(&source, &source, None, &mut verify)?;
        if manifest != verify {
            return Err("source changed during snapshot; retry".into());
        }
        let bytes = manifest_bytes(&manifest)?;
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let content_hash = format!("sha256:{hash}");
        if expected.is_some_and(|value| value != content_hash) {
            return Err("source content does not match pinned snapshot".into());
        }
        let target = snapshots.join(&hash);
        if target.exists() {
            let mut existing = Vec::new();
            let code = fs::canonicalize(target.join("code")).map_err(|e| e.to_string())?;
            if code != target.join("code") {
                return Err("snapshot symlink forbidden".into());
            }
            entries(&code, &code, None, &mut existing)?;
            if existing != manifest {
                return Err("owned snapshot content corrupted".into());
            }
        } else {
            fs::write(tmp.join("manifest.json"), bytes).map_err(|e| e.to_string())?;
            fs::set_permissions(tmp.join("manifest.json"), fs::Permissions::from_mode(0o444))
                .map_err(|e| e.to_string())?;
            for item in manifest.iter().filter(|entry| entry.sha256.is_none()).rev() {
                fs::set_permissions(
                    tmp.join("code").join(&item.path),
                    fs::Permissions::from_mode(0o555),
                )
                .map_err(|e| e.to_string())?;
            }
            fs::set_permissions(tmp.join("code"), fs::Permissions::from_mode(0o555))
                .map_err(|e| e.to_string())?;
            fs::set_permissions(&tmp, fs::Permissions::from_mode(0o555))
                .map_err(|e| e.to_string())?;
            if let Err(e) = fs::rename(&tmp, &target) {
                let _ = fs::set_permissions(&tmp, fs::Permissions::from_mode(0o755));
                let _ = fs::set_permissions(tmp.join("code"), fs::Permissions::from_mode(0o755));
                for item in manifest.iter().filter(|entry| entry.sha256.is_none()) {
                    let _ = fs::set_permissions(
                        tmp.join("code").join(&item.path),
                        fs::Permissions::from_mode(0o755),
                    );
                }
                return Err(format!(
                    "snapshot publish failed (retry if concurrent): {e}"
                ));
            }
        }
        Ok(Snapshot {
            path: target.join("code").to_string_lossy().into_owned(),
            content_hash,
        })
    })();
    if tmp.exists() {
        let _ = fs::remove_dir_all(tmp);
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{symlink, MetadataExt, PermissionsExt};

    #[test]
    fn excludes_weight_payloads_and_records_policy() {
        let base = std::env::temp_dir().join(format!("eb-snapshot-weights-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let src = base.join("source");
        fs::create_dir_all(src.join("openvla/data")).unwrap();
        fs::write(src.join("openvla/data/loader.py"), b"source").unwrap();
        fs::create_dir_all(src.join("assets")).unwrap();
        fs::write(src.join("assets/mesh.bin"), b"mesh").unwrap();
        fs::write(src.join("openvla/pytorch_model-00001.bin"), b"weights").unwrap();
        fs::write(src.join("openvla/adapter_model.bin"), b"weights").unwrap();
        for extension in [
            "safetensors",
            "bin",
            "pt",
            "pth",
            "ckpt",
            "onnx",
            "gguf",
            "msgpack",
            "h5",
        ] {
            fs::write(
                src.join(format!("openvla/model.{extension}")),
                b"model payload",
            )
            .unwrap();
        }
        let snapshot = prepare_snapshot(&src, &base.join("owned"), None).unwrap();
        let code = Path::new(&snapshot.path);
        assert!(code.join("openvla/data/loader.py").is_file());
        assert_eq!(fs::read(code.join("assets/mesh.bin")).unwrap(), b"mesh");
        assert!(!code.join("openvla/model.safetensors").exists());
        assert_eq!(fs::read_dir(code.join("openvla")).unwrap().count(), 1);
        let manifest: serde_json::Value = serde_json::from_slice(
            &fs::read(code.parent().unwrap().join("manifest.json")).unwrap(),
        )
        .unwrap();
        assert!(manifest["exclusions"]["weight_extensions"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("safetensors")));
        assert_eq!(
            prepare_snapshot(code, &base.join("owned"), Some(&snapshot.content_hash))
                .unwrap()
                .content_hash,
            snapshot.content_hash
        );
    }

    #[test]
    fn snapshot_freezes_content_and_rejects_escape() {
        let base = std::env::temp_dir().join(format!("eb-snapshot-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let src = base.join("source");
        std::fs::create_dir_all(src.join(".git")).unwrap();
        std::fs::create_dir_all(src.join(".agents/skills/example")).unwrap();
        std::fs::create_dir_all(src.join(".claude/skills")).unwrap();
        std::fs::write(src.join("AGENTS.md"), b"instructions").unwrap();
        symlink("AGENTS.md", src.join("CLAUDE.md")).unwrap();
        symlink("../../.agents/skills/example", src.join(".claude/skills/example")).unwrap();
        std::fs::write(src.join("code.py"), b"old").unwrap();
        std::fs::create_dir_all(src.join("pkg/datasets")).unwrap();
        std::fs::write(src.join("pkg/datasets/__init__.py"), b"source").unwrap();
        std::fs::create_dir_all(src.join("rlinf/envs/venv")).unwrap();
        std::fs::write(src.join("rlinf/envs/venv/__init__.py"), b"source").unwrap();
        std::fs::create_dir_all(src.join("venv")).unwrap();
        std::fs::write(src.join("venv/pyvenv.cfg"), b"home = /python").unwrap();
        let first = prepare_snapshot(&src, &base.join("owned"), None).unwrap();
        let path = std::path::Path::new(&first.path);
        assert_eq!(std::fs::read(path.join("code.py")).unwrap(), b"old");
        assert!(!path.join(".git").exists());
        assert!(!path.join(".claude").exists());
        assert!(!path.join(".agents").exists());
        assert!(!path.join("CLAUDE.md").exists());
        assert!(path.join("pkg/datasets/__init__.py").is_file());
        assert!(path.join("rlinf/envs/venv/__init__.py").is_file());
        assert!(!path.join("venv").exists());
        assert_ne!(
            std::fs::metadata(src.join("code.py")).unwrap().ino(),
            std::fs::metadata(path.join("code.py")).unwrap().ino()
        );
        assert_eq!(
            std::fs::metadata(path.join("code.py"))
                .unwrap()
                .permissions()
                .mode()
                & 0o222,
            0
        );
        assert_eq!(
            prepare_snapshot(&src, &base.join("owned"), None)
                .unwrap()
                .content_hash,
            first.content_hash
        );
        assert_eq!(
            prepare_snapshot(path, &base.join("owned"), Some(&first.content_hash))
                .unwrap()
                .path,
            first.path
        );
        std::fs::write(src.join("code.py"), b"new").unwrap();
        assert!(prepare_snapshot(&src, &base.join("owned"), Some(&first.content_hash)).is_err());
        assert_ne!(
            prepare_snapshot(&src, &base.join("owned"), None)
                .unwrap()
                .content_hash,
            first.content_hash
        );
        assert!(prepare_snapshot(&src, &base.join("owned"), Some("HEAD")).is_err());
        symlink("/etc/passwd", src.join("escape")).unwrap();
        assert!(prepare_snapshot(&src, &base.join("owned"), None).is_err());
        assert_eq!(std::fs::read(path.join("code.py")).unwrap(), b"old");
    }
}
