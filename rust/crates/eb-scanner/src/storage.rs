use eb_artifact::{ArtifactError, fd_path, open_directory, open_relative};
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

pub fn hash_at(base: &Path, path: &str) -> Result<Value, ArtifactError> {
    let _pin = eb_artifact::pin_paths(&[base.join(path)])?;
    let mut file = eb_artifact::open_member(&open_directory(base)?, path)?;
    if file.metadata()?.len() > 1024u64.pow(4) {
        return Err(ArtifactError::new("limit", "file hash limit exceeded"));
    }
    let (digest, bytes) = eb_artifact::hash_file(&mut file)?;
    Ok(json!({"schema_version":"eb/storage-hash/v1","bytes":bytes,"sha256":digest}))
}

fn walk(
    directory: &File,
    prefix: &str,
    recursive: bool,
    items: &mut Vec<Value>,
    exclude_dirs: &[String],
) -> Result<(), ArtifactError> {
    for entry in fs::read_dir(fd_path(directory))? {
        if items.len() >= 100000 {
            return Err(ArtifactError::new(
                "limit",
                "storage inventory exceeds 100000 entries",
            ));
        }
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| ArtifactError::new("invalid", "non-UTF8 storage name"))?;
        let relative = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        eb_artifact::validate_relative(&relative)?;
        let metadata = fs::symlink_metadata(entry.path())?;
        if !metadata.is_dir() && !metadata.is_file() && !metadata.file_type().is_symlink() {
            return Err(ArtifactError::new(
                "forbidden",
                "special files cannot be inspected",
            ));
        }
        let is_dir = metadata.is_dir();
        if is_dir && exclude_dirs.contains(&name) {
            continue;
        }
        items.push(json!({"path":relative,"is_dir":is_dir,"is_symlink":metadata.file_type().is_symlink(),
            "bytes":metadata.len(),"mtime":metadata.mtime(),"mtime_ns":metadata.mtime() as i128 * 1_000_000_000 + metadata.mtime_nsec() as i128}));
        if recursive && is_dir {
            let child = open_relative(directory, &name)?;
            walk(&child, &relative, true, items, exclude_dirs)?;
        }
    }
    Ok(())
}

pub fn inspect_at(
    base: &Path,
    path: &str,
    mode: &str,
    cursor: Option<&str>,
    limit: usize,
    recursive: bool,
    exclude_dirs: &[String],
) -> Result<Value, ArtifactError> {
    if !matches!(mode, "inventory" | "size" | "disk") || !(1..=1000).contains(&limit) {
        return Err(ArtifactError::new(
            "invalid",
            "invalid inspection mode or page size",
        ));
    }
    if !path.is_empty() {
        eb_artifact::validate_relative(path)?;
    }
    if exclude_dirs.len() > 64
        || exclude_dirs
            .iter()
            .any(|name| name.contains('/') || eb_artifact::validate_relative(name).is_err())
    {
        return Err(ArtifactError::new("invalid", "invalid directory exclusion"));
    }
    let _pin = eb_artifact::pin_paths(&[base.join(path)])?;
    let root = open_directory(base)?;
    let mut relative = path.to_owned();
    let file = loop {
        let result = if relative.is_empty() {
            Ok(root.try_clone()?)
        } else {
            open_relative(&root, &relative)
        };
        match result {
            Ok(file) => break file,
            Err(error) if mode == "disk" && error.code == "missing" && !relative.is_empty() => {
                relative = Path::new(&relative)
                    .parent()
                    .unwrap_or_else(|| Path::new(""))
                    .to_string_lossy()
                    .into_owned();
            }
            Err(error) => return Err(error),
        }
    };
    if mode == "disk" {
        let (free, total) = eb_probe::disk_space(&fd_path(&file))?;
        return Ok(
            json!({"schema_version":"eb/storage-inspection/v1","mode":mode,"free_bytes":free,"total_bytes":total}),
        );
    }
    if !file.metadata()?.is_dir() {
        if mode == "size" {
            return Ok(
                json!({"schema_version":"eb/storage-inspection/v1","mode":mode,"bytes":file.metadata()?.len()}),
            );
        }
        return Err(ArtifactError::new(
            "invalid",
            "inventory requires a directory",
        ));
    }
    let mut items = Vec::new();
    walk(
        &file,
        "",
        recursive || mode == "size",
        &mut items,
        exclude_dirs,
    )?;
    if mode == "size" {
        let bytes: u64 = items
            .iter()
            .filter(|item| item["is_dir"] == false && item["is_symlink"] == false)
            .map(|item| item["bytes"].as_u64().unwrap())
            .sum();
        return Ok(json!({"schema_version":"eb/storage-inspection/v1","mode":mode,"bytes":bytes}));
    }
    items.sort_by_key(|item| {
        (
            !item["is_dir"].as_bool().unwrap(),
            item["path"].as_str().unwrap().to_lowercase(),
            item["path"].as_str().unwrap().to_owned(),
        )
    });
    let offset: usize = match cursor {
        Some(value) => value
            .parse()
            .map_err(|_| ArtifactError::new("invalid", "invalid inventory cursor"))?,
        None => 0,
    };
    let total = items.len();
    let next = (offset.saturating_add(limit) < total).then(|| (offset + limit).to_string());
    Ok(
        json!({"schema_version":"eb/storage-inspection/v1","mode":mode,
        "items":items.into_iter().skip(offset).take(limit).collect::<Vec<_>>(),"total":total,"next_cursor":next}),
    )
}

pub fn read_text_at(base: &Path, path: &str, max_bytes: usize) -> Result<Value, ArtifactError> {
    if !(1..=16 * 1024 * 1024).contains(&max_bytes) {
        return Err(ArtifactError::new("limit", "text limit exceeded"));
    }
    let _pin = eb_artifact::pin_paths(&[base.join(path)])?;
    let file = eb_artifact::open_member(&open_directory(base)?, path)?;
    if file.metadata()?.len() > max_bytes as u64 {
        return Err(ArtifactError::new("limit", "text file too large"));
    }
    let mut bytes = Vec::new();
    file.take(max_bytes as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > max_bytes {
        return Err(ArtifactError::new("limit", "text file grew beyond limit"));
    }
    let text = String::from_utf8(bytes)
        .map_err(|_| ArtifactError::new("invalid", "storage text is not UTF-8"))?;
    Ok(json!({"schema_version":"eb/storage-text/v1","text":text}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::symlink;

    #[test]
    fn inspection_lists_pages_sizes_and_refuses_linked_text() {
        let root = std::env::temp_dir().join(format!("eb-storage-io-{}", std::process::id()));
        fs::create_dir_all(root.join("nested")).unwrap();
        fs::write(root.join("robot.urdf"), b"<robot/>").unwrap();
        fs::write(root.join("nested/model"), b"state").unwrap();
        symlink("/etc/passwd", root.join("linked")).unwrap();
        let page = inspect_at(&root, "", "inventory", None, 1, false, &[]).unwrap();
        assert_eq!(page["total"], 3);
        assert_eq!(page["items"][0]["path"], "nested");
        assert!(page["next_cursor"].is_string());
        let size = inspect_at(&root, "", "size", None, 50, true, &[]).unwrap();
        assert_eq!(size["bytes"], 13);
        assert_eq!(
            read_text_at(&root, "robot.urdf", 100).unwrap()["text"],
            "<robot/>"
        );
        assert!(read_text_at(&root, "linked", 100).is_err());
        assert_eq!(hash_at(&root, "robot.urdf").unwrap()["bytes"], 8);
        assert!(hash_at(&root, "linked").is_err());
        assert!(read_text_at(&root, "robot.urdf", 2).is_err());
        assert!(inspect_at(&root, "../", "inventory", None, 50, false, &[]).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}

