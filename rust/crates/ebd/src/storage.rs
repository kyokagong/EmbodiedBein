use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

#[derive(Deserialize)]
pub struct DeleteRequest {
    pub paths: Vec<String>,
    pub run_ids: Vec<String>,
}

fn valid_id(value: &str, prefix: &str) -> bool {
    value.starts_with(prefix) && value.len() > prefix.len()
        && value.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn checked_path(root: &Path, models: &Path, path: &Path) -> Result<(), String> {
    if !path.is_absolute() || path.components().any(|p| matches!(p, Component::ParentDir | Component::CurDir)) {
        return Err("invalid cleanup path".into());
    }
    let allowed = [root.join("runs"), root.join("tsdb"), models.join("runs")]
        .iter().any(|base| path.strip_prefix(base).is_ok_and(|p| {
            let parts: Vec<_> = p.components().collect();
            parts.first().is_some_and(|part| valid_id(&part.as_os_str().to_string_lossy(), "run_"))
                && !parts.is_empty()
        })) || path.strip_prefix(models).is_ok_and(|p| p.components().next()
            .is_some_and(|part| valid_id(&part.as_os_str().to_string_lossy(), "mdl_")));
    if !allowed { return Err("cleanup path is not platform-owned Run/model storage".into()); }
    let mut current = PathBuf::new();
    for part in path.components() {
        current.push(part);
        match fs::symlink_metadata(&current) {
            Ok(meta) if meta.file_type().is_symlink() => return Err("cleanup path contains a symlink".into()),
            Ok(_) => (),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(())
}

fn bytes(path: &Path) -> Result<u64, String> {
    let meta = match fs::symlink_metadata(path) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e.to_string()),
    };
    if !meta.is_dir() { return Ok(meta.len()); }
    let mut total = 0;
    for entry in fs::read_dir(path).map_err(|e| e.to_string())? {
        total += bytes(&entry.map_err(|e| e.to_string())?.path())?;
    }
    Ok(total)
}

fn remove(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => fs::remove_dir_all(path).map_err(|e| e.to_string()),
        Ok(_) => fs::remove_file(path).map_err(|e| e.to_string()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

fn check_containers(containers: &[Value], paths: &[PathBuf], run_ids: &[String]) -> Result<Vec<String>, String> {
    let mut stopped = Vec::new();
    for container in containers {
        let id = container["Id"].as_str().ok_or("invalid container inspection")?;
        let run_id = container["Config"]["Labels"]["eb.run_id"].as_str().unwrap_or("");
        let state = container["State"]["Status"].as_str().ok_or("missing container state")?;
        let active = !matches!(state, "exited" | "dead");
        if run_ids.iter().any(|rid| rid == run_id) {
            if active { return Err(format!("Run container is still active: {run_id}")); }
            stopped.push(id.to_string());
        }
        if active {
            let mounts = container["Mounts"].as_array().ok_or("missing container mounts")?;
            for mount in mounts {
                let source = PathBuf::from(mount["Source"].as_str().ok_or("invalid container mount")?);
                let source = fs::canonicalize(&source).unwrap_or(source);
                if paths.iter().any(|p| p.starts_with(&source) || source.starts_with(p)) {
                    return Err(format!("cleanup path is mounted by active Run: {run_id}"));
                }
            }
        }
    }
    Ok(stopped)
}

pub fn delete_storage(root: &Path, models: &Path, req: DeleteRequest) -> Result<Value, String> {
    if req.paths.len() > 1000 || req.run_ids.len() > 1000
        || req.run_ids.iter().any(|id| !valid_id(id, "run_")) {
        return Err("invalid Run cleanup request".into());
    }
    let mut paths: Vec<PathBuf> = req.paths.iter().map(PathBuf::from).collect();
    for path in &paths { checked_path(root, models, path)?; }
    paths.sort();
    paths.dedup();
    let all = paths.clone();
    paths.retain(|p| !all.iter().any(|other| other != p && p.starts_with(other)));
    let list = Command::new("docker").args(["ps", "-aq", "--filter", "label=eb.run_id"])
        .output().map_err(|e| e.to_string())?;
    if !list.status.success() { return Err("could not inspect task containers".into()); }
    let ids: Vec<_> = String::from_utf8_lossy(&list.stdout).split_whitespace().map(str::to_owned).collect();
    let stopped = if ids.is_empty() { Vec::new() } else {
        let inspect = Command::new("docker").arg("inspect").args(&ids).output().map_err(|e| e.to_string())?;
        if !inspect.status.success() { return Err("could not confirm task container state".into()); }
        let containers: Vec<Value> = serde_json::from_slice(&inspect.stdout).map_err(|e| e.to_string())?;
        check_containers(&containers, &paths, &req.run_ids)?
    };
    let mut total = 0;
    for path in &paths { total += bytes(path)?; }
    // Never force-remove a container: a concurrent start must fail cleanup safely.
    if !stopped.is_empty() {
        let removed = Command::new("docker").arg("rm").args(&stopped).output().map_err(|e| e.to_string())?;
        if !removed.status.success() { return Err("stopped container removal failed; retry cleanup".into()); }
    }
    // The collector shares this lock so it cannot recreate deleted TSDB files.
    eb_collector::with_storage_lock(|| {
        for path in &paths {
            checked_path(root, models, path)?;
            remove(path)?;
        }
        Ok(())
    })?;
    Ok(json!({"paths": paths, "removed_bytes": total, "removed_containers": stopped}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleanup_is_owned_bounded_and_does_not_follow_links() {
        let root = std::env::temp_dir().join(format!("eb-delete-{}", std::process::id()));
        let state = root.join("state");
        let models = root.join("models");
        fs::create_dir_all(state.join("runs/run_test")).unwrap();
        fs::create_dir_all(root.join("outside")).unwrap();
        fs::write(root.join("outside/keep"), b"keep").unwrap();
        std::os::unix::fs::symlink(root.join("outside"), state.join("runs/run_test/link")).unwrap();
        assert!(checked_path(&state, &models, &state.join("runs/run_test")).is_ok());
        assert!(checked_path(&state, &models, &state.join("runs")).is_err());
        assert!(checked_path(&state, &models, &state.join("runs/run_test/link/keep")).is_err());
        assert!(checked_path(&state, &models, &state.join("runs/run_test/../other")).is_err());
        assert!(checked_path(&state, &models, &models.join("sources/base")).is_err());
        remove(&state.join("runs/run_test")).unwrap();
        remove(&state.join("runs/run_test")).unwrap();
        assert!(root.join("outside/keep").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn active_paused_and_shared_mounts_block_cleanup() {
        let container = |state: &str, source: &str| json!({"Id":"abc", "State":{"Status":state},
            "Config":{"Labels":{"eb.run_id":"run_test"}}, "Mounts":[{"Source":source}]});
        let paths = vec![PathBuf::from("/managed/runs/run_source")];
        for state in ["running", "paused", "restarting", "created"] {
            assert!(check_containers(&[container(state, "/other")], &paths, &["run_test".into()]).is_err());
            assert!(check_containers(&[container(state, "/managed/runs/run_source/output")], &paths, &[]).is_err());
        }
        assert_eq!(check_containers(&[container("exited", "/other")], &paths, &["run_test".into()]).unwrap(), vec!["abc"]);
    }
}
