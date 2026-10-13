use std::path::{Path, PathBuf};

pub fn preview_path(root: &Path, run: &str, clip: &str) -> Option<PathBuf> {
    if !run.strip_prefix("run_").is_some_and(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_hexdigit()))
        || clip.is_empty() || clip.len() > 160
        || !clip.bytes().next().is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        || !clip.bytes().all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b)) {
        return None;
    }
    let base = root.join("runs").join(run).join("outputs/motion-preview");
    let path = base.join(format!("{clip}.json"));
    let canonical = path.canonicalize().ok()?;
    if !canonical.starts_with(root.canonicalize().ok()?.join("runs").join(run).join("outputs/motion-preview")) {
        return None;
    }
    let metadata = canonical.metadata().ok()?;
    (metadata.is_file() && metadata.len() <= 8 * 1024 * 1024).then_some(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_is_bounded_to_a_run_and_rejects_escape() {
        let root = std::env::temp_dir().join(format!("eb-motion-preview-{}", std::process::id()));
        let output = root.join("runs/run_abc/outputs/motion-preview");
        std::fs::create_dir_all(&output).unwrap();
        std::fs::write(output.join("walk.json"), "{}").unwrap();
        assert!(preview_path(&root, "run_abc", "walk").is_some());
        assert!(preview_path(&root, "../run_abc", "walk").is_none());
        assert!(preview_path(&root, "run_abc", "../walk").is_none());
        assert!(preview_path(&root, "run_abc", "missing").is_none());
        std::os::unix::fs::symlink(root.join("outside.json"), output.join("escape.json")).unwrap();
        std::fs::write(root.join("outside.json"), "{}").unwrap();
        assert!(preview_path(&root, "run_abc", "escape").is_none());
        std::fs::remove_dir_all(root).unwrap();
    }
}
