use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    pub step: i64,
    pub value: f32,
}

fn tag_file(root: &Path, run_id: &str, tag: &str) -> PathBuf {
    let safe: String = tag
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    root.join("tsdb").join(run_id).join(format!("{safe}.jsonl"))
}

pub fn append(root: &Path, run_id: &str, tag: &str, step: i64, value: f32) -> std::io::Result<()> {
    let path = tag_file(root, run_id, tag);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    let rec = serde_json::to_string(&Point { step, value }).unwrap();
    writeln!(f, "{rec}")?;
    Ok(())
}

pub fn query(root: &Path, run_id: &str, tag: &str) -> std::io::Result<Vec<(i64, f32)>> {
    let path = tag_file(root, run_id, tag);
    if !path.exists() {
        return Ok(vec![]);
    }
    let f = fs::File::open(path)?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(p) = serde_json::from_str::<Point>(&line) {
            out.push((p.step, p.value));
        }
    }
    Ok(out)
}

pub fn list_tags(root: &Path, run_id: &str) -> std::io::Result<Vec<String>> {
    let dir = root.join("tsdb").join(run_id);
    if !dir.exists() {
        return Ok(vec![]);
    }
    let mut tags = Vec::new();
    for ent in fs::read_dir(dir)? {
        let ent = ent?;
        let name = ent.file_name().to_string_lossy().to_string();
        if let Some(stem) = name.strip_suffix(".jsonl") {
            tags.push(stem.replace('_', "/"));
        }
    }
    tags.sort();
    Ok(tags)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn roundtrip_100_points() {
        let dir = env::temp_dir().join(format!("eb-tsdb-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for i in 0..100 {
            append(&dir, "run_t", "Train/mean_reward", i, i as f32 * 0.1).unwrap();
        }
        let pts = query(&dir, "run_t", "Train/mean_reward").unwrap();
        assert_eq!(pts.len(), 100);
        assert_eq!(pts[0], (0, 0.0));
        assert!((pts[99].1 - 9.9).abs() < 1e-4);
        let _ = fs::remove_dir_all(&dir);
    }
}
