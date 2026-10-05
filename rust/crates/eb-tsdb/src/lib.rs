use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    pub step: i64,
    pub value: f32,
}

#[derive(Serialize, Deserialize)]
struct TaggedPoint {
    tag: String,
    #[serde(flatten)]
    point: Point,
}

fn legacy_stem(tag: &str) -> String {
    tag.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn tag_file(root: &Path, run_id: &str, tag: &str) -> PathBuf {
    root.join("tsdb")
        .join(run_id)
        .join("v2")
        .join(format!("{}.jsonl", eb_core::sha256_hex(tag.as_bytes())))
}

pub fn append(root: &Path, run_id: &str, tag: &str, step: i64, value: f32) -> std::io::Result<()> {
    let path = tag_file(root, run_id, tag);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    // The original tag is authoritative, even if two hashes ever collide.
    let record = TaggedPoint {
        tag: tag.into(),
        point: Point { step, value },
    };
    writeln!(file, "{}", serde_json::to_string(&record).unwrap())?;
    Ok(())
}

/// Exact v2 points only: collectors must not use ambiguous legacy tags for deduplication.
pub fn query_exact(root: &Path, run_id: &str, tag: &str) -> std::io::Result<Vec<(i64, f32)>> {
    let path = tag_file(root, run_id, tag);
    if !path.exists() {
        return Ok(vec![]);
    }
    let mut points = Vec::new();
    for line in BufReader::new(fs::File::open(path)?).lines() {
        if let Ok(record) = serde_json::from_str::<TaggedPoint>(&line?) {
            if record.tag == tag {
                points.push((record.point.step, record.point.value));
            }
        }
    }
    Ok(points)
}

pub fn query(root: &Path, run_id: &str, tag: &str) -> std::io::Result<Vec<(i64, f32)>> {
    if tag_file(root, run_id, tag).exists() {
        return query_exact(root, run_id, tag);
    }
    // Legacy names were lossy. Read them only until this exact tag is re-ingested as v2.
    let path = root
        .join("tsdb")
        .join(run_id)
        .join(format!("{}.jsonl", legacy_stem(tag)));
    if !path.exists() {
        return Ok(vec![]);
    }
    let mut points = Vec::new();
    for line in BufReader::new(fs::File::open(path)?).lines() {
        if let Ok(point) = serde_json::from_str::<Point>(&line?) {
            points.push((point.step, point.value));
        }
    }
    Ok(points)
}

pub fn list_tags(root: &Path, run_id: &str) -> std::io::Result<Vec<String>> {
    let dir = root.join("tsdb").join(run_id);
    if !dir.exists() {
        return Ok(vec![]);
    }
    let mut tags = BTreeSet::new();
    if dir.join("v2").exists() {
        // ponytail: scan records for exact names; add a tag index if metrics history makes this costly.
        for entry in fs::read_dir(dir.join("v2"))? {
            let entry = entry?;
            if !entry.file_type()?.is_file()
                || entry.path().extension().and_then(|e| e.to_str()) != Some("jsonl")
            {
                continue;
            }
            for line in BufReader::new(fs::File::open(entry.path())?).lines() {
                if let Ok(record) = serde_json::from_str::<TaggedPoint>(&line?) {
                    tags.insert(record.tag);
                }
            }
        }
    }
    let replaced: BTreeSet<String> = tags.iter().map(|tag| legacy_stem(tag)).collect();
    for entry in fs::read_dir(dir)? {
        let name = entry?.file_name().to_string_lossy().to_string();
        if let Some(stem) = name.strip_suffix(".jsonl") {
            if !replaced.contains(stem) {
                tags.insert(stem.replace('_', "/"));
            }
        }
    }
    Ok(tags.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn exact_tags_do_not_collide_and_legacy_stays_read_only() {
        let root = env::temp_dir().join(format!("eb-tsdb-tags-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("tsdb/run_t")).unwrap();
        let legacy = root.join("tsdb/run_t/foo_bar.jsonl");
        let old = b"{\"step\":0,\"value\":99.0}\n";
        fs::write(&legacy, old).unwrap();
        assert_eq!(query(&root, "run_t", "foo/bar").unwrap(), vec![(0, 99.0)]);
        for (i, tag) in ["foo/bar", "foo_bar", "中文/损失"].iter().enumerate() {
            append(&root, "run_t", tag, 0, i as f32).unwrap();
        }
        assert_eq!(
            list_tags(&root, "run_t").unwrap(),
            vec!["foo/bar", "foo_bar", "中文/损失"]
        );
        for (i, tag) in ["foo/bar", "foo_bar", "中文/损失"].iter().enumerate() {
            assert_eq!(query(&root, "run_t", tag).unwrap(), vec![(0, i as f32)]);
        }
        assert_eq!(fs::read(legacy).unwrap(), old);
        fs::remove_dir_all(root).unwrap();
    }

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
