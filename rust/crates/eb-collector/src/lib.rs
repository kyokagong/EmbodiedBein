//! Watch run logs for tfevents files and append scalars into eb-tsdb.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;
use walkdir::WalkDir;

static INGEST_LOCK: Mutex<()> = Mutex::new(());

pub fn with_storage_lock(operation: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
    let _guard = INGEST_LOCK.lock().map_err(|_| "collector lock poisoned")?;
    operation()
}

/// Stream complete TFRecords, bounding each payload rather than the log file.
/// Oversized non-scalar records are skipped; partial tails are retried next scan.
pub fn parse_tfevents(
    path: &Path,
    max_record_bytes: usize,
) -> impl Iterator<Item = (String, i64, f32)> {
    let file = fs::File::open(path).ok();
    let file_len = file
        .as_ref()
        .and_then(|f| f.metadata().ok())
        .map_or(0, |m| m.len());
    let mut reader = file.map(BufReader::new);
    let mut offset = 0u64;
    std::iter::from_fn(move || {
        let reader = reader.as_mut()?;
        loop {
            let mut header = [0; 12];
            reader.read_exact(&mut header).ok()?;
            let length = u64::from_le_bytes(header[..8].try_into().ok()?);
            let end = offset.checked_add(16)?.checked_add(length)?;
            if end > file_len {
                return None;
            }
            offset = end;
            if length > max_record_bytes as u64 {
                reader.seek(SeekFrom::Start(end)).ok()?;
                continue;
            }
            let mut payload = vec![0; length as usize];
            reader.read_exact(&mut payload).ok()?;
            let mut crc = [0; 4];
            reader.read_exact(&mut crc).ok()?;
            let (step, values) = parse_event(&payload);
            return Some(
                values
                    .into_iter()
                    .map(move |(tag, value)| (tag, step, value)),
            );
        }
    })
    .flatten()
}

fn parse_event(buf: &[u8]) -> (i64, Vec<(String, f32)>) {
    let mut step = 0i64;
    let mut values = Vec::new();
    let mut i = 0usize;
    while i < buf.len() {
        let Some((field, wire, ni)) = key(buf, i) else {
            break;
        };
        i = ni;
        match wire {
            0 => {
                let Some((val, ni)) = varint(buf, i) else {
                    break;
                };
                i = ni;
                if field == 2 {
                    step = val as i64;
                }
            }
            1 => i += 8,
            5 => i += 4,
            2 => {
                let Some((ln, ni)) = varint(buf, i) else {
                    break;
                };
                i = ni;
                let Some(end) = i.checked_add(ln as usize).filter(|end| *end <= buf.len()) else {
                    break;
                };
                let inner = &buf[i..end];
                i = end;
                if field == 5 {
                    values.extend(parse_summary(inner));
                }
            }
            _ => break,
        }
    }
    (step, values)
}

fn parse_summary(buf: &[u8]) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < buf.len() {
        let Some((field, wire, ni)) = key(buf, i) else {
            break;
        };
        i = ni;
        if wire != 2 {
            match wire {
                0 => {
                    let Some((_, ni)) = varint(buf, i) else { break };
                    i = ni;
                }
                1 => i += 8,
                5 => i += 4,
                _ => break,
            }
            continue;
        }
        let Some((ln, ni)) = varint(buf, i) else {
            break;
        };
        i = ni;
        let Some(end) = i.checked_add(ln as usize).filter(|end| *end <= buf.len()) else {
            break;
        };
        let inner = &buf[i..end];
        i = end;
        if field == 1 {
            if let Some((tag, val)) = parse_value(inner) {
                out.push((tag, val));
            }
        }
    }
    out
}

fn parse_value(buf: &[u8]) -> Option<(String, f32)> {
    let mut tag = None;
    let mut val = None;
    let mut i = 0usize;
    while i < buf.len() {
        let (field, wire, ni) = key(buf, i)?;
        i = ni;
        if field == 1 && wire == 2 {
            let (ln, ni) = varint(buf, i)?;
            i = ni;
            let end = i.checked_add(usize::try_from(ln).ok()?)?;
            buf.get(i..end)?;
            tag = Some(String::from_utf8_lossy(&buf[i..end]).into_owned());
            i = end;
        } else if field == 2 && wire == 5 {
            if i + 4 > buf.len() {
                break;
            }
            val = Some(f32::from_le_bytes(buf[i..i + 4].try_into().ok()?));
            i += 4;
        } else {
            match wire {
                0 => {
                    let (_, ni) = varint(buf, i)?;
                    i = ni;
                }
                1 => i += 8,
                5 => i += 4,
                2 => {
                    let (ln, ni) = varint(buf, i)?;
                    i = ni.checked_add(usize::try_from(ln).ok()?)?;
                }
                _ => break,
            }
        }
    }
    Some((tag?, val?))
}

fn varint(buf: &[u8], mut i: usize) -> Option<(u64, usize)> {
    let mut x = 0u64;
    let mut s = 0;
    while i < buf.len() {
        let b = buf[i];
        i += 1;
        if s == 63 && b > 1 {
            return None;
        }
        x |= u64::from(b & 0x7f) << s;
        if b & 0x80 == 0 {
            return Some((x, i));
        }
        s += 7;
        if s >= 64 {
            break;
        }
    }
    None
}

fn key(buf: &[u8], i: usize) -> Option<(u32, u32, usize)> {
    let (k, ni) = varint(buf, i)?;
    Some(((k >> 3) as u32, (k & 7) as u32, ni))
}

pub fn ingest_run_logs(root: &Path, run_id: &str) -> std::io::Result<usize> {
    // ponytail: one collector lock; use per-run locks and file offsets if scan throughput matters.
    let _guard = INGEST_LOCK
        .lock()
        .map_err(|_| std::io::Error::other("collector lock poisoned"))?;
    let run = root.join("runs").join(run_id);
    let mut n = 0usize;
    let mut seen: HashMap<String, HashSet<i64>> = HashMap::new();
    for ent in ["logs", "output", "tb"].iter().flat_map(|name| {
        WalkDir::new(run.join(name))
            .follow_root_links(false)
            .into_iter()
            .flatten()
    }) {
        if !ent.file_type().is_file() {
            continue;
        }
        let name = ent.file_name().to_string_lossy();
        if !name.contains("tfevents") {
            continue;
        }
        for (tag, step, value) in parse_tfevents(ent.path(), 16 * 1024 * 1024) {
            if !seen.contains_key(&tag) {
                seen.insert(
                    tag.clone(),
                    eb_tsdb::query_exact(root, run_id, &tag)?
                        .into_iter()
                        .map(|point| point.0)
                        .collect(),
                );
            }
            if !seen.get_mut(&tag).unwrap().insert(step) {
                continue;
            }
            eb_tsdb::append(root, run_id, &tag, step, value)?;
            n += 1;
        }
    }
    Ok(n)
}

pub async fn spawn(root: PathBuf) {
    loop {
        let runs_dir = root.join("runs");
        if runs_dir.exists() {
            if let Ok(rd) = fs::read_dir(&runs_dir) {
                for ent in rd.flatten() {
                    if ent.path().is_dir() {
                        let run_id = ent.file_name().to_string_lossy().to_string();
                        let _ = ingest_run_logs(&root, &run_id);
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn streams_past_large_records_and_retries_partial_tail() {
        let path = std::env::temp_dir().join(format!("eb-large-events-{}", std::process::id()));
        let mut file = fs::File::create(&path).unwrap();
        let oversized = 17 * 1024 * 1024u64;
        file.write_all(&oversized.to_le_bytes()).unwrap();
        file.write_all(&[0; 4]).unwrap();
        file.seek(SeekFrom::Start(oversized + 12)).unwrap();
        file.write_all(&[0; 4]).unwrap();
        file.write_all(&scalar_record(1, b"train/loss")).unwrap();
        let next = scalar_record(2, b"train/loss");
        file.write_all(&next[..next.len() - 2]).unwrap();
        assert_eq!(
            parse_tfevents(&path, 16 * 1024 * 1024).collect::<Vec<_>>(),
            vec![("train/loss".into(), 1, 0.25)]
        );
        file.write_all(&next[next.len() - 2..]).unwrap();
        assert_eq!(parse_tfevents(&path, 16 * 1024 * 1024).count(), 2);
        assert!(varint(&[0xff; 11], 0).is_none());
        let mut corrupt = vec![0x2a];
        corrupt.extend([0xff; 9]);
        corrupt.push(1);
        assert!(parse_event(&corrupt).1.is_empty());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn exact_tags_reingest_legacy_without_duplicates() {
        let root = std::env::temp_dir().join(format!("eb-collector-tags-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("tsdb/run_test")).unwrap();
        let legacy = root.join("tsdb/run_test/foo_bar.jsonl");
        let old = b"{\"step\":0,\"value\":99.0}\n";
        fs::write(&legacy, old).unwrap();
        let output = root.join("runs/run_test/output");
        fs::create_dir_all(&output).unwrap();
        let tags = ["foo/bar", "foo_bar", "中文/损失"];
        let records: Vec<u8> = tags
            .iter()
            .flat_map(|tag| scalar_record(0, tag.as_bytes()))
            .collect();
        fs::write(output.join("events.out.tfevents.test"), records).unwrap();
        assert_eq!(ingest_run_logs(&root, "run_test").unwrap(), 3);
        assert_eq!(ingest_run_logs(&root, "run_test").unwrap(), 0);
        assert_eq!(eb_tsdb::list_tags(&root, "run_test").unwrap(), tags);
        for tag in tags {
            assert_eq!(
                eb_tsdb::query(&root, "run_test", tag).unwrap(),
                vec![(0, 0.25)]
            );
        }
        assert_eq!(fs::read(legacy).unwrap(), old);
        fs::remove_dir_all(root).unwrap();
    }

    fn scalar_record(step: u8, tag: &[u8]) -> Vec<u8> {
        // Minimal TFRecord containing Event(step, Summary.Value(tag, simple_value)).
        let mut value = vec![0x0a, tag.len() as u8];
        value.extend(tag);
        value.push(0x15);
        value.extend(0.25f32.to_le_bytes());
        let mut summary = vec![0x0a, value.len() as u8];
        summary.extend(value);
        let mut event = vec![0x10, step, 0x2a, summary.len() as u8];
        event.extend(summary);
        let mut record = (event.len() as u64).to_le_bytes().to_vec();
        record.extend([0; 4]); // Existing extractor does not verify masked CRCs.
        record.extend(event);
        record.extend([0; 4]);
        record
    }

    #[test]
    fn ingests_only_output_roots_once() {
        let root = std::env::temp_dir().join(format!("eb-collector-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        for (step, dir) in [
            "logs",
            "output/experiment/tensorboard",
            "tb",
            "inputs",
            "code",
        ]
        .iter()
        .enumerate()
        {
            let path = root.join("runs/run_test").join(dir);
            fs::create_dir_all(&path).unwrap();
            fs::write(
                path.join("events.out.tfevents.test"),
                scalar_record(step as u8, b"train/loss"),
            )
            .unwrap();
        }
        assert_eq!(ingest_run_logs(&root, "run_test").unwrap(), 3);
        assert_eq!(ingest_run_logs(&root, "run_test").unwrap(), 0);
        let mut points = eb_tsdb::query(&root, "run_test", "train/loss").unwrap();
        points.sort_by_key(|point| point.0);
        assert_eq!(points, vec![(0, 0.25), (1, 0.25), (2, 0.25)]);
        fs::remove_dir_all(root).unwrap();
    }
}
