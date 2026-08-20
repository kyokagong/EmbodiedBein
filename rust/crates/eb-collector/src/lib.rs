//! Watch run logs for tfevents files and append scalars into eb-tsdb.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use walkdir::WalkDir;

/// Best-effort TFRecord + Event protobuf scalar extractor (no tensorflow).
pub fn parse_tfevents(path: &Path, max_bytes: usize) -> Vec<(String, i64, f32)> {
    let Ok(data) = fs::read(path) else {
        return vec![];
    };
    let data = if data.len() > max_bytes {
        &data[..max_bytes]
    } else {
        &data
    };
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 12 <= data.len() {
        let length = u64::from_le_bytes(data[i..i + 8].try_into().unwrap()) as usize;
        i += 12;
        if length == 0 || i + length + 4 > data.len() {
            break;
        }
        let payload = &data[i..i + length];
        i += length + 4;
        let (step, values) = parse_event(payload);
        for (tag, val) in values {
            out.push((tag, step, val));
        }
    }
    out
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
                let Some((val, ni)) = varint(buf, i) else { break };
                i = ni;
                if field == 2 {
                    step = val as i64;
                }
            }
            1 => i += 8,
            5 => i += 4,
            2 => {
                let Some((ln, ni)) = varint(buf, i) else { break };
                i = ni;
                let end = (i + ln as usize).min(buf.len());
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
        let Some((ln, ni)) = varint(buf, i) else { break };
        i = ni;
        let end = (i + ln as usize).min(buf.len());
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
            let end = (i + ln as usize).min(buf.len());
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
                    i = ni + ln as usize;
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
        x |= u64::from(b & 0x7f) << s;
        if b & 0x80 == 0 {
            return Some((x, i));
        }
        s += 7;
        if s > 70 {
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
    let logs = root.join("runs").join(run_id).join("logs");
    if !logs.exists() {
        return Ok(0);
    }
    let mut n = 0usize;
    let mut seen = HashSet::new();
    for ent in WalkDir::new(&logs).into_iter().flatten() {
        if !ent.file_type().is_file() {
            continue;
        }
        let name = ent.file_name().to_string_lossy();
        if !name.contains("tfevents") {
            continue;
        }
        let key = ent.path().display().to_string();
        if !seen.insert(key) {
            continue;
        }
        for (tag, step, value) in parse_tfevents(ent.path(), 16 * 1024 * 1024) {
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
