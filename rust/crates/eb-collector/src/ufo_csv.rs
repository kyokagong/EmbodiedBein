use std::collections::HashSet;
use std::fs;
use std::io::{self, BufReader, Read};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

const MAX_RECORD_BYTES: usize = 1024 * 1024;
const MAX_COLUMNS: usize = 4096;

/// Training tags keep their CSV names, with agent updates as the TSDB axis.
/// `timestep` / `distributed/global_env_steps` and `local_timestep` retain the
/// native collect-before-update boundary as values, without realigning it.
/// Tracking tags are `ufo_tracking/<motion_name>/<CSV name>` on the native
/// `timestep` (global environment steps) axis; per-motion rows are never averaged.
pub(super) fn collect(
    run: &Path,
    mut append: impl FnMut(&str, i64, f32) -> io::Result<()>,
) -> io::Result<()> {
    let native = run.join("outputs/native");
    // Only managed outputs in this Run, never input/source logs or linked Runs.
    for directory in [run.to_path_buf(), run.join("outputs"), native.clone()] {
        if !fs::symlink_metadata(directory).is_ok_and(|meta| meta.is_dir()) {
            return Ok(());
        }
    }
    for (filename, training) in [("train_log.txt", true), ("ufo_tracking_eval.csv", false)] {
        let path = native.join(filename);
        if !fs::symlink_metadata(&path).is_ok_and(|meta| meta.is_file()) {
            continue;
        }
        let file = fs::OpenOptions::new()
            .read(true)
            .custom_flags(0x20000) // Linux O_NOFOLLOW
            .open(&path)?;
        // Read an open inode to its original length. Replacement/header extension
        // is picked up by rescanning; no stale byte cursor can hide new columns.
        let length = file.metadata()?.len();
        let mut reader = BufReader::new(file.take(length));
        let Some(Record::Complete(header)) = read_record(&mut reader, MAX_RECORD_BYTES)? else {
            continue;
        };
        let Some(fields) = parse_record(&header) else {
            continue;
        };
        if fields.iter().any(String::is_empty)
            || fields.iter().collect::<HashSet<_>>().len() != fields.len()
        {
            continue;
        }
        let axis = if training {
            "distributed/optimizer_steps"
        } else {
            "timestep"
        };
        let Some(axis) = fields.iter().position(|field| field == axis) else {
            tracing::warn!(path = %path.display(), "UFO CSV has no native step column");
            continue;
        };
        let motion = fields.iter().position(|field| field == "motion_name");
        if !training && motion.is_none() {
            tracing::warn!(path = %path.display(), "UFO tracking CSV has no motion identity");
            continue;
        }
        let mut nonfinite = 0usize;
        let mut invalid = 0usize;
        while let Some(record) = read_record(&mut reader, MAX_RECORD_BYTES)? {
            let Record::Complete(record) = record else {
                invalid += 1;
                continue;
            };
            let Some(values) = parse_record(&record).filter(|values| values.len() == fields.len())
            else {
                invalid += 1;
                continue;
            };
            let Some(step) = values[axis]
                .trim()
                .parse::<i64>()
                .ok()
                .filter(|step| *step >= 0)
            else {
                invalid += 1;
                continue;
            };
            let prefix = if training {
                String::new()
            } else {
                let name = &values[motion.unwrap()];
                if name.is_empty() {
                    invalid += 1;
                    continue;
                }
                format!("ufo_tracking/{name}/")
            };
            for (field, value) in fields.iter().zip(&values) {
                if field == "motion_name" {
                    continue;
                }
                // Empty/string columns stay absent; nonfinite values never become zero.
                let Ok(value) = value.trim().parse::<f32>() else {
                    continue;
                };
                if !value.is_finite() {
                    nonfinite += 1;
                    continue;
                }
                append(&format!("{prefix}{field}"), step, value)?;
            }
        }
        if nonfinite > 0 || invalid > 0 {
            tracing::warn!(path = %path.display(), nonfinite, invalid, "skipped invalid UFO CSV values/rows");
        }
    }
    Ok(())
}

enum Record {
    Complete(Vec<u8>),
    Oversized,
}

/// CSV records may contain quoted newlines. A tail without its final newline
/// remains uncommitted and is retried on the next scan, including a closing quote.
fn read_record(reader: &mut impl Read, limit: usize) -> io::Result<Option<Record>> {
    let mut record = Vec::new();
    let mut quoted = false;
    for byte in reader.bytes() {
        let byte = byte?;
        if byte == b'"' {
            quoted = !quoted;
        }
        if byte == b'\n' && !quoted {
            return Ok(Some(if record.len() > limit {
                Record::Oversized
            } else {
                Record::Complete(record)
            }));
        }
        // Retain one sentinel byte so oversized rows can be skipped without
        // allocating in proportion to the file or losing the next row boundary.
        if record.len() <= limit {
            record.push(byte);
        }
    }
    Ok(None)
}

fn parse_record(record: &[u8]) -> Option<Vec<String>> {
    let record = record.strip_suffix(b"\r").unwrap_or(record);
    let mut fields = Vec::new();
    let mut field = Vec::new();
    let mut quoted = false;
    let mut closed = false;
    let mut i = 0;
    while i < record.len() {
        let byte = record[i];
        if quoted {
            if byte == b'"' {
                if record.get(i + 1) == Some(&b'"') {
                    field.push(b'"');
                    i += 1;
                } else {
                    quoted = false;
                    closed = true;
                }
            } else {
                field.push(byte);
            }
        } else if byte == b',' {
            fields.push(String::from_utf8(std::mem::take(&mut field)).ok()?);
            if fields.len() >= MAX_COLUMNS {
                return None;
            }
            closed = false;
        } else if byte == b'"' && field.is_empty() && !closed {
            quoted = true;
        } else if closed || byte == b'"' {
            return None;
        } else {
            field.push(byte);
        }
        i += 1;
    }
    if quoted {
        return None;
    }
    fields.push(String::from_utf8(field).ok()?);
    Some(fields)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn fixture(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("eb-ufo-csv-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("runs/run_a/outputs/native")).unwrap();
        root
    }

    #[test]
    fn reingests_replaced_header_and_partial_tail_without_duplicates() {
        let root = fixture("rewrite");
        let native = root.join("runs/run_a/outputs/native");
        let path = native.join("train_log.txt");
        let original = b"distributed/optimizer_steps,fb_loss,timestep\n16,2.5,11264\n32,1.5,12288";
        fs::write(&path, original).unwrap();
        assert_eq!(crate::ingest_run_logs(&root, "run_a").unwrap(), 3);
        assert_eq!(crate::ingest_run_logs(&root, "run_a").unwrap(), 0);
        assert_eq!(fs::read(&path).unwrap(), original);
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"\n")
            .unwrap();
        assert_eq!(crate::ingest_run_logs(&root, "run_a").unwrap(), 3);
        let replacement = native.join(".train_log.txt.new");
        let extended = b"actor_loss,distributed/optimizer_steps,fb_loss,timestep\n,16,2.5,11264\n,32,1.5,12288\n0.8,48,1.0,13312\n";
        fs::write(&replacement, extended).unwrap();
        fs::rename(replacement, &path).unwrap();
        assert_eq!(crate::ingest_run_logs(&root, "run_a").unwrap(), 4);
        assert_eq!(crate::ingest_run_logs(&root, "run_a").unwrap(), 0);
        assert_eq!(
            eb_tsdb::query(&root, "run_a", "fb_loss").unwrap(),
            vec![(16, 2.5), (32, 1.5), (48, 1.0)]
        );
        assert_eq!(
            eb_tsdb::query(&root, "run_a", "actor_loss").unwrap(),
            vec![(48, 0.8)]
        );
        assert_eq!(
            eb_tsdb::query(&root, "run_a", "timestep").unwrap(),
            vec![(16, 11264.0), (32, 12288.0), (48, 13312.0)]
        );
        assert_eq!(fs::read(&path).unwrap(), extended);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn skips_missing_string_nonfinite_values_and_keeps_run_streams_separate() {
        let root = fixture("values");
        let csv = b"distributed/optimizer_steps,fb_loss,critic_loss,execution/matmul_precision,timestep\n16,NaN,,high,11264\n32,inf,1.5,high,12288\n48,-inf,1e100,high,13312\n64,0,2.5,high,14336\n";
        for run in ["run_a", "run_b"] {
            let native = root.join("runs").join(run).join("outputs/native");
            fs::create_dir_all(&native).unwrap();
            fs::write(native.join("train_log.txt"), csv).unwrap();
            crate::ingest_run_logs(&root, run).unwrap();
            assert_eq!(crate::ingest_run_logs(&root, run).unwrap(), 0);
            assert_eq!(
                eb_tsdb::query(&root, run, "fb_loss").unwrap(),
                vec![(64, 0.0)]
            );
            assert_eq!(
                eb_tsdb::query(&root, run, "critic_loss").unwrap(),
                vec![(32, 1.5), (64, 2.5)]
            );
            assert!(!eb_tsdb::list_tags(&root, run)
                .unwrap()
                .contains(&"execution/matmul_precision".into()));
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn tracking_keeps_every_motion_on_the_native_env_step_axis() {
        let root = fixture("tracking");
        let native = root.join("runs/run_a/outputs/native");
        fs::write(native.join("ufo_tracking_eval.csv"), b"emd,motion_name,timestep\n1.5,walk,11264\n2.5,\"turn,\"\"left\"\"\",11264\n3.5,\"two\nlines\",12288\n").unwrap();
        assert_eq!(crate::ingest_run_logs(&root, "run_a").unwrap(), 6);
        assert_eq!(crate::ingest_run_logs(&root, "run_a").unwrap(), 0);
        for (motion, step, value) in [
            ("walk", 11264, 1.5),
            ("turn,\"left\"", 11264, 2.5),
            ("two\nlines", 12288, 3.5),
        ] {
            assert_eq!(
                eb_tsdb::query(&root, "run_a", &format!("ufo_tracking/{motion}/emd")).unwrap(),
                vec![(step, value)]
            );
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bounds_records_and_columns_and_retries_quoted_tail() {
        let mut input = &b"\"too\nlong\",1\nnext,2\n\"tail\""[..];
        assert!(matches!(
            read_record(&mut input, 6).unwrap(),
            Some(Record::Oversized)
        ));
        assert!(
            matches!(read_record(&mut input, 6).unwrap(), Some(Record::Complete(record)) if record == b"next,2")
        );
        assert!(read_record(&mut input, 6).unwrap().is_none());
        assert!(parse_record(b"a,\"bad\"tail").is_none());
        assert!(parse_record(b"a,\"unterminated").is_none());
        assert!(parse_record(",".repeat(MAX_COLUMNS).as_bytes()).is_none());
    }

    #[test]
    fn refuses_linked_outputs_and_does_not_scan_source_copies() {
        let root = fixture("isolation");
        let native = root.join("runs/run_a/outputs/native");
        let csv = b"distributed/optimizer_steps,fb_loss,timestep\n16,2.5,11264\n";
        for directory in ["inputs", "code", "output", "logs"] {
            let path = root.join("runs/run_a").join(directory);
            fs::create_dir_all(&path).unwrap();
            fs::write(path.join("train_log.txt"), csv).unwrap();
        }
        let external = root.join("train_log.txt");
        fs::write(&external, csv).unwrap();
        std::os::unix::fs::symlink(&external, native.join("train_log.txt")).unwrap();
        assert_eq!(crate::ingest_run_logs(&root, "run_a").unwrap(), 0);
        fs::remove_file(native.join("train_log.txt")).unwrap();
        fs::remove_dir(&native).unwrap();
        std::os::unix::fs::symlink(root.join("runs/run_a/logs"), &native).unwrap();
        assert_eq!(crate::ingest_run_logs(&root, "run_a").unwrap(), 0);
        fs::remove_dir_all(root).unwrap();
    }
}
