//! Read-only resource evidence for live native UFO training containers.
//! TSDB axes are Unix seconds. GPU counters describe the explicitly leased
//! device, not per-process allocation; Docker counters describe the container.

use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

const MAX_OUTPUT: u64 = 64 * 1024;
const SCHEMA: &str = "eb/ufo-native-train-plan/v1";
const CSV_HEADER: &str = "wall_unix_s,monitor_start_unix_s,elapsed_s,container_id,assigned_devices,gpu_index,gpu_uuid,gpu_memory_mib,gpu_utilization_percent,cpu_percent,memory_mib,shm_used_mib,shm_capacity_mib,gpu_scope,status\n";
const INSPECT_FORMAT: &str = "{\"id\":{{json .Id}},\"image\":{{json .Image}},\"run_id\":{{json (index .Config.Labels \"eb.run_id\")}},\"running\":{{json .State.Running}},\"pid\":{{json .State.Pid}},\"devices\":{{json .HostConfig.DeviceRequests}}}";

#[derive(Deserialize)]
struct Plan {
    schema_version: String,
    run_id: String,
    image_digest: String,
    world_size: usize,
}

#[derive(Deserialize)]
struct Inspection {
    id: String,
    image: String,
    run_id: String,
    running: bool,
    pid: u32,
    devices: Option<Vec<DeviceRequest>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct DeviceRequest {
    #[serde(default)]
    driver: String,
    count: i64,
    #[serde(rename = "DeviceIDs")]
    device_ids: Option<Vec<String>>,
    capabilities: Vec<Vec<String>>,
}

#[derive(Clone, Debug, PartialEq)]
struct GpuSample {
    index: u32,
    uuid: String,
    memory_mib: f32,
    utilization_percent: f32,
}

#[derive(Debug)]
struct ContainerSample {
    cpu_percent: f32,
    memory_mib: f32,
}

fn valid_id(value: &str, prefix: &str) -> bool {
    value.starts_with(prefix)
        && value.len() > prefix.len()
        && value.len() <= 128
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn hex_id(value: &str) -> bool {
    (12..=64).contains(&value.len()) && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn unlinked(path: &Path) -> bool {
    !path
        .ancestors()
        .any(|part| fs::symlink_metadata(part).is_ok_and(|m| m.file_type().is_symlink()))
}

fn read_plan(root: &Path, run_id: &str) -> Option<Plan> {
    if !valid_id(run_id, "run_") {
        return None;
    }
    let path = root.join("runs").join(run_id).join("ufo-train-plan.json");
    if !unlinked(&path) {
        return None;
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .ok()?;
    let meta = file.metadata().ok()?;
    if !meta.is_file() || meta.len() > 1024 * 1024 {
        return None;
    }
    let mut bytes = Vec::new();
    file.take(1024 * 1024 + 1).read_to_end(&mut bytes).ok()?;
    if bytes.len() > 1024 * 1024 {
        return None;
    }
    let plan: Plan = serde_json::from_slice(&bytes).ok()?;
    (plan.schema_version == SCHEMA
        && plan.run_id == run_id
        && (1..=4).contains(&plan.world_size)
        && plan
            .image_digest
            .strip_prefix("sha256:")
            .is_some_and(|h| h.len() == 64 && hex_id(h)))
    .then_some(plan)
}

fn explicit_devices(plan: &Plan, inspect: &Inspection, container_id: &str) -> Option<Vec<String>> {
    if !inspect.running
        || inspect.pid == 0
        || !hex_id(&inspect.id)
        || !inspect.id.starts_with(container_id)
        || inspect.run_id != plan.run_id
        || inspect.image != plan.image_digest
    {
        return None;
    }
    let requests: Vec<_> = inspect
        .devices
        .as_ref()?
        .iter()
        .filter(|request| {
            request
                .capabilities
                .iter()
                .flatten()
                .any(|capability| capability == "gpu")
        })
        .collect();
    if requests.len() != 1 {
        return None;
    }
    let request = requests[0];
    let ids = request.device_ids.as_ref()?;
    if !matches!(request.driver.as_str(), "" | "nvidia")
        || request.count != 0
        || ids.len() != plan.world_size
        || ids.iter().collect::<HashSet<_>>().len() != ids.len()
        || ids
            .iter()
            .any(|id| !(id.parse::<u32>().is_ok() || valid_id(id, "GPU-")))
    {
        return None;
    }
    Some(ids.clone())
}

fn nonnegative(text: &str) -> Option<f32> {
    text.trim()
        .parse::<f32>()
        .ok()
        .filter(|value| value.is_finite() && *value >= 0.0)
}

fn parse_gpus(text: &str) -> Vec<GpuSample> {
    text.lines()
        .take(4)
        .filter_map(|line| {
            let fields: Vec<_> = line.split(',').map(str::trim).collect();
            if fields.len() != 4 || !valid_id(fields[1], "GPU-") {
                return None;
            }
            let utilization_percent = nonnegative(fields[3]).filter(|value| *value <= 100.0)?;
            Some(GpuSample {
                index: fields[0].parse().ok()?,
                uuid: fields[1].into(),
                memory_mib: nonnegative(fields[2])?,
                utilization_percent,
            })
        })
        .collect()
}

fn attributed_gpus(ids: &[String], samples: &[GpuSample]) -> Option<Vec<GpuSample>> {
    if samples.len() != ids.len() {
        return None;
    }
    let selected: Vec<_> = ids
        .iter()
        .map(|id| {
            samples
                .iter()
                .find(|sample| id.parse::<u32>().ok() == Some(sample.index) || id == &sample.uuid)
                .cloned()
        })
        .collect::<Option<Vec<_>>>()?;
    (selected
        .iter()
        .map(|sample| sample.index)
        .collect::<HashSet<_>>()
        .len()
        == ids.len())
    .then_some(selected)
}

fn memory_mib(value: &str) -> Option<f32> {
    let value = value.trim();
    let split = value.find(|c: char| !c.is_ascii_digit() && c != '.')?;
    let amount = nonnegative(&value[..split])?;
    let scale = match value[split..].trim() {
        "B" => 1.0 / 1048576.0,
        "kB" => 1000.0 / 1048576.0,
        "MB" => 1000000.0 / 1048576.0,
        "GB" => 1000000000.0 / 1048576.0,
        "KiB" => 1.0 / 1024.0,
        "MiB" => 1.0,
        "GiB" => 1024.0,
        "TiB" => 1048576.0,
        _ => return None,
    };
    let result = amount * scale;
    result.is_finite().then_some(result)
}

fn parse_stats(text: &str, inspect: &Inspection) -> Option<ContainerSample> {
    let value: Value = serde_json::from_str(text.trim()).ok()?;
    let id = value["ID"].as_str()?;
    if !hex_id(id)
        || !inspect.id.starts_with(id)
        || value["Name"].as_str()? != format!("eb-run-{}", inspect.run_id)
    {
        return None;
    }
    Some(ContainerSample {
        cpu_percent: nonnegative(value["CPUPerc"].as_str()?.strip_suffix('%')?)?,
        memory_mib: memory_mib(value["MemUsage"].as_str()?.split('/').next()?)?,
    })
}

async fn command(program: &str, args: &[String]) -> Result<String, String> {
    let mut child = Command::new(program)
        .args(args)
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| e.to_string())?;
    tokio::time::timeout(Duration::from_secs(3), async move {
        let mut bytes = Vec::new();
        child
            .stdout
            .take()
            .ok_or("missing resource probe stdout")?
            .take(MAX_OUTPUT + 1)
            .read_to_end(&mut bytes)
            .await
            .map_err(|e| e.to_string())?;
        if bytes.len() as u64 > MAX_OUTPUT {
            return Err("resource probe output exceeds limit".into());
        }
        if !child.wait().await.map_err(|e| e.to_string())?.success() {
            return Err(format!("{program} resource probe failed"));
        }
        String::from_utf8(bytes).map_err(|e| e.to_string())
    })
    .await
    .map_err(|_| format!("{program} resource probe timed out"))?
}

async fn inspect(id: &str) -> Option<Inspection> {
    let output = command(
        "docker",
        &[
            "inspect".into(),
            "--format".into(),
            INSPECT_FORMAT.into(),
            id.into(),
        ],
    )
    .await
    .ok()?;
    serde_json::from_str(output.trim()).ok()
}

fn shm_mib(pid: u32, container_id: &str) -> Option<(f32, f32)> {
    let mut membership = String::new();
    fs::File::open(format!("/proc/{pid}/cgroup"))
        .ok()?
        .take(4096)
        .read_to_string(&mut membership)
        .ok()?;
    if !membership.contains(container_id) {
        return None;
    }
    let path = CString::new(format!("/proc/{pid}/root/dev/shm")).ok()?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // PID comes only from a confirmed live task container in ebd's host PID namespace.
    if unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return None;
    }
    let stats = unsafe { stats.assume_init() };
    let scale = stats.f_frsize as f64 / 1048576.0;
    Some((
        ((stats.f_blocks.saturating_sub(stats.f_bfree)) as f64 * scale) as f32,
        (stats.f_blocks as f64 * scale) as f32,
    ))
}

struct Sample {
    inspect: Inspection,
    devices: Vec<String>,
    gpus: Vec<GpuSample>,
    container: Option<ContainerSample>,
    wall: i64,
    monitor_start: i64,
    elapsed: f64,
}

fn write_sample(root: &Path, sample: &Sample) -> Result<(), String> {
    eb_collector::with_storage_lock(|| {
        let output = root
            .join("runs")
            .join(&sample.inspect.run_id)
            .join("outputs");
        if !output.is_dir() || !unlinked(&output) {
            return Ok(());
        }
        let path = output.join("resources.csv");
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&path)
            .map_err(|e| e.to_string())?;
        let meta = file.metadata().map_err(|e| e.to_string())?;
        if !meta.is_file() {
            return Err("resource output is not a regular file".into());
        }
        if meta.len() == 0 {
            file.write_all(CSV_HEADER.as_bytes())
                .map_err(|e| e.to_string())?;
        } else {
            let mut header = vec![0; CSV_HEADER.len()];
            file.read_exact(&mut header).map_err(|e| e.to_string())?;
            if header != CSV_HEADER.as_bytes() {
                return Err("resource CSV header differs".into());
            }
        }
        let shm = shm_mib(sample.inspect.pid, &sample.inspect.id);
        let number = |value: Option<f32>| value.map(|n| n.to_string()).unwrap_or_default();
        let cpu = sample.container.as_ref().map(|s| s.cpu_percent);
        let memory = sample.container.as_ref().map(|s| s.memory_mib);
        let mut status = Vec::new();
        if sample.gpus.is_empty() {
            status.push("gpu_unavailable");
        }
        if sample.container.is_none() {
            status.push("container_stats_unavailable");
        }
        if shm.is_none() {
            status.push("shm_unavailable");
        }
        let status = if status.is_empty() {
            "ok".into()
        } else {
            status.join(";")
        };
        let gpus: Vec<_> = if sample.gpus.is_empty() {
            vec![None]
        } else {
            sample.gpus.iter().map(Some).collect()
        };
        for gpu in gpus {
            writeln!(
                file,
                "{},{},{:.3},{},{},{},{},{},{},{},{},{},{},assigned_device,{}",
                sample.wall,
                sample.monitor_start,
                sample.elapsed,
                sample.inspect.id,
                sample.devices.join(";"),
                gpu.map(|g| g.index.to_string()).unwrap_or_default(),
                gpu.map(|g| g.uuid.as_str()).unwrap_or(""),
                number(gpu.map(|g| g.memory_mib)),
                number(gpu.map(|g| g.utilization_percent)),
                number(cpu),
                number(memory),
                number(shm.map(|s| s.0)),
                number(shm.map(|s| s.1)),
                status
            )
            .map_err(|e| e.to_string())?;
        }
        let append = |tag: &str, value: f32| {
            eb_tsdb::append(root, &sample.inspect.run_id, tag, sample.wall, value)
                .map_err(|e| e.to_string())
        };
        for gpu in &sample.gpus {
            append(
                &format!("resource/gpu_{}/memory_mib", gpu.index),
                gpu.memory_mib,
            )?;
            append(
                &format!("resource/gpu_{}/utilization_percent", gpu.index),
                gpu.utilization_percent,
            )?;
        }
        if let Some(value) = cpu {
            append("resource/cpu_percent", value)?;
        }
        if let Some(value) = memory {
            append("resource/memory_mib", value)?;
        }
        if let Some((used, capacity)) = shm {
            append("resource/shm_used_mib", used)?;
            append("resource/shm_capacity_mib", capacity)?;
        }
        Ok(())
    })
}

pub async fn spawn(root: PathBuf) {
    let mut observed: HashMap<String, (i64, Instant)> = HashMap::new();
    loop {
        let listed = command(
            "docker",
            &[
                "ps".into(),
                "--filter".into(),
                "label=eb.run_id".into(),
                "--filter".into(),
                "status=running".into(),
                "--format".into(),
                "{{.ID}}\t{{.Label \"eb.run_id\"}}".into(),
            ],
        )
        .await;
        let mut active = HashSet::new();
        if let Ok(listed) = listed {
            for line in listed.lines().take(128) {
                let Some((container_id, run_id)) = line.split_once('\t') else {
                    continue;
                };
                if !hex_id(container_id) || !valid_id(run_id, "run_") {
                    continue;
                }
                let (plan_root, plan_run) = (root.clone(), run_id.to_owned());
                let Ok(Some(plan)) =
                    tokio::task::spawn_blocking(move || read_plan(&plan_root, &plan_run)).await
                else {
                    continue;
                };
                let Some(before) = inspect(container_id).await else {
                    continue;
                };
                let Some(devices) = explicit_devices(&plan, &before, container_id) else {
                    continue;
                };
                active.insert(run_id.to_owned());
                let gpu_args = [
                    format!("--id={}", devices.join(",")),
                    "--query-gpu=index,uuid,memory.used,utilization.gpu".into(),
                    "--format=csv,noheader,nounits".into(),
                ];
                let stats_args = [
                    "stats".into(),
                    "--no-stream".into(),
                    "--format".into(),
                    "{{json .}}".into(),
                    before.id.clone(),
                ];
                let (gpu_result, stats_result) = tokio::join!(
                    command("nvidia-smi", &gpu_args),
                    command("docker", &stats_args)
                );
                let gpus = gpu_result
                    .ok()
                    .and_then(|text| attributed_gpus(&devices, &parse_gpus(&text)))
                    .unwrap_or_default();
                let container = stats_result
                    .ok()
                    .and_then(|text| parse_stats(&text, &before));
                // A completed/replaced task must not acquire the next owner's GPU evidence.
                let Some(after) = inspect(&before.id).await else {
                    continue;
                };
                if explicit_devices(&plan, &after, &before.id).as_ref() != Some(&devices)
                    || after.pid != before.pid
                {
                    continue;
                }
                let wall = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64;
                let (monitor_start, instant) = observed
                    .entry(run_id.to_owned())
                    .or_insert((wall, Instant::now()));
                let sample = Sample {
                    inspect: after,
                    devices,
                    gpus,
                    container,
                    wall,
                    monitor_start: *monitor_start,
                    elapsed: instant.elapsed().as_secs_f64(),
                };
                let write_root = root.clone();
                match tokio::task::spawn_blocking(move || write_sample(&write_root, &sample)).await
                {
                    Ok(Ok(())) => (),
                    error => {
                        tracing::warn!(?error, run_id, "native training resource evidence failed")
                    }
                }
            }
        }
        observed.retain(|run_id, _| active.contains(run_id));
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn plan() -> Plan {
        Plan {
            schema_version: SCHEMA.into(),
            run_id: "run_test".into(),
            world_size: 2,
            image_digest: format!("sha256:{}", "a".repeat(64)),
        }
    }

    fn inspection() -> Inspection {
        serde_json::from_value(json!({"id":"a".repeat(64), "image":plan().image_digest,
        "run_id":"run_test", "running":true, "pid":123, "devices":[{
            "Driver":"", "Count":0, "DeviceIDs":["1", "GPU-abc"], "Capabilities":[["gpu"]]
        }]}))
        .unwrap()
    }

    #[test]
    fn requires_live_matching_container_and_explicit_gpu_lease() {
        let mut inspect = inspection();
        assert_eq!(
            explicit_devices(&plan(), &inspect, &"a".repeat(12)),
            Some(vec!["1".into(), "GPU-abc".into()])
        );
        inspect.run_id = "run_other".into();
        assert!(explicit_devices(&plan(), &inspect, &"a".repeat(12)).is_none());
        inspect = inspection();
        inspect.devices.as_mut().unwrap()[0].count = -1;
        assert!(explicit_devices(&plan(), &inspect, &"a".repeat(12)).is_none());
        inspect = inspection();
        inspect.devices.as_mut().unwrap()[0].device_ids = Some(vec!["1".into(), "1".into()]);
        assert!(explicit_devices(&plan(), &inspect, &"a".repeat(12)).is_none());
        inspect = inspection();
        inspect.running = false;
        assert!(explicit_devices(&plan(), &inspect, &"a".repeat(12)).is_none());
    }

    #[test]
    fn gpu_values_are_finite_and_attributed_to_exact_index_or_uuid() {
        let rows = parse_gpus(
            "0, GPU-unassigned, 999, 99\n1, GPU-assigned, 1024, 40\n2, GPU-abc, 512, 20\n",
        );
        let assigned = attributed_gpus(&["1".into(), "GPU-abc".into()], &rows[1..]).unwrap();
        assert_eq!(
            assigned.iter().map(|g| g.index).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert!(attributed_gpus(&["1".into(), "GPU-assigned".into()], &rows[1..]).is_none());
        assert!(attributed_gpus(&["3".into()], &rows[1..2]).is_none());
        assert!(parse_gpus("1, GPU-a, N/A, 5\n2, GPU-b, NaN, 5\n3, GPU-c, 100, 101\n").is_empty());
    }

    #[test]
    fn docker_stats_preserve_real_units_and_refuse_other_container() {
        let inspect = inspection();
        let text = json!({"ID":"a".repeat(12), "Name":"eb-run-run_test", "CPUPerc":"350.5%", "MemUsage":"1.5GiB / 16GiB"}).to_string();
        let sample = parse_stats(&text, &inspect).unwrap();
        assert_eq!(sample.cpu_percent, 350.5);
        assert_eq!(sample.memory_mib, 1536.0);
        assert_eq!(memory_mib("1024KiB"), Some(1.0));
        assert!(memory_mib("NaNMiB").is_none());
        assert!(parse_stats(&text.replace("run_test", "run_other"), &inspect).is_none());
    }

    #[test]
    fn only_bounded_owned_native_plans_qualify() {
        let root =
            std::env::temp_dir().join(format!("eb-native-resources-plan-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let run = root.join("runs/run_test");
        fs::create_dir_all(&run).unwrap();
        let path = run.join("ufo-train-plan.json");
        fs::write(
            &path,
            json!({"schema_version":SCHEMA, "run_id":"run_test", "world_size":2,
            "image_digest":plan().image_digest})
            .to_string(),
        )
        .unwrap();
        assert!(read_plan(&root, "run_test").is_some());
        assert!(read_plan(&root, "../run_test").is_none());
        fs::write(&path, vec![b' '; 1024 * 1024 + 1]).unwrap();
        assert!(read_plan(&root, "run_test").is_none());
        fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink("/etc/passwd", &path).unwrap();
        assert!(read_plan(&root, "run_test").is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn raw_missing_metrics_stay_blank_and_no_gpu_points_are_fabricated() {
        let root =
            std::env::temp_dir().join(format!("eb-native-resources-csv-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("runs/run_test/outputs")).unwrap();
        let sample = Sample {
            inspect: Inspection {
                pid: u32::MAX,
                ..inspection()
            },
            devices: vec!["1".into(), "GPU-abc".into()],
            gpus: vec![],
            container: None,
            wall: 1700000000,
            monitor_start: 1700000000,
            elapsed: 0.0,
        };
        write_sample(&root, &sample).unwrap();
        let raw = fs::read_to_string(root.join("runs/run_test/outputs/resources.csv")).unwrap();
        assert!(raw.contains(
            ",,,,,,,,assigned_device,gpu_unavailable;container_stats_unavailable;shm_unavailable"
        ));
        assert!(eb_tsdb::list_tags(&root, "run_test").unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn writes_resource_points_on_unix_seconds_with_exact_gpu_tags() {
        let root =
            std::env::temp_dir().join(format!("eb-native-resources-points-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("runs/run_test/outputs")).unwrap();
        let sample = Sample {
            inspect: Inspection {
                pid: u32::MAX,
                ..inspection()
            },
            devices: vec!["1".into()],
            gpus: parse_gpus("1, GPU-test, 512, 25\n"),
            container: Some(ContainerSample {
                cpu_percent: 150.0,
                memory_mib: 2048.0,
            }),
            wall: 1700000001,
            monitor_start: 1700000000,
            elapsed: 1.0,
        };
        write_sample(&root, &sample).unwrap();
        assert_eq!(
            eb_tsdb::query(&root, "run_test", "resource/gpu_1/memory_mib").unwrap(),
            vec![(1700000001, 512.0)]
        );
        assert_eq!(
            eb_tsdb::query(&root, "run_test", "resource/cpu_percent").unwrap(),
            vec![(1700000001, 150.0)]
        );
        assert!(eb_tsdb::list_tags(&root, "run_test")
            .unwrap()
            .iter()
            .all(|tag| !tag.contains("gpu_0")));
        assert!(eb_tsdb::list_tags(&root, "run_other").unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn command_output_and_wait_are_bounded_without_blocking_timers() {
        let output = command("head", &["-c".into(), "65537".into(), "/dev/zero".into()]).await;
        assert!(output.unwrap_err().contains("exceeds limit"));
        let task = tokio::spawn(async { command("sleep", &["10".into()]).await });
        let started = Instant::now();
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(started.elapsed() < Duration::from_millis(500));
        assert!(task.await.unwrap().unwrap_err().contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}
