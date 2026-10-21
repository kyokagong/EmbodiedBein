pub mod snapshot;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mount {
    pub src: String,
    pub dst: String,
    pub mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRequest {
    pub run_id: String,
    pub image: String,
    #[serde(default)]
    pub command: Vec<String>,
    #[serde(default)]
    pub env: Vec<(String, String)>,
    #[serde(default)]
    pub gpus: Vec<u32>,
    #[serde(default = "default_shm")]
    pub shm_gb: u32,
    #[serde(default)]
    pub devices: Vec<String>,
    #[serde(default)]
    pub mounts: Vec<Mount>,
    #[serde(default)]
    pub workdir: String,
    #[serde(default)]
    pub network: Option<String>,
    #[serde(default)]
    pub hostname: Option<String>,
    #[serde(default = "default_cpus")]
    pub cpus: u32,
    #[serde(default = "default_memory")]
    pub memory_gb: u32,
}

fn default_cpus() -> u32 { 4 }
fn default_memory() -> u32 { 8 }

fn default_shm() -> u32 {
    2
}

#[derive(Debug, Error)]
pub enum StartError {
    #[error("image not present: {0}")]
    ImageMissing(String),
    #[error("docker: {0}")]
    Docker(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerInfo {
    pub run_id: String,
    pub container_id: String,
    pub status: String,
    pub exit_code: Option<i32>,
    pub image_digest: Option<String>,
    #[serde(default)]
    pub oom_killed: bool,
}

#[derive(Clone)]
pub struct Runner {
    data_root: PathBuf,
    inner: Arc<Mutex<Inner>>,
}

struct Inner {
    gpu_to_run: HashMap<u32, String>,
    runs: HashMap<String, ContainerInfo>,
}

impl Runner {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            inner: Arc::new(Mutex::new(Inner {
                gpu_to_run: HashMap::new(),
                runs: HashMap::new(),
            })),
        }
    }

    pub fn gpu_occupant(&self, index: u32) -> Option<String> {
        self.inner.lock().unwrap().gpu_to_run.get(&index).cloned()
    }

    pub fn get(&self, run_id: &str) -> Option<ContainerInfo> {
        let cached = self.inner.lock().unwrap().runs.get(run_id).cloned();
        if let Some(mut info) = cached {
            if let Some(fresh) = inspect_container(run_id) {
                info.status = fresh.status;
                info.exit_code = fresh.exit_code;
                info.oom_killed = fresh.oom_killed;
                if info.status == "exited" {
                    self.release(run_id);
                }
                self.inner
                    .lock()
                    .unwrap()
                    .runs
                    .insert(run_id.to_string(), info.clone());
            }
            return Some(info);
        }
        inspect_container(run_id)
    }

    pub fn ps(&self) -> Vec<ContainerInfo> {
        docker_ps()
    }

    pub fn image_digest(image: &str) -> Result<String, StartError> {
        Self::image_info(image).map(|(digest, _)| digest)
    }

    pub fn image_info(image: &str) -> Result<(String, u64), StartError> {
        let out = Command::new("docker")
            .args(["image", "inspect", "--format", "{{.Id}}\t{{.Size}}", image])
            .output()
            .map_err(|e| StartError::Docker(e.to_string()))?;
        if !out.status.success() {
            return Err(StartError::ImageMissing(format!("image not present: {image}")));
        }
        parse_image_inspect(&String::from_utf8_lossy(&out.stdout))
    }

    pub async fn start(&self, req: &RunRequest) -> Result<String, StartError> {
        validate_resources(req)?;
        if req.image.is_empty() {
            return Err(StartError::ImageMissing("image not present: (empty)".into()));
        }
        let digest = Self::image_digest(&req.image)?;
        let run_root = self.data_root.join("runs").join(&req.run_id);
        fs::create_dir_all(run_root.join("logs")).map_err(|e| StartError::Docker(e.to_string()))?;
        fs::create_dir_all(run_root.join("outputs")).map_err(|e| StartError::Docker(e.to_string()))?;
        fs::create_dir_all(run_root.join("configs")).map_err(|e| StartError::Docker(e.to_string()))?;

        let args = build_docker_args(req);
        let out = Command::new("docker")
            .args(&args)
            .output()
            .map_err(|e| StartError::Docker(e.to_string()))?;
        if !out.status.success() {
            return Err(StartError::Docker(String::from_utf8_lossy(&out.stderr).to_string()));
        }
        let container_id = String::from_utf8_lossy(&out.stdout).trim().to_string();
        {
            let mut g = self.inner.lock().unwrap();
            for gpu in &req.gpus {
                g.gpu_to_run.insert(*gpu, req.run_id.clone());
            }
            g.runs.insert(
                req.run_id.clone(),
                ContainerInfo {
                    run_id: req.run_id.clone(),
                    container_id: container_id.clone(),
                    status: "running".into(),
                    exit_code: None,
                    image_digest: Some(digest),
                    oom_killed: false,
                },
            );
        }
        self.spawn_log_pump(&req.run_id);
        self.spawn_waiter(&req.run_id);
        Ok(container_id)
    }

    pub async fn cancel(&self, run_id: &str) -> Result<(), String> {
        let name = format!("eb-run-{run_id}");
        let removed = Command::new("docker").args(["rm", "-f", &name]).output()
            .map_err(|e| format!("docker removal failed: {e}"))?;
        let inspected = if removed.status.success() { None } else {
            Some(Command::new("docker").args(["inspect", &name]).output()
                .map_err(|e| format!("could not confirm container removal: {e}"))?)
        };
        check_cancel_result(&name, &removed, inspected.as_ref())?;
        self.release(run_id);
        Ok(())
    }

    fn release(&self, run_id: &str) {
        let mut g = self.inner.lock().unwrap();
        g.gpu_to_run.retain(|_, v| v != run_id);
        if let Some(info) = g.runs.get_mut(run_id) {
            if info.status != "exited" {
                info.status = "exited".into();
            }
        }
    }

    fn spawn_log_pump(&self, run_id: &str) {
        let name = format!("eb-run-{run_id}");
        let log_path = self.data_root.join("runs").join(run_id).join("console.log");
        tokio::spawn(async move {
            let _ = tokio::task::spawn_blocking(move || {
                let file = File::create(&log_path).ok();
                let Some(file) = file else {
                    return;
                };
                let _ = Command::new("docker")
                    .args(["logs", "-f", &name])
                    .stdout(Stdio::from(file.try_clone().unwrap()))
                    .stderr(Stdio::from(file))
                    .status();
            })
            .await;
        });
    }

    fn spawn_waiter(&self, run_id: &str) {
        let name = format!("eb-run-{run_id}");
        let inner = self.inner.clone();
        let run_id = run_id.to_string();
        tokio::spawn(async move {
            let _ = tokio::task::spawn_blocking(move || {
                let out = Command::new("docker").args(["wait", &name]).output();
                let code = out.ok().and_then(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .trim()
                        .parse::<i32>()
                        .ok()
                });
                let mut g = inner.lock().unwrap();
                g.gpu_to_run.retain(|_, v| v != &run_id);
                if let Some(info) = g.runs.get_mut(&run_id) {
                    info.status = "exited".into();
                    info.exit_code = code;
                }
            })
            .await;
        });
    }
}

fn check_cancel_result(
    name: &str, removed: &std::process::Output, inspected: Option<&std::process::Output>,
) -> Result<(), String> {
    if removed.status.success() { return Ok(()); }
    if let Some(result) = inspected {
        let error = String::from_utf8_lossy(&result.stderr);
        if !result.status.success() && result.stdout.is_empty()
            && [format!("Error: No such object: {name}"), format!("Error: No such container: {name}")]
                .iter().any(|missing| error.trim() == missing)
        { return Ok(()); }
    }
    Err(format!("container removal not confirmed: {}", String::from_utf8_lossy(&removed.stderr).trim()))
}

pub fn build_docker_args(req: &RunRequest) -> Vec<String> {
    let mut args = vec![
        "run".into(),
        "-d".into(),
        "--name".into(),
        format!("eb-run-{}", req.run_id),
        "--label".into(),
        format!("eb.run_id={}", req.run_id),
    ];
    args.extend(["--cpus".into(), req.cpus.to_string(), "--memory".into(), format!("{}g", req.memory_gb)]);
    if !req.gpus.is_empty() {
        let devices = req
            .gpus
            .iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
            .join(",");
        args.push("--gpus".into());
        // Docker parses this argv value as CSV; quote the entire device field.
        args.push(format!("\"device={devices}\""));
    }
    args.push("--shm-size".into());
    args.push(format!("{}g", req.shm_gb));
    for d in &req.devices {
        args.push("--device".into());
        args.push(d.clone());
    }
    for m in &req.mounts {
        args.push("-v".into());
        args.push(format!("{}:{}:{}", m.src, m.dst, m.mode));
    }
    for (k, v) in &req.env {
        args.push("-e".into());
        args.push(format!("{k}={v}"));
    }
    if !req.workdir.is_empty() {
        args.push("-w".into());
        args.push(req.workdir.clone());
    }
    args.push("--network".into());
    args.push(req.network.clone().unwrap_or_else(|| "bridge".into()));
    if let Some(hostname) = &req.hostname {
        args.extend(["--hostname".into(), hostname.clone()]);
    }
    args.push(req.image.clone());
    args.extend(req.command.clone());
    args
}

fn validate_resources(req: &RunRequest) -> Result<(), StartError> {
    if req.hostname.as_deref().is_some_and(|hostname| hostname != "localhost") {
        return Err(StartError::Docker("only the localhost task hostname is supported".into()));
    }
    if req.cpus == 0 || req.memory_gb == 0 || req.shm_gb == 0 || req.shm_gb > req.memory_gb {
        return Err(StartError::Docker("cpus, memory_gb and shm_gb must be positive; shm_gb must not exceed memory_gb".into()));
    }
    if !matches!(req.network.as_deref(), None | Some("bridge") | Some("none")) {
        return Err(StartError::Docker("task network must be bridge or none".into()));
    }
    Ok(())
}

fn inspect_container(run_id: &str) -> Option<ContainerInfo> {
    let name = format!("eb-run-{run_id}");
    let out = Command::new("docker")
        .args([
            "inspect",
            "--format",
            "{{.Id}} {{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}",
            &name,
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    parse_container_inspect(run_id, &text)
}

fn parse_container_inspect(run_id: &str, text: &str) -> Option<ContainerInfo> {
    let mut parts = text.split_whitespace();
    let id = parts.next()?.to_string();
    let status = parts.next()?.to_string();
    let code = parts.next().and_then(|s| s.parse().ok());
    let oom_killed = parts.next()?.parse().ok()?;
    Some(ContainerInfo {
        run_id: run_id.to_string(),
        container_id: id,
        status: if status == "running" { "running".into() } else { "exited".into() },
        exit_code: if status == "running" { None } else { code },
        image_digest: None,
        oom_killed,
    })
}

fn parse_image_inspect(text: &str) -> Result<(String, u64), StartError> {
    let (digest, size) = text.trim().split_once('\t')
        .ok_or_else(|| StartError::Docker("invalid docker image inspect output".into()))?;
    let size = size.parse()
        .map_err(|_| StartError::Docker("invalid docker image size".into()))?;
    Ok((digest.to_string(), size))
}

fn docker_ps() -> Vec<ContainerInfo> {
    let out = Command::new("docker")
        .args([
            "ps",
            "-a",
            "--filter",
            "label=eb.run_id",
            "--format",
            "{{.ID}}\t{{.Label \"eb.run_id\"}}\t{{.State}}\t{{.Status}}",
        ])
        .output();
    let Ok(out) = out else {
        return vec![];
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let mut p = line.split('\t');
            let id = p.next()?.to_string();
            let run_id = p.next()?.to_string();
            let state = p.next().unwrap_or("").to_string();
            Some(ContainerInfo {
                run_id,
                container_id: id,
                status: if state == "running" {
                    "running".into()
                } else {
                    "exited".into()
                },
                exit_code: None,
                image_digest: None,
                oom_killed: false,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_requires_removal_or_confirmed_absence() {
        use std::os::unix::process::ExitStatusExt;
        let output = |success: bool, stderr: &str| std::process::Output {
            status: std::process::ExitStatus::from_raw(if success { 0 } else { 256 }),
            stdout: vec![], stderr: stderr.as_bytes().to_vec(),
        };
        let failed = output(false, "daemon unavailable");
        assert!(check_cancel_result("eb-run-x", &output(true, ""), None).is_ok());
        assert!(check_cancel_result("eb-run-x", &failed, Some(&failed)).is_err());
        assert!(check_cancel_result("eb-run-x", &failed, Some(&output(true, ""))).is_err());
        assert!(check_cancel_result("eb-run-x", &failed, Some(&output(false, "Error: No such object: eb-run-x\n"))).is_ok());
        assert!(check_cancel_result("eb-run-x", &failed, Some(&output(false, "Error: No such object: another"))).is_err());
    }

    #[test]
    fn inspect_preserves_oom_while_container_is_running() {
        let info = parse_container_inspect("run_x", "abc running 0 true").unwrap();
        assert_eq!(info.status, "running");
        assert_eq!(info.exit_code, None);
        assert!(info.oom_killed);
        let exited = parse_container_inspect("run_x", "abc exited 137 true").unwrap();
        assert_eq!(exited.exit_code, Some(137));
        assert!(exited.oom_killed);
        assert!(!parse_container_inspect("run_x", "abc running 0 false").unwrap().oom_killed);
    }

    #[test]
    fn parses_image_digest_and_size() {
        assert_eq!(parse_image_inspect("sha256:abc\t123\n").unwrap(), ("sha256:abc".into(), 123));
        assert!(parse_image_inspect("sha256:abc").is_err());
    }

    #[test]
    fn docker_args_include_gpu_and_shm() {
        let args = build_docker_args(&RunRequest {
            run_id: "run_x".into(),
            image: "eb/isaac-lab:2.3.2".into(),
            command: vec!["python".into(), "scripts/rsl_rl/train.py".into()],
            env: vec![("ACCEPT_EULA".into(), "Y".into())],
            gpus: vec![0],
            shm_gb: 2,
            devices: vec!["/dev/dri".into()],
            mounts: vec![Mount {
                src: "/wt".into(),
                dst: "/workspace/isaaclab_extension_template".into(),
                mode: "ro".into(),
            }],
            workdir: "/workspace/isaaclab_extension_template".into(),
            network: None,
            hostname: None,
            cpus: 6,
            memory_gb: 24,
        });
        let s = args.join(" ");
        assert!(s.contains("--gpus"));
        assert!(s.contains("device=0"));
        assert!(s.contains("--shm-size"));
        assert!(s.contains("2g"));
        assert!(s.contains("--label"));
        assert!(s.contains("eb.run_id=run_x"));
        assert!(s.contains("--cpus 6 --memory 24g"));
        assert!(s.contains("--network bridge"));
    }

    #[test]
    fn isolated_task_can_resolve_localhost_without_opening_a_network() {
        let mut req: RunRequest = serde_json::from_value(serde_json::json!({
            "run_id":"run_native", "image":"native", "network":"none", "hostname":"localhost"
        })).unwrap();
        assert!(validate_resources(&req).is_ok());
        let args = build_docker_args(&req);
        assert!(args.windows(2).any(|items| items == ["--hostname", "localhost"]));
        assert!(args.windows(2).any(|items| items == ["--network", "none"]));
        req.hostname = Some("external.example".into());
        assert!(validate_resources(&req).is_err());
    }

    #[test]
    fn rejects_invalid_limits_and_host_network() {
        let mut req: RunRequest = serde_json::from_str(r#"{"run_id":"x","image":"test"}"#).unwrap();
        assert!(validate_resources(&req).is_ok());
        req.cpus = 0;
        assert!(validate_resources(&req).is_err());
        req.cpus = 4;
        req.network = Some("host".into());
        assert!(validate_resources(&req).is_err());
        req.network = Some("bridge".into());
        req.shm_gb = 9;
        assert!(validate_resources(&req).is_err());
    }

    #[test]
    fn multiple_gpus_are_one_csv_field() {
        let mut req: RunRequest = serde_json::from_str(r#"{"run_id":"x","image":"test"}"#).unwrap();
        req.gpus = vec![0, 1, 2, 3];
        let args = build_docker_args(&req);
        let position = args.iter().position(|arg| arg == "--gpus").unwrap();
        assert_eq!(args[position + 1], "\"device=0,1,2,3\"");
    }
}
