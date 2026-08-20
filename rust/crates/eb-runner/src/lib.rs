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
}

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
            if info.status != "exited" {
                if let Some(fresh) = inspect_container(run_id) {
                    info.status = fresh.status;
                    info.exit_code = fresh.exit_code;
                    if info.status == "exited" {
                        self.release(run_id);
                    }
                    self.inner
                        .lock()
                        .unwrap()
                        .runs
                        .insert(run_id.to_string(), info.clone());
                }
            }
            return Some(info);
        }
        inspect_container(run_id)
    }

    pub fn ps(&self) -> Vec<ContainerInfo> {
        docker_ps()
    }

    pub fn image_digest(image: &str) -> Result<String, StartError> {
        let out = Command::new("docker")
            .args(["image", "inspect", "--format", "{{.Id}}", image])
            .output()
            .map_err(|e| StartError::Docker(e.to_string()))?;
        if !out.status.success() {
            return Err(StartError::ImageMissing(format!("image not present: {image}")));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    pub async fn start(&self, req: &RunRequest) -> Result<String, StartError> {
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
                },
            );
        }
        self.spawn_log_pump(&req.run_id);
        self.spawn_waiter(&req.run_id);
        Ok(container_id)
    }

    pub async fn cancel(&self, run_id: &str) -> Result<(), String> {
        let name = format!("eb-run-{run_id}");
        let _ = Command::new("docker").args(["rm", "-f", &name]).output();
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

pub fn build_docker_args(req: &RunRequest) -> Vec<String> {
    let mut args = vec![
        "run".into(),
        "-d".into(),
        "--name".into(),
        format!("eb-run-{}", req.run_id),
        "--label".into(),
        format!("eb.run_id={}", req.run_id),
    ];
    if !req.gpus.is_empty() {
        let devices = req
            .gpus
            .iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
            .join(",");
        args.push("--gpus".into());
        args.push(format!("device={devices}"));
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
    if let Some(net) = &req.network {
        args.push("--network".into());
        args.push(net.clone());
    }
    args.push(req.image.clone());
    args.extend(req.command.clone());
    args
}

fn inspect_container(run_id: &str) -> Option<ContainerInfo> {
    let name = format!("eb-run-{run_id}");
    let out = Command::new("docker")
        .args([
            "inspect",
            "--format",
            "{{.Id}} {{.State.Status}} {{.State.ExitCode}}",
            &name,
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut parts = text.split_whitespace();
    let id = parts.next()?.to_string();
    let status = parts.next()?.to_string();
    let code = parts.next().and_then(|s| s.parse().ok());
    Some(ContainerInfo {
        run_id: run_id.to_string(),
        container_id: id,
        status: if status == "running" { "running".into() } else { "exited".into() },
        exit_code: if status == "running" { None } else { code },
        image_digest: None,
    })
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
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        });
        let s = args.join(" ");
        assert!(s.contains("--gpus"));
        assert!(s.contains("device=0"));
        assert!(s.contains("--shm-size"));
        assert!(s.contains("2g"));
        assert!(s.contains("--label"));
        assert!(s.contains("eb.run_id=run_x"));
    }
}
