use axum::body::Body;
use axum::extract::{Path, Request, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::{Parser, Subcommand};
use eb_core::{sha256_hex, GpuInfo, HostProfile};
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use sysinfo::System;
use tokio::net::UnixListener;
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "ebd")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Serve,
    Doctor,
    Scan { path: String },
    Health,
}

#[derive(Clone)]
struct App {
    data_root: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Doctor => {
            println!("{}", serde_json::to_string_pretty(&doctor())?);
        }
        Cmd::Health => {
            println!("{{\"ok\":true}}");
        }
        Cmd::Scan { path } => {
            println!("{}", serde_json::to_string_pretty(&scan(&path))?);
        }
        Cmd::Serve => serve().await?,
    }
    Ok(())
}

async fn serve() -> Result<(), Box<dyn std::error::Error>> {
    let data_root = PathBuf::from(std::env::var("EB_DATA_ROOT").unwrap_or_else(|_| "/var/lib/eb".into()));
    fs::create_dir_all(&data_root)?;
    fs::create_dir_all(data_root.join("cas"))?;
    fs::create_dir_all(data_root.join("tsdb"))?;
    fs::create_dir_all(data_root.join("runs"))?;
    let app_state = Arc::new(App { data_root: data_root.clone() });

    let control = Router::new()
        .route("/healthz", get(health))
        .route("/v1/doctor", get(doctor_handler))
        .route("/v1/runs", post(submit_run))
        .route("/v1/runs/{id}/cancel", post(cancel_run))
        .route("/v1/scan", post(scan_handler))
        .with_state(app_state.clone());

    let media = Router::new()
        .route("/healthz", get(health))
        .route("/media/{*path}", get(media_get))
        .with_state(app_state);

    let control_path = std::env::var("EB_EBD_UDS").unwrap_or_else(|_| "/run/eb/ebd.sock".into());
    let media_path = std::env::var("EB_EBD_MEDIA_UDS").unwrap_or_else(|_| "/run/eb/ebd-http.sock".into());
    bind_unix(&control_path, control).await?;
    bind_unix(&media_path, media).await?;
    tracing::info!("ebd listening on {control_path} and {media_path}");
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
    }
}

async fn bind_unix(path: &str, router: Router) -> Result<(), Box<dyn std::error::Error>> {
    let p = FsPath::new(path);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent)?;
    }
    let _ = fs::remove_file(p);
    let listener = UnixListener::bind(p)?;
    let _ = fs::set_permissions(p, fs::Permissions::from_mode(0o660));
    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, router).await {
            tracing::error!("uds server: {e}");
        }
    });
    Ok(())
}

async fn health() -> Json<Value> {
    Json(json!({"ok": true, "service": "ebd"}))
}

async fn doctor_handler() -> Json<HostProfile> {
    Json(doctor())
}

fn doctor() -> HostProfile {
    let mut sys = System::new_all();
    sys.refresh_all();
    let gpus = nvidia_smi_gpus();
    HostProfile {
        hostname: System::host_name().unwrap_or_else(|| "unknown".into()),
        gpus,
        memory_total_mb: sys.total_memory() / 1024 / 1024,
        has_dri: FsPath::new("/dev/dri").exists(),
    }
}

fn nvidia_smi_gpus() -> Vec<GpuInfo> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(out) = out else {
        return vec![];
    };
    if !out.status.success() {
        return vec![];
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let parts: Vec<_> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() < 4 {
                return None;
            }
            Some(GpuInfo {
                index: parts[0].parse().unwrap_or(0),
                name: parts[1].to_string(),
                memory_total_mb: parts[2].parse().unwrap_or(0),
                memory_free_mb: parts[3].parse().unwrap_or(0),
                run_id: None,
            })
        })
        .collect()
}

#[derive(Deserialize)]
struct RunReq {
    run_id: String,
    image: Option<String>,
    command: Option<Vec<String>>,
    kind: Option<String>,
}

enum StartFail {
    ImageMissing(String),
    Docker(String),
}

async fn submit_run(Json(req): Json<RunReq>) -> Json<Value> {
    match try_docker_run(&req).await {
        Ok(id) => Json(json!({"container_id": id, "status": "started"})),
        Err(StartFail::ImageMissing(reason)) => Json(json!({
            "container_id": null,
            "status": "image_missing",
            "reason": reason
        })),
        Err(StartFail::Docker(reason)) => Json(json!({
            "container_id": null,
            "status": "docker_error",
            "reason": reason
        })),
    }
}

async fn try_docker_run(req: &RunReq) -> Result<String, StartFail> {
    let image = req.image.clone().unwrap_or_default();
    if image.is_empty() {
        return Err(StartFail::ImageMissing("image not present: (empty)".into()));
    }
    let inspect = std::process::Command::new("docker")
        .args(["image", "inspect", &image])
        .output()
        .map_err(|e| StartFail::Docker(e.to_string()))?;
    if !inspect.status.success() {
        return Err(StartFail::ImageMissing(format!("image not present: {image}")));
    }
    let name = format!("eb-run-{}", req.run_id);
    let mut cmd = std::process::Command::new("docker");
    cmd.args(["run", "-d", "--name", &name, "--label", &format!("eb.run_id={}", req.run_id)]);
    cmd.arg(&image);
    if let Some(c) = &req.command {
        cmd.args(c);
    }
    let out = cmd.output().map_err(|e| StartFail::Docker(e.to_string()))?;
    if !out.status.success() {
        return Err(StartFail::Docker(String::from_utf8_lossy(&out.stderr).to_string()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

async fn cancel_run(Path(id): Path<String>) -> Json<Value> {
    let name = format!("eb-run-{id}");
    let _ = std::process::Command::new("docker")
        .args(["rm", "-f", &name])
        .output();
    Json(json!({"ok": true, "run_id": id}))
}

#[derive(Deserialize)]
struct ScanReq {
    path: String,
}

async fn scan_handler(Json(req): Json<ScanReq>) -> Json<Value> {
    Json(scan(&req.path))
}

fn scan(path: &str) -> Value {
    let mut files = 0u64;
    let mut bytes = 0u64;
    let mut samples = vec![];
    if FsPath::new(path).exists() {
        for ent in WalkDir::new(path).max_depth(6).into_iter().flatten() {
            if ent.file_type().is_file() {
                files += 1;
                let sz = ent.metadata().map(|m| m.len()).unwrap_or(0);
                bytes += sz;
                if samples.len() < 20 {
                    samples.push(json!({
                        "path": ent.path().display().to_string(),
                        "size": sz
                    }));
                }
            }
        }
    }
    json!({"path": path, "files": files, "bytes": bytes, "samples": samples})
}

async fn media_get(State(st): State<Arc<App>>, Path(path): Path<String>, req: Request) -> Response {
    let full = st.data_root.join("cas").join(path.trim_start_matches('/'));
    if !full.exists() {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    match fs::read(&full) {
        Ok(buf) => {
            let mut res = Response::new(Body::from(buf));
            res.headers_mut()
                .insert(header::CONTENT_TYPE, "application/octet-stream".parse().unwrap());
            let _ = req;
            res
        }
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "read").into_response(),
    }
}

#[allow(dead_code)]
fn cas_put(root: &FsPath, data: &[u8]) -> PathBuf {
    let digest = sha256_hex(data);
    let dir = root.join("cas").join(&digest[..2]);
    let _ = fs::create_dir_all(&dir);
    let p = dir.join(&digest);
    let _ = fs::write(&p, data);
    p
}
