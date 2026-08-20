use axum::body::Body;
use axum::extract::{Path, Query, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::{Parser, Subcommand};
use eb_core::HostProfile;
use eb_runner::{RunRequest, Runner, StartError};
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use tokio::net::UnixListener;

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
    runner: Runner,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Doctor => {
            println!("{}", serde_json::to_string_pretty(&eb_probe::doctor())?);
        }
        Cmd::Health => {
            println!("{{\"ok\":true}}");
        }
        Cmd::Scan { path } => {
            println!("{}", serde_json::to_string_pretty(&eb_scanner::scan(&path))?);
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
    fs::create_dir_all(data_root.join("worktrees"))?;
    let runner = Runner::new(data_root.clone());
    let app_state = Arc::new(App {
        data_root: data_root.clone(),
        runner,
    });

    let control = Router::new()
        .route("/healthz", get(health))
        .route("/v1/doctor", get(doctor_handler))
        .route("/v1/runs", post(submit_run))
        .route("/v1/runs/{id}", get(get_run))
        .route("/v1/runs/{id}/cancel", post(cancel_run))
        .route("/v1/runs/{id}/metrics", get(run_metrics))
        .route("/v1/runs/{id}/logs", get(run_logs))
        .route("/v1/ps", get(ps_handler))
        .route("/v1/scan", post(scan_handler))
        .with_state(app_state.clone());

    let media = Router::new()
        .route("/healthz", get(health))
        .route("/media/{*path}", get(media_get))
        .with_state(app_state.clone());

    let control_path = std::env::var("EB_EBD_UDS").unwrap_or_else(|_| "/run/eb/ebd.sock".into());
    let media_path = std::env::var("EB_EBD_MEDIA_UDS").unwrap_or_else(|_| "/run/eb/ebd-http.sock".into());
    bind_unix(&control_path, control).await?;
    bind_unix(&media_path, media).await?;
    tracing::info!("ebd listening on {control_path} and {media_path}");
    let root = data_root.clone();
    tokio::spawn(async move { eb_collector::spawn(root).await });
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

async fn doctor_handler(State(st): State<Arc<App>>) -> Json<HostProfile> {
    let mut profile = eb_probe::doctor();
    for gpu in &mut profile.gpus {
        gpu.run_id = st.runner.gpu_occupant(gpu.index);
    }
    Json(profile)
}

#[derive(Deserialize)]
struct SubmitBody {
    run_id: String,
    #[serde(default)]
    image: String,
    #[serde(default)]
    command: Vec<String>,
    #[serde(default)]
    env: Value,
    #[serde(default)]
    gpus: Vec<u32>,
    #[serde(default)]
    shm_gb: Option<u32>,
    #[serde(default)]
    devices: Vec<String>,
    #[serde(default)]
    mounts: Vec<eb_runner::Mount>,
    #[serde(default)]
    workdir: String,
    #[serde(default)]
    network: Option<String>,
}

fn env_pairs(env: &Value) -> Vec<(String, String)> {
    match env {
        Value::Object(map) => map
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string()),
                )
            })
            .collect(),
        Value::Array(arr) => arr
            .iter()
            .filter_map(|v| {
                if let Some(a) = v.as_array() {
                    if a.len() >= 2 {
                        return Some((a[0].as_str()?.to_string(), a[1].as_str()?.to_string()));
                    }
                }
                None
            })
            .collect(),
        _ => vec![],
    }
}

async fn submit_run(State(st): State<Arc<App>>, Json(body): Json<SubmitBody>) -> Json<Value> {
    let req = RunRequest {
        run_id: body.run_id.clone(),
        image: body.image,
        command: body.command,
        env: env_pairs(&body.env),
        gpus: body.gpus,
        shm_gb: body.shm_gb.unwrap_or(2),
        devices: body.devices,
        mounts: body.mounts,
        workdir: body.workdir,
        network: body.network,
    };
    let digest = eb_runner::Runner::image_digest(&req.image).ok();
    match st.runner.start(&req).await {
        Ok(id) => Json(json!({
            "container_id": id,
            "status": "started",
            "image_digest": digest,
        })),
        Err(StartError::ImageMissing(reason)) => Json(json!({
            "container_id": null,
            "status": "image_missing",
            "reason": reason
        })),
        Err(StartError::Docker(reason)) => Json(json!({
            "container_id": null,
            "status": "docker_error",
            "reason": reason
        })),
    }
}

async fn get_run(State(st): State<Arc<App>>, Path(id): Path<String>) -> Json<Value> {
    match st.runner.get(&id) {
        Some(info) => Json(serde_json::to_value(info).unwrap_or(json!({"run_id": id}))),
        None => Json(json!({"run_id": id, "status": "unknown"})),
    }
}

async fn cancel_run(State(st): State<Arc<App>>, Path(id): Path<String>) -> Json<Value> {
    let _ = st.runner.cancel(&id).await;
    Json(json!({"ok": true, "run_id": id}))
}

async fn ps_handler(State(st): State<Arc<App>>) -> Json<Value> {
    Json(json!({"items": st.runner.ps()}))
}

#[derive(Deserialize)]
struct MetricsQ {
    tag: Option<String>,
}

async fn run_metrics(
    State(st): State<Arc<App>>,
    Path(id): Path<String>,
    Query(q): Query<MetricsQ>,
) -> Json<Value> {
    let _ = eb_collector::ingest_run_logs(&st.data_root, &id);
    if let Some(tag) = q.tag {
        let points = eb_tsdb::query(&st.data_root, &id, &tag).unwrap_or_default();
        Json(json!({
            "tag": tag,
            "points": points.into_iter().map(|(step, value)| json!({"step": step, "value": value})).collect::<Vec<_>>()
        }))
    } else {
        let tags = eb_tsdb::list_tags(&st.data_root, &id).unwrap_or_default();
        let mut series = serde_json::Map::new();
        for tag in tags {
            let points = eb_tsdb::query(&st.data_root, &id, &tag).unwrap_or_default();
            series.insert(
                tag,
                json!(points.into_iter().map(|(step, value)| json!({"step": step, "value": value})).collect::<Vec<_>>()),
            );
        }
        Json(json!({"run_id": id, "series": series}))
    }
}

#[derive(Deserialize)]
struct LogsQ {
    tail: Option<usize>,
}

async fn run_logs(
    State(st): State<Arc<App>>,
    Path(id): Path<String>,
    Query(q): Query<LogsQ>,
) -> Json<Value> {
    let path = st.data_root.join("runs").join(&id).join("console.log");
    let text = fs::read_to_string(&path).unwrap_or_default();
    let tail = q.tail.unwrap_or(200);
    let lines: Vec<&str> = text.lines().collect();
    let slice = if lines.len() > tail {
        &lines[lines.len() - tail..]
    } else {
        &lines
    };
    Json(json!({"run_id": id, "lines": slice}))
}

#[derive(Deserialize)]
struct ScanReq {
    path: String,
}

async fn scan_handler(Json(req): Json<ScanReq>) -> Json<Value> {
    let path = req.path;
    let value = tokio::task::spawn_blocking(move || eb_scanner::scan(&path))
        .await
        .unwrap_or_else(|e| json!({"error": e.to_string()}));
    Json(value)
}

async fn media_get(State(st): State<Arc<App>>, Path(path): Path<String>, req: Request) -> Response {
    let full = st.data_root.join("cas").join(path.trim_start_matches('/'));
    if !full.exists() {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    let Ok(meta) = fs::metadata(&full) else {
        return (StatusCode::INTERNAL_SERVER_ERROR, "stat").into_response();
    };
    let len = meta.len();
    let headers = req.headers();
    if let Some(range) = parse_range(headers, len) {
        let Ok(buf) = fs::read(&full) else {
            return (StatusCode::INTERNAL_SERVER_ERROR, "read").into_response();
        };
        let (start, end) = range;
        let slice = buf.get(start as usize..=end as usize).unwrap_or(&[]);
        let mut res = Response::new(Body::from(slice.to_vec()));
        *res.status_mut() = StatusCode::PARTIAL_CONTENT;
        res.headers_mut().insert(
            header::CONTENT_TYPE,
            "application/octet-stream".parse().unwrap(),
        );
        res.headers_mut().insert(
            header::CONTENT_RANGE,
            format!("bytes {start}-{end}/{len}").parse().unwrap(),
        );
        res.headers_mut()
            .insert(header::ACCEPT_RANGES, "bytes".parse().unwrap());
        return res;
    }
    match fs::read(&full) {
        Ok(buf) => {
            let mut res = Response::new(Body::from(buf));
            res.headers_mut()
                .insert(header::CONTENT_TYPE, "application/octet-stream".parse().unwrap());
            res.headers_mut()
                .insert(header::ACCEPT_RANGES, "bytes".parse().unwrap());
            res
        }
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "read").into_response(),
    }
}

fn parse_range(headers: &HeaderMap, len: u64) -> Option<(u64, u64)> {
    let raw = headers.get(header::RANGE)?.to_str().ok()?;
    let spec = raw.strip_prefix("bytes=")?;
    let (a, b) = spec.split_once('-')?;
    if a.is_empty() {
        let n: u64 = b.parse().ok()?;
        let start = len.saturating_sub(n);
        return Some((start, len.saturating_sub(1)));
    }
    let start: u64 = a.parse().ok()?;
    let end = if b.is_empty() {
        len.saturating_sub(1)
    } else {
        b.parse().ok()?
    };
    if start >= len || end >= len || start > end {
        return None;
    }
    Some((start, end))
}
