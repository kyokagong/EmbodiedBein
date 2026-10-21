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
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use tokio::net::UnixListener;
mod storage;
mod run_output;
mod native_live;
mod native_training_resources;

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

struct App {
    data_root: PathBuf,
    runner: Runner,
    storage_lock: tokio::sync::Mutex<()>,
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
        storage_lock: tokio::sync::Mutex::new(()),
    });

    let control = Router::new()
        .route("/healthz", get(health))
        .route("/v1/doctor", get(doctor_handler))
        .route("/v1/images/inspect", get(image_handler))
        .route("/v1/runs", post(submit_run))
        .route("/v1/runs/{id}", get(get_run))
        .route("/v1/runs/{id}/cancel", post(cancel_run))
        .route("/v1/runs/{id}/live/control", post(native_live::control))
        .route("/v1/runs/{id}/live/heartbeat", post(native_live::heartbeat))
        .route("/v1/runs/{id}/metrics", get(run_metrics))
        .route("/v1/runs/{id}/logs", get(run_logs))
        .route("/v1/ps", get(ps_handler))
        .route("/v1/scan", post(scan_handler))
        .route("/v1/snapshots", post(snapshot_handler))
        .route("/v1/checkpoints/freeze", post(freeze_checkpoint_handler))
        .route("/v1/storage/delete", post(delete_storage_handler))
        .with_state(app_state.clone());

    let media = Router::new()
        .route("/healthz", get(health))
        .route("/media/{*path}", get(media_get))
        .route("/v1/runs/{id}/outputs/{*path}", get(run_output::get))
        .route("/v1/runs/{id}/live/stream", get(native_live::stream))
        .route("/v1/runs/{id}/live/{kind}", get(native_live::get))
        .with_state(app_state.clone());

    let control_path = std::env::var("EB_EBD_UDS").unwrap_or_else(|_| "/run/eb/ebd.sock".into());
    let media_path = std::env::var("EB_EBD_MEDIA_UDS").unwrap_or_else(|_| "/run/eb/ebd-http.sock".into());
    bind_unix(&control_path, control).await?;
    bind_unix(&media_path, media).await?;
    tracing::info!("ebd listening on {control_path} and {media_path}");
    let root = data_root.clone();
    tokio::spawn(async move { eb_collector::spawn(root).await });
    let resource_root = data_root.clone();
    tokio::spawn(async move { native_training_resources::spawn(resource_root).await });
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
struct ImageQuery {
    name: String,
}

async fn image_handler(Query(query): Query<ImageQuery>) -> impl IntoResponse {
    match Runner::image_info(&query.name) {
        Ok((digest, size_bytes)) => (StatusCode::OK, Json(json!({
            "ok": true, "name": query.name, "digest": digest, "size_bytes": size_bytes
        }))),
        Err(StartError::ImageMissing(reason)) => (StatusCode::NOT_FOUND, Json(json!({
            "ok": false, "name": query.name, "error": reason
        }))),
        Err(StartError::Docker(reason)) => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({
            "ok": false, "name": query.name, "error": reason
        }))),
    }
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
    #[serde(default)]
    hostname: Option<String>,
    #[serde(default)]
    cpus: Option<u32>,
    #[serde(default)]
    memory_gb: Option<u32>,
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
    let _guard = st.storage_lock.lock().await;
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
        hostname: body.hostname,
        cpus: body.cpus.unwrap_or(4),
        memory_gb: body.memory_gb.unwrap_or(8),
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

async fn delete_storage_handler(State(st): State<Arc<App>>, Json(req): Json<storage::DeleteRequest>) -> Response {
    let _guard = st.storage_lock.lock().await;
    let root = st.data_root.clone();
    let models = PathBuf::from(std::env::var("EB_MODELS_ROOT").unwrap_or_else(|_| "/data/kk/models".into()));
    match tokio::task::spawn_blocking(move || storage::delete_storage(&root, &models, req)).await {
        Ok(Ok(result)) => Json(result).into_response(),
        Ok(Err(error)) => (StatusCode::CONFLICT, Json(json!({"error": error}))).into_response(),
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response(),
    }
}

async fn get_run(State(st): State<Arc<App>>, Path(id): Path<String>) -> Json<Value> {
    match st.runner.get(&id) {
        Some(info) => Json(serde_json::to_value(info).unwrap_or(json!({"run_id": id}))),
        None => Json(json!({"run_id": id, "status": "unknown"})),
    }
}

async fn cancel_run(State(st): State<Arc<App>>, Path(id): Path<String>) -> Response {
    match st.runner.cancel(&id).await {
        Ok(()) => Json(json!({"ok": true, "run_id": id})).into_response(),
        Err(error) => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"ok": false, "run_id": id, "error": error}))).into_response(),
    }
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

#[derive(Deserialize)]
struct SnapshotReq {
    path: String,
    content_hash: Option<String>,
}

async fn snapshot_handler(State(st): State<Arc<App>>, Json(req): Json<SnapshotReq>) -> Response {
    // Only platform-owned source trees or configured external repositories can be read.
    let source = match fs::canonicalize(&req.path) {
        Ok(path) => path,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    let allowed = [
        std::env::var("EB_BHL_REPO").unwrap_or_else(|_| "/data/kk/Berkeley-Humanoid-Lite".into()),
        std::env::var("EB_RLINF_REPO").unwrap_or_else(|_| "/data/kk/RLinf".into()),
        std::env::var("EB_UFO_NATIVE_ROOT").unwrap_or_else(|_| "/data/kk/UFO".into()),
        std::env::var("EB_ASSETS_ROOT").unwrap_or_else(|_| "/data/kk/assets".into()),
        st.data_root.to_string_lossy().into_owned(),
    ];
    if !allowed.iter().filter_map(|p| fs::canonicalize(p).ok()).any(|root| source.starts_with(root)) {
        return (StatusCode::FORBIDDEN, "snapshot source outside allowed roots").into_response();
    }
    let root = st.data_root.clone();
    match tokio::task::spawn_blocking(move || eb_runner::snapshot::prepare_snapshot(&source, &root, req.content_hash.as_deref())).await {
        Ok(Ok(snapshot)) => Json(snapshot).into_response(),
        Ok(Err(e)) => (StatusCode::BAD_REQUEST, e).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

#[derive(Deserialize)]
struct FreezeCheckpointReq {
    run_id: String,
    step: u64,
}

async fn freeze_checkpoint_handler(State(st): State<Arc<App>>, Json(req): Json<FreezeCheckpointReq>) -> Response {
    let root = st.data_root.clone();
    match tokio::task::spawn_blocking(move || freeze_checkpoint(&root, &req.run_id, req.step)).await {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(e)) => (StatusCode::BAD_REQUEST, e).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

fn freeze_checkpoint(root: &FsPath, run_id: &str, step: u64) -> Result<Value, String> {
    if !run_id.starts_with("run_") || !run_id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') {
        return Err("invalid source Run ID".into());
    }
    let runs = root.join("runs");
    let run_dir = runs.join(run_id);
    let output = run_dir.join("output");
    for path in [&runs, &run_dir, &output] {
        if fs::symlink_metadata(path).map_err(|e| e.to_string())?.file_type().is_symlink() {
            return Err("checkpoint source contains a symlink".into());
        }
    }
    let mut weights = Vec::new();
    for entry in fs::read_dir(&output).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        if !entry.file_type().map_err(|e| e.to_string())?.is_dir() { continue; }
        let mut current = entry.path();
        let step_dir = format!("global_step_{step}");
        for component in ["checkpoints", step_dir.as_str(), "actor", "model_state_dict", "full_weights.pt"] {
            current = current.join(component);
            if fs::symlink_metadata(&current).is_ok_and(|m| m.file_type().is_symlink()) {
                return Err("checkpoint contains symlink".into());
            }
        }
        let weight = current;
        if weight.is_file() { weights.push(weight); }
    }
    if weights.len() != 1 { return Err("checkpoint must resolve to one full_weights.pt".into()); }
    let source = fs::OpenOptions::new().read(true).custom_flags(0x20000).open(&weights[0]).map_err(|e| e.to_string())?;
    let before = source.metadata().map_err(|e| e.to_string())?;
    if before.len() == 0 { return Err("empty checkpoint".into()); }
    let cas_dir = root.join("cas");
    let tmp_dir = cas_dir.join("tmp");
    fs::create_dir_all(&cas_dir).map_err(|e| e.to_string())?;
    fs::create_dir_all(&tmp_dir).map_err(|e| e.to_string())?;
    for path in [&cas_dir, &tmp_dir] {
        if fs::symlink_metadata(path).map_err(|e| e.to_string())?.file_type().is_symlink() {
            return Err("checkpoint CAS contains a symlink".into());
        }
    }
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?.as_nanos();
    let tmp = tmp_dir.join(format!("checkpoint-{}-{nonce}", std::process::id()));
    let result = (|| -> Result<Value, String> {
        let mut out = fs::OpenOptions::new().write(true).create_new(true).open(&tmp).map_err(|e| e.to_string())?;
        let mut input = source;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 1024 * 1024];
        let mut bytes = 0u64;
        loop {
            let n = input.read(&mut buffer).map_err(|e| e.to_string())?;
            if n == 0 { break; }
            out.write_all(&buffer[..n]).map_err(|e| e.to_string())?;
            hash.update(&buffer[..n]);
            bytes += n as u64;
        }
        out.sync_all().map_err(|e| e.to_string())?;
        let after = input.metadata().map_err(|e| e.to_string())?;
        if bytes != before.len() || before.len() != after.len() || before.modified().ok() != after.modified().ok() {
            return Err("checkpoint changed during freeze".into());
        }
        let digest = format!("{:x}", hash.finalize());
        let dir = cas_dir.join(&digest[..2]);
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        if fs::symlink_metadata(&dir).map_err(|e| e.to_string())?.file_type().is_symlink() {
            return Err("checkpoint CAS digest directory is a symlink".into());
        }
        let dest = dir.join(&digest);
        fs::set_permissions(&tmp, fs::Permissions::from_mode(0o444)).map_err(|e| e.to_string())?;
        match fs::hard_link(&tmp, &dest) {
            Ok(()) => {},
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let meta = fs::symlink_metadata(&dest).map_err(|e| e.to_string())?;
                if !meta.is_file() || meta.len() != bytes {
                    return Err("CAS entry conflicts with checkpoint".into());
                }
                let mut existing = fs::File::open(&dest).map_err(|e| e.to_string())?;
                let mut current = Sha256::new();
                loop {
                    let n = existing.read(&mut buffer).map_err(|e| e.to_string())?;
                    if n == 0 { break; }
                    current.update(&buffer[..n]);
                }
                if format!("{:x}", current.finalize()) != digest {
                    return Err("CAS entry digest conflicts with checkpoint".into());
                }
            },
            Err(e) => return Err(e.to_string()),
        }
        fs::remove_file(&tmp).map_err(|e| e.to_string())?;
        Ok(json!({"path": dest, "sha256": digest, "bytes": bytes, "run_id": run_id, "step": step}))
    })();
    if tmp.exists() { let _ = fs::remove_file(&tmp); }
    result
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
