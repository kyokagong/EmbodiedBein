use axum::body::Bytes;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as RoutePath, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const TASKS: [&str; 3] = ["move-ego-0-0", "move-ego-0-0.3", "move-ego-0-0.7"];
const STATE_LIMIT: usize = 256 * 1024;
const SCENE_LIMIT: usize = 512 * 1024;
const FRAME_LIMIT: usize = 2 * 1024 * 1024;
const FRAME_WINDOW: usize = 8;
// Large unrelated storage operations must never starve the 15-second live lease.
static LIVE_CONTROL_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub async fn control(
    State(state): State<Arc<crate::App>>,
    RoutePath(run): RoutePath<String>,
    bytes: Bytes,
) -> Response {
    if bytes.len() > 4096 {
        return response(Err(StatusCode::PAYLOAD_TOO_LARGE));
    }
    let value: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(_) => return response(Err(StatusCode::BAD_REQUEST)),
    };
    if let Err(status) = validate_command(&value) {
        return response(Err(status));
    }
    mutate(state, run, Some(value)).await
}

pub async fn heartbeat(
    State(state): State<Arc<crate::App>>,
    RoutePath(run): RoutePath<String>,
    bytes: Bytes,
) -> Response {
    if !bytes.is_empty() {
        return response(Err(StatusCode::BAD_REQUEST));
    }
    mutate(state, run, None).await
}

async fn mutate(state: Arc<crate::App>, run: String, command: Option<Value>) -> Response {
    response(mutate_result(state, run, command, None).await)
}

async fn mutate_result(
    state: Arc<crate::App>,
    run: String,
    command: Option<Value>,
    camera: Option<Value>,
) -> Result<Value, StatusCode> {
    let root = state.data_root.clone();
    let runner = state.runner.clone();
    let checked_root = root.clone();
    let checked_run = run.clone();
    let checked = tokio::task::spawn_blocking(move || {
        // Validate the frozen operation before querying Docker-backed run liveness.
        open_live_run(&checked_root, &checked_run)?;
        if !runner
            .get(&checked_run)
            .is_some_and(|info| info.status == "running")
        {
            return Err(StatusCode::CONFLICT);
        }
        Ok(())
    })
    .await
    .unwrap_or(Err(StatusCode::INTERNAL_SERVER_ERROR));
    if let Err(status) = checked {
        return Err(status);
    }
    match camera {
        None => write_live_update(root, run, command).await,
        Some(camera) => {
            let guard = LIVE_CONTROL_LOCK.lock().await;
            tokio::task::spawn_blocking(move || {
                let _guard = guard;
                write_camera(&root, &run, &camera)
            })
            .await
            .unwrap_or(Err(StatusCode::INTERNAL_SERVER_ERROR))
        }
    }
}

async fn write_live_update(
    root: PathBuf,
    run: String,
    command: Option<Value>,
) -> Result<Value, StatusCode> {
    let guard = LIVE_CONTROL_LOCK.lock().await;
    tokio::task::spawn_blocking(move || {
        // Keep serialization until the write finishes even if the HTTP request disconnects.
        let _guard = guard;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .as_secs_f64();
        match command {
            Some(value) => write_command(&root, &run, value, now),
            None => heartbeat_file(&root, &run, now),
        }
    })
    .await
    .unwrap_or(Err(StatusCode::INTERNAL_SERVER_ERROR))
}

pub async fn get(
    State(state): State<Arc<crate::App>>,
    RoutePath((run, kind)): RoutePath<(String, String)>,
) -> Response {
    let root = state.data_root.clone();
    response(
        tokio::task::spawn_blocking(move || read_snapshot(&root, &run, &kind))
            .await
            .unwrap_or(Err(StatusCode::INTERNAL_SERVER_ERROR)),
    )
}

fn response(result: Result<Value, StatusCode>) -> Response {
    let headers = [(header::CACHE_CONTROL, "private, no-store")];
    match result {
        Ok(value) => (headers, Json(value)).into_response(),
        Err(status) => (
            status,
            headers,
            Json(json!({"error":"live simulation unavailable"})),
        )
            .into_response(),
    }
}

fn validate_command(value: &Value) -> Result<(), StatusCode> {
    let bad = StatusCode::BAD_REQUEST;
    let object = value.as_object().ok_or(bad)?;
    if object
        .keys()
        .any(|key| !matches!(key.as_str(), "id" | "action" | "task"))
    {
        return Err(bad);
    }
    let id = value["id"].as_str().ok_or(bad)?;
    if id.is_empty()
        || id.len() > 64
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err(bad);
    }
    match value["action"].as_str().ok_or(bad)? {
        "task"
            if value["task"]
                .as_str()
                .is_some_and(|task| TASKS.contains(&task)) =>
        {
            Ok(())
        }
        "pause" | "resume" | "reset" | "stop" if !object.contains_key("task") => Ok(()),
        _ => Err(bad),
    }
}

fn valid_run(run: &str) -> bool {
    run.len() == 24
        && run.starts_with("run_")
        && run[4..]
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn io_status(error: io::Error) -> StatusCode {
    match error.kind() {
        io::ErrorKind::NotFound | io::ErrorKind::NotADirectory => StatusCode::NOT_FOUND,
        io::ErrorKind::PermissionDenied => StatusCode::FORBIDDEN,
        _ if error.raw_os_error() == Some(libc::ELOOP) => StatusCode::FORBIDDEN,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn anchored(directory: &File, name: &str) -> PathBuf {
    PathBuf::from(format!("/proc/self/fd/{}", directory.as_raw_fd())).join(name)
}

fn open_child(directory: &File, name: &str, is_directory: bool) -> Result<File, StatusCode> {
    let path = anchored(directory, name);
    let metadata = fs::symlink_metadata(&path).map_err(io_status)?;
    if metadata.file_type().is_symlink() {
        return Err(StatusCode::FORBIDDEN);
    }
    if !(if is_directory {
        metadata.is_dir()
    } else {
        metadata.is_file()
    }) {
        return Err(StatusCode::NOT_FOUND);
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(
            libc::O_NOFOLLOW | libc::O_NONBLOCK | if is_directory { libc::O_DIRECTORY } else { 0 },
        )
        .open(path)
        .map_err(io_status)?;
    let metadata = file.metadata().map_err(io_status)?;
    if !(if is_directory {
        metadata.is_dir()
    } else {
        metadata.is_file()
    }) {
        return Err(StatusCode::NOT_FOUND);
    }
    Ok(file)
}

fn open_live_run(root: &Path, run: &str) -> Result<File, StatusCode> {
    if !valid_run(run) {
        return Err(StatusCode::BAD_REQUEST);
    }
    let root = if root.is_absolute() {
        root.to_owned()
    } else {
        std::env::current_dir().map_err(io_status)?.join(root)
    };
    let mut directory = File::open("/").map_err(io_status)?;
    for component in root.components().skip(1) {
        let Component::Normal(name) = component else {
            return Err(StatusCode::BAD_REQUEST);
        };
        directory = open_child(
            &directory,
            name.to_str().ok_or(StatusCode::BAD_REQUEST)?,
            true,
        )?;
    }
    directory = open_child(&directory, "runs", true)?;
    directory = open_child(&directory, run, true)?;
    let plan = read_json(&directory, "native-plan.json", SCENE_LIMIT)?;
    if plan["operation"] != "ufo_native_live"
        || !plan["hash"].as_str().is_some_and(|hash| !hash.is_empty())
        || plan["tasks"] != json!(TASKS)
    {
        return Err(StatusCode::CONFLICT);
    }
    Ok(directory)
}

fn read_json(directory: &File, name: &str, limit: usize) -> Result<Value, StatusCode> {
    let file = open_child(directory, name, false)?;
    if file.metadata().map_err(io_status)?.len() > limit as u64 {
        return Err(StatusCode::PAYLOAD_TOO_LARGE);
    }
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(io_status)?;
    if bytes.len() > limit {
        return Err(StatusCode::PAYLOAD_TOO_LARGE);
    }
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|_| StatusCode::UNPROCESSABLE_ENTITY)?;
    if !value.is_object() {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }
    Ok(value)
}

fn optional_json(directory: &File, name: &str, limit: usize) -> Result<Option<Value>, StatusCode> {
    match read_json(directory, name, limit) {
        Ok(value) => Ok(Some(value)),
        Err(StatusCode::NOT_FOUND) => {
            // A FIFO/directory is not an absent file and must not be replaced.
            match fs::symlink_metadata(anchored(directory, name)) {
                Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
                _ => Err(StatusCode::FORBIDDEN),
            }
        }
        Err(status) => Err(status),
    }
}

fn atomic_json(directory: &File, name: &str, value: &Value) -> Result<(), StatusCode> {
    let destination = anchored(directory, name);
    match fs::symlink_metadata(&destination) {
        Ok(metadata) if !metadata.is_file() || metadata.file_type().is_symlink() => {
            return Err(StatusCode::FORBIDDEN)
        }
        Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(io_status(error)),
        _ => (),
    }
    let temporary = anchored(directory, &format!(".{name}.tmp"));
    match fs::symlink_metadata(&temporary) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {
            fs::remove_file(&temporary).map_err(io_status)?
        }
        Ok(_) => return Err(StatusCode::FORBIDDEN),
        Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(io_status(error)),
        _ => (),
    }
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o660)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&temporary)
            .map_err(io_status)?;
        file.write_all(&serde_json::to_vec(value).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?)
            .map_err(io_status)?;
        file.sync_all().map_err(io_status)?;
        fs::rename(&temporary, destination).map_err(io_status)?;
        directory.sync_all().map_err(io_status)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn write_command(root: &Path, run: &str, mut value: Value, now: f64) -> Result<Value, StatusCode> {
    validate_command(&value)?;
    let directory = open_live_run(root, run)?;
    writable_state(&directory)?;
    let control = open_child(&directory, "live-control", true)?;
    let previous = optional_json(&control, "command.json", 4096)?;
    let mut history = optional_json(&control, "commands.json", SCENE_LIMIT)?.unwrap_or(json!({}));
    if history.as_object().unwrap().len() > 1024 {
        return Err(StatusCode::CONFLICT);
    }
    // Recover the last accepted receipt if a crash happened between the two atomic writes.
    if let Some(previous) = &previous {
        let id = previous["id"].as_str().ok_or(StatusCode::CONFLICT)?;
        history
            .as_object_mut()
            .unwrap()
            .entry(id.to_owned())
            .or_insert_with(|| previous.clone());
    }
    let id = value["id"].as_str().unwrap().to_owned();
    if let Some(receipt) = history.get(&id) {
        let mut payload = receipt.clone();
        payload
            .as_object_mut()
            .ok_or(StatusCode::CONFLICT)?
            .remove("received_at");
        return if payload == value {
            Ok(receipt.clone())
        } else {
            Err(StatusCode::CONFLICT)
        };
    }
    if history.as_object().unwrap().len() >= 1024 {
        return Err(StatusCode::CONFLICT);
    }
    if let Some(previous) = &previous {
        if value["action"] != "stop" {
            let acknowledged = match read_state(&directory) {
                Ok(state) => state["ack"]["id"]
                    .as_str()
                    .is_some_and(|id| Some(id) == previous["id"].as_str()),
                Err(StatusCode::NOT_FOUND) => false,
                Err(status) => return Err(status),
            };
            if !acknowledged {
                return Err(StatusCode::CONFLICT);
            }
        }
    }
    value
        .as_object_mut()
        .unwrap()
        .insert("received_at".to_owned(), json!(now));
    history.as_object_mut().unwrap().insert(id, value.clone());
    atomic_json(&control, "command.json", &value)?;
    atomic_json(&control, "commands.json", &history)?;
    Ok(value)
}

fn heartbeat_file(root: &Path, run: &str, now: f64) -> Result<Value, StatusCode> {
    let directory = open_live_run(root, run)?;
    writable_state(&directory)?;
    let control = open_child(&directory, "live-control", true)?;
    atomic_json(&control, "lease.json", &json!({"updated_at":now}))?;
    Ok(json!({"ok":true,"updated_at":now}))
}

fn read_state(run: &File) -> Result<Value, StatusCode> {
    let outputs = open_child(run, "outputs", true)?;
    let live = open_child(&outputs, "live", true)?;
    read_json(&live, "state.json", STATE_LIMIT)
}

fn read_snapshot(root: &Path, run: &str, kind: &str) -> Result<Value, StatusCode> {
    if !matches!(kind, "state" | "scene") {
        return Err(StatusCode::BAD_REQUEST);
    }
    let directory = open_live_run(root, run)?;
    if kind == "state" {
        return read_state(&directory);
    }
    let outputs = open_child(&directory, "outputs", true)?;
    let live = open_child(&outputs, "live", true)?;
    read_json(&live, "scene.json", SCENE_LIMIT)
}

fn writable_state(directory: &File) -> Result<(), StatusCode> {
    match read_state(directory) {
        Ok(state)
            if state["status"].as_str().is_some_and(|status| {
                !matches!(status, "initializing" | "running" | "paused" | "fallen")
            }) =>
        {
            Err(StatusCode::CONFLICT)
        }
        Ok(_) | Err(StatusCode::NOT_FOUND) => Ok(()),
        Err(status) => Err(status),
    }
}

fn stream_supported(directory: &File) -> Result<(), StatusCode> {
    let plan = read_json(directory, "native-plan.json", SCENE_LIMIT)?;
    let stream = &plan["suite_spec"]["stream"];
    if stream["transport"] != "websocket-jpeg"
        || stream["width"] != 640
        || stream["height"] != 480
        || stream["max_frame_bytes"] != FRAME_LIMIT
    {
        return Err(StatusCode::CONFLICT);
    }
    Ok(())
}

fn validate_message(value: &Value) -> Result<(), StatusCode> {
    let bad = StatusCode::BAD_REQUEST;
    let object = value.as_object().ok_or(bad)?;
    let fields: &[&str] = match value["type"].as_str().ok_or(bad)? {
        "heartbeat" => &["type"],
        "frame_ack" => &["type", "seq"],
        "control" => &["type", "command"],
        "camera" => &["type", "seq", "azimuth", "elevation", "distance"],
        _ => return Err(bad),
    };
    if object.len() != fields.len() || object.keys().any(|key| !fields.contains(&key.as_str())) {
        return Err(bad);
    }
    match value["type"].as_str().unwrap() {
        "control" => validate_command(&value["command"]),
        "frame_ack" => value["seq"]
            .as_u64()
            .filter(|seq| *seq <= 9007199254740991)
            .map(|_| ())
            .ok_or(bad),
        "camera" => {
            if !value["seq"]
                .as_u64()
                .is_some_and(|seq| seq <= i32::MAX as u64)
            {
                return Err(bad);
            }
            for (key, low, high) in [
                ("azimuth", -360.0, 360.0),
                ("elevation", -85.0, -5.0),
                ("distance", 1.0, 8.0),
            ] {
                if !value[key]
                    .as_f64()
                    .is_some_and(|n| n.is_finite() && (low..=high).contains(&n))
                {
                    return Err(bad);
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn write_camera(root: &Path, run: &str, camera: &Value) -> Result<Value, StatusCode> {
    validate_message(camera)?;
    if camera["type"] != "camera" {
        return Err(StatusCode::BAD_REQUEST);
    }
    let directory = open_live_run(root, run)?;
    writable_state(&directory)?;
    let control = open_child(&directory, "live-control", true)?;
    atomic_json(&control, "camera.json", camera)?;
    Ok(json!({"ok":true,"seq":camera["seq"]}))
}

fn read_frame(root: &Path, run: &str) -> Result<(u64, Vec<u8>), StatusCode> {
    let directory = open_live_run(root, run)?;
    let outputs = open_child(&directory, "outputs", true)?;
    let live = open_child(&outputs, "live", true)?;
    let file = open_child(&live, "frame.bin", false)?;
    if file.metadata().map_err(io_status)?.len() > FRAME_LIMIT as u64 {
        return Err(StatusCode::PAYLOAD_TOO_LARGE);
    }
    let mut bytes = Vec::new();
    file.take(FRAME_LIMIT as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(io_status)?;
    if bytes.len() > FRAME_LIMIT {
        return Err(StatusCode::PAYLOAD_TOO_LARGE);
    }
    let bad = StatusCode::UNPROCESSABLE_ENTITY;
    let length =
        u32::from_be_bytes(bytes.get(..4).ok_or(bad)?.try_into().map_err(|_| bad)?) as usize;
    if length == 0 || length > 4096 || bytes.len() < 4 + length + 4 {
        return Err(bad);
    }
    let metadata: Value = serde_json::from_slice(&bytes[4..4 + length]).map_err(|_| bad)?;
    let seq = metadata["seq"]
        .as_u64()
        .filter(|seq| *seq <= 9007199254740991)
        .ok_or(bad)?;
    if metadata["width"] != 640
        || metadata["height"] != 480
        || metadata["step"].as_u64().is_none()
        || metadata["episode"].as_u64().is_none()
        || !metadata["camera_seq"]
            .as_u64()
            .is_some_and(|seq| seq <= i32::MAX as u64)
        || ["updated_at", "rendered_at", "render_ms", "encode_ms"]
            .iter()
            .any(|key| {
                !metadata[key]
                    .as_f64()
                    .is_some_and(|value| value.is_finite() && value >= 0.0)
            })
        || bytes[4 + length..4 + length + 2] != [0xff, 0xd8]
        || bytes[bytes.len() - 2..] != [0xff, 0xd9]
    {
        return Err(bad);
    }
    Ok((seq, bytes))
}

#[derive(Default)]
struct FrameFlow {
    last_seq: Option<u64>,
    // Only receipts are retained: JPEG bytes have no transport history queue.
    pending: VecDeque<(u64, Instant)>,
}

impl FrameFlow {
    fn can_send(&self, seq: u64) -> bool {
        self.pending.len() < FRAME_WINDOW && self.last_seq.is_none_or(|last| seq > last)
    }
    fn sent(&mut self, seq: u64, now: Instant) {
        debug_assert!(self.can_send(seq));
        self.last_seq = Some(seq);
        self.pending.push_back((seq, now));
    }
    fn ack(&mut self, seq: u64) -> Result<(), StatusCode> {
        if let Some(index) = self.pending.iter().position(|(pending, _)| seq == *pending) {
            self.pending.drain(..=index);
            Ok(())
        } else {
            Err(StatusCode::BAD_REQUEST)
        }
    }
    fn expired(&self, now: Instant) -> bool {
        self.pending
            .front()
            .is_some_and(|(_, sent)| now.duration_since(*sent) >= Duration::from_secs(2))
    }
}

async fn send(socket: &mut WebSocket, message: Message) -> Result<(), ()> {
    tokio::time::timeout(Duration::from_secs(1), socket.send(message))
        .await
        .map_err(|_| ())?
        .map_err(|_| ())
}

async fn send_json(socket: &mut WebSocket, value: Value) -> Result<(), ()> {
    send(socket, Message::Text(value.to_string().into())).await
}

pub async fn stream(
    State(state): State<Arc<crate::App>>,
    RoutePath(run): RoutePath<String>,
    websocket: WebSocketUpgrade,
) -> Response {
    let checked_state = state.clone();
    let checked_run = run.clone();
    let checked = tokio::task::spawn_blocking(move || {
        let directory = open_live_run(&checked_state.data_root, &checked_run)?;
        stream_supported(&directory)?;
        writable_state(&directory)?;
        if !checked_state
            .runner
            .get(&checked_run)
            .is_some_and(|info| info.status == "running")
        {
            return Err(StatusCode::CONFLICT);
        }
        Ok(())
    })
    .await
    .unwrap_or(Err(StatusCode::INTERNAL_SERVER_ERROR));
    if let Err(status) = checked {
        return response(Err(status));
    }
    websocket
        .max_message_size(4096)
        .max_frame_size(4096)
        .max_write_buffer_size(FRAME_LIMIT + 65536)
        .on_upgrade(move |socket| stream_loop(socket, state, run))
}

async fn stream_loop(mut socket: WebSocket, state: Arc<crate::App>, run: String) {
    let mut tick = tokio::time::interval(Duration::from_millis(33));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut flow = FrameFlow::default();
    let mut last_state = Instant::now() - Duration::from_secs(1);
    let mut last_liveness = Instant::now();
    let mut last_camera = Instant::now() - Duration::from_secs(1);
    let mut camera_seq = None;
    let mut camera = None;
    let mut window = Instant::now();
    let mut inputs = 0;
    let mut initializing = false;
    loop {
        tokio::select! {
            message = socket.recv() => {
                let Some(Ok(message)) = message else { break };
                if window.elapsed() >= Duration::from_secs(1) {
                    window = Instant::now(); inputs = 0;
                }
                inputs += 1;
                if inputs > 120 { break }
                let text = match message {
                    Message::Text(text) => text,
                    Message::Close(_) => break,
                    Message::Ping(_) | Message::Pong(_) => continue,
                    _ => break,
                };
                if text.len() > 4096 { break }
                let value: Value = match serde_json::from_str(&text) {
                    Ok(value) => value,
                    Err(_) => break,
                };
                if validate_message(&value).is_err() { break }
                match value["type"].as_str().unwrap() {
                    "frame_ack" => if flow.ack(value["seq"].as_u64().unwrap()).is_err() { break },
                    "camera" => {
                        let seq = value["seq"].as_u64().unwrap();
                        if camera_seq.is_none_or(|last| seq > last) {
                            camera_seq = Some(seq); camera = Some(value);
                        }
                    }
                    "control" | "heartbeat" => {
                        let command = (value["type"] == "control").then(|| value["command"].clone());
                        let result = mutate_result(state.clone(), run.clone(), command, None).await;
                        let (ok, status, detail) = match result {
                            Ok(receipt) => (true, StatusCode::OK, receipt),
                            Err(status) => (false, status, json!("live simulation unavailable or command conflict")),
                        };
                        let mut reply = json!({"type":"heartbeat_result","ok":ok,"status":status.as_u16(),"detail":detail});
                        if value["type"] == "control" {
                            reply["type"] = json!("control_result"); reply["id"] = value["command"]["id"].clone();
                        }
                        if send_json(&mut socket, reply).await.is_err() { break }
                    }
                    _ => break,
                }
            }
            _ = tick.tick() => {
                if flow.expired(Instant::now()) { break }
                if last_state.elapsed() >= Duration::from_millis(100) {
                    last_state = Instant::now();
                    let root = state.data_root.clone(); let id = run.clone();
                    let snapshot = tokio::task::spawn_blocking(move || read_snapshot(&root,&id,"state")).await;
                    match snapshot {
                        Ok(Ok(snapshot)) => {
                            let ended = snapshot["status"].as_str().is_some_and(|status| !matches!(status,"initializing"|"running"|"paused"|"fallen"));
                            if send_json(&mut socket,json!({"type":"state","state":snapshot})).await.is_err() { break }
                            if ended { break }
                        }
                        Ok(Err(StatusCode::NOT_FOUND)) => if !initializing {
                            initializing = true;
                            if send_json(&mut socket,json!({"type":"initializing"})).await.is_err() { break }
                        },
                        _ => break,
                    }
                }
                if last_liveness.elapsed() >= Duration::from_secs(1) {
                    last_liveness = Instant::now();
                    let runner = state.runner.clone(); let id = run.clone();
                    if !tokio::task::spawn_blocking(move || runner.get(&id).is_some_and(|info| info.status == "running")).await.unwrap_or(false) {
                        let _ = send_json(&mut socket,json!({"type":"error","status":409,"detail":"worker ended"})).await;
                        break
                    }
                }
                if camera.is_some() && last_camera.elapsed() >= Duration::from_secs_f64(1.0/30.0) {
                    last_camera = Instant::now();
                    if let Err(status) = mutate_result(state.clone(),run.clone(),None,camera.take()).await {
                        let _ = send_json(&mut socket,json!({"type":"error","status":status.as_u16(),"detail":"camera unavailable"})).await;
                        break
                    }
                }
                if flow.pending.len() < FRAME_WINDOW {
                    let root = state.data_root.clone(); let id = run.clone();
                    match tokio::task::spawn_blocking(move || read_frame(&root,&id)).await {
                        Ok(Ok((seq, bytes))) if flow.can_send(seq) => {
                            if send(&mut socket,Message::Binary(bytes.into())).await.is_err() { break }
                            flow.sent(seq,Instant::now());
                        }
                        Ok(Ok(_)) | Ok(Err(StatusCode::NOT_FOUND)) => (),
                        _ => break,
                    }
                }
            }
        }
    }
    // Closing a viewer never renews the lease or writes a command on its behalf.
    let _ = send(&mut socket, Message::Close(None)).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use axum::routing::{get as get_route, post};
    use axum::Router;
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt;
    const RUN: &str = "run_0123456789abcdef0123";
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    #[tokio::test]
    async fn heartbeat_validation_is_not_blocked_by_unrelated_storage_work() {
        let f = Fixture::new();
        let app = Arc::new(crate::App {
            data_root: f.root.clone(),
            runner: eb_runner::Runner::new(f.root.clone()),
            storage_lock: tokio::sync::Mutex::new(()),
        });
        let _storage_guard = app.storage_lock.lock().await;
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            heartbeat(
                State(app.clone()),
                RoutePath("../invalid".to_owned()),
                Bytes::new(),
            ),
        )
        .await
        .expect("unrelated storage operation blocked the live heartbeat handler");
        assert_eq!(result.status(), StatusCode::BAD_REQUEST);
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            write_live_update(f.root.clone(), RUN.to_owned(), None),
        )
        .await
        .expect("unrelated storage operation blocked lease writing")
        .unwrap();
        assert_eq!(result["ok"], true);
        let lease: Value =
            serde_json::from_slice(&fs::read(f.run().join("live-control/lease.json")).unwrap())
                .unwrap();
        assert!(lease["updated_at"].as_f64().unwrap() > 0.0);
    }

    struct Fixture {
        root: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "eb-live-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let run = root.join("runs").join(RUN);
            fs::create_dir_all(run.join("outputs/live")).unwrap();
            fs::create_dir(run.join("live-control")).unwrap();
            fs::write(run.join("native-plan.json"), json!({"operation":"ufo_native_live", "hash":"abc", "tasks":["move-ego-0-0","move-ego-0-0.3","move-ego-0-0.7"]}).to_string()).unwrap();
            Self { root }
        }
        fn run(&self) -> PathBuf {
            self.root.join("runs").join(RUN)
        }
        fn state(&self, value: Value) {
            fs::write(
                self.run().join("outputs/live/state.json"),
                value.to_string(),
            )
            .unwrap();
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[tokio::test]
    async fn media_returns_json_without_cache_or_liveness_requirement() {
        let f = Fixture::new();
        f.state(json!({"status":"completed","step":100}));
        let app = Router::new()
            .route("/v1/runs/{id}/live/{kind}", get_route(get))
            .route("/v1/runs/{id}/live/control", post(control))
            .route("/v1/runs/{id}/live/heartbeat", post(heartbeat))
            .with_state(Arc::new(crate::App {
                data_root: f.root.clone(),
                runner: eb_runner::Runner::new(f.root.clone()),
                storage_lock: tokio::sync::Mutex::new(()),
            }));
        let request = Request::builder()
            .uri(format!("/v1/runs/{RUN}/live/state"))
            .body(Body::empty())
            .unwrap();
        let result = app.clone().oneshot(request).await.unwrap();
        assert_eq!(result.status(), StatusCode::OK);
        assert_eq!(result.headers()[header::CACHE_CONTROL], "private, no-store");
        assert_eq!(result.headers()[header::CONTENT_TYPE], "application/json");
        let value: Value =
            serde_json::from_slice(&to_bytes(result.into_body(), STATE_LIMIT).await.unwrap())
                .unwrap();
        assert_eq!(value["step"], 100);
        for (suffix, body) in [
            ("control", "{\"id\":\"x\",\"action\":\"shell\"}"),
            ("heartbeat", "{}"),
            ("control", "["),
        ] {
            let request = Request::builder()
                .method("POST")
                .uri(format!("/v1/runs/{RUN}/live/{suffix}"))
                .body(Body::from(body))
                .unwrap();
            let result = app.clone().oneshot(request).await.unwrap();
            assert_eq!(result.status(), StatusCode::BAD_REQUEST);
        }
        assert!(!f.run().join("live-control/command.json").exists());
        assert!(!f.run().join("live-control/lease.json").exists());
    }

    #[test]
    fn strict_commands_reject_unknown_fields_actions_tasks_and_unsafe_ids() {
        for value in [
            json!({"id":"", "action":"pause"}),
            json!({"id":"../x", "action":"pause"}),
            json!({"id":"a".repeat(65),"action":"pause"}),
            json!({"id":"a","action":"shell"}),
            json!({"id":"a","action":"task"}),
            json!({"id":"a","action":"task","task":"other"}),
            json!({"id":"a","action":"pause","task":"move-ego-0-0"}),
            json!({"id":"a","action":"pause","extra":true}),
            json!({"id":"a","action":"pause","task":null}),
            json!([]),
        ] {
            assert_eq!(
                validate_command(&value),
                Err(StatusCode::BAD_REQUEST),
                "{value}"
            );
        }
        for value in [
            json!({"id":"A_0-9", "action":"pause"}),
            json!({"id":"x".repeat(64),"action":"task","task":"move-ego-0-0.7"}),
        ] {
            assert_eq!(validate_command(&value), Ok(()));
        }
    }

    #[test]
    fn commands_are_idempotent_busy_until_ack_and_independent_of_lease() {
        let f = Fixture::new();
        let first = json!({"id":"one","action":"task","task":"move-ego-0-0.3"});
        let written = write_command(&f.root, RUN, first.clone(), 1.5).unwrap();
        assert_eq!(written["received_at"], 1.5);
        assert_eq!(
            write_command(&f.root, RUN, first.clone(), 2.0).unwrap(),
            written
        );
        assert_eq!(
            write_command(&f.root, RUN, json!({"id":"one","action":"pause"}), 2.0),
            Err(StatusCode::CONFLICT)
        );
        assert_eq!(
            write_command(&f.root, RUN, json!({"id":"two","action":"pause"}), 2.0),
            Err(StatusCode::CONFLICT)
        );
        assert_eq!(
            heartbeat_file(&f.root, RUN, 3.0).unwrap(),
            json!({"ok":true,"updated_at":3.0})
        );
        let persisted: Value =
            serde_json::from_slice(&fs::read(f.run().join("live-control/command.json")).unwrap())
                .unwrap();
        assert_eq!(persisted, written);
        f.state(json!({"ack":{"id":"one","action":"task","applied_step":40,"task":"move-ego-0-0.3","episode":0}}));
        assert!(write_command(&f.root, RUN, json!({"id":"two","action":"pause"}), 4.0).is_ok());
        assert!(write_command(&f.root, RUN, json!({"id":"stop","action":"stop"}), 5.0).is_ok());
        assert_eq!(write_command(&f.root, RUN, first, 6.0).unwrap(), written);
        let last: Value =
            serde_json::from_slice(&fs::read(f.run().join("live-control/command.json")).unwrap())
                .unwrap();
        assert_eq!(last["id"], "stop");
        assert_eq!(
            write_command(&f.root, RUN, json!({"id":"one","action":"reset"}), 7.0),
            Err(StatusCode::CONFLICT)
        );
    }

    #[test]
    fn command_history_is_bounded_without_disabling_heartbeat_or_retries() {
        let f = Fixture::new();
        let history: serde_json::Map<String, Value> = (0..1024)
            .map(|i| {
                (
                    format!("id{i}"),
                    json!({"id":format!("id{i}"),"action":"stop","received_at":1.0}),
                )
            })
            .collect();
        fs::write(
            f.run().join("live-control/commands.json"),
            Value::Object(history).to_string(),
        )
        .unwrap();
        assert_eq!(
            write_command(&f.root, RUN, json!({"id":"new","action":"stop"}), 2.0),
            Err(StatusCode::CONFLICT)
        );
        assert_eq!(
            write_command(&f.root, RUN, json!({"id":"id0","action":"stop"}), 2.0).unwrap()
                ["received_at"],
            1.0
        );
        assert!(heartbeat_file(&f.root, RUN, 3.0).is_ok());
    }

    #[test]
    fn old_plans_and_invalid_runs_fail_closed() {
        let f = Fixture::new();
        assert_eq!(
            heartbeat_file(&f.root, "../bad", 1.0),
            Err(StatusCode::BAD_REQUEST)
        );
        fs::write(
            f.run().join("native-plan.json"),
            r#"{"operation":"ufo_native_eval","hash":"abc"}"#,
        )
        .unwrap();
        assert_eq!(
            write_command(&f.root, RUN, json!({"id":"x","action":"pause"}), 1.0),
            Err(StatusCode::CONFLICT)
        );
        assert_eq!(
            read_snapshot(&f.root, RUN, "state"),
            Err(StatusCode::CONFLICT)
        );
    }

    #[test]
    fn snapshots_are_bounded_objects_with_fixed_kinds() {
        let f = Fixture::new();
        assert_eq!(
            read_snapshot(&f.root, RUN, "state"),
            Err(StatusCode::NOT_FOUND)
        );
        f.state(json!({"step":42}));
        assert_eq!(
            read_snapshot(&f.root, RUN, "state").unwrap(),
            json!({"step":42})
        );
        assert_eq!(
            read_snapshot(&f.root, RUN, "../native-plan"),
            Err(StatusCode::BAD_REQUEST)
        );
        fs::write(
            f.run().join("outputs/live/state.json"),
            " ".repeat(256 * 1024 + 1),
        )
        .unwrap();
        assert_eq!(
            read_snapshot(&f.root, RUN, "state"),
            Err(StatusCode::PAYLOAD_TOO_LARGE)
        );
        fs::write(f.run().join("outputs/live/state.json"), "[]").unwrap();
        assert_eq!(
            read_snapshot(&f.root, RUN, "state"),
            Err(StatusCode::UNPROCESSABLE_ENTITY)
        );
    }

    #[test]
    fn symlink_ancestors_and_nonregular_files_never_escape_or_block() {
        let f = Fixture::new();
        let external = f.root.join("external");
        fs::create_dir(&external).unwrap();
        fs::remove_dir(f.run().join("live-control")).unwrap();
        symlink(&external, f.run().join("live-control")).unwrap();
        assert_eq!(
            heartbeat_file(&f.root, RUN, 1.0),
            Err(StatusCode::FORBIDDEN)
        );
        assert!(!external.join("lease.json").exists());
        symlink(
            f.run().join("native-plan.json"),
            f.run().join("outputs/live/state.json"),
        )
        .unwrap();
        assert_eq!(
            read_snapshot(&f.root, RUN, "state"),
            Err(StatusCode::FORBIDDEN)
        );
        fs::remove_file(f.run().join("outputs/live/state.json")).unwrap();
        let fifo = std::ffi::CString::new(
            f.run()
                .join("outputs/live/state.json")
                .as_os_str()
                .as_encoded_bytes(),
        )
        .unwrap();
        assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
        assert_eq!(
            read_snapshot(&f.root, RUN, "state"),
            Err(StatusCode::NOT_FOUND)
        );
    }

    fn frame(seq: u64) -> Vec<u8> {
        let metadata = serde_json::to_vec(&json!({
            "seq":seq,"step":8,"episode":0,"updated_at":1.0,"rendered_at":1.1,
            "camera_seq":0,"width":640,"height":480,"render_ms":2.0,"encode_ms":0.3
        }))
        .unwrap();
        let mut bytes = (metadata.len() as u32).to_be_bytes().to_vec();
        bytes.extend(metadata);
        bytes.extend([0xff, 0xd8, 0xff, 0xd9]);
        bytes
    }

    #[test]
    fn frame_protocol_preserves_bytes_and_rejects_corruption_and_oversized_metadata() {
        let f = Fixture::new();
        let path = f.run().join("outputs/live/frame.bin");
        let bytes = frame(9);
        fs::write(&path, &bytes).unwrap();
        assert_eq!(read_frame(&f.root, RUN).unwrap(), (9, bytes.clone()));
        for broken in [
            vec![],
            vec![0; 4],
            vec![0xff; 4],
            vec![0; FRAME_LIMIT + 1],
            {
                let mut v = bytes.clone();
                v.pop();
                v
            },
        ] {
            fs::write(&path, broken).unwrap();
            assert!(read_frame(&f.root, RUN).is_err());
        }
        fs::remove_file(&path).unwrap();
        symlink(f.run().join("native-plan.json"), &path).unwrap();
        assert_eq!(read_frame(&f.root, RUN), Err(StatusCode::FORBIDDEN));
    }

    #[test]
    fn camera_validation_and_atomic_write_do_not_touch_command_history_or_lease() {
        let f = Fixture::new();
        let camera = json!({"type":"camera","seq":1,"azimuth":90,"elevation":-30,"distance":3});
        assert!(validate_message(&camera).is_ok());
        write_camera(&f.root, RUN, &camera).unwrap();
        let stored: Value =
            serde_json::from_slice(&fs::read(f.run().join("live-control/camera.json")).unwrap())
                .unwrap();
        assert_eq!(stored["seq"], 1);
        assert!(!f.run().join("live-control/commands.json").exists());
        assert!(!f.run().join("live-control/lease.json").exists());
        for (key, value) in [
            ("seq", json!(-1)),
            ("seq", json!(true)),
            ("seq", json!(2147483648u64)),
            ("azimuth", json!(361)),
            ("elevation", json!(0)),
            ("distance", json!(0)),
            ("path", json!("/tmp/x")),
        ] {
            let mut broken = camera.clone();
            broken[key] = value;
            assert_eq!(validate_message(&broken), Err(StatusCode::BAD_REQUEST));
        }
        f.state(json!({"status":"finalizing"}));
        assert_eq!(
            write_camera(&f.root, RUN, &camera),
            Err(StatusCode::CONFLICT)
        );
        assert_eq!(heartbeat_file(&f.root, RUN, 3.0), Err(StatusCode::CONFLICT));
        assert_eq!(
            write_command(&f.root, RUN, json!({"id":"x","action":"pause"}), 3.0),
            Err(StatusCode::CONFLICT)
        );
        fs::remove_file(f.run().join("live-control/camera.json")).unwrap();
        symlink(
            f.run().join("native-plan.json"),
            f.run().join("live-control/camera.json"),
        )
        .unwrap();
        fs::remove_file(f.run().join("outputs/live/state.json")).unwrap();
        assert_eq!(
            write_camera(&f.root, RUN, &camera),
            Err(StatusCode::FORBIDDEN)
        );
    }

    #[test]
    fn frame_flow_bounds_window_and_accepts_only_known_cumulative_ack() {
        let now = std::time::Instant::now();
        let mut flow = FrameFlow::default();
        assert_eq!(flow.ack(0), Err(StatusCode::BAD_REQUEST));
        for seq in [0, 2, 4, 6, 8, 10, 12, 14] {
            assert!(flow.can_send(seq));
            flow.sent(seq, now);
        }
        assert!(!flow.can_send(16));
        assert_eq!(flow.ack(1), Err(StatusCode::BAD_REQUEST));
        assert_eq!(flow.ack(16), Err(StatusCode::BAD_REQUEST));
        assert_eq!(flow.ack(8), Ok(()));
        assert_eq!(flow.ack(4), Err(StatusCode::BAD_REQUEST));
        assert_eq!(flow.ack(8), Err(StatusCode::BAD_REQUEST));
        assert!(!flow.can_send(14));
        assert!(flow.can_send(99));
        flow.sent(99, now);
        assert_eq!(flow.ack(99), Ok(()));
        assert!(flow.can_send(100));
    }

    #[test]
    fn frame_flow_deadline_tracks_oldest_unacknowledged_frame() {
        let now = std::time::Instant::now();
        let mut flow = FrameFlow::default();
        flow.sent(0, now);
        flow.sent(2, now + Duration::from_millis(100));
        flow.sent(4, now + Duration::from_millis(200));
        assert!(!flow.expired(now + std::time::Duration::from_millis(1999)));
        assert!(flow.expired(now + std::time::Duration::from_secs(2)));
        assert_eq!(flow.ack(0), Ok(()));
        assert!(!flow.expired(now + Duration::from_millis(2099)));
        assert!(flow.expired(now + Duration::from_millis(2100)));
        assert_eq!(flow.ack(4), Ok(()));
        assert!(!flow.expired(now + Duration::from_secs(30)));
        assert_eq!(flow.ack(0), Err(StatusCode::BAD_REQUEST));
    }

    #[test]
    fn websocket_protocol_rejects_unknown_binary_like_and_extra_fields() {
        for value in [
            json!([]),
            json!({"type":"shell"}),
            json!({"type":"heartbeat","extra":true}),
            json!({"type":"frame_ack","seq":-1}),
            json!({"type":"frame_ack","seq":true}),
            json!({"type":"control","command":{"id":"x","action":"shell"}}),
        ] {
            assert_eq!(validate_message(&value), Err(StatusCode::BAD_REQUEST));
        }
        assert!(validate_message(&json!({"type":"heartbeat"})).is_ok());
    }

    #[test]
    fn historical_plans_cannot_open_native_frame_stream() {
        let f = Fixture::new();
        let directory = open_live_run(&f.root, RUN).unwrap();
        assert_eq!(stream_supported(&directory), Err(StatusCode::CONFLICT));
        let mut plan = read_json(&directory, "native-plan.json", SCENE_LIMIT).unwrap();
        plan["suite_spec"] = json!({"stream":{"transport":"websocket-jpeg","width":640,"height":480,"max_frame_bytes":2097152}});
        fs::write(f.run().join("native-plan.json"), plan.to_string()).unwrap();
        assert!(stream_supported(&directory).is_ok());
    }

    #[tokio::test]
    async fn unix_websocket_bounds_eight_frames_and_skips_latest_when_window_full() {
        use futures_util::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite::Message as ClientMessage;
        let f = Fixture::new();
        f.state(json!({"status":"running","step":0}));
        fs::write(f.run().join("outputs/live/frame.bin"), frame(0)).unwrap();
        let app = Arc::new(crate::App {
            data_root: f.root.clone(),
            runner: eb_runner::Runner::new(f.root.clone()),
            storage_lock: tokio::sync::Mutex::new(()),
        });
        let router = Router::new()
            .route(
                "/stream",
                get_route(
                    |State(app): State<Arc<crate::App>>, websocket: WebSocketUpgrade| async move {
                        websocket.on_upgrade(move |socket| stream_loop(socket, app, RUN.to_owned()))
                    },
                ),
            )
            .with_state(app);
        let socket_path = f.root.join("test.sock");
        let listener = tokio::net::UnixListener::bind(&socket_path).unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let stream = tokio::net::UnixStream::connect(&socket_path).await.unwrap();
        let (mut client, _) = tokio_tungstenite::client_async("ws://ebd/stream", stream)
            .await
            .unwrap();
        async fn next_binary(
            client: &mut tokio_tungstenite::WebSocketStream<tokio::net::UnixStream>,
        ) -> Vec<u8> {
            loop {
                match client.next().await.unwrap().unwrap() {
                    ClientMessage::Binary(bytes) => return bytes.to_vec(),
                    ClientMessage::Text(_) => (),
                    other => panic!("unexpected message {other:?}"),
                }
            }
        }
        assert_eq!(
            tokio::time::timeout(Duration::from_millis(500), next_binary(&mut client))
                .await
                .unwrap(),
            frame(0)
        );
        for seq in [2, 4, 6, 8, 10, 12, 14] {
            fs::write(f.run().join("outputs/live/frame.bin"), frame(seq)).unwrap();
            assert_eq!(
                tokio::time::timeout(Duration::from_millis(500), next_binary(&mut client))
                    .await
                    .unwrap(),
                frame(seq)
            );
        }
        fs::write(f.run().join("outputs/live/frame.bin"), frame(16)).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(150), next_binary(&mut client))
                .await
                .is_err()
        );
        fs::write(f.run().join("outputs/live/frame.bin"), frame(30)).unwrap();
        client
            .send(ClientMessage::Text(
                json!({"type":"frame_ack","seq":8}).to_string().into(),
            ))
            .await
            .unwrap();
        assert_eq!(
            tokio::time::timeout(Duration::from_millis(500), next_binary(&mut client))
                .await
                .unwrap(),
            frame(30)
        );
        f.state(json!({"status":"finalizing","step":8}));
        let terminal = tokio::time::timeout(Duration::from_millis(500), async {
            loop {
                if let ClientMessage::Text(value) = client.next().await.unwrap().unwrap() {
                    let value: Value = serde_json::from_str(&value).unwrap();
                    if value["state"]["status"] == "finalizing" {
                        break value;
                    }
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(terminal["state"]["step"], 8);
        assert!(!f.run().join("live-control/lease.json").exists());
        server.abort();
    }
}
