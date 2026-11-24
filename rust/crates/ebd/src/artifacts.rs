use axum::extract::{rejection::JsonRejection, DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use eb_artifact::{ArtifactError, ArtifactFile, VerifyLimits};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};
use tokio::sync::Semaphore;

static IO_SLOTS: LazyLock<Arc<Semaphore>> = LazyLock::new(|| Arc::new(Semaphore::new(2)));

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRoot {
    pub namespace: String,
    pub run_id: String,
    pub path: String,
}

impl ArtifactRoot {
    pub fn resolve(&self, data_root: &std::path::Path) -> Result<PathBuf, ArtifactError> {
        if !self.run_id.starts_with("run_")
            || self.run_id.len() <= 4
            || self.run_id.len() > 128
            || !self
                .run_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err(ArtifactError::new("forbidden", "invalid Run reference"));
        }
        if !self.path.is_empty() {
            eb_artifact::validate_relative(&self.path)?;
        }
        let base = match self.namespace.as_str() {
            "run" => data_root.to_owned(),
            "model_run" => PathBuf::from(
                std::env::var("EB_MODELS_ROOT").unwrap_or_else(|_| "/data/kk/models".into()),
            ),
            _ => return Err(ArtifactError::new("forbidden", "unknown artifact root")),
        };
        Ok(base.join("runs").join(&self.run_id).join(&self.path))
    }
}

fn max_files() -> usize {
    200
}
fn max_file_bytes() -> u64 {
    2 * 1024_u64.pow(3)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VerifyRequest {
    root: ArtifactRoot,
    files: Vec<ArtifactFile>,
    #[serde(default = "max_files")]
    max_files: usize,
    #[serde(default = "max_file_bytes")]
    max_file_bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PublishRequest {
    root: ArtifactRoot,
    path: String,
    expected_bytes: u64,
    expected_sha256: Option<String>,
}

pub fn routes() -> Router<Arc<crate::App>> {
    Router::new()
        .route("/v1/artifacts/verify", post(verify))
        .route("/v1/artifacts/publish-file", post(publish))
        .route("/v1/artifacts/pack-bundle", post(pack_bundle))
        .route("/v1/storage/prepare-resume", post(prepare_resume))
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
}

pub fn failure(error: ArtifactError) -> Response {
    let status = match error.code {
        "invalid" => StatusCode::BAD_REQUEST,
        "forbidden" => StatusCode::FORBIDDEN,
        "missing" => StatusCode::NOT_FOUND,
        "conflict" => StatusCode::CONFLICT,
        "limit" => StatusCode::PAYLOAD_TOO_LARGE,
        "no_space" => StatusCode::INSUFFICIENT_STORAGE,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (
        status,
        Json(json!({"error":{"code":error.code, "detail":error.detail, "path":error.path}})),
    )
        .into_response()
}

pub(crate) fn json_failure(error: JsonRejection) -> Response {
    let code = if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
        "limit"
    } else {
        "invalid"
    };
    failure(ArtifactError::new(code, "invalid bounded artifact request"))
}

pub async fn run_io(
    operation: impl FnOnce() -> Result<Value, ArtifactError> + Send + 'static,
) -> Response {
    let permit = match IO_SLOTS.clone().acquire_owned().await {
        Ok(permit) => permit,
        Err(_) => return failure(ArtifactError::new("io", "artifact IO unavailable")),
    };
    match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        operation()
    })
    .await
    {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(error)) => failure(error),
        Err(_) => failure(ArtifactError::new("io", "artifact IO worker failed")),
    }
}

async fn verify(
    State(state): State<Arc<crate::App>>,
    body: Result<Json<VerifyRequest>, JsonRejection>,
) -> Response {
    let Json(body) = match body {
        Ok(body) => body,
        Err(error) => return json_failure(error),
    };
    run_io(move || {
        let root = body.root.resolve(&state.data_root)?;
        let result = eb_artifact::verify_manifest(
            &root,
            &body.files,
            VerifyLimits {
                max_files: body.max_files,
                max_file_bytes: body.max_file_bytes,
            },
        )?;
        Ok(
            json!({"schema_version":"eb/artifact-verification/v1", "verified":true,
            "file_count":result.file_count,"total_bytes":result.total_bytes}),
        )
    })
    .await
}

async fn publish(
    State(state): State<Arc<crate::App>>,
    body: Result<Json<PublishRequest>, JsonRejection>,
) -> Response {
    let Json(body) = match body {
        Ok(body) => body,
        Err(error) => return json_failure(error),
    };
    run_io(move || {
        let root = body.root.resolve(&state.data_root)?;
        let result = eb_artifact::publish_file(
            &root,
            &body.path,
            &state.data_root.join("cas"),
            eb_artifact::FileExpectation {
                bytes: body.expected_bytes,
                sha256: body.expected_sha256,
            },
        )?;
        Ok(
            json!({"schema_version":"eb/artifact-publication/v1", "sha256":result.sha256,
            "size_bytes":result.size_bytes,"storage":result.storage}),
        )
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BundleRequest {
    root: ArtifactRoot,
    members: Vec<ArtifactFile>,
}

async fn pack_bundle(
    State(state): State<Arc<crate::App>>,
    body: Result<Json<BundleRequest>, JsonRejection>,
) -> Response {
    let Json(body) = match body {
        Ok(body) => body,
        Err(error) => return json_failure(error),
    };
    run_io(move || {
        if body.root.namespace != "run" || body.root.path != "outputs/bundle-staging" {
            return Err(ArtifactError::new(
                "forbidden",
                "bundle packing requires Run bundle staging",
            ));
        }
        let result = eb_artifact::pack_bundle(
            &body.root.resolve(&state.data_root)?,
            &body.members,
            &state.data_root.join("cas"),
        )?;
        Ok(
            json!({"schema_version":"eb/artifact-publication/v1", "sha256":result.sha256,
            "size_bytes":result.size_bytes,"storage":result.storage}),
        )
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResumeRequest {
    source_run_id: String,
    target_run_id: String,
    files: Vec<eb_artifact::ResumeFile>,
}

async fn prepare_resume(
    State(state): State<Arc<crate::App>>,
    body: Result<Json<ResumeRequest>, JsonRejection>,
) -> Response {
    let Json(body) = match body {
        Ok(body) => body,
        Err(error) => return json_failure(error),
    };
    run_io(move || {
        if body.source_run_id == body.target_run_id {
            return Err(ArtifactError::new(
                "invalid",
                "resume source and target must differ",
            ));
        }
        let source = ArtifactRoot {
            namespace: "run".into(),
            run_id: body.source_run_id.clone(),
            path: String::new(),
        }
        .resolve(&state.data_root)?;
        let target = ArtifactRoot {
            namespace: "run".into(),
            run_id: body.target_run_id.clone(),
            path: String::new(),
        }
        .resolve(&state.data_root)?;
        let containers = crate::storage::inspect_task_containers()
            .map_err(|_| ArtifactError::new("io", "cannot verify resume source state"))?;
        crate::storage::check_containers(
            &containers,
            &[source.clone(), target.clone()],
            &[body.source_run_id.clone(), body.target_run_id.clone()],
        )
        .map_err(|_| ArtifactError::new("conflict", "resume source or target is active"))?;
        let files = eb_artifact::prepare_resume(&source, &target, &body.files)?;
        Ok(
            json!({"schema_version":"eb/resume-preparation/v1", "prepared":true,
            "source_run_id":body.source_run_id,"target_run_id":body.target_run_id,"files":files}),
        )
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt;
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    #[tokio::test]
    async fn verify_http_validates_metadata_and_actual_bytes() {
        let root = std::env::temp_dir().join(format!(
            "eb-batch-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(root.join("runs/run_test/outputs")).unwrap();
        fs::write(root.join("runs/run_test/outputs/model"), b"state").unwrap();
        let app = routes().with_state(Arc::new(crate::App {
            data_root: root.clone(),
            runner: eb_runner::Runner::new(root.clone()),
            storage_lock: tokio::sync::Mutex::new(()),
        }));
        let mut payload = json!({"root":{"namespace":"run","run_id":"run_test","path":"outputs"},
            "files":[{"path":"model","bytes":5,"sha256":eb_core::sha256_hex(b"state")}]});
        for (change, status) in [
            ("valid", StatusCode::OK),
            ("changed", StatusCode::CONFLICT),
            ("bad-root", StatusCode::FORBIDDEN),
            ("bad-count", StatusCode::BAD_REQUEST),
        ] {
            if change == "changed" {
                fs::write(root.join("runs/run_test/outputs/model"), b"wrong").unwrap();
            }
            if change == "bad-root" {
                payload["root"]["path"] = json!("../outputs");
            }
            if change == "bad-count" {
                payload["max_files"] = json!(true);
            }
            let response = app
                .clone()
                .oneshot(
                    Request::post("/v1/artifacts/verify")
                        .header("content-type", "application/json")
                        .body(Body::from(payload.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), status, "{change}");
            let value: Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap())
                    .unwrap();
            if change == "valid" {
                assert_eq!(value["file_count"], 1);
            } else {
                assert!(value["error"]["code"].is_string());
            }
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn artifact_io_does_not_stall_live_heartbeat() {
        let work = tokio::spawn(run_io(|| {
            std::thread::sleep(std::time::Duration::from_millis(180));
            Ok(json!({"ok":true}))
        }));
        let start = std::time::Instant::now();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        assert!(start.elapsed() < std::time::Duration::from_millis(100));
        assert_eq!(work.await.unwrap().status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn verify_http_accepts_long_manifest_and_rejects_oversized_body() {
        let root = std::env::temp_dir().join(format!(
            "eb-long-index-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(root.join("runs/run_test/outputs")).unwrap();
        let app = routes().with_state(Arc::new(crate::App {
            data_root: root.clone(),
            runner: eb_runner::Runner::new(root.clone()),
            storage_lock: tokio::sync::Mutex::new(()),
        }));
        let files: Vec<Value> = (0..10000).map(|i| json!({
            "path":format!("{}/{}/{}/{}/file-{i}", "a".repeat(240), "b".repeat(240), "c".repeat(240), "d".repeat(240)), "bytes":0, "sha256":eb_core::sha256_hex(b"")
        })).collect();
        let payload = json!({"root":{"namespace":"run","run_id":"run_test","path":"outputs"},
            "max_files":10000,"files":files})
        .to_string();
        assert!(payload.len() > 8 * 1024 * 1024);
        let response = app
            .clone()
            .oneshot(
                Request::post("/v1/artifacts/verify")
                    .header("content-type", "application/json")
                    .body(Body::from(payload))
                    .unwrap(),
            )
            .await
            .unwrap();
        // Parsing and bounds succeeded; the fixture intentionally contains no such file.
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let response = app
            .oneshot(
                Request::post("/v1/artifacts/verify")
                    .header("content-type", "application/json")
                    .body(Body::from(" ".repeat(16 * 1024 * 1024 + 1)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        fs::remove_dir_all(root).unwrap();
    }
}
