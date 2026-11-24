use eb_scanner::{hash_at, inspect_at, read_text_at};
use axum::extract::{rejection::JsonRejection, State};
use axum::response::Response;
use axum::routing::post;
use axum::{Json, Router};
use eb_artifact::{open_directory, ArtifactError};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageRoot {
    pub namespace: String,
    #[serde(default)]
    pub path: String,
    pub run_id: Option<String>,
    pub source: Option<String>,
}

const SOURCE_ROOTS: &str = "/data/kk/embodiedbein,/data/kk/models,/data/kk/data,/data/kk/assets,/data/kk/Berkeley-Humanoid-Lite,/data/kk/RLinf,/data/kk/UFO,/data/kk/P-BFM,/data/kk/berkeley_ego_data,/data/kk/roboparty/roboparty_train/robolab/data/motions/rpo_lab";

impl StorageRoot {
    pub fn location(&self, state: &Path) -> Result<(PathBuf, String), ArtifactError> {
        if !self.path.is_empty() {
            eb_artifact::validate_relative(&self.path)?;
        }
        if matches!(self.namespace.as_str(), "run" | "model_run") {
            if self.source.is_some() {
                return Err(ArtifactError::new("invalid", "unexpected source reference"));
            }
            let root = crate::artifacts::ArtifactRoot {
                namespace: self.namespace.clone(),
                run_id: self.run_id.clone().unwrap_or_default(),
                path: String::new(),
            };
            return Ok((root.resolve(state)?, self.path.clone()));
        }
        if self.run_id.is_some() {
            return Err(ArtifactError::new("invalid", "unexpected Run reference"));
        }
        let configured = |key: &str, default: &str| {
            PathBuf::from(std::env::var(key).unwrap_or_else(|_| default.into()))
        };
        let base = match self.namespace.as_str() {
            "state" => state.to_owned(),
            "models" => configured("EB_MODELS_ROOT", "/data/kk/models"),
            "data" => configured("EB_DATASETS_ROOT", "/data/kk/data"),
            "assets" => configured("EB_ASSETS_ROOT", "/data/kk/assets"),
            "source" => {
                let source = self.source.as_ref().ok_or_else(|| {
                    ArtifactError::new("forbidden", "missing configured source root")
                })?;
                if !std::env::var("EB_ALLOWED_PATH_PREFIXES")
                    .unwrap_or_else(|_| SOURCE_ROOTS.into())
                    .split(',')
                    .any(|allowed| Path::new(allowed.trim()) == Path::new(source))
                {
                    return Err(ArtifactError::new(
                        "forbidden",
                        "source root is not configured",
                    ));
                }
                PathBuf::from(source)
            }
            _ => return Err(ArtifactError::new("forbidden", "unknown storage root")),
        };
        if self.namespace != "source" && self.source.is_some() {
            return Err(ArtifactError::new("invalid", "unexpected source reference"));
        }
        Ok((base, self.path.clone()))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InspectRequest {
    root: StorageRoot,
    #[serde(default)]
    path: String,
    mode: String,
    cursor: Option<String>,
    #[serde(default = "page_limit")]
    limit: usize,
    #[serde(default)]
    recursive: bool,
    #[serde(default)]
    exclude_dirs: Vec<String>,
}
fn page_limit() -> usize {
    50
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TextRequest {
    root: StorageRoot,
    path: String,
    #[serde(default = "text_limit")]
    max_bytes: usize,
}
fn text_limit() -> usize {
    1024 * 1024
}

pub fn routes() -> Router<Arc<crate::App>> {
    Router::new()
        .route("/v1/storage/inspect", post(inspect))
        .route("/v1/storage/read-text", post(read_text))
        .route("/v1/storage/hash-file", post(hash_file))
}

fn joined(first: &str, second: &str) -> Result<String, ArtifactError> {
    if !second.is_empty() {
        eb_artifact::validate_relative(second)?;
    }
    Ok([first, second]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("/"))
}

async fn inspect(
    State(state): State<Arc<crate::App>>,
    request: Result<Json<InspectRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match request {
        Ok(value) => value,
        Err(error) => return crate::artifacts::json_failure(error),
    };
    crate::artifacts::run_io(move || {
        let (base, path) = request.root.location(&state.data_root)?;
        inspect_at(
            &base,
            &joined(&path, &request.path)?,
            &request.mode,
            request.cursor.as_deref(),
            request.limit,
            request.recursive,
            &request.exclude_dirs,
        )
    })
    .await
}

async fn read_text(
    State(state): State<Arc<crate::App>>,
    request: Result<Json<TextRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match request {
        Ok(value) => value,
        Err(error) => return crate::artifacts::json_failure(error),
    };
    crate::artifacts::run_io(move || {
        let (base, path) = request.root.location(&state.data_root)?;
        read_text_at(&base, &joined(&path, &request.path)?, request.max_bytes)
    })
    .await
}

async fn hash_file(
    State(state): State<Arc<crate::App>>,
    request: Result<Json<TextRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match request {
        Ok(value) => value,
        Err(error) => return crate::artifacts::json_failure(error),
    };
    crate::artifacts::run_io(move || {
        let (base, path) = request.root.location(&state.data_root)?;
        hash_at(&base, &joined(&path, &request.path)?)
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileQuery {
    root: String,
    path: String,
}

pub async fn get_file(
    State(state): State<Arc<crate::App>>,
    axum::extract::Query(query): axum::extract::Query<FileQuery>,
    request: axum::extract::Request,
) -> Response {
    let root: StorageRoot = match serde_json::from_str(&query.root) {
        Ok(root) => root,
        Err(_) => {
            return crate::artifacts::failure(ArtifactError::new("invalid", "invalid file root"))
        }
    };
    let name = query.path.clone();
    let opened = tokio::task::spawn_blocking(move || -> Result<_, ArtifactError> {
        let (base, path) = root.location(&state.data_root)?;
        let file = eb_artifact::open_member(&open_directory(&base)?, &joined(&path, &query.path)?)?;
        let len = file.metadata()?.len();
        Ok((file, len))
    })
    .await;
    match opened {
        Ok(Ok((file, len))) => crate::run_output::file_response(file, len, &name, request).await,
        Ok(Err(error)) => crate::artifacts::failure(error),
        Err(_) => crate::run_output::failure(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
    }
}
