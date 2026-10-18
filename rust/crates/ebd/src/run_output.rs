use axum::body::Body;
use axum::extract::{Path, Request, State};
use axum::http::{header, HeaderMap, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use futures_util::stream;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path as FsPath, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

pub async fn get(
    State(state): State<Arc<crate::App>>,
    Path((run, path)): Path<(String, String)>,
    request: Request,
) -> Response {
    if run.len() != 24
        || !run.starts_with("run_")
        || !run[4..]
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        || path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
        || path
            .chars()
            .any(|c| c == '%' || c == '\\' || c.is_control())
    {
        return failure(StatusCode::BAD_REQUEST);
    }
    let root = state.data_root.clone();
    let relative = path.clone();
    let (file, len) =
        match tokio::task::spawn_blocking(move || open_output(&root, &run, &relative)).await {
            Ok(Ok(result)) => result,
            Ok(Err(status)) => return failure(status),
            Err(_) => return failure(StatusCode::INTERNAL_SERVER_ERROR),
        };
    let mut response = Response::new(Body::empty());
    let headers = response.headers_mut();
    headers.insert(header::ACCEPT_RANGES, "bytes".parse().unwrap());
    headers.insert(header::CACHE_CONTROL, "private, no-store".parse().unwrap());
    let range = match requested_range(request.headers(), len) {
        Ok(range) => range,
        Err(()) => {
            headers.insert(
                header::CONTENT_RANGE,
                format!("bytes */{len}").parse().unwrap(),
            );
            headers.insert(header::CONTENT_LENGTH, "0".parse().unwrap());
            *response.status_mut() = StatusCode::RANGE_NOT_SATISFIABLE;
            return response;
        }
    };
    let (start, count) = match range {
        Some((start, end)) => {
            headers.insert(
                header::CONTENT_RANGE,
                format!("bytes {start}-{end}/{len}").parse().unwrap(),
            );
            (start, end - start + 1)
        }
        None => (0, len),
    };
    headers.insert(header::CONTENT_LENGTH, count.to_string().parse().unwrap());
    let mime = match FsPath::new(&path).extension().and_then(|s| s.to_str()) {
        Some("mp4") => "video/mp4",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("json") => "application/json",
        Some("jsonl") => "application/x-ndjson",
        _ => "application/octet-stream",
    };
    headers.insert(header::CONTENT_TYPE, mime.parse().unwrap());
    if range.is_some() {
        *response.status_mut() = StatusCode::PARTIAL_CONTENT;
    }
    if request.method() == Method::HEAD || count == 0 {
        return response;
    }
    let mut file = tokio::fs::File::from_std(file);
    if file.seek(io::SeekFrom::Start(start)).await.is_err() {
        return failure(StatusCode::INTERNAL_SERVER_ERROR);
    }
    *response.body_mut() = Body::from_stream(stream::try_unfold(
        (file, count),
        |(mut file, remaining)| async move {
            if remaining == 0 {
                return Ok::<_, io::Error>(None);
            }
            let mut chunk = vec![0; remaining.min(64 * 1024) as usize];
            let read = file.read(&mut chunk).await?;
            if read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "run output truncated during download",
                ));
            }
            chunk.truncate(read);
            Ok(Some((chunk, (file, remaining - read as u64))))
        },
    ));
    response
}

fn failure(status: StatusCode) -> Response {
    (
        status,
        [(header::CACHE_CONTROL, "private, no-store")],
        "run output unavailable",
    )
        .into_response()
}

fn io_status(error: io::Error) -> StatusCode {
    match error.kind() {
        io::ErrorKind::NotFound | io::ErrorKind::NotADirectory => StatusCode::NOT_FOUND,
        io::ErrorKind::PermissionDenied => StatusCode::FORBIDDEN,
        _ if error.raw_os_error() == Some(libc::ELOOP) => StatusCode::FORBIDDEN,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn open_output(root: &FsPath, run: &str, relative: &str) -> Result<(File, u64), StatusCode> {
    let root = if root.is_absolute() {
        root.to_owned()
    } else {
        std::env::current_dir().map_err(io_status)?.join(root)
    };
    let base = root.join("runs").join(run).join("outputs");
    let full = base.join(relative);
    let mut current = File::open("/").map_err(io_status)?;
    let mut components = full.components().skip(1).peekable();
    while let Some(component) = components.next() {
        let Component::Normal(name) = component else {
            return Err(StatusCode::BAD_REQUEST);
        };
        // Anchor each open to its parent descriptor so concurrent symlink swaps cannot escape.
        let next = PathBuf::from(format!("/proc/self/fd/{}", current.as_raw_fd())).join(name);
        let metadata = fs::symlink_metadata(&next).map_err(io_status)?;
        if metadata.file_type().is_symlink() {
            return Err(StatusCode::FORBIDDEN);
        }
        if components.peek().is_none() && !metadata.is_file() {
            return Err(StatusCode::NOT_FOUND);
        }
        let flags = libc::O_NOFOLLOW
            | libc::O_NONBLOCK
            | if components.peek().is_some() {
                libc::O_DIRECTORY
            } else {
                0
            };
        current = OpenOptions::new()
            .read(true)
            .custom_flags(flags)
            .open(next)
            .map_err(io_status)?;
    }
    let metadata = current.metadata().map_err(io_status)?;
    if !metadata.is_file() {
        return Err(StatusCode::NOT_FOUND);
    }
    let canonical =
        fs::canonicalize(format!("/proc/self/fd/{}", current.as_raw_fd())).map_err(io_status)?;
    let expected = root
        .canonicalize()
        .map_err(io_status)?
        .join("runs")
        .join(run)
        .join("outputs");
    if !canonical.starts_with(expected) || full.canonicalize().map_err(io_status)? != canonical {
        return Err(StatusCode::FORBIDDEN);
    }
    Ok((current, metadata.len()))
}

fn requested_range(headers: &HeaderMap, len: u64) -> Result<Option<(u64, u64)>, ()> {
    let mut values = headers.get_all(header::RANGE).iter();
    let Some(value) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() || len == 0 {
        return Err(());
    }
    let spec = value
        .to_str()
        .map_err(|_| ())?
        .strip_prefix("bytes=")
        .ok_or(())?;
    let (first, last) = spec.split_once('-').ok_or(())?;
    let number = |s: &str| -> Result<u64, ()> {
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            return Err(());
        }
        s.parse().map_err(|_| ())
    };
    if first.is_empty() {
        let suffix = number(last)?;
        if suffix == 0 {
            return Err(());
        }
        return Ok(Some((len.saturating_sub(suffix), len - 1)));
    }
    let start = number(first)?;
    let end = if last.is_empty() {
        len - 1
    } else {
        number(last)?
    };
    if start >= len || end < start {
        return Err(());
    }
    Ok(Some((start, end.min(len - 1))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{header, Method};
    use axum::routing;
    use axum::Router;
    use futures_util::StreamExt;
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt;

    const RUN: &str = "run_0123456789abcdef0123";
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    struct Fixture {
        base: PathBuf,
        root: PathBuf,
        output: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let base = std::env::temp_dir().join(format!(
                "eb-run-output-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let root = base.join("state");
            let output = root.join("runs").join(RUN).join("outputs");
            fs::create_dir_all(output.join("nested")).unwrap();
            fs::write(output.join("nested/policy.ebp"), b"0123456789").unwrap();
            Self { base, root, output }
        }

        async fn request(
            &self,
            method: Method,
            run: &str,
            path: &str,
            ranges: &[&str],
        ) -> Response {
            let app = Router::new()
                .route("/v1/runs/{id}/outputs/{*path}", routing::get(get))
                .with_state(Arc::new(crate::App {
                    data_root: self.root.clone(),
                    runner: eb_runner::Runner::new(self.root.clone()),
                    storage_lock: tokio::sync::Mutex::new(()),
                }));
            let mut request = Request::builder()
                .method(method)
                .uri(format!("/v1/runs/{run}/outputs/{path}"));
            for range in ranges {
                request = request.header(header::RANGE, *range);
            }
            app.oneshot(request.body(Body::empty()).unwrap())
                .await
                .unwrap()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.base);
        }
    }

    #[tokio::test]
    async fn get_returns_private_file_bytes_and_mime_types() {
        let fixture = Fixture::new();
        for (name, mime) in [
            ("movie.mp4", "video/mp4"),
            ("chart.svg", "image/svg+xml"),
            ("frame.png", "image/png"),
            ("report.json", "application/json"),
            ("cases.jsonl", "application/x-ndjson"),
            ("traces.npz", "application/octet-stream"),
            ("package.ebp", "application/octet-stream"),
        ] {
            fs::write(fixture.output.join(name), b"\0data\xff").unwrap();
            let response = fixture.request(Method::GET, RUN, name, &[]).await;
            assert_eq!(response.status(), StatusCode::OK, "{name}");
            assert_eq!(response.headers()[header::CONTENT_TYPE], mime);
            assert_eq!(response.headers()[header::CONTENT_LENGTH], "6");
            assert_eq!(response.headers()[header::ACCEPT_RANGES], "bytes");
            assert_eq!(
                response.headers()[header::CACHE_CONTROL],
                "private, no-store"
            );
            assert_eq!(
                to_bytes(response.into_body(), 100).await.unwrap(),
                b"\0data\xff"[..]
            );
        }
    }

    #[tokio::test]
    async fn single_ranges_return_exact_bytes_and_headers() {
        let fixture = Fixture::new();
        for (range, content_range, expected) in [
            ("bytes=2-5", "bytes 2-5/10", "2345"),
            ("bytes=6-", "bytes 6-9/10", "6789"),
            ("bytes=-4", "bytes 6-9/10", "6789"),
            ("bytes=-99", "bytes 0-9/10", "0123456789"),
            ("bytes=8-99", "bytes 8-9/10", "89"),
        ] {
            let response = fixture
                .request(Method::GET, RUN, "nested/policy.ebp", &[range])
                .await;
            assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT, "{range}");
            assert_eq!(response.headers()[header::CONTENT_RANGE], content_range);
            assert_eq!(
                response.headers()[header::CONTENT_LENGTH],
                expected.len().to_string()
            );
            assert_eq!(to_bytes(response.into_body(), 100).await.unwrap(), expected);
        }
    }

    #[tokio::test]
    async fn invalid_multiple_and_unsatisfiable_ranges_return_416() {
        let fixture = Fixture::new();
        for ranges in [
            vec!["bytes=-0"],
            vec!["bytes=10-"],
            vec!["bytes=7-3"],
            vec!["bytes=0-1,4-5"],
            vec!["bytes="],
            vec!["bytes=-"],
            vec!["bytes=abc-3"],
            vec!["items=0-1"],
            vec!["bytes=+1-3"],
            vec!["bytes=18446744073709551616-"],
            vec!["bytes=0-1", "bytes=4-5"],
        ] {
            let response = fixture
                .request(Method::GET, RUN, "nested/policy.ebp", &ranges)
                .await;
            assert_eq!(
                response.status(),
                StatusCode::RANGE_NOT_SATISFIABLE,
                "{ranges:?}"
            );
            assert_eq!(response.headers()[header::CONTENT_RANGE], "bytes */10");
            assert_eq!(response.headers()[header::ACCEPT_RANGES], "bytes");
            assert_eq!(
                response.headers()[header::CACHE_CONTROL],
                "private, no-store"
            );
        }
    }

    #[tokio::test]
    async fn head_has_get_headers_without_a_body() {
        let fixture = Fixture::new();
        for (ranges, status, length) in [
            (vec![], StatusCode::OK, "10"),
            (vec!["bytes=2-5"], StatusCode::PARTIAL_CONTENT, "4"),
        ] {
            let response = fixture
                .request(Method::HEAD, RUN, "nested/policy.ebp", &ranges)
                .await;
            assert_eq!(response.status(), status);
            assert_eq!(response.headers()[header::CONTENT_LENGTH], length);
            assert!(to_bytes(response.into_body(), 100)
                .await
                .unwrap()
                .is_empty());
        }
    }

    #[tokio::test]
    async fn empty_files_have_zero_length_and_no_satisfiable_range() {
        let fixture = Fixture::new();
        fs::write(fixture.output.join("empty.ebp"), []).unwrap();
        let response = fixture.request(Method::GET, RUN, "empty.ebp", &[]).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_LENGTH], "0");
        assert!(to_bytes(response.into_body(), 100)
            .await
            .unwrap()
            .is_empty());
        for range in ["bytes=0-", "bytes=-1"] {
            let response = fixture
                .request(Method::GET, RUN, "empty.ebp", &[range])
                .await;
            assert_eq!(response.status(), StatusCode::RANGE_NOT_SATISFIABLE);
            assert_eq!(response.headers()[header::CONTENT_RANGE], "bytes */0");
        }
    }

    #[tokio::test]
    async fn run_ids_and_encoded_paths_cannot_escape_output() {
        let fixture = Fixture::new();
        for run in [
            "run_abc",
            "run_0123456789abcdef012A",
            "run_0123456789abcdef01234",
            "other_0123456789abcdef0123",
        ] {
            let response = fixture
                .request(Method::GET, run, "nested/policy.ebp", &[])
                .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{run}");
        }
        for path in [
            "../secret",
            "./nested/policy.ebp",
            "nested/../secret",
            "nested//policy.ebp",
            "/etc/passwd",
            "%2e%2e/secret",
            "%252e%252e/secret",
            "%2fetc/passwd",
            "nested/%2e%2e/secret",
            "nested%5cpolicy.ebp",
            "nested/%00policy.ebp",
        ] {
            let response = fixture.request(Method::GET, RUN, path, &[]).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
        }
        assert_eq!(
            fixture
                .request(Method::GET, RUN, "missing.ebp", &[])
                .await
                .status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            fixture
                .request(Method::GET, RUN, "nested", &[])
                .await
                .status(),
            StatusCode::NOT_FOUND
        );
    }

    #[tokio::test]
    async fn symlinks_in_every_directory_or_file_are_rejected_even_inside_output() {
        for component in [
            "",
            "runs",
            "runs/run_0123456789abcdef0123",
            "runs/run_0123456789abcdef0123/outputs",
            "runs/run_0123456789abcdef0123/outputs/nested",
            "runs/run_0123456789abcdef0123/outputs/nested/policy.ebp",
        ] {
            let fixture = Fixture::new();
            let source = if component.is_empty() {
                fixture.root.clone()
            } else {
                fixture.root.join(component)
            };
            let saved = fixture.base.join("saved");
            fs::rename(&source, &saved).unwrap();
            symlink(&saved, &source).unwrap();
            let response = fixture
                .request(Method::GET, RUN, "nested/policy.ebp", &[])
                .await;
            assert_eq!(response.status(), StatusCode::FORBIDDEN, "{component}");
        }
    }

    #[tokio::test]
    async fn nonregular_output_is_rejected_before_opening() {
        let fixture = Fixture::new();
        let _socket =
            std::os::unix::net::UnixListener::bind(fixture.output.join("socket")).unwrap();
        assert_eq!(
            fixture
                .request(Method::GET, RUN, "socket", &[])
                .await
                .status(),
            StatusCode::NOT_FOUND
        );
    }

    #[tokio::test]
    async fn truncation_during_download_fails_the_body() {
        let fixture = Fixture::new();
        let response = fixture
            .request(Method::GET, RUN, "nested/policy.ebp", &[])
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        fs::OpenOptions::new()
            .write(true)
            .open(fixture.output.join("nested/policy.ebp"))
            .unwrap()
            .set_len(3)
            .unwrap();
        assert!(to_bytes(response.into_body(), 100).await.is_err());
    }

    #[tokio::test]
    async fn package_is_streamed_in_bounded_chunks_and_open_descriptor_survives_replacement() {
        let fixture = Fixture::new();
        let bytes = vec![42; 2 * 1024 * 1024];
        let path = fixture.output.join("large.ebp");
        fs::write(&path, &bytes).unwrap();
        let response = fixture.request(Method::GET, RUN, "large.ebp", &[]).await;
        assert_eq!(response.status(), StatusCode::OK);
        fs::rename(&path, fixture.output.join("original.ebp")).unwrap();
        fs::write(fixture.base.join("secret"), b"secret").unwrap();
        symlink(fixture.base.join("secret"), &path).unwrap();
        let mut stream = response.into_body().into_data_stream();
        let mut received = Vec::new();
        let mut chunks = 0;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            assert!(chunk.len() <= 128 * 1024);
            received.extend_from_slice(&chunk);
            chunks += 1;
        }
        assert!(chunks > 1);
        assert_eq!(received, bytes);
    }
}
