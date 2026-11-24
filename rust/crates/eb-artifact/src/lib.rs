use eb_core::sha256_hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

#[derive(Debug)]
pub struct ArtifactError {
    pub code: &'static str,
    pub detail: String,
    pub path: Option<String>,
}

impl ArtifactError {
    pub fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
            path: None,
        }
    }
}

impl std::fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.detail)
    }
}
impl std::error::Error for ArtifactError {}
impl From<std::io::Error> for ArtifactError {
    fn from(error: std::io::Error) -> Self {
        let code = match error.raw_os_error() {
            Some(libc::ENOENT) => "missing",
            Some(libc::ELOOP | libc::ENOTDIR | libc::EACCES | libc::EPERM) => "forbidden",
            Some(libc::ENOSPC) => "no_space",
            _ => "io",
        };
        Self::new(code, error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactFile {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Copy, Debug)]
pub struct VerifyLimits {
    pub max_files: usize,
    pub max_file_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct Verification {
    pub file_count: usize,
    pub total_bytes: u64,
}

pub fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub fn validate_relative(value: &str) -> Result<(), ArtifactError> {
    if value.is_empty()
        || value.len() > 1024
        || value.contains('\\')
        || value.chars().any(char::is_control)
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(ArtifactError::new(
            "forbidden",
            "unsafe relative artifact path",
        ));
    }
    Ok(())
}

pub fn fd_path(file: &File) -> PathBuf {
    PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()))
}

pub fn open_directory(path: &Path) -> Result<File, ArtifactError> {
    if !path.is_absolute() {
        return Err(ArtifactError::new(
            "forbidden",
            "storage root must be absolute",
        ));
    }
    let mut current = File::open("/")?;
    for component in path.components().skip(1) {
        let std::path::Component::Normal(name) = component else {
            return Err(ArtifactError::new("forbidden", "invalid storage root"));
        };
        current = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_DIRECTORY | libc::O_NONBLOCK)
            .open(fd_path(&current).join(name))?;
    }
    Ok(current)
}

pub fn open_member(root: &File, relative: &str) -> Result<File, ArtifactError> {
    let current = open_relative(root, relative)?;
    if !current.metadata()?.is_file() {
        return Err(ArtifactError::new(
            "forbidden",
            "artifact must be a regular file",
        ));
    }
    Ok(current)
}

pub fn open_relative(root: &File, relative: &str) -> Result<File, ArtifactError> {
    validate_relative(relative)?;
    let mut current = root.try_clone()?;
    let mut parts = relative.split('/').peekable();
    while let Some(part) = parts.next() {
        let directory = parts.peek().is_some();
        current = OpenOptions::new()
            .read(true)
            .custom_flags(
                libc::O_NOFOLLOW | libc::O_NONBLOCK | if directory { libc::O_DIRECTORY } else { 0 },
            )
            .open(fd_path(&current).join(part))?;
    }
    let metadata = current.metadata()?;
    if !metadata.is_file() && !metadata.is_dir() {
        return Err(ArtifactError::new(
            "forbidden",
            "special storage file forbidden",
        ));
    }
    Ok(current)
}

fn identity(metadata: &fs::Metadata) -> (u64, u64, u64, i64, i64, i64, i64) {
    (
        metadata.dev(),
        metadata.ino(),
        metadata.len(),
        metadata.mtime(),
        metadata.mtime_nsec(),
        metadata.ctime(),
        metadata.ctime_nsec(),
    )
}

pub fn hash_file(file: &mut File) -> Result<(String, u64), ArtifactError> {
    read_and_hash(file, &mut std::io::sink())
}

fn copy_and_hash(file: &mut File, output: &mut impl Write) -> Result<(String, u64), ArtifactError> {
    let copied = read_and_hash(file, output)?;
    // Filesystem timestamps may not change for a same-tick rewrite.
    file.rewind()?;
    if hash_file(file)? != copied {
        return Err(ArtifactError::new("conflict", "source changed during copy"));
    }
    Ok(copied)
}

fn read_and_hash(file: &mut File, output: &mut impl Write) -> Result<(String, u64), ArtifactError> {
    let before = file.metadata()?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    let mut bytes = 0;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
        bytes += count as u64;
        if bytes > before.len() {
            return Err(ArtifactError::new("conflict", "artifact grew during read"));
        }
        output.write_all(&buffer[..count])?;
    }
    if bytes != before.len() || identity(&before) != identity(&file.metadata()?) {
        return Err(ArtifactError::new(
            "conflict",
            "artifact changed during read",
        ));
    }
    Ok((format!("{:x}", hash.finalize()), bytes))
}

pub fn verify_manifest(
    root: &Path,
    files: &[ArtifactFile],
    limits: VerifyLimits,
) -> Result<Verification, ArtifactError> {
    if limits.max_files == 0
        || limits.max_files > 10000
        || limits.max_file_bytes == 0
        || limits.max_file_bytes > 1024_u64.pow(4)
        || files.is_empty()
        || files.len() > limits.max_files
    {
        return Err(ArtifactError::new(
            "limit",
            "artifact verification limit exceeded",
        ));
    }
    let mut seen = std::collections::HashSet::new();
    for item in files {
        validate_relative(&item.path)?;
        if !seen.insert(&item.path) || !valid_digest(&item.sha256) {
            return Err(ArtifactError::new(
                "invalid",
                "duplicate member or invalid file digest",
            ));
        }
        if item.bytes > limits.max_file_bytes {
            return Err(ArtifactError::new("limit", "artifact exceeds file limit"));
        }
    }
    let _pin = pin_paths(&[root.to_owned()])?;
    let directory = open_directory(root)?;
    let mut total_bytes = 0;
    for item in files {
        let checked = (|| {
            let mut file = open_member(&directory, &item.path)?;
            if file.metadata()?.len() != item.bytes {
                return Err(ArtifactError::new("conflict", "artifact size differs"));
            }
            let (digest, bytes) = hash_file(&mut file)?;
            if digest != item.sha256 {
                return Err(ArtifactError::new("conflict", "artifact hash differs"));
            }
            Ok(bytes)
        })();
        total_bytes += checked.map_err(|mut e: ArtifactError| {
            e.path = Some(item.path.clone());
            e
        })?;
    }
    Ok(Verification {
        file_count: files.len(),
        total_bytes,
    })
}

static PINS: LazyLock<Mutex<std::collections::HashMap<PathBuf, usize>>> =
    LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

pub struct StoragePin(Vec<PathBuf>);

pub fn pin_paths(paths: &[PathBuf]) -> Result<StoragePin, ArtifactError> {
    let mut pins = PINS
        .lock()
        .map_err(|_| ArtifactError::new("io", "storage protection unavailable"))?;
    for path in paths {
        *pins.entry(path.clone()).or_default() += 1;
    }
    Ok(StoragePin(paths.to_vec()))
}

impl Drop for StoragePin {
    fn drop(&mut self) {
        let mut pins = PINS.lock().unwrap_or_else(|error| error.into_inner());
        for path in &self.0 {
            if let Some(count) = pins.get_mut(path) {
                *count -= 1;
                if *count == 0 {
                    pins.remove(path);
                }
            }
        }
    }
}

pub fn with_unpinned_paths<T>(
    paths: &[PathBuf],
    operation: impl FnOnce() -> Result<T, ArtifactError>,
) -> Result<T, ArtifactError> {
    let pins = PINS
        .lock()
        .map_err(|_| ArtifactError::new("io", "storage protection unavailable"))?;
    if paths.iter().any(|path| {
        pins.keys()
            .any(|pinned| path.starts_with(pinned) || pinned.starts_with(path))
    }) {
        return Err(ArtifactError::new(
            "conflict",
            "storage is in use by artifact IO",
        ));
    }
    operation()
}

#[derive(Debug)]
pub struct FileExpectation {
    pub bytes: u64,
    pub sha256: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Publication {
    pub sha256: String,
    pub size_bytes: u64,
    pub storage: PathBuf,
}

pub fn child_directory(parent: &File, name: &str) -> Result<File, ArtifactError> {
    validate_relative(name)?;
    if name.contains('/') {
        return Err(ArtifactError::new(
            "forbidden",
            "directory name must be a single component",
        ));
    }
    let path = fd_path(parent).join(name);
    match fs::create_dir(&path) {
        Ok(()) => {
            parent.sync_all()?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
        Err(error) => return Err(error.into()),
    }
    Ok(OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_DIRECTORY | libc::O_NONBLOCK)
        .open(path)?)
}

static NEXT_STAGING: AtomicU64 = AtomicU64::new(0);

struct StagingFile {
    parent: File,
    name: String,
    file: File,
}

impl StagingFile {
    fn new(parent: File) -> Result<Self, ArtifactError> {
        for _ in 0..100 {
            let name = format!(
                "copy-{}-{}",
                std::process::id(),
                NEXT_STAGING.fetch_add(1, Ordering::Relaxed)
            );
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW)
                .open(fd_path(&parent).join(&name))
            {
                Ok(file) => return Ok(Self { parent, name, file }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(ArtifactError::new("conflict", "staging names exhausted"))
    }
}

impl Drop for StagingFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(fd_path(&self.parent).join(&self.name));
        let _ = self.parent.sync_all();
    }
}

pub fn publish_file(
    root: &Path,
    relative: &str,
    cas_root: &Path,
    expected: FileExpectation,
) -> Result<Publication, ArtifactError> {
    if expected.bytes > 1024_u64.pow(4)
        || expected
            .sha256
            .as_ref()
            .is_some_and(|hash| !valid_digest(hash))
    {
        return Err(ArtifactError::new(
            "invalid",
            "invalid publication expectation",
        ));
    }
    let _pin = pin_paths(&[root.to_owned(), cas_root.to_owned()])?;
    let mut source = open_member(&open_directory(root)?, relative)?;
    if source.metadata()?.len() != expected.bytes {
        return Err(ArtifactError::new("conflict", "source size differs"));
    }
    let cas = open_directory(cas_root)?;
    let mut temporary = StagingFile::new(child_directory(&cas, ".staging")?)?;
    let (digest, size_bytes) = copy_and_hash(&mut source, &mut temporary.file)?;
    if expected.sha256.is_some_and(|hash| hash != digest) {
        return Err(ArtifactError::new("conflict", "source hash differs"));
    }
    temporary
        .file
        .set_permissions(fs::Permissions::from_mode(0o444))?;
    temporary.file.sync_all()?;
    publish_staging(&temporary, cas_root, digest, size_bytes)
}

fn publish_staging(
    temporary: &StagingFile,
    cas_root: &Path,
    digest: String,
    size_bytes: u64,
) -> Result<Publication, ArtifactError> {
    let cas = open_directory(cas_root)?;
    let destination = child_directory(&cas, &digest[..2])?;
    let old = std::ffi::CString::new(temporary.name.as_bytes()).unwrap();
    let new = std::ffi::CString::new(digest.as_bytes()).unwrap();
    // Rename private CAS staging: later cleanup must not change a published inode's ctime.
    let renamed = unsafe {
        libc::renameat2(
            temporary.parent.as_raw_fd(),
            old.as_ptr(),
            destination.as_raw_fd(),
            new.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if renamed != 0 {
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::AlreadyExists {
            return Err(error.into());
        }
        let mut existing = open_member(&destination, &digest)?;
        let (existing_hash, existing_bytes) = hash_file(&mut existing)?;
        if existing_hash != digest || existing_bytes != size_bytes {
            return Err(ArtifactError::new(
                "conflict",
                "existing CAS object is corrupt",
            ));
        }
    }
    destination.sync_all()?;
    temporary.parent.sync_all()?;
    Ok(Publication {
        storage: cas_root.join(&digest[..2]).join(&digest),
        sha256: digest,
        size_bytes,
    })
}

#[derive(Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResumeFile {
    pub source_path: String,
    pub target_path: String,
    pub expected_bytes: u64,
    pub expected_sha256: Option<String>,
}

struct StagingDirectory {
    parent: File,
    name: String,
}
impl Drop for StagingDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(fd_path(&self.parent).join(&self.name));
    }
}

pub fn prepare_resume(
    source: &Path,
    target: &Path,
    files: &[ResumeFile],
) -> Result<Vec<ArtifactFile>, ArtifactError> {
    if files.is_empty() || files.len() > 10000 {
        return Err(ArtifactError::new("limit", "invalid resume file count"));
    }
    let _pins = pin_paths(&[source.to_owned(), target.to_owned()])?;
    let source_fd = open_directory(source)?;
    let target_fd = open_directory(target)?;
    let mut prefix = String::new();
    let mut seen = std::collections::HashSet::new();
    for item in files {
        validate_relative(&item.source_path)?;
        validate_relative(&item.target_path)?;
        let parts: Vec<_> = item.target_path.split('/').collect();
        let src_parts: Vec<_> = item.source_path.split('/').collect();
        if parts.len() < 5
            || parts[..2] != ["logs", "rsl_rl"]
            || src_parts.len() < 5
            || src_parts[..2] != ["logs", "rsl_rl"]
            || item.expected_bytes > 1024u64.pow(4)
            || item
                .expected_sha256
                .as_ref()
                .is_some_and(|s| !valid_digest(s))
        {
            return Err(ArtifactError::new(
                "forbidden",
                "resume members must be bounded RSL files",
            ));
        }
        let directory = parts[..4].join("/");
        if prefix.is_empty() {
            prefix = directory;
        } else if prefix != directory {
            return Err(ArtifactError::new(
                "invalid",
                "resume files require one destination directory",
            ));
        }
        if !seen.insert(item.target_path.clone()) {
            return Err(ArtifactError::new("invalid", "duplicate resume target"));
        }
    }
    let parts: Vec<_> = prefix.split('/').collect();
    let mut parent = target_fd;
    for part in &parts[..3] {
        parent = child_directory(&parent, part)?;
    }
    let name = format!(
        ".resume-{}-{}",
        std::process::id(),
        NEXT_STAGING.fetch_add(1, Ordering::Relaxed)
    );
    fs::create_dir(fd_path(&parent).join(&name))?;
    let temporary = StagingDirectory {
        parent: parent.try_clone()?,
        name,
    };
    let staging = open_relative(&parent, &temporary.name)?;
    let mut records = Vec::new();
    for item in files {
        let relative = item
            .target_path
            .strip_prefix(&(prefix.clone() + "/"))
            .unwrap();
        let mut folder = staging.try_clone()?;
        let target_parts: Vec<_> = relative.split('/').collect();
        for part in &target_parts[..target_parts.len() - 1] {
            folder = child_directory(&folder, part)?;
        }
        let mut output = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(fd_path(&folder).join(target_parts.last().unwrap()))?;
        let mut input = open_member(&source_fd, &item.source_path)?;
        if input.metadata()?.len() != item.expected_bytes {
            return Err(ArtifactError::new("conflict", "resume source size differs"));
        }
        let (digest, bytes) = copy_and_hash(&mut input, &mut output)?;
        if bytes != item.expected_bytes
            || item.expected_sha256.as_ref().is_some_and(|s| *s != digest)
        {
            return Err(ArtifactError::new(
                "conflict",
                "resume source identity differs",
            ));
        }
        output.set_permissions(fs::Permissions::from_mode(0o444))?;
        output.sync_all()?;
        folder.sync_all()?;
        records.push(ArtifactFile {
            path: relative.into(),
            bytes,
            sha256: digest,
        });
    }
    records.sort_by(|a, b| a.path.cmp(&b.path));
    let receipt = serde_json::to_vec(&records)
        .map_err(|_| ArtifactError::new("io", "cannot encode resume receipt"))?;
    let mut receipt_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o444)
        .open(fd_path(&staging).join(".resume-manifest.json"))?;
    receipt_file.write_all(&receipt)?;
    receipt_file.sync_all()?;
    staging.sync_all()?;
    let old = std::ffi::CString::new(temporary.name.as_str()).unwrap();
    let new = std::ffi::CString::new(parts[3]).unwrap();
    let renamed = unsafe {
        libc::renameat2(
            parent.as_raw_fd(),
            old.as_ptr(),
            parent.as_raw_fd(),
            new.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if renamed != 0 {
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::AlreadyExists {
            return Err(error.into());
        }
        let existing = open_relative(&parent, parts[3])?;
        let mut old_receipt = Vec::new();
        open_member(&existing, ".resume-manifest.json")?
            .take(16 * 1024 * 1024 + 1)
            .read_to_end(&mut old_receipt)?;
        if old_receipt != receipt {
            return Err(ArtifactError::new("conflict", "resume destination differs"));
        }
        verify_manifest(
            &target.join(&prefix),
            &records,
            VerifyLimits {
                max_files: 10000,
                max_file_bytes: 1024u64.pow(4),
            },
        )?;
    }
    parent.sync_all()?;
    Ok(records)
}

struct HashReader<'a> {
    file: &'a mut File,
    digest: Sha256,
    bytes: u64,
}
impl Read for HashReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let count = self.file.read(buffer)?;
        self.digest.update(&buffer[..count]);
        self.bytes += count as u64;
        Ok(count)
    }
}

pub fn pack_bundle(
    root: &Path,
    members: &[ArtifactFile],
    cas_root: &Path,
) -> Result<Publication, ArtifactError> {
    if members.is_empty() || members.len() > 10000 {
        return Err(ArtifactError::new("limit", "invalid bundle members"));
    }
    let mut paths = std::collections::HashSet::new();
    for member in members {
        validate_relative(&member.path)?;
        if member.bytes > 1024u64.pow(4)
            || !valid_digest(&member.sha256)
            || !paths.insert(&member.path)
        {
            return Err(ArtifactError::new(
                "invalid",
                "invalid bundle member identity",
            ));
        }
    }
    let _pins = pin_paths(&[root.to_owned(), cas_root.to_owned()])?;
    let source = open_directory(root)?;
    let mut temporary = StagingFile::new(child_directory(&open_directory(cas_root)?, ".staging")?)?;
    {
        let encoder = zstd::stream::write::Encoder::new(&mut temporary.file, 3)?;
        let mut archive = tar::Builder::new(encoder);
        for member in members {
            let mut file = open_member(&source, &member.path)?;
            if file.metadata()?.len() != member.bytes {
                return Err(ArtifactError::new("conflict", "bundle member size differs"));
            }
            let mut reader = HashReader {
                file: &mut file,
                digest: Sha256::new(),
                bytes: 0,
            };
            let mut header = tar::Header::new_gnu();
            header.set_size(member.bytes);
            header.set_mode(0o444);
            header.set_uid(0);
            header.set_gid(0);
            header.set_mtime(0);
            header.set_cksum();
            archive.append_data(&mut header, &member.path, &mut reader)?;
            if reader.bytes != member.bytes
                || format!("{:x}", reader.digest.finalize()) != member.sha256
            {
                return Err(ArtifactError::new("conflict", "bundle member hash differs"));
            }
        }
        archive.into_inner()?.finish()?;
    }
    temporary
        .file
        .set_permissions(fs::Permissions::from_mode(0o444))?;
    temporary.file.sync_all()?;
    temporary.file.rewind()?;
    let (digest, size_bytes) = hash_file(&mut temporary.file)?;
    publish_staging(&temporary, cas_root, digest, size_bytes)
}

/// Store bytes at `cas/<ab>/<sha256>` and return the path.
pub fn put(root: &Path, bytes: &[u8]) -> PathBuf {
    let digest = sha256_hex(bytes);
    let dir = root.join("cas").join(&digest[..2]);
    let _ = fs::create_dir_all(&dir);
    let p = dir.join(&digest);
    if !p.exists() {
        let _ = fs::write(&p, bytes);
    }
    p
}

pub fn get(root: &Path, sha256: &str) -> Option<PathBuf> {
    if sha256.len() < 2 {
        return None;
    }
    let p = root.join("cas").join(&sha256[..2]).join(sha256);
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

pub fn put_file(root: &Path, src: &Path) -> std::io::Result<(String, PathBuf, u64)> {
    let bytes = fs::read(src)?;
    let digest = sha256_hex(&bytes);
    let dest = put(root, &bytes);
    Ok((digest, dest, bytes.len() as u64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn put_get_roundtrip() {
        let dir = env::temp_dir().join(format!("eb-cas-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let p = put(&dir, b"hello-cas");
        let digest = p.file_name().unwrap().to_str().unwrap();
        assert_eq!(digest.len(), 64);
        assert!(get(&dir, digest).unwrap().exists());
        let _ = fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod integrity_tests {
    use super::*;
    use std::os::unix::fs::symlink;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    fn fixture() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "eb-artifact-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn record(path: &str, bytes: &[u8]) -> ArtifactFile {
        ArtifactFile {
            path: path.into(),
            bytes: bytes.len() as u64,
            sha256: sha256_hex(bytes),
        }
    }

    fn limits() -> VerifyLimits {
        VerifyLimits {
            max_files: 200,
            max_file_bytes: 2 * 1024_u64.pow(3),
        }
    }

    #[test]
    fn verify_manifest_rejects_same_size_mutation() {
        let root = fixture();
        fs::write(root.join("model"), b"state").unwrap();
        let files = vec![record("model", b"state")];
        assert_eq!(
            verify_manifest(&root, &files, limits())
                .unwrap()
                .total_bytes,
            5
        );
        fs::write(root.join("model"), b"wrong").unwrap();
        assert!(verify_manifest(&root, &files, limits()).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn verify_manifest_rejects_link_swap_and_escape() {
        let root = fixture();
        let outside = fixture();
        fs::write(outside.join("secret"), b"state").unwrap();
        symlink(outside.join("secret"), root.join("model")).unwrap();
        assert!(verify_manifest(&root, &[record("model", b"state")], limits()).is_err());
        for path in ["../secret", "/etc/passwd", "a//b", "a/./b", "a\\b"] {
            assert!(
                verify_manifest(&root, &[record(path, b"state")], limits()).is_err(),
                "{path}"
            );
        }
        fs::remove_dir_all(&root).unwrap();
        symlink(&outside, &root).unwrap();
        assert!(verify_manifest(&root, &[record("secret", b"state")], limits()).is_err());
        fs::remove_file(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[test]
    fn verify_manifest_keeps_empty_file_and_size_bounds() {
        let root = fixture();
        fs::write(root.join("__init__.py"), b"").unwrap();
        fs::write(root.join("model"), b"state").unwrap();
        let files = vec![record("__init__.py", b""), record("model", b"state")];
        let verified = verify_manifest(&root, &files, limits()).unwrap();
        assert_eq!((verified.file_count, verified.total_bytes), (2, 5));
        assert!(verify_manifest(
            &root,
            &files,
            VerifyLimits {
                max_files: 1,
                max_file_bytes: 10
            }
        )
        .is_err());
        assert!(verify_manifest(
            &root,
            &files,
            VerifyLimits {
                max_files: 2,
                max_file_bytes: 4
            }
        )
        .is_err());
        assert!(verify_manifest(&root, &[files[1].clone(), files[1].clone()], limits()).is_err());
        assert!(verify_manifest(&root, &[], limits()).is_err());
        assert!(verify_manifest(&root, &[record("missing", b"")], limits()).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn publish_file_is_streamed_and_idempotent() {
        let source = fixture();
        let cas = fixture();
        fs::write(source.join("model"), b"state").unwrap();
        let expected = || FileExpectation {
            bytes: 5,
            sha256: Some(sha256_hex(b"state")),
        };
        let first = publish_file(&source, "model", &cas, expected()).unwrap();
        let second = publish_file(&source, "model", &cas, expected()).unwrap();
        assert_eq!(first.storage, second.storage);
        assert_eq!(fs::read(&first.storage).unwrap(), b"state");
        assert_eq!(first.size_bytes, 5);
        assert!(first
            .storage
            .ends_with(format!("{}/{}", &first.sha256[..2], first.sha256)));
        assert_eq!(fs::read_dir(cas.join(".staging")).unwrap().count(), 0);
        assert!(source.join("model").exists());
        fs::remove_dir_all(source).unwrap();
        fs::remove_dir_all(cas).unwrap();
    }

    #[test]
    fn publish_file_never_overwrites_corrupt_cas() {
        let source = fixture();
        let cas = fixture();
        fs::write(source.join("model"), b"state").unwrap();
        let digest = sha256_hex(b"state");
        fs::create_dir_all(cas.join(&digest[..2])).unwrap();
        fs::write(cas.join(&digest[..2]).join(&digest), b"wrong").unwrap();
        assert!(publish_file(
            &source,
            "model",
            &cas,
            FileExpectation {
                bytes: 5,
                sha256: None
            }
        )
        .is_err());
        assert_eq!(
            fs::read(cas.join(&digest[..2]).join(&digest)).unwrap(),
            b"wrong"
        );
        fs::remove_dir_all(source).unwrap();
        fs::remove_dir_all(cas).unwrap();
    }

    #[test]
    fn publish_file_failure_keeps_source_and_no_final_object() {
        let source = fixture();
        let cas = fixture();
        fs::write(source.join("model"), b"state").unwrap();
        assert!(publish_file(
            &source,
            "model",
            &cas,
            FileExpectation {
                bytes: 5,
                sha256: Some(sha256_hex(b"wrong"))
            }
        )
        .is_err());
        assert!(source.join("model").exists());
        assert!(!cas.join(&sha256_hex(b"wrong")[..2]).exists());
        assert_eq!(fs::read_dir(cas.join(".staging")).unwrap().count(), 0);
        fs::remove_dir_all(source).unwrap();
        fs::remove_dir_all(cas).unwrap();
    }

    #[test]
    fn publish_file_rejects_source_mutation() {
        use std::io::Write;
        struct Mutator(PathBuf);
        impl Write for Mutator {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                fs::write(&self.0, b"wrong")?;
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let root = fixture();
        fs::write(root.join("model"), b"state").unwrap();
        let mut file = File::open(root.join("model")).unwrap();
        let before = identity(&file.metadata().unwrap());
        let result = copy_and_hash(&mut file, &mut Mutator(root.join("model")));
        assert_eq!(fs::read(root.join("model")).unwrap(), b"wrong");
        eprintln!(
            "mutation metadata: before={before:?}, after={:?}",
            identity(&file.metadata().unwrap())
        );
        assert!(result.is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn storage_pin_blocks_only_overlapping_deletions() {
        let root = fixture();
        let _pin = pin_paths(&[root.join("runs/run_a/output")]).unwrap();
        assert!(with_unpinned_paths(&[root.join("runs/run_a")], || Ok(())).is_err());
        assert!(with_unpinned_paths(&[root.join("runs/run_b")], || Ok(())).is_ok());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn copy_is_bounded_and_propagates_enospc() {
        struct Counter {
            bytes: usize,
            max_chunk: usize,
        }
        impl Write for Counter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.bytes += bytes.len();
                self.max_chunk = self.max_chunk.max(bytes.len());
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        struct Full;
        impl Write for Full {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::from_raw_os_error(libc::ENOSPC))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let root = fixture();
        fs::write(root.join("model"), vec![42; 3 * 1024 * 1024 + 1]).unwrap();
        let mut counter = Counter {
            bytes: 0,
            max_chunk: 0,
        };
        copy_and_hash(&mut File::open(root.join("model")).unwrap(), &mut counter).unwrap();
        assert_eq!(counter.bytes, 3 * 1024 * 1024 + 1);
        assert!(counter.max_chunk <= 1024 * 1024);
        let error =
            copy_and_hash(&mut File::open(root.join("model")).unwrap(), &mut Full).unwrap_err();
        assert_eq!(error.code, "no_space");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn concurrent_publication_reuses_one_complete_object() {
        let source = fixture();
        let cas = fixture();
        fs::write(source.join("model"), vec![42; 5 * 1024 * 1024]).unwrap();
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let (source, cas) = (source.clone(), cas.clone());
                std::thread::spawn(move || {
                    publish_file(
                        &source,
                        "model",
                        &cas,
                        FileExpectation {
                            bytes: 5 * 1024 * 1024,
                            sha256: None,
                        },
                    )
                })
            })
            .collect();
        let publications: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap().unwrap())
            .collect();
        assert!(publications
            .iter()
            .all(|p| p.storage == publications[0].storage));
        assert_eq!(fs::read_dir(cas.join(".staging")).unwrap().count(), 0);
        fs::remove_dir_all(source).unwrap();
        fs::remove_dir_all(cas).unwrap();
    }

    #[test]
    fn publication_supports_cross_filesystem_source_and_refuses_linked_destination() {
        let source = PathBuf::from("/dev/shm").join(format!(
            "eb-artifact-cross-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&source).unwrap();
        let cas = fixture();
        fs::write(source.join("model"), b"state").unwrap();
        let first = publish_file(
            &source,
            "model",
            &cas,
            FileExpectation {
                bytes: 5,
                sha256: None,
            },
        )
        .unwrap();
        fs::remove_file(&first.storage).unwrap();
        symlink(source.join("model"), &first.storage).unwrap();
        assert!(publish_file(
            &source,
            "model",
            &cas,
            FileExpectation {
                bytes: 5,
                sha256: None
            }
        )
        .is_err());
        assert_eq!(fs::read(source.join("model")).unwrap(), b"state");
        fs::remove_dir_all(source).unwrap();
        fs::remove_dir_all(cas).unwrap();
    }
}

#[cfg(test)]
mod resume_tests {
    use super::*;
    #[test]
    fn resume_files_publish_together_retry_and_reject_conflicts() {
        let root = std::env::temp_dir().join(format!("eb-resume-{}", std::process::id()));
        let source = root.join("source");
        let target = root.join("target");
        fs::create_dir_all(source.join("logs/rsl_rl/exp/source/params")).unwrap();
        fs::create_dir_all(&target).unwrap();
        fs::write(
            source.join("logs/rsl_rl/exp/source/model_10.pt"),
            b"checkpoint",
        )
        .unwrap();
        let file = ResumeFile {
            source_path: "logs/rsl_rl/exp/source/model_10.pt".into(),
            target_path: "logs/rsl_rl/exp/source/model_10.pt".into(),
            expected_bytes: 10,
            expected_sha256: None,
        };
        let result = prepare_resume(&source, &target, &[file.clone()]).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            prepare_resume(&source, &target, &[file.clone()]).unwrap(),
            result
        );
        fs::write(source.join(&file.source_path), b"xxxxxxxxxx").unwrap();
        assert!(prepare_resume(&source, &target, &[file.clone()]).is_err());
        assert_eq!(
            fs::read(target.join(&file.target_path)).unwrap(),
            b"checkpoint"
        );
        let mut bad = file.clone();
        bad.target_path = "../escape".into();
        assert!(prepare_resume(&source, &target, &[bad]).is_err());
        let mut missing = file;
        missing.source_path = "logs/rsl_rl/exp/source/missing.pt".into();
        assert!(prepare_resume(&source, &target, &[missing]).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod bundle_tests {
    use super::*;
    #[test]
    fn bundle_archive_is_zstd_verified_and_compatible() {
        let root = std::env::temp_dir().join(format!("eb-pack-{}", std::process::id()));
        fs::create_dir_all(root.join("staging")).unwrap();
        fs::create_dir_all(root.join("cas")).unwrap();
        fs::write(root.join("staging/policy.onnx"), b"model").unwrap();
        let member = ArtifactFile {
            path: "policy.onnx".into(),
            bytes: 5,
            sha256: sha256_hex(b"model"),
        };
        let published =
            pack_bundle(&root.join("staging"), &[member.clone()], &root.join("cas")).unwrap();
        let decoder =
            zstd::stream::read::Decoder::new(File::open(&published.storage).unwrap()).unwrap();
        let mut archive = tar::Archive::new(decoder);
        let mut entries = archive.entries().unwrap();
        let mut entry = entries.next().unwrap().unwrap();
        assert_eq!(entry.path().unwrap(), Path::new("policy.onnx"));
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"model");
        let mut bad = member;
        bad.sha256 = "0".repeat(64);
        assert!(pack_bundle(&root.join("staging"), &[bad], &root.join("cas")).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod publication_lifetime_tests {
    use super::*;
    #[test]
    fn published_cas_is_independent_of_staging_cleanup() {
        let root=std::env::temp_dir().join(format!("eb-publication-lifetime-{}",std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let mut temporary=StagingFile::new(child_directory(&open_directory(&root).unwrap(),".staging").unwrap()).unwrap();
        temporary.file.write_all(b"model").unwrap();
        temporary.file.sync_all().unwrap();
        let publication=publish_staging(&temporary,&root,sha256_hex(b"model"),5).unwrap();
        let before=fs::metadata(&publication.storage).unwrap();
        assert_eq!(before.nlink(),1,"published inode must have no link still awaiting staging cleanup");
        drop(temporary);
        assert_eq!(identity(&before),identity(&fs::metadata(&publication.storage).unwrap()));
        fs::remove_dir_all(root).unwrap();
    }
}
