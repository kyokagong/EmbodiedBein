use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub memory_total_mb: u64,
    pub memory_free_mb: u64,
    pub run_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HostProfile {
    pub hostname: String,
    pub gpus: Vec<GpuInfo>,
    pub memory_total_mb: u64,
    pub has_dri: bool,
    #[serde(default)]
    pub memory_available_mb: u64,
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex::encode(h.finalize())
}
