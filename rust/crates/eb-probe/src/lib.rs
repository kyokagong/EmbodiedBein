use eb_core::{GpuInfo, HostProfile};
use std::path::Path as FsPath;
use sysinfo::System;

pub fn doctor() -> HostProfile {
    let mut sys = System::new();
    sys.refresh_memory();
    HostProfile {
        hostname: System::host_name().unwrap_or_else(|| "unknown".into()),
        gpus: nvidia_smi_gpus(),
        memory_total_mb: sys.total_memory() / 1024 / 1024,
        memory_available_mb: sys.available_memory() / 1024 / 1024,
        has_dri: FsPath::new("/dev/dri").exists(),
    }
}

pub fn nvidia_smi_gpus() -> Vec<GpuInfo> {
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

#[cfg(test)]
mod tests {
    #[test]
    fn doctor_reports_available_host_memory() {
        let profile = super::doctor();
        assert!(profile.memory_total_mb > 0);
        assert!(profile.memory_available_mb > 0);
        assert!(profile.memory_available_mb <= profile.memory_total_mb);
    }
}
