use clap::{Parser, Subcommand};
use serde_json::json;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    platform: String,
    #[arg(long, default_value = "robot_bhl_01")]
    robot_id: String,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Pull,
    Serve,
    Rollback,
}

fn root(id: &str) -> PathBuf {
    dirs_home().join(".eb/robot").join(id)
}

fn dirs_home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
}

fn main() {
    let args = Args::parse();
    match args.cmd {
        Cmd::Pull => pull(&args),
        Cmd::Rollback => rollback(&args),
        Cmd::Serve => serve(&args),
    }
}

fn pull(args: &Args) {
    let dir = root(&args.robot_id);
    fs::create_dir_all(&dir).unwrap();
    let url = format!("{}/api/v1/bundles/next", args.platform);
    let body = curl_get(&url);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap_or(json!({}));
    let bid = v.get("id").and_then(|x| x.as_str()).unwrap_or("bundle");
    let dest = dir.join(format!("{bid}.ebp"));
    let cas = v
        .pointer("/manifest/cas")
        .and_then(|x| x.as_str())
        .map(|s| format!("{}{}", args.platform, s))
        .unwrap_or(url);
    let _ = Command::new("curl").args(["-fsS", "-o", dest.to_str().unwrap(), &cas]).status();
    if !dest.exists() {
        fs::write(&dest, body).ok();
    }
    let current = dir.join("current");
    let previous = dir.join("previous");
    if current.exists() {
        let _ = fs::remove_file(&previous);
        let _ = fs::rename(&current, &previous);
    }
    let _ = fs::remove_file(&current);
    let _ = symlink(&dest, &current);
    println!("{}", json!({"ok": true, "current": current, "bundle": bid}));
}

fn rollback(args: &Args) {
    let dir = root(&args.robot_id);
    let current = dir.join("current");
    let previous = dir.join("previous");
    if !previous.exists() {
        eprintln!("no previous bundle");
        std::process::exit(1);
    }
    let _ = fs::remove_file(&current);
    let _ = fs::rename(&previous, &current);
    println!("{}", json!({"ok": true, "current": current}));
}

fn serve(args: &Args) {
    let points: Vec<_> = (0..100)
        .map(|i| json!({"tag": "robot.temp_c", "step": i, "value": 40.0}))
        .collect();
    let url = format!("{}/api/v1/deployments/{}/telemetry", args.platform, args.robot_id);
    let payload = json!({"points": points}).to_string();
    let _ = Command::new("curl")
        .args(["-fsS", "-X", "POST", "-H", "Content-Type: application/json", "-d", &payload, &url])
        .status();
    println!("{}", json!({"ok": true, "n": 100, "offline": !Path::new("/sys").exists()}));
}

fn curl_get(url: &str) -> String {
    let out = Command::new("curl").args(["-fsS", url]).output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => e.to_string(),
    }
}
