use clap::{Parser, Subcommand};
use std::os::unix::net::UnixStream;
use std::io::{Read, Write};

#[derive(Parser)]
#[command(name = "ebctl")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Health,
    Doctor,
    Ps,
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Health => {
            match uds_get("/run/eb/ebd.sock", "/healthz") {
                Ok(body) => {
                    print!("{body}");
                    std::process::exit(0);
                }
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            }
        }
        Cmd::Doctor => {
            match uds_get("/run/eb/ebd.sock", "/v1/doctor") {
                Ok(body) => println!("{body}"),
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            }
        }
        Cmd::Ps => {
            let _ = std::process::Command::new("docker")
                .args(["ps", "--filter", "label=eb.run_id"])
                .status();
        }
    }
}

fn uds_get(sock: &str, path: &str) -> Result<String, String> {
    let mut stream = UnixStream::connect(sock).map_err(|e| e.to_string())?;
    let req = format!("GET {path} HTTP/1.1\r\nHost: ebd\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    let mut buf = String::new();
    stream.read_to_string(&mut buf).map_err(|e| e.to_string())?;
    if let Some(idx) = buf.find("\r\n\r\n") {
        let body = buf[idx + 4..].to_string();
        if buf.starts_with("HTTP/1.1 200") || buf.contains(" 200 ") {
            return Ok(body);
        }
        return Err(body);
    }
    Err(buf)
}
