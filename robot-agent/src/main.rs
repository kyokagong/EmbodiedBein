use clap::Parser;
use serde_json::json;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    platform: String,
    #[arg(long, default_value = "robot_bhl_01")]
    robot_id: String,
}

fn main() {
    let args = Args::parse();
    println!(
        "{}",
        json!({
            "service": "eb-robot-agent",
            "robot_id": args.robot_id,
            "platform": args.platform,
            "endpoints": ["GET /api/v1/bundles/next", "POST /api/v1/deployments/{id}/telemetry", "POST /events", "GET /policy/active"],
            "note": "first version reports identity only; pull/telemetry wired in M4"
        })
    );
}
