#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
model_dir=${RPO_MODEL_DIR:-/data/kk/embodiedbein/runs/run_d886ddc9b1834ddd9994/output/exported}
robot_dir=${RPO_DESCRIPTION_DIR:-/data/kk/assets/roboparty/rpo-v2/rpo_description}
image=roboparty-sim:local
display_needed=true
build=false
args=()
for arg in "$@"; do
    case "$arg" in
        --headless|--self-test|--help|-h) display_needed=false; args+=("$arg") ;;
        --build) build=true ;;
        *) args+=("$arg") ;;
    esac
done
if [[ $display_needed == true && -z ${DISPLAY:-} ]]; then
    echo 'No DISPLAY. Start from a desktop/remote desktop terminal, or use --headless --seconds 10.' >&2
    exit 2
fi
if [[ ! -d $model_dir || ! -d $robot_dir ]]; then
    echo 'Missing model/robot directory. Set RPO_MODEL_DIR and RPO_DESCRIPTION_DIR.' >&2
    exit 2
fi
if [[ $build == true ]] || ! docker image inspect "$image" >/dev/null 2>&1; then
    docker build -t "$image" -f "$script_dir/roboparty_sim.Dockerfile" "$script_dir"
fi
options=(--rm --init --read-only --tmpfs /tmp:rw,nosuid,size=256m
    --mount "type=bind,src=$model_dir,dst=/models,readonly"
    --mount "type=bind,src=$robot_dir,dst=/robot,readonly"
    --mount "type=bind,src=$script_dir/roboparty_sim.py,dst=/app/roboparty_sim.py,readonly")
if [[ $display_needed == true ]]; then
    options+=(--network host --hostname "$(hostname)" --user "$(id -u):$(id -g)"
        -e "DISPLAY=$DISPLAY" -e LIBGL_ALWAYS_SOFTWARE=1
        -e __GLX_VENDOR_LIBRARY_NAME=mesa -e XDG_CACHE_HOME=/tmp/graphics-cache)
    if [[ -d /tmp/.X11-unix ]]; then
        options+=(--mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly)
    fi
    authority=${XAUTHORITY:-${HOME}/.Xauthority}
    if [[ -f $authority ]]; then
        options+=(--mount "type=bind,src=$authority,dst=/tmp/xauthority,readonly" -e XAUTHORITY=/tmp/xauthority)
    fi
else
    options+=(--network none)
fi
exec docker run "${options[@]}" "$image" "${args[@]}"
