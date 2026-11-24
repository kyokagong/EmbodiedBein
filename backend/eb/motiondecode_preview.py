"""Bounded CSV tables and G1 kinematic replay in an isolated CPU Job."""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from eb.motiondecode import HEADER, JOINTS, read_motion, write_json
from eb.motiondecode_view import file_hash

OPERATION = "preview_motiondecode_v1"
IMAGE = "eb/motiondecode-preview:latest"
PAGE_SIZE = 100


def frame_selection(frames: int, start: int) -> tuple[int, list[int]]:
    if type(start) is not int or not 0 <= start < frames:
        raise ValueError("preview start frame is outside the source motion")
    end = min(frames, start + 7200)
    return end, list(range(start, end, 4))


def preview_file(path: str, receipt: dict) -> str:
    if path in {"replay.mp4", "preview-receipt.json"}:
        return path
    match = re.fullmatch(r"table/(\d{6})\.json", path)
    if not match or not 0 <= int(match[1]) < receipt["table_pages"]:
        raise ValueError("file is outside the frozen preview")
    return path


def qpos_columns(model) -> list[int]:
    import mujoco

    free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if free < 0 or model.nq != 36 or model.njnt != 30 or model.jnt_qposadr[free] != 0:
        raise ValueError("G1 model does not have the expected free root and 29 hinges")
    columns = list(range(36))
    for index, name in enumerate(JOINTS):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"G1 model joint differs from CSV: {name}")
        columns[int(model.jnt_qposadr[joint])] = 7 + index
    if sorted(columns) != list(range(36)):
        raise ValueError("G1 joint mapping is not one-to-one")
    return columns


def render(
    source: Path,
    expected_sha: str,
    robot_root: Path,
    start: int,
    output: Path,
    *,
    source_path: str,
    asset_commit: str,
) -> dict:
    import imageio.v2 as imageio
    import mujoco
    import numpy as np

    if source.is_symlink() or not source.is_file() or source.stat().st_size > 32 * 1024**2:
        raise ValueError("preview source must be an unlinked CSV smaller than 32 MiB")
    if file_hash(source) != expected_sha:
        raise ValueError("preview source hash differs from the validated member")
    motion = read_motion(source)
    if len(motion) > 500000:
        raise ValueError("preview source exceeds the 500000-frame budget")
    end, indices = frame_selection(len(motion), start)
    output.mkdir(parents=True, exist_ok=True)
    table = output / "table"
    table.mkdir(exist_ok=True)
    files = []

    def record(path):
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
        )

    for offset in range(0, len(motion), PAGE_SIZE):
        page = table / f"{offset // PAGE_SIZE:06d}.json"
        write_json(
            page,
            {
                "columns": HEADER,
                "rows": motion[offset : offset + PAGE_SIZE].tolist(),
                "frame_start": offset,
                "total_frames": len(motion),
            },
        )
        record(page)
    xml = ET.parse(robot_root / "g1_29dof.xml")
    xml.getroot().find("compiler").set("meshdir", str(robot_root / "meshes"))
    world = xml.getroot().find("worldbody")
    ET.SubElement(world, "geom", name="preview_floor", type="plane", size="0 0 .1", rgba=".18 .22 .26 1")
    ET.SubElement(world, "light", pos="0 0 5", directional="true", diffuse=".8 .8 .8")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding="unicode"))
    model.vis.quality.offsamples = 0
    columns = qpos_columns(model)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.0, 135, -18
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    video = output / "replay.mp4"
    try:
        with imageio.get_writer(
            video,
            fps=30,
            codec="libx264",
            pixelformat="yuv420p",
            ffmpeg_params=["-movflags", "+faststart", "-threads", "2"],
        ) as writer:
            for ordinal, source_index in enumerate(indices):
                data.qpos[:] = motion[source_index, columns]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = motion[source_index, :3] + np.array([0.0, 0.0, 0.15])
                renderer.update_scene(data, camera=camera, scene_option=options)
                writer.append_data(renderer.render())
                if ordinal % 300 == 0:
                    print(json.dumps({"video_frames": ordinal, "total": len(indices)}), flush=True)
    finally:
        renderer.close()
    with imageio.get_reader(video) as reader:
        if reader.count_frames() != len(indices) or reader.get_data(0).shape != (480, 640, 3):
            raise ValueError("encoded video failed frame-count/shape verification")
    record(video)
    receipt = {
        "schema_version": "eb/motiondecode-preview/v1",
        "source_sha256": expected_sha,
        "source_path": source_path,
        "asset_commit": asset_commit,
        "columns": HEADER,
        "frames": len(motion),
        "start_frame": start,
        "end_frame": end,
        "source_fps": 120,
        "fps": 30,
        "video_frames": len(indices),
        "source_frame_indices": indices,
        "table_page_size": PAGE_SIZE,
        "table_pages": math.ceil(len(motion) / PAGE_SIZE),
        "files": files,
        "timebase": "declared_120hz_no_source_timestamps",
        "kinematic_only": True,
        "dynamics_validated": False,
        "source_units": {"position": "m", "joint": "rad", "quaternion": "wxyz"},
    }
    write_json(output / "preview-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--robot-root", required=True)
    parser.add_argument("--asset-commit", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(
        Path(args.source),
        args.expected_sha,
        Path(args.robot_root),
        args.start_frame,
        Path(args.output),
        source_path=args.source_path,
        asset_commit=args.asset_commit,
    )


if __name__ == "__main__":
    main()
