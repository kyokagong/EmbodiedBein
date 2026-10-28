"""CPU-only actual-upstream RobotState, MotionLib/FK and namespaced RPO geometry check.

Run in the PBFM task image with repository at /workspace and frozen upstream at
/sources/pbfm, managed RPO asset snapshot at /robot-assets. No training is run.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import mujoco
import numpy as np

from eb.runners.pbfm_source import prepare_pbfm_source, rpo_robot_config


def main():
    from types import SimpleNamespace

    from eb.runners.pbfm_eval import terrain_height

    terrain_model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="floor" type="plane" size="5 5 .1" group="0"/>
      <body name="static_stairs"><geom name="lower" type="box" pos="0 0 .1" size=".5 .5 .1" group="0"/>
      <geom name="upper" type="box" pos="1.5 0 .2" size=".5 .5 .2" group="0"/></body>
      <body name="robot/floating" pos="0 0 1"><freejoint/>
      <geom name="robot/torso" type="box" size=".2 .2 .2" group="0"/></body>
      </worldbody></mujoco>""")
    terrain_data = mujoco.MjData(terrain_model)
    mujoco.mj_forward(terrain_model, terrain_data)
    ray_scene = SimpleNamespace(model=terrain_model, data=terrain_data)
    assert abs(terrain_height(ray_scene, [0.0, 0.0, 1.0]) - 0.2) < 1e-8
    assert abs(terrain_height(ray_scene, [1.5, 0.0, 1.0]) - 0.4) < 1e-8
    assert (
        terrain_model.geom_group[mujoco.mj_name2id(terrain_model, mujoco.mjtObj.mjOBJ_GEOM, "robot/torso")]
        == 0
    )
    print(json.dumps({"group0_terrain_ray_verified": True, "floating_robot_excluded": True}), flush=True)
    private = Path("/tmp/pbfm-evaluation-source")
    prepare_pbfm_source(Path("/sources/pbfm"), private)
    sys.path.insert(0, str(private))
    from pbfm.motion.data.robot_state import RobotStateMotion
    from pbfm.motion.data.robot_state_convert import robot_state_to_ufo_motion
    from pbfm.robots import load_robot_training_spec

    from eb.runners.pbfm_data import qualify_native, validate_records
    from eb.runners.pbfm_eval import foot_geometry

    native = load_robot_training_spec(rpo_robot_config(private))
    robot = native.robot
    tree = ET.parse(robot.xml_path)
    body_names = {body.get("name") for body in tree.findall(".//body")}
    for body in tree.findall(".//body"):
        body.set("name", "robot/" + body.get("name"))
    for element in tree.iter():
        for key in ("body", "body1", "body2"):
            if element.get(key) in body_names:
                element.set(key, "robot/" + element.get(key))
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
    geometry = foot_geometry(model, ["left_ankle_roll_link", "right_ankle_roll_link"])
    assert all(row["toe_local"][0] > row["heel_local"][0] for row in geometry.values())
    print(json.dumps({"geometry": geometry, "mujoco": mujoco.__version__}, sort_keys=True), flush=True)
    if "--geometry-only" in sys.argv:
        raise SystemExit(0)
    dof = np.asarray(
        [native.default_joint_angles[joint] for joint in robot.control_joint_names], dtype=np.float32
    )
    trajectory = np.tile(dof, (3, 1))
    for index, joint in enumerate(robot.control_joint_names):
        low, high = robot.joint_ranges[joint]
        for frame, sign in ((1, 1), (2, -1)):
            trajectory[frame, index] = np.clip(dof[index] + sign * 0.025 * np.sin(index + 0.3), low, high)
    yaws = np.asarray([0.0, 0.1, -0.2])
    state = RobotStateMotion(
        motion_key="static-rpo",
        root_pos=np.asarray([[0.0, 0.0, 0.8], [0.01, -0.02, 0.81], [0.03, 0.01, 0.79]]),
        root_quat=np.stack([np.zeros(3), np.zeros(3), np.sin(yaws / 2), np.cos(yaws / 2)], axis=1),
        dof_pos=trajectory,
        fps=50.0,
        joint_names=list(robot.control_joint_names),
        metadata={"ancestor_id": "cpu-fixture"},
    )
    records = {"static-rpo": robot_state_to_ufo_motion(state, robot, "cpu-fixture")}
    validate_records(records, robot, {"cpu-fixture"})
    motion = Path("/tmp/pbfm-rpo-cpu.pkl")
    joblib.dump(records, motion)
    loader, fk = qualify_native(motion, records, native)
    assert loader["all_motions_loaded"] is True and fk["frames_checked"] == 3
    print(
        json.dumps(
            {"loader": loader, "fk": fk, "geometry": geometry, "mujoco": mujoco.__version__}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
