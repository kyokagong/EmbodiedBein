"""Negative geometry fixture; run in the scientific task image, no physics rollout."""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, "/repo/backend")
from eb.rpo_bfm import BODY_ORDER, RPO_LAB_JOINT_ORDER
from eb.runners.rpo_bfm_motion import geometry_screen

model = mujoco.MjModel.from_xml_path("/assets/rpo_description/mjcf/rpo.xml")
data = mujoco.MjData(model)
q_native = np.random.default_rng(42).uniform(-2, 2, 23)
data.qpos[:7] = [0, 0, 2, 1, 0, 0, 0]
data.qpos[7:] = q_native
mujoco.mj_forward(model, data)
addresses = model.jnt_qposadr[[model.joint(n).id for n in RPO_LAB_JOINT_ORDER]]
q = np.tile(data.qpos[addresses], (9, 1))
positions = np.tile(data.xpos[[model.body(n).id for n in BODY_ORDER]], (9, 1, 1))
valid, report = geometry_screen(model, data, q, np.tile([0, 0, 2], (9, 1)),
    np.tile([1, 0, 0, 0], (9, 1)), positions, RPO_LAB_JOINT_ORDER, BODY_ORDER)
assert not valid.any(), report
assert report["max_nonadjacent_self_penetration_m"] > .03, report
print({"negative_geometry_fixture_passed":True, **report})
