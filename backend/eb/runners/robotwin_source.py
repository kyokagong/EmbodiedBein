"""Prepare a run-private RoboTwin source overlay with exact action evidence."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ORIGINAL_BASE_TASK_SHA256 = "aa9d717ad214b9d68c6eeb83ed03f2491db6addc98deaa988a48a3c074178bdd"
REPLACEMENTS = (
    (
        '''    def gen_sparse_reward_data(self, chunk_actions, action_type="qpos"):  # action_type: qpos or ee

        infos = {
            "success": False,
        }
''',
        '''    def gen_sparse_reward_data(self, chunk_actions, action_type="qpos"):  # action_type: qpos or ee

        infos = {
            "success": False,
            "chunk_requested": int(chunk_actions.shape[0]),
            "chunk_executed": 0,
            "sim_steps_executed": 0,
            "sim_timestep": float(self.scene.get_timestep()),
        }
''',
    ),
    (
        '''        steps_executed = 0

        while now_left_id < left_n_step or now_right_id < right_n_step:
''',
        '''        steps_executed = 0
        action_steps_executed = 0

        while now_left_id < left_n_step or now_right_id < right_n_step:
''',
    ),
    (
        '''            steps_executed += 1

            if self.check_success():
''',
        '''            steps_executed += 1
            progress = max(now_left_id / left_n_step, now_right_id / right_n_step)
            action_steps_executed = min(len(actions), math.ceil(progress * len(actions)))

            if self.check_success():
''',
    ),
    (
        '''                termination = np.array([1], dtype=np.int32)
                return reward, termination, truncation, infos

        if getattr(self, "eval_success", False):
''',
        '''                termination = np.array([1], dtype=np.int32)
                infos["chunk_executed"] = action_steps_executed
                infos["sim_steps_executed"] = steps_executed
                return reward, termination, truncation, infos

        if getattr(self, "eval_success", False):
''',
    ),
    (
        '''        self._update_render()
        if self.render_freq:  # UI
            self.viewer.render()
        return reward, termination, truncation, infos


    def save_camera_images''',
        '''        infos["chunk_executed"] = len(actions)
        infos["sim_steps_executed"] = steps_executed
        self._update_render()
        if self.render_freq:  # UI
            self.viewer.render()
        return reward, termination, truncation, infos


    def save_camera_images''',
    ),
)
OPTIONAL_DEPENDENCY_REPLACEMENTS = {
    "envs/robot/planner.py": (
        (
            "    print('[planner.py]: Something wrong happened when importing CuroboPlanner! Please check if Curobo is installed correctly. If the problem still exists, you can install Curobo from https://github.com/NVlabs/curobo manually.')\n",
            "",
        ),
        (
            "    print('Exception traceback:')\n    traceback.print_exc()",
            '    print(f"[planner.py] optional CuroboPlanner disabled: {type(e).__name__}")',
        ),
    ),
    "envs/camera/camera.py": ((
        'except:\n    print("missing pytorch3d")\n\n    def fps(points, num_points=1024, use_cuda=True):\n'
        '        print("fps error: missing pytorch3d")\n        exit()',
        'except ImportError:\n\n    def fps(points, num_points=1024, use_cuda=True):\n'
        '        raise RuntimeError("pytorch3d is required for point-cloud sampling")',
    ),),
}
PATCH_SHA256 = "sha256:" + hashlib.sha256(json.dumps(
    {"evidence":REPLACEMENTS, "optional_dependencies":OPTIONAL_DEPENDENCY_REPLACEMENTS},
    sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
PATCHED_BASE_TASK_SHA256 = "sha256:c72b26917a2e2c35911049218d141768b9570d73d6c9d5d9fe49cc561cb4a835"


def prepare_robotwin_source(source: Path, output: Path, expected_patch: str) -> tuple[Path, str]:
    if expected_patch != PATCH_SHA256:
        raise ValueError("RoboTwin evidence patch 与冻结计划不一致")
    original = source / "envs/_base_task.py"
    if hashlib.sha256(original.read_bytes()).hexdigest() != ORIGINAL_BASE_TASK_SHA256:
        raise ValueError("RoboTwin evidence patch 的上游源码不匹配")
    target = output / "runtime/robotwin-source"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", "assets", "__pycache__", "*.pyc"))
    patched = target / "envs/_base_task.py"
    text = patched.read_text()
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise ValueError("RoboTwin evidence patch 无法唯一应用")
        text = text.replace(old, new)
    patched.write_text(text)
    for relative, replacements in OPTIONAL_DEPENDENCY_REPLACEMENTS.items():
        path = target / relative
        text = path.read_text()
        for old, new in replacements:
            if text.count(old) != 1:
                raise ValueError(f"RoboTwin optional dependency patch 无法唯一应用: {relative}")
            text = text.replace(old, new)
        path.write_text(text)
    digest = "sha256:" + hashlib.sha256(patched.read_bytes()).hexdigest()
    if digest != PATCHED_BASE_TASK_SHA256:
        raise ValueError("RoboTwin evidence patch 输出摘要不一致")
    return target, digest
