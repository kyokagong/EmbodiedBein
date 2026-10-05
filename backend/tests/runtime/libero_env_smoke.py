"""Run one real LIBERO reset and control step in the job image (no training)."""
from pathlib import Path

# Installs upstream's robosuite/MuJoCo compatibility hook before importing LIBERO.
import rlinf.envs.libero.libero_env
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
assert bddl.is_file()
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
try:
    env.reset()
    env.set_init_state(suite.get_task_init_states(0)[0])
    obs, reward, done, info = env.step([0.0] * 7)
    assert obs["agentview_image"].shape == (64, 64, 3)
    assert obs["robot0_eye_in_hand_image"].shape == (64, 64, 3)
    print("PASS: real LIBERO spatial task 0 reset + one action + both cameras", reward, done)
finally:
    env.close()
