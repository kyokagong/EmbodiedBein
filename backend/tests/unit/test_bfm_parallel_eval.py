import numpy as np
import pytest


def test_parallel_case_partition_covers_frozen_suite_once():
    from eb.runners.rpo_bfm_parallel_eval import partition_seeds
    groups = partition_seeds(list(range(50000, 50100)), 4)
    assert [len(group) for group in groups] == [25] * 4
    assert sorted(seed for group in groups for seed in group) == list(range(50000, 50100))
    with pytest.raises(ValueError):
        partition_seeds([1, 1, 2, 3], 4)


def test_bfm_batch_observation_and_command_frontend_keep_rows_independent():
    from eb.runners.rpo_bfm_eval import pack_observation, CommandActorSession
    obs = {key: np.zeros((4, width), np.float32) for key, width in
           (("states", 52), ("last_action", 23), ("history_actor", 300), ("privileged_state", 358))}
    obs["states"][:, 0] = np.arange(4)
    packed = pack_observation(obs)
    assert packed["policy"].shape == (4, 375)
    assert np.array_equal(packed["policy"][:, 0], np.arange(4))
    class Graph:
        def __init__(self, command=False): self.command = command
        def run(self, _, inputs):
            if self.command:
                z = np.zeros((4, 256), np.float32)
                z[np.arange(4), np.arange(4)] = 16
                return [z]
            assert inputs["z"].shape == (4, 256)
            return [np.zeros((4, 23), np.float32)]
    session = CommandActorSession(Graph(), Graph(True))
    session.command = lambda: np.zeros((4, 3), np.float32)
    assert session.run(None, {"observation": packed["policy"]})[0].shape == (4, 23)


def test_case_rng_does_not_consume_other_case_or_global_stream():
    torch = pytest.importorskip("torch")
    from eb.runners.rpo_bfm_parallel_eval import case_rng
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    with case_rng(50000, "cpu"):
        first = (np.random.rand(), torch.rand(2))
    assert torch.equal(before, torch.random.get_rng_state())
    with case_rng(50001, "cpu"):
        other = (np.random.rand(), torch.rand(2))
    with case_rng(50000, "cpu"):
        repeat = (np.random.rand(), torch.rand(2))
    assert first[0] == repeat[0] and torch.equal(first[1], repeat[1])
    assert first[0] != other[0] and not torch.equal(first[1], other[1])


def test_batch_early_failure_does_not_stop_or_reset_other_case(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace as NS
    torch=pytest.importorskip("torch")
    from eb.runners import rpo_bfm_parallel_eval as batch
    monkeypatch.setitem(sys.modules,"rpo_wbc",NS(RPO_LAB_JOINT_ORDER=("q",)))
    class Env:
        device="cpu"
        step_dt=.02
        resets=0
        steps=0
        def __init__(self):
            self.scene=NS(env_origins=torch.zeros(2,3))
            self.command_generator=NS(command=torch.zeros(2,3),heading_target=torch.zeros(2),is_standing_env=torch.zeros(2,dtype=torch.bool))
            self.frozen=[]
            self.robot=NS(joint_names=("q",),data=NS(root_state_w=torch.zeros(2,13),joint_pos=torch.zeros(2,23),joint_vel=torch.zeros(2,23)),
                          write_root_state_to_sim=lambda state,env_ids:self.frozen.extend(env_ids.tolist()),
                          write_joint_state_to_sim=lambda *a,**k:None)
        def obs(self):return {k:torch.zeros(2,w) for k,w in (("states",52),("last_action",23),("history_actor",300),("privileged_state",358))}
        def reset(self):self.resets+=1;return self.obs(),{}
        def step(self,actions):
            self.steps+=1
            return self.obs(),None,torch.tensor([True,self.steps==3]),torch.zeros(2,dtype=torch.bool),{}
    class Session:
        last_latent=np.ones((2,256),np.float32)
        def get_inputs(self):return [NS(name="observation")]
        def run(self,*args):return [np.zeros((2,23),np.float32)]
    monkeypatch.setattr(batch.flat,"_hard_limits",lambda _: {})
    monkeypatch.setattr(batch.flat,"_initial_state",lambda *a: {})
    monkeypatch.setattr(batch.flat,"_violates_hard_limit",lambda *a:False)
    monkeypatch.setattr(batch.flat,"_state",lambda *a:dict(root_pos_w=np.array([0.,0.,.8]),root_quat_w=np.array([1.,0.,0.,0.]),
            root_lin_vel_w=np.zeros(3),root_ang_vel_w=np.zeros(3),projected_gravity_b=np.array([0.,0.,-1.])))
    env=Env()
    rows=batch.evaluate_batch(Session(),env,[50000,50001],tmp_path,False,"standard")
    assert [r["actual_steps"] for r in rows]==[1,3]
    assert [r["metric_steps"] for r in rows]==[1,3]
    assert all(r["completed"] and r["fall"] and not r["success"] for r in rows)
    assert env.resets==1 and env.steps==3 and env.frozen==[0,0]


def test_camera_uses_usd_pose_view_without_changing_render_or_buffer_device(monkeypatch):
    import sys
    from types import SimpleNamespace as NS
    from eb.runners.rpo_bfm_parallel_eval import usd_camera_type
    class Camera:
        cfg=NS(prim_path="/World/envs/env_.*/Camera")
        stage="stage"
        _device="cuda:3"
        def _initialize_impl(self):self._view=NS(count=4,device="cuda:3",_use_fabric=True)
    monkeypatch.setitem(sys.modules,"isaaclab.sensors",NS(Camera=Camera))
    camera=usd_camera_type()()
    camera._initialize_impl()
    assert camera._device=="cuda:3" and camera._view.device=="cuda:3"
    assert camera._view._use_fabric is False


def test_video_integrity_decodes_real_frames_and_rejects_corruption(tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    from eb.runners.rpo_bfm_parallel_eval import video_integrity
    path = tmp_path / "0.mp4"
    with imageio.get_writer(path, fps=10, codec="libx264", macro_block_size=1) as writer:
        for value in (0, 30, 60): writer.append_data(np.full((240, 320, 3), value, dtype=np.uint8))
    receipt = video_integrity(path, "standard")
    assert receipt["decoded_frames"] == 3 and receipt["resolution"] == [320, 240]
    assert receipt["fps"] == 10 and len(receipt["sha256"]) == 64
    path.write_bytes(b"damaged nonempty mp4")
    with pytest.raises(ValueError): video_integrity(path, "standard")
