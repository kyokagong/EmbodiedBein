"""Native MJCF boundary checks; no CUDA or PhysX import."""
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")
ASSET = Path("/data/kk/assets/roboparty/rpo-v2")


def test_rlinf_numpy_action_boundary():
    from types import SimpleNamespace
    from eb.runners.rpo_ufo_mjlab import RPOUFOMJLabPhysics
    env = SimpleNamespace(num_envs=2, _device="cpu")
    with pytest.raises(ValueError, match="Invalid native RPO action"):
        RPOUFOMJLabPhysics.step(env, np.full((2, 23), np.nan, dtype=np.float32))
    class ReachedPhysics(Exception):
        pass
    env.get_observations = lambda: (_ for _ in ()).throw(ReachedPhysics())
    env.history = SimpleNamespace(append=lambda *args: None)
    with pytest.raises(ReachedPhysics):
        RPOUFOMJLabPhysics.step(env, np.zeros((2, 23), dtype=np.float32))


def test_contact_force_is_on_robot_body_in_codec_order():
    from types import SimpleNamespace
    from eb.runners.rpo_ufo_mjlab import RPOUFOMJLabPhysics
    # Actual native floor contacts are negative Z in the primary→secondary convention.
    sensor = SimpleNamespace(data=SimpleNamespace(force=torch.tensor([[[1., 2., -100.], [3., 4., -200.]]])))
    env = SimpleNamespace(sensor=sensor, contact_ids=[1, 0])
    torch.testing.assert_close(RPOUFOMJLabPhysics._forces(env),
        torch.tensor([[[-3., -4., 200.], [-1., -2., 100.]]]))


def test_native_rollout_accepts_full_trace_and_counts_strict_limit(tmp_path):
    pytest.importorskip("rlinf")
    from types import SimpleNamespace
    from eb.runners.bfm_ufo_worker import UFOActorWorker
    trace = torch.zeros(1, 131)
    trace[:, 116] = 1
    worker = SimpleNamespace(
        cfg=SimpleNamespace(actor=SimpleNamespace(model={"physics_profile": "native_mjcf_v12",
            "contact_profile": "native_mjcf_v12", "learning_profile": "physical_limits_v2"}),
            runner=SimpleNamespace(logger=SimpleNamespace(log_path=str(tmp_path)))),
        learner=SimpleNamespace(joint_updates=0, sampling_rounds=1, train_round=lambda _: {}),
        rollout_batch={"next_obs": {"ufo_action_trace": trace}},
        transitions={"action": torch.zeros(1, 23), "episode_id": torch.zeros(1, 1),
            "step_id": torch.zeros(1, 1), "next": {"terminated": torch.zeros(1, 1, dtype=torch.bool)}})
    result = UFOActorWorker.run_training(worker)
    assert result["ufo/hard_velocity_count"] == 1
    assert np.load(tmp_path / "action-traces/round_1.npz")["values"].shape == (1, 131)


def test_native_observation_trace_uses_strict_not_relaxed_velocity_flags():
    from types import SimpleNamespace
    from eb.runners.rpo_ufo_mjlab import RPOUFOMJLabPhysics
    from eb.runners.rpo_ufo_isaac import ObservationCache, PhysicsInterval
    from eb.runners.rpo_bfm_codec import History
    limits = {"joint_pos": torch.tensor([[-10., 10.]] * 23),
              "joint_vel": torch.full((23,), 8.), "applied_torque": torch.full((23,), 27.)}
    q = torch.zeros(1, 23); dq = q.clone(); dq[:, 0] = 9.; q[:, 1] = 9.5
    interval = PhysicsInterval(1, limits, device="cpu", velocity_multiplier=1.3)
    interval.begin()
    for index in range(8):
        interval.sample(index, q, dq, q, q, torch.zeros(1, 2), torch.zeros(1, 5, 3))
    assert interval.strict_flags[:, 1].all() and not interval.training_flags[:, 1].any()
    quat = torch.zeros(1, 24, 4); quat[..., 0] = 1
    state = {"q": q, "dq": dq, "torques": q, "body_quat": quat,
        **{key: torch.zeros(1, 24, 3) for key in ("body_pos", "body_lin_vel", "body_ang_vel")}}
    env = SimpleNamespace(cache=ObservationCache(), _state=lambda: state,
        _forces=lambda: torch.zeros(1, 24, 3), limits=limits, num_envs=1, _device="cpu",
        feet=[22, 23], protected=[0, 1, 2, 3, 4], joint_ids=list(range(23)), last=q, previous=q, requested=q, target=q,
        interval=interval, interval_valid=torch.ones(1, dtype=torch.bool),
        steps=torch.ones(1, 1, dtype=torch.long), episode=torch.ones(1, 1, dtype=torch.long),
        peak_force=torch.zeros(1, 2), macro=torch.zeros(1, 3), lift=torch.zeros(1),
        history=History(1, device="cpu"), reference=torch.zeros(1, 3), tracking=torch.zeros(1, 1),
        robot=SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=limits["joint_pos"][None] * .9)),
        lab=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(1, 3))))
    obs = RPOUFOMJLabPhysics.get_observations(env)
    assert obs["ufo_action_trace"][0, 116] == 1
    assert obs["ufo_interval_trace"][0, 169] == 0
    assert obs["ufo_aux_rewards"][0, 2] == .5


def test_completed_substep_keeps_force_that_was_integrated():
    from types import SimpleNamespace
    from eb.runners.rpo_ufo_mjlab import RPOUFOMJLabPhysics
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep=".0025" integrator="implicitfast" solver="Newton"/>
      <worldbody><geom type="plane" size="1 1 .1"/>
        <body pos="0 0 .099"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model); data.qvel[2] = -.5
    expected = mujoco.MjData(model); expected.qpos[:] = data.qpos; expected.qvel[:] = data.qvel
    def force(d):
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, d, 0, wrench)
        return torch.tensor([[0., 0., wrench[0]]])
    mujoco.mj_step(model, expected)
    integrated = force(expected)
    sim = SimpleNamespace(step=lambda: mujoco.mj_step(model, data), sense=lambda: None,
        data=SimpleNamespace(nacon=torch.tensor([1]), naconmax=128,
                             nefc=torch.tensor([4]), njmax=1024))
    scene = SimpleNamespace(update=lambda dt: None)
    env = SimpleNamespace(lab=SimpleNamespace(sim=sim, scene=scene),
        robot=SimpleNamespace(data=SimpleNamespace(qfrc_actuator=torch.from_numpy(data.qfrc_actuator)[None])),
        joint_ids=[0], interval=SimpleNamespace(physics_dt=.0025), _forces=lambda: force(data),
        _refresh=lambda dt=0.: mujoco.mj_forward(model, data))
    actual, _ = RPOUFOMJLabPhysics._completed_substep(env)
    torch.testing.assert_close(actual, integrated)
    assert not torch.allclose(actual, force(data)), "force after extra forward substituted for impulse"
    sim.data.naconmax = 1
    with pytest.raises(ValueError, match="capacity"):
        RPOUFOMJLabPhysics._completed_substep(env)


def test_native_mjcf_preserves_inertia_geometry_and_order():
    from eb.runners.rpo_ufo_mjlab import prepare_spec, actuator_parameters
    from eb.rpo_bfm import RPO_LAB_JOINT_ORDER, JOINT_POSITION_MIN, JOINT_POSITION_MAX
    path = ASSET / "rpo_description/mjcf/rpo.xml"
    before = mujoco.MjModel.from_xml_path(str(path))
    after = prepare_spec(ASSET).compile()
    assert after.nq == before.nq == 30
    assert after.nv == before.nv == 29
    assert after.nu == 0
    np.testing.assert_allclose(after.body_mass, before.body_mass)
    np.testing.assert_allclose(after.body_inertia, before.body_inertia)
    assert after.nmesh == before.nmesh
    assert after.opt.timestep == .0025
    assert after.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    for name, low, high in zip(RPO_LAB_JOINT_ORDER, JOINT_POSITION_MIN, JOINT_POSITION_MAX):
        joint = after.joint(name)
        assert after.dof_armature[joint.dofadr[0]] == pytest.approx(.01)
        np.testing.assert_allclose(joint.range, [low, high])
    assert actuator_parameters("right_ankle_pitch_joint") == (40., 2.5, 27., 8.)
    collision = after.geom_contype != 0
    assert (after.geom_conaffinity[collision] == 0).all(), "native self-collision masks changed"


def test_native_profile_does_not_modify_frozen_codec():
    from eb.rpo_bfm import CODEC_V2_HASH, spec
    from eb.runners.rpo_ufo_mjlab import PHYSICS_PROFILE
    assert CODEC_V2_HASH == "dce25d554ceb7fa69287515946d2f5cd9d41d14f1b1ef38715b819d7e962b6ef"
    assert spec("ee_rpo_v2_bfm_23dof_v2")["physics_hz"] == 200
    assert PHYSICS_PROFILE["physics_dt_s"] == .0025
    assert PHYSICS_PROFILE["control_dt_s"] == .02


@pytest.mark.parametrize('recipe_id', ['rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_v12',
    'rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_viability_v13'])
def test_resolved_native_config_rejects_cross_backend(recipe_id):
    pytest.importorskip("rlinf")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from eb.domain.rlinf_form import build_overrides
    from eb.domain.training_recipe import UFO_MJLAB_RECIPE
    from eb.rpo_bfm import CODEC_V2_HASH
    from eb.runners.bfm_ufo_worker import validate_ufo_runtime_config
    overrides = build_overrides(recipe_id, 4, "native-check", ["actor.seed=42"])
    # The adapter adds the frozen dataset/interface identity before Hydra composition.
    overrides.append("+actor.model.codec_sha256=" + CODEC_V2_HASH)
    with initialize_config_dir(version_base="1.1", config_dir="/workspace/eb_runner/config"):
        cfg = compose(config_name="rpo_bfm_ufo_isaac", overrides=overrides)
        OmegaConf.resolve(cfg)
        validate_ufo_runtime_config(cfg)
        cfg.env.train.env_type = "isaaclab"
        with pytest.raises(ValueError, match="runtime"):
            validate_ufo_runtime_config(cfg)
