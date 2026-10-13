import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eb.runners.rpo_ufo_isaac import ObservationCache, raw_aux_rewards, reference_state_batch, terminal_transition


def test_bootstrap_observation_is_idempotent():
    cache = ObservationCache()
    calls = []
    def read():
        calls.append(1)
        return {"states": torch.zeros(2, 52), "history_actor": torch.zeros(2, 300)}
    first = cache.read(read)
    first["history_actor"] += 1
    assert cache.read(read)["history_actor"].sum() == 0
    assert len(calls) == 1
    cache.invalidate()
    cache.read(read)
    assert len(calls) == 2


def test_reference_reset_is_train_only_and_partial():
    from eb.runners.bfm_ufo_replay import UFOExpertSlicer
    clips = []
    for source in ("official_rpo", "g1"):
        arrays = {"states": np.zeros((270, 52), np.float32), "privileged_state": np.zeros((270, 358), np.float32),
            "valid": np.ones(270, bool), "root_pos": np.ones((270, 3), np.float32),
            "root_quat": np.tile(np.array([1, 0, 0, 0], np.float32), (270, 1)),
            "q": np.full((270, 23), .3, np.float32), "dq": np.full((270, 23), .4, np.float32),
            "root_lin_vel_w": np.full((270, 3), .5, np.float32), "root_ang_vel_w": np.full((270, 3), .6, np.float32)}
        clips.append({"id": source, "source": source, "source_group": source, "split": "train", "arrays": arrays})
    sampler = UFOExpertSlicer(clips, manifest_sha256="a" * 64, seed=0)
    root, q, dq = reference_state_batch(sampler, 1, device="cpu")
    assert root.shape == (1, 13)
    assert (root[:, 7:10] == .5).all() and (root[:, 10:] == .6).all()
    assert (q == .3).all() and (dq == .4).all()
    target = torch.zeros(3, 13)
    target[[1]] = root
    assert target[[0, 2]].sum() == 0


def test_timeout_uses_terminal_observation():
    terminal = {"states": torch.full((2, 52), 7.)}
    reset = {"states": torch.zeros(2, 52)}
    result = terminal_transition(reset, {"final_observation": terminal,
        "_final_observation": torch.tensor([True, True])},
        torch.tensor([True, False]), torch.tensor([False, True]))
    assert (result["observation"]["states"] == 7).all()
    assert result["terminated"].tolist() == [[True], [False]]


def test_aux_requires_all_raw_fields_and_is_not_dt_weighted():
    with pytest.raises(ValueError, match="raw"):
        raw_aux_rewards({}, torch.zeros(2, 23), torch.zeros(2, 23), {}, {})


def test_reference_write_preserves_link_velocity_and_other_environments():
    from eb.runners.rpo_ufo_isaac import write_reference_state

    class Robot:
        root = torch.zeros(3, 13)
        q = torch.zeros(3, 23)
        dq = torch.zeros(3, 23)

        def write_root_link_state_to_sim(self, value, *, env_ids):
            self.root[env_ids] = value

        def write_joint_state_to_sim(self, q, dq, *, env_ids):
            self.q[env_ids], self.dq[env_ids] = q, dq

    robot = Robot()
    root = torch.ones(1, 13)
    root[:, 7:10] = .5
    write_reference_state(robot, root, torch.ones(1, 23), torch.full((1, 23), .4),
        torch.tensor([1]), torch.full((3, 3), 10.))
    assert (robot.root[1, :3] == 11).all()
    assert (robot.root[1, 7:10] == .5).all()
    assert robot.root[[0, 2]].sum() == 0
    assert (robot.dq[1] == .4).all()


def test_physics_step_decodes_cpu_rpc_action_on_environment_device(monkeypatch):
    import sys
    from types import SimpleNamespace
    from eb.runners import rpo_ufo_isaac
    from eb import rpo_bfm
    from rpo_flat_extension import FlatResetCompat

    class Parent:
        def step(self, action):
            return action

    class BfmBase(FlatResetCompat, Parent):
        pass

    monkeypatch.setitem(sys.modules, "rpo_bfm_extension", SimpleNamespace(RPOBFMIsaacEnv=BfmBase))
    monkeypatch.setitem(sys.modules, "rpo_bfm", rpo_bfm)
    physics = rpo_ufo_isaac.__getattr__("RPOUFOIsaacPhysics")
    env = object.__new__(physics)
    env.ufo_contact = False
    env.device, env.num_envs = "meta", 2
    env.bfm_last_action = torch.zeros(2, 23, device="meta")
    env.bfm_step_id = torch.zeros(2, 1, device="meta", dtype=torch.long)
    env.bfm_history = SimpleNamespace(append=lambda *args: None)
    env.ufo_cache = ObservationCache()
    env._get_observations = lambda: {"states": torch.zeros(2, 52)}
    # Meta exercises the real cross-device tensor boundary without requiring a GPU unit run.
    action = torch.zeros(2, 23)
    decoded = env.step(action)
    assert decoded.device.type == "meta" and decoded.shape == action.shape
    assert action.device.type == "cpu"


def test_physical_limits_match_position_velocity_and_torque_acceptance():
    from eb.runners.rpo_ufo_isaac import physical_limit_flags
    q = torch.zeros(4, 23)
    dq, torque = q.clone(), q.clone()
    q[0, 1] = 1.002
    dq[1, 5] = 8.002
    torque[2, 7] = 27.002
    q[3, 1], dq[3, 5], torque[3, 7] = 1.0005, 8.0005, 27.0005
    limits = {"joint_pos": torch.tensor([[-1., 1.]] * 23),
              "joint_vel": torch.full((23,), 8.), "applied_torque": torch.full((23,), 27.)}
    flags = physical_limit_flags(q, dq, torque, limits)
    assert flags.tolist() == [[True, False, False], [False, True, False],
                              [False, False, True], [False, False, False]]
    dq[3, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        physical_limit_flags(q, dq, torque, limits)


def test_safe_reference_reset_resamples_velocity_violation_without_mutating_data():
    import numpy as np
    from eb.runners.rpo_ufo_isaac import reference_state_batch

    arrays = {key: np.zeros((2, width), np.float32) for key, width in
              (("root_pos", 3), ("root_quat", 4), ("root_lin_vel_w", 3),
               ("root_ang_vel_w", 3), ("q", 23), ("dq", 23))}
    arrays["root_quat"][:, 0] = 1
    arrays["valid"] = np.ones(2, bool)
    arrays["dq"][0, 5] = 9

    class Sampler:
        clips = [{"split": "train", "arrays": arrays}]
        draws = 0

        def sample_numpy(self, count):
            frame = 0 if self.draws == 0 else 1
            self.draws += 1
            return {"sample_ids": np.tile([0, frame], (count, 1))}

    sampler = Sampler()
    limits = {"joint_pos": torch.tensor([[-1., 1.]] * 23),
              "joint_vel": torch.full((23,), 8.), "applied_torque": torch.full((23,), 27.)}
    _, _, dq = reference_state_batch(sampler, 1, device="cpu", limits=limits)
    assert dq.abs().max() == 0 and sampler.draws == 2
    assert arrays["dq"][0, 5] == 9 and arrays["valid"].all()
    arrays["dq"][1, 5] = 9
    sampler.draws = 0
    with pytest.raises(ValueError, match="frozen draw budget"):
        reference_state_batch(sampler, 1, device="cpu", limits=limits)
    assert sampler.draws == 33


def test_contact_latch_covers_only_new_physics_frames(monkeypatch):
    import sys
    from types import SimpleNamespace
    from eb import rpo_bfm
    from eb.runners import rpo_ufo_isaac

    class Base:
        def _apply_action(self):
            pass

        def _get_dones(self):
            return torch.zeros(1,dtype=torch.bool),torch.zeros(1,dtype=torch.bool)

    monkeypatch.setitem(sys.modules,'rpo_bfm_extension',SimpleNamespace(RPOBFMIsaacEnv=Base))
    monkeypatch.setitem(sys.modules,'rpo_bfm',rpo_bfm)
    cls=rpo_ufo_isaac.__getattr__('RPOUFOIsaacPhysics');env=object.__new__(cls)
    env.ufo_contact=True;env.ufo_safe=False;env.ufo_apply_count=0
    env.ufo_substep_hard=torch.zeros(1,3,dtype=torch.bool)
    env.ufo_substep_contact=torch.zeros(1,dtype=torch.bool)
    sampled=[];env._sample_substep=lambda:sampled.append(env.ufo_apply_count)
    for _ in range(8):env._apply_action()
    env._get_dones()
    assert sampled==list(range(1,9)), 'Previous interval/reset contact must not be latched'
