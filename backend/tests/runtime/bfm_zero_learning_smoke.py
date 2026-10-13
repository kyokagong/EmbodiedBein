"""Focused torch behavior checks, executed in the real RLinf task image."""

from __future__ import annotations

import argparse


def check_codec():
    import torch
    from rpo_bfm import ACTION_GAIN, DEFAULT_Q
    from rpo_bfm_codec import History, encode_privileged, encode_state, policy_to_target

    q = torch.tensor(DEFAULT_Q).repeat(2, 1)
    dq = torch.zeros_like(q)
    gravity = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
    omega = torch.tensor([[4., 8., 12.]]).repeat(2, 1)
    state = encode_state(q, dq, gravity, omega)
    assert state.shape == (2, 52)
    torch.testing.assert_close(state[:, :23], torch.zeros_like(q))
    torch.testing.assert_close(state[:, -3:], omega / 4)
    history = History(2, device="cpu")
    empty = history.snapshot().clone()
    history.append(state, torch.zeros_like(q))
    assert history.snapshot().shape == (2, 300)
    history.append(state + 1, torch.ones_like(q))
    untouched = history.snapshot()[1].clone()
    history.reset(torch.tensor([0]))
    assert torch.equal(history.snapshot()[0], empty[0])
    assert torch.equal(history.snapshot()[1], untouched)
    torch.testing.assert_close(policy_to_target(torch.eye(23)),
                               torch.tensor(DEFAULT_Q) + .25 * torch.eye(23) * torch.tensor(ACTION_GAIN))

    pos = torch.zeros(2, 24, 3)
    pos[:, 0, 2] = .75
    pos[:, 1, 0] = 1
    quat = torch.zeros(2, 24, 4)
    quat[:, :, 0] = 1  # wxyz
    vel = torch.zeros_like(pos)
    privileged = encode_privileged(pos, quat, vel, vel)
    assert privileged.shape == (2, 358)
    assert torch.equal(privileged[:, 0], torch.full((2,), .75))
    torch.testing.assert_close(privileged[:, 1:4], torch.tensor([[1., 0., -.75]]).repeat(2, 1))
    pos[:, :, :2] += 100  # Translation does not affect local state.
    torch.testing.assert_close(privileged, encode_privileged(pos, quat, vel, vel))
    try:
        encode_state(q[:, :22], dq[:, :22], gravity, omega)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong joint dimension accepted")


def check_motion():
    import copy
    import hashlib
    import json
    import tempfile
    from pathlib import Path

    import numpy as np
    from rpo_bfm import BODY_ORDER, CODEC_HASH, MOTION_CONVERTER_REVISION, RPO_LAB_JOINT_ORDER
    from rpo_bfm_motion import load_motion_view

    # Synthetic fixture tests sampling only; it is never published as expert data.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        states = np.repeat(np.arange(24, dtype=np.float32)[:, None], 52, axis=1)
        privileged = np.repeat(np.arange(24, dtype=np.float32)[:, None], 358, axis=1)
        valid = np.ones(24, dtype=bool)
        valid[12] = False
        file = root / "stand.npz"
        np.savez(file, states=states, privileged_state=privileged,
                 reward_features=np.zeros((24, 6), dtype=np.float32), valid=valid)
        manifest = {"schema_version":"eb/rpo-bfm-motion/v1", "purpose":"behavior_prior", "usage":"train",
                    "joint_order":list(RPO_LAB_JOINT_ORDER), "body_order":list(BODY_ORDER),
                    "converter_revision":MOTION_CONVERTER_REVISION,
                    "codec_sha256":CODEC_HASH, "asset_sha256":"a"*64, "qualification":"diagnostic",
                    "source_rights":"repository_license_only", "dynamic_replay_verified":False,
                    "source_refs":[{"uri":"fixture", "revision":"b"*40, "sha256":"c"*64, "license":"test"}],
                    "clips":[{"id":"stand", "file":"stand.npz", "sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
                              "fps":50, "split":"train", "source_group":"stand", "frames":24}]}
        (root / "manifest.json").write_text(json.dumps(manifest))
        sampler = load_motion_view(root, asset_sha256="a"*64, seed=42)
        rng = copy.deepcopy(sampler.state_dict())
        batch = sampler.sample_sequences(16, length=8)
        assert batch["states"].shape == (16, 52)
        current = batch["states"][:, 0].reshape(2, 8)
        assert np.array_equal(np.diff(current, axis=1), np.ones((2, 7)))
        assert not np.any(current == 12)
        assert np.array_equal(batch["next_states"][:, 0], batch["states"][:, 0]+1)
        sampler.load_state_dict(rng)
        again = sampler.sample_sequences(16, length=8)
        assert np.array_equal(batch["sample_ids"], again["sample_ids"])
        try:
            sampler.sample_sequences(15, length=8)
        except ValueError:
            pass
        else:
            raise AssertionError("non-divisible expert batch accepted")
        manifest["clips"][0]["sha256"] = "d"*64
        (root / "manifest.json").write_text(json.dumps(manifest))
        try:
            load_motion_view(root, asset_sha256="a"*64)
        except ValueError:
            pass
        else:
            raise AssertionError("changed motion content accepted")


def check_networks():
    import torch
    from bfm_zero_policy import BFMZeroPolicy, project_z
    from rpo_bfm import OBS_DIMS

    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = BFMZeroPolicy({"bfm_profile":"smoke"}).eval()
    observations = {key:torch.randn(16, width) for key, width in OBS_DIMS.items()}
    obs = model.normalize(observations)
    z = project_z(torch.randn(16, 256))
    action = model.actor(obs, z, .05).mean
    assert action.shape == (16, 23) and bool((action.abs() <= 1).all())
    assert model.forward_map(obs, z, action).shape == (2, 16, 256)
    assert model.critic(obs, z, action).shape == (2, 16, 1)
    assert model.aux_critic(obs, z, action).shape == (2, 16, 1)
    assert model.backward_map(obs).shape == (16, 256)
    assert model.discriminator.compute_logits(obs, z).shape == (16, 1)
    assert not any(p.requires_grad for p in model.targets.parameters())
    # Verify the actual builders, including asymmetric embedding depth, without allocating 3 GB.
    with torch.device("meta"):
        main = BFMZeroPolicy({"bfm_profile":"research"})
    assert sum(p.numel() for module in main.online_modules().values() for p in module.parameters()) == 435116484
    assert sum(p.numel() for p in main.targets.parameters()) == 400806352
    assert len(model.actor.embed_s) == 2
    assert len(model.forward_map.embed_z) == 3  # Smoke has one embedding residual + two projections.
    before = {k:v.clone() for k,v in model.obs_normalizer.state_dict().items()}
    model.normalize(observations)
    assert all(torch.equal(v, model.obs_normalizer.state_dict()[k]) for k,v in before.items())
    try:
        project_z(torch.zeros(2, 256))
    except ValueError:
        pass
    else:
        raise AssertionError("degenerate prompt accepted")


def check_losses():
    import torch
    from bfm_zero_learning import (
        actor_objective,
        bellman_discount,
        critic_objective,
        discriminator_objective,
        encode_expert,
        fb_objective,
        mixed_z,
        soft_update_targets,
    )
    from bfm_zero_policy import BFMZeroPolicy, project_z
    from rpo_bfm import OBS_DIMS

    torch.set_num_threads(2)
    forward = torch.tensor([[[1., 2.], [3., 4.]], [[2., 0.], [0., 2.]]], requires_grad=True)
    backward = torch.eye(2, requires_grad=True)
    target_forward = torch.zeros_like(forward, requires_grad=True)
    target_backward = torch.eye(2, requires_grad=True)
    loss, _ = fb_objective(forward, backward, target_forward, target_backward,
                           torch.tensor([[.98], [0.]]), ortho_coef=1.)
    torch.testing.assert_close(loss, torch.tensor(-2.25))
    loss.backward()
    assert forward.grad is not None and backward.grad is not None
    assert target_forward.grad is None and target_backward.grad is None
    torch.testing.assert_close(bellman_discount(torch.tensor([[False], [False], [True]]), .98),
                               torch.tensor([[.98], [.98], [0.]]))

    torch.manual_seed(7)
    model = BFMZeroPolicy({"bfm_profile":"smoke"}).eval()
    raw = {key:torch.randn(16, width) for key, width in OBS_DIMS.items()}
    obs = model.normalize(raw)
    z = project_z(torch.randn(16, 256))
    action = torch.randn(16, 23).clamp(-1, 1)
    online = model.online_modules()
    optimizers = {name:torch.optim.Adam(module.parameters(), lr=1e-4) for name,module in online.items()}
    original = {name:next(module.parameters()).detach().clone() for name,module in online.items()}
    discriminator_loss, _ = discriminator_objective(model.discriminator, obs, z, obs, z, gradient_penalty=10.)
    optimizers["discriminator"].zero_grad()
    discriminator_loss.backward()
    optimizers["discriminator"].step()
    next_action = model.actor(obs, z, .05).mean.detach()
    loss, _ = fb_objective(model.forward_map(obs, z, action), model.backward_map(obs),
                           model.targets["forward_map"](obs, z, next_action), model.targets["backward_map"](obs),
                           torch.full((16, 1), .98))
    optimizers["forward"].zero_grad()
    optimizers["backward"].zero_grad()
    loss.backward()
    optimizers["forward"].step()
    optimizers["backward"].step()
    for name, target_name, module in (("critic", "critic", model.critic), ("aux_critic", "aux_critic", model.aux_critic)):
        q_loss, _ = critic_objective(module(obs, z, action), model.targets[target_name](obs, z, next_action),
                                     torch.ones(16, 1), torch.full((16, 1), .98))
        optimizers[name].zero_grad()
        q_loss.backward()
        optimizers[name].step()
    current_action = model.actor(obs, z, .05).sample(clip=.3)
    a_loss, _ = actor_objective(model.forward_map(obs, z, current_action), model.critic(obs, z, current_action),
                                model.aux_critic(obs, z, current_action), z)
    optimizers["actor"].zero_grad()
    a_loss.backward()
    optimizers["actor"].step()
    for name,module in online.items():
        assert not torch.equal(next(module.parameters()).detach(), original[name]), name
        assert all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all()) for p in module.parameters()), name
    assert not any(p.grad is not None for p in model.targets.parameters())

    expert_z = encode_expert(model, obs, sequence_length=8)
    assert expert_z.shape == (16, 256)
    torch.testing.assert_close(expert_z[:8], expert_z[:1].expand(8, -1))
    relabeled_z = mixed_z(model, obs, expert_z)
    torch.testing.assert_close(relabeled_z.norm(dim=-1), torch.full((16,), 16.))

    target_before = {name:[parameter.clone() for parameter in module.parameters()]
                     for name,module in model.targets.items()}
    with torch.no_grad():
        for module in online.values():
            for parameter in module.parameters():
                parameter.add_(2.)
    soft_update_targets(model, fb_tau=.2, critic_tau=.4)
    for name,online_name,tau in (("forward_map", "forward", .2), ("backward_map", "backward", .2),
                                 ("critic", "critic", .4), ("aux_critic", "aux_critic", .4)):
        for before,target,current in zip(target_before[name], model.targets[name].parameters(),
                                         online[online_name].parameters(), strict=True):
            torch.testing.assert_close(target, before+2*tau, atol=5e-5, rtol=1e-4)


def check_replay():
    import copy

    import torch
    from bfm_zero_learning import TransitionReplay
    from rpo_bfm import OBS_DIMS

    replay = TransitionReplay(capacity=3, seed=42)
    def batch(start, count):
        ids = torch.arange(start, start+count, dtype=torch.float32)
        return {"obs":{key:ids[:, None].expand(count, width).clone() for key,width in OBS_DIMS.items()},
                "next_obs":{key:(ids+1)[:, None].expand(count, width).clone() for key,width in OBS_DIMS.items()},
                "action":torch.zeros(count, 23), "z":torch.ones(count, 256),
                "terminated":torch.zeros(count, 1, dtype=torch.bool),
                "truncated":torch.zeros(count, 1, dtype=torch.bool), "aux_rewards":torch.zeros(count, 3)}
    replay.add(batch(0, 2))
    replay.add(batch(2, 3))
    assert len(replay) == 3 and replay.total_seen == 5
    checkpoint = replay.state_dict()
    sample = replay.sample(16)
    assert sample["obs"]["states"][:, 0].min() >= 2
    restored = TransitionReplay(capacity=3, seed=99)
    restored.load_state_dict(checkpoint)
    again = restored.sample(16)
    assert torch.equal(sample["sample_ids"], again["sample_ids"])
    assert torch.equal(sample["next_obs"]["states"]-sample["obs"]["states"], torch.ones(16, 52))
    malformed = copy.deepcopy(checkpoint)
    del malformed["storage"]["obs/states"]
    try:
        restored.load_state_dict(malformed)
    except ValueError as exc:
        assert "schema" in str(exc) or "required" in str(exc)
    else:
        raise AssertionError("replay checkpoint missing a required observation was accepted")
    assert len(restored) == 3  # A rejected restore must leave the current replay usable.
    invalid = batch(5, 1)
    invalid["next_obs"].pop("privileged_state")
    try:
        restored.add(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("missing terminal privileged state accepted")


def check_inference(motion_root):
    import copy
    import json

    import torch
    from bfm_zero_policy import BFMZeroPolicy, command_to_latent
    from rpo_bfm import OBS_DIMS

    torch.set_num_threads(2)
    torch.manual_seed(7)
    manifest = json.loads((motion_root / "manifest.json").read_text())
    cfg = {"bfm_profile":"smoke", "motion_root":str(motion_root), "asset_sha256":manifest["asset_sha256"]}
    model = BFMZeroPolicy(cfg).eval()
    reference = {key:torch.from_numpy(model._reference[key]) for key in ("states", "privileged_state")}
    goal = model.goal_inference({key:value[:1] for key,value in reference.items()})
    tracking = model.tracking_inference({key:value[:8] for key,value in reference.items()})
    reward = model.reward_inference(reference, torch.ones(len(reference["states"]), 1))
    for z in (goal, tracking, reward):
        assert torch.isfinite(z).all()
        torch.testing.assert_close(z.norm(dim=-1), torch.full((len(z),), 16.))
    try:
        model.reward_inference(reference, torch.zeros(len(reference["states"]), 1))
    except ValueError:
        pass
    else:
        raise AssertionError("zero reward produced a valid prompt")
    beta = torch.zeros(2, 256)
    beta[0, 0], beta[1, 1] = 16, 16
    features = torch.tensor([[0., 0., 0., 1., .75, 0.], [.3, 0., 0., 1., .75, 0.]])
    latent = command_to_latent(torch.tensor([[0., 0., 0.], [.3, 0., 0.]]), beta, features)
    assert latent[0, 0] > latent[0, 1] and latent[1, 1] > latent[1, 0]
    try:
        command_to_latent(torch.tensor([[1.01, 0., 0.]]), beta, features)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range command accepted")

    obs = {key:torch.zeros(4, width) for key,width in OBS_DIMS.items()}
    obs.update(episode_id=torch.zeros(4, 1, dtype=torch.long), step_id=torch.zeros(4, 1, dtype=torch.long))
    actions, result = model.predict_action_batch(obs, mode="train")
    assert actions.shape == (4, 1, 23) and result["prev_logprobs"].shape == (4, 1)
    assert result["prev_values"].shape == (4, 1)
    torch.testing.assert_close(result["forward_inputs"]["action"], actions[:, 0])
    assert torch.equal(result["forward_inputs"]["bfm_z_pool_count"], torch.zeros(4, 1, dtype=torch.long))
    saved_z = result["forward_inputs"]["z"].clone()
    rng = copy.deepcopy(model.motion_view.state_dict())
    _, again = model.predict_action_batch(obs, mode="train")
    assert torch.equal(again["forward_inputs"]["z"], saved_z)
    assert rng == model.motion_view.state_dict()  # Repeated/bootstrap read cannot advance expert cursor.
    weights = copy.deepcopy(model.state_dict())
    state = model.rollout_state_dict()
    obs["step_id"] += 1
    next_actions, next_result = model.predict_action_batch(obs, mode="train")
    restored = BFMZeroPolicy(cfg).eval()
    restored.load_state_dict(weights)
    restored.load_rollout_state_dict(state)
    restored_actions, restored_result = restored.predict_action_batch(obs, mode="train")
    torch.testing.assert_close(next_actions, restored_actions)
    torch.testing.assert_close(next_result["forward_inputs"]["z"], restored_result["forward_inputs"]["z"])
    before = {key:value.clone() for key,value in restored.obs_normalizer.state_dict().items()}
    eval_actions, _ = restored.predict_action_batch(obs, mode="eval")
    obs["privileged_state"] += 1000
    other_actions, _ = restored.predict_action_batch(obs, mode="eval")
    torch.testing.assert_close(eval_actions, other_actions)  # Real robot privileged fields never enter deployed actor/prompt.
    assert all(torch.equal(value, restored.obs_normalizer.state_dict()[key]) for key,value in before.items())


def check_worker(motion_root):
    import torch
    from bfm_zero_policy import BFMZeroPolicy, project_z
    from bfm_zero_worker import BFMZeroUpdater
    from rpo_bfm import OBS_DIMS

    torch.set_num_threads(2)
    torch.manual_seed(19)
    manifest = __import__("json").loads((motion_root / "manifest.json").read_text())
    model = BFMZeroPolicy({"bfm_profile":"smoke", "motion_root":str(motion_root),
                           "asset_sha256":manifest["asset_sha256"]})
    updater = BFMZeroUpdater(model, batch_size=8, updates_per_round=1)
    count = 16
    current = {key:torch.randn(count, width) for key,width in OBS_DIMS.items()}
    following = {key:value + .01 for key,value in current.items()}
    transitions = {"obs":current, "next_obs":following, "action":torch.rand(count,23)*2-1,
                   "z":project_z(torch.randn(count,256)), "aux_rewards":torch.randn(count,3)*.01,
                   "terminated":torch.zeros(count,1,dtype=torch.bool),
                   "truncated":torch.zeros(count,1,dtype=torch.bool)}
    before = {name:next(module.parameters()).detach().clone()
              for name,module in model.online_modules().items()}
    metrics = updater.train_round(transitions)
    assert updater.update_count == 1 and len(updater.replay) == count
    assert int(model.z_pool_count) == 8, "learner mixed-z never reaches the rollout pool"
    latent = model._sample_rollout_z()
    assert bool(torch.isclose(model.z_pool[:8], latent).all(-1).any())
    required = {"disc_expert_logit", "disc_online_logit", "fb_offdiag", "orth_loss",
                "actor_Q_fb", "actor_Q_aux", "actor_action_saturation", "B_effective_rank",
                "replay_sample_age", "replay_draws_per_new_transition", "updates_this_round"}
    assert required <= metrics.keys(), required - metrics.keys()
    assert all(__import__("math").isfinite(v) for v in metrics.values())
    assert {"disc_loss", "fb_loss", "critic_loss", "aux_critic_loss", "actor_loss"} <= metrics.keys()
    for name,module in model.online_modules().items():
        assert not torch.equal(before[name], next(module.parameters()).detach()), name
    state = updater.state_dict()
    restored = BFMZeroUpdater(model, batch_size=8, updates_per_round=1)
    restored.load_state_dict(state)
    assert restored.update_count == 1 and len(restored.replay) == count
    multi = BFMZeroUpdater(model, batch_size=8, updates_per_round=8, optimizer=updater.optimizer)
    multi.load_state_dict(state)
    metrics = multi.train_round(transitions)
    assert multi.update_count == 9 and metrics["updates_this_round"] == 8
    assert metrics["replay_draws_per_new_transition"] == 4
    assert int(model.z_pool_count) == 72
    from bfm_zero_worker import verify_optimizer_steps
    assert len(verify_optimizer_steps(multi.optimizer, 9)) == 6
    parameter = multi.optimizer.param_groups[0]["params"][0]
    multi.optimizer.state[parameter]["step"] -= 1
    try:
        verify_optimizer_steps(multi.optimizer, 9)
    except ValueError as exc:
        assert "optimizer" in str(exc)
    else:
        raise AssertionError("lagging optimizer shard accepted")


def check_command_frontend(motion_root):
    import json

    import torch
    from bfm_zero_policy import BFMZeroPolicy, command_to_latent
    from rpo_bfm_artifact import _CommandLatentGraph

    manifest = json.loads((motion_root / "manifest.json").read_text())
    model = BFMZeroPolicy({"bfm_profile":"smoke", "motion_root":str(motion_root),
                           "asset_sha256":manifest["asset_sha256"]}).eval()
    reference = model._reference
    observations = {key:torch.from_numpy(reference[key]) for key in ("states", "privileged_state")}
    beta = model.backward_map(model.normalize(observations))
    features = torch.from_numpy(reference["reward_features"])
    graph = _CommandLatentGraph(beta, features).eval()
    low = torch.tensor([-.6, -.5, -1.57])
    high = torch.tensor([1., .5, 1.57])
    commands = torch.stack((low, torch.zeros(3), torch.tensor([.3, 0., 0.]), high))
    torch.testing.assert_close(graph(commands), command_to_latent(commands, beta, features), atol=1e-6, rtol=1e-6)
    assert graph(commands).shape == (4, 256)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["codec", "motion", "networks", "losses", "replay", "inference", "worker", "command"], action="append")
    parser.add_argument("--motion-root", type=__import__("pathlib").Path)
    args = parser.parse_args()
    cases = args.case or ["codec"]
    if "inference" in cases:
        check_inference(args.motion_root)
    if "worker" in cases:
        check_worker(args.motion_root)
    if "command" in cases:
        check_command_frontend(args.motion_root)
    for case, check in {"codec":check_codec, "motion":check_motion, "networks":check_networks,
                        "losses":check_losses, "replay":check_replay}.items():
        if case in cases:
            check()
    print(f"BFM {','.join(cases)} behavior checks passed")
