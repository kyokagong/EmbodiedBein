"""Diagnostic evidence must survive reset and preserve the old reward."""
import importlib.util
import pytest
import numpy as np


def module():
    name = 'eb.runners.rpo_ufo_learning_diagnostics'
    assert importlib.util.find_spec(name), 'bounded learning diagnostics are missing'
    return __import__(name, fromlist=['*'])


def test_event_keeps_pre_reset_state_and_integrated_force():
    audit = module().CompletedFrames()
    q = np.zeros((2, 23), np.float32)
    force = np.full((2, 24, 3), 12., np.float32)
    for i in range(8):
        q[:, 0] = i
        audit.append(i, {'q': q, 'integrated_force': force})
    q[:] = 0; force[:] = 99
    saved = audit.finish()
    assert saved['q'][3, 0, 0] == 3
    assert saved['integrated_force'][7, 0, 0, 0] == 12
    with pytest.raises(ValueError):
        audit.append(7, {'q': q, 'integrated_force': force})


def test_substep_position_cost_detects_transient_endpoint_miss():
    frames = np.zeros((8, 2, 23), np.float32)
    frames[2, 0, 0] = 1.2
    result = module().position_audit(frames, np.tile([-1., 1.], (23, 1)))
    assert result['endpoint'][0] == 0
    assert result['mean'][0] == pytest.approx(.2 / 8)
    assert result['peak'][0] == pytest.approx(.2)
    assert result['transient_missed'][0]


def test_weighted_reward_audit_does_not_change_raw_columns():
    raw = np.arange(18, dtype=np.float32).reshape(2, 9)
    before = raw.copy()
    weights = np.array([0., -.1, -10., 0., -1., -.4, -4., -2., -25.])
    result = module().reward_audit(raw, weights)
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_allclose(result['weighted_sum'], raw @ weights)
    assert result['zero_weight_columns'] == [0, 3]
    with pytest.raises(ValueError):
        module().reward_audit(raw * np.nan, weights)


def test_event_windows_are_bounded_and_do_not_cross_reset():
    audit = module().EventWindows(capacity=1, radius=2)
    for step in range(7):
        episode = np.array([1 if step <= 3 else 2])
        frame = {'q': np.full((8, 1, 23), step, np.float32)}
        audit.append(step, episode, frame, {'position': [0] if step == 3 else []})
    event = audit.events[0]
    assert event['category'] == 'position'
    assert [row['control_step'] for row in event['frames']] == [1, 2, 3]
    assert event['post_reset_truncated']
    assert len(audit.events) == 1


def test_four_card_probe_accepts_ebd_without_cuda_visible_devices(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    m=module()
    assert hasattr(m,'visible_devices'), 'ebd four-card mapping must work without CUDA_VISIBLE_DEVICES'
    assert m.visible_devices() == ['0','1','2','3']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-a,GPU-b,GPU-c,GPU-d')
    assert m.visible_devices()[1] == 'GPU-b'


def test_frozen_ema_cost_does_not_advance_statistics_in_eval():
    torch=pytest.importorskip('torch')
    from humanoidverse.agents.nn_models import EMA
    ema=EMA(scale=True,translate=True);ema(torch.tensor([[1.],[3.]]));ema.eval()
    before={k:v.clone() for k,v in ema.state_dict().items()}
    m=module()
    assert hasattr(m,'frozen_aux_reward'), 'EMA eval still updates; a frozen diagnostic transform is required'
    result=m.frozen_aux_reward(ema,torch.tensor([[1.],[3.]]))
    torch.testing.assert_close(result,torch.tensor([[-1.],[1.]]),atol=1e-4,rtol=1e-4)
    assert all(torch.equal(before[k],v) for k,v in ema.state_dict().items())


def test_inference_model_actor_gradients_are_enabled_only_for_readonly_audit():
    torch=pytest.importorskip('torch')
    from eb.runners.bfm_ufo_policy import UFOBFMPolicy
    from eb.runners.bfm_ufo_agent import OBS_DIMS
    torch.set_num_threads(1)
    policy=UFOBFMPolicy({'hidden_dim':64,'hidden_layers':2,'batch_size':16})
    before={k:v.clone() for k,v in policy.model.state_dict().items()}
    assert all(not p.requires_grad for p in policy.model.parameters())
    m=module()
    assert hasattr(m,'actor_objective_audit'), 'inference model has gradients disabled by construction'
    obs={k:torch.zeros(4,v) for k,v in OBS_DIMS.items()}
    rng=torch.get_rng_state().clone()
    result=m.actor_objective_audit(policy.native,obs,torch.ones(4,256))
    assert torch.equal(rng,torch.get_rng_state())
    assert all(result['actor_gradient_norms'][k]>0 for k in ('FB','D','aux'))
    assert all(torch.equal(before[k],v) for k,v in policy.model.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in policy.model.parameters())
