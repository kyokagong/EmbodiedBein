"""Training exploration must survive the per-environment tracking partition."""
import numpy as np
import pytest
torch=pytest.importorskip('torch')


@pytest.mark.parametrize('transitions',[0,10240])
def test_tracking_partition_does_not_replace_train_action_mode(transitions):
    from eb.runners.bfm_ufo_policy import UFOBFMPolicy
    from eb.runners.bfm_ufo_replay import UFOExpertSlicer
    from eb.runners.bfm_ufo_agent import OBS_DIMS
    from eb.runners.bfm_ufo_rollout_context import tracking_mask
    torch.set_num_threads(1)
    policy=UFOBFMPolicy({'hidden_dim':64,'hidden_layers':2,'batch_size':16,
        'seed':42,'context_profile':'environment_tracking_v10','ankle_control_profile':'kd25'})
    policy.expert=UFOExpertSlicer([
        {'id':s,'source':s,'source_group':s,'split':'train','arrays':{
            'states':np.zeros((270,52),np.float32),'privileged_state':np.zeros((270,358),np.float32),
            'valid':np.ones(270,bool)}} for s in ('official_rpo','g1')],manifest_sha256='a'*64)
    policy.online_transitions=transitions
    obs={('states' if k=='state' else k):torch.zeros(4,v) for k,v in OBS_DIMS.items()}
    obs.update(episode_id=torch.zeros(4,1,dtype=torch.long),step_id=torch.zeros(4,1,dtype=torch.long),
        ufo_tracking_mode=torch.as_tensor(tracking_mask(4,42))[:,None],
        ufo_reset_reference=torch.tensor([[0,0,0],[0,0,0],[1,1,0],[1,1,0]]))
    snapshot=policy.rollout_state_dict()
    train=policy.predict_action_batch(env_obs=obs,mode='train')[1]['forward_inputs']['action']
    policy.load_rollout_state_dict(snapshot,reset_episode=True); policy.cached=None
    evaluation=policy.predict_action_batch(env_obs=obs,mode='eval')[1]['forward_inputs']['action']
    assert not torch.equal(train,evaluation), 'train unexpectedly uses deterministic evaluation action'
    if transitions==0:
        assert float(train.abs().max()) > .5, 'warmup random action was bypassed'
    # Same state/version with another mode must not reuse the previous cached action.
    policy.cached=None
    eval_again=policy.predict_action_batch(env_obs=obs,mode='eval')[1]['forward_inputs']['action']
    train_again=policy.predict_action_batch(env_obs=obs,mode='train')[1]['forward_inputs']['action']
    assert not torch.equal(eval_again,train_again)
