"""Behavior checks for the derived tracking/contact recipe, without GPU physics."""
import copy
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def interval():
    from eb.runners import rpo_ufo_isaac
    assert hasattr(rpo_ufo_isaac, "PhysicsInterval"), "completed-substep accumulator is missing"
    limits = {"joint_pos": torch.tensor([[-1., 1.]] * 23),
              "joint_vel": torch.ones(23), "applied_torque": torch.full((23,), 27.)}
    return rpo_ufo_isaac.PhysicsInterval(2, limits, device="cpu", velocity_multiplier=1.3)


def test_substeps_count_completed_frames_once():
    value = interval()
    value.begin()
    q = torch.zeros(2, 23)
    for i in range(8):
        dq = q.clone()
        dq[0, 0] = 1.2 if i == 3 else .6
        value.sample(i, q, dq, q, q, torch.ones(2, 2), torch.zeros(2, 1, 3))
    # Re-reading the eighth completed physics frame is idempotent.
    value.sample(7, q, torch.full_like(q, .6), q, q, torch.ones(2, 2), torch.zeros(2, 1, 3))
    assert value.samples == 8
    assert value.strict_flags.tolist() == [[False, True, False], [False, False, False]]
    assert not value.training_flags.any()
    torch.testing.assert_close(value.velocity_penalty(), torch.tensor([.03125, 0.]))
    torch.testing.assert_close(value.force_impulse, torch.full((2, 2), .02))
    assert value.peak_index[0, 0] == 3
    value.reset(torch.tensor([0]))
    assert value.first_violation.tolist() == [0, 0]


def test_strict_shadow_stops_at_first_violation():
    value = interval()
    q = torch.zeros(2, 23)
    value.begin()
    for i in range(8):
        value.sample(i, q, torch.full_like(q, 1.2), q, q, torch.zeros(2, 2), torch.zeros(2, 1, 3))
    value.finish(torch.tensor([4, 8]), torch.zeros(2, dtype=torch.bool))
    assert value.first_violation.tolist() == [4, 8]
    value.begin()
    for i in range(8):
        value.sample(i, q, q, q, q, torch.zeros(2, 2), torch.zeros(2, 1, 3))
    value.finish(torch.tensor([5, 9]), torch.zeros(2, dtype=torch.bool))
    assert value.first_violation.tolist() == [4, 8]
    value.reset(torch.tensor([0]))
    assert value.first_violation.tolist() == [0, 8]


def test_substep_penalty_preserves_constant_velocity_scale():
    value = interval()
    q = torch.zeros(2, 23)
    value.begin()
    for i in range(8):
        value.sample(i, q, torch.full_like(q, 1.), q, q, torch.zeros(2, 2), torch.zeros(2, 1, 3))
    torch.testing.assert_close(value.velocity_penalty(), torch.full((2,), 2.07))


def test_contact_point_speed_includes_angular_velocity():
    from eb.runners import rpo_ufo_contact
    assert hasattr(rpo_ufo_contact, "contact_point_normal_velocity"), "collision point velocity is missing"
    fn = rpo_ufo_contact.contact_point_normal_velocity
    result = fn(np.array([0., 0., 0.]), np.array([0., 2., 0.]),
                np.array([.1, 0., 0.]), np.array([0., 0., 1.]))
    assert result == pytest.approx(-.2)


def test_ankle_filter_step_response_and_slew():
    from eb.runners import rpo_bfm_codec
    assert hasattr(rpo_bfm_codec, "filter_ankle_action"), "ankle target filter is missing"
    previous = torch.zeros(2, 23)
    requested = torch.ones_like(previous)
    filtered = rpo_bfm_codec.filter_ankle_action(requested, previous, cutoff_hz=15.)
    from eb.rpo_bfm import RPO_LAB_JOINT_ORDER
    ankle = [i for i, name in enumerate(RPO_LAB_JOINT_ORDER) if "ankle" in name]
    others = [i for i in range(23) if i not in ankle]
    torch.testing.assert_close(filtered[:, ankle], torch.full((2, 4), .8481642))
    assert (filtered[:, others] == 1).all()


def clips():
    values = []
    for source in ("official_rpo", "g1"):
        state = np.zeros((300, 52), np.float32)
        state[:, 0] = np.arange(300) / 300
        values.append({"id": source, "source": source, "source_group": source, "split": "train",
                       "arrays": {"states": state, "privileged_state": np.zeros((300, 358), np.float32),
                                  "valid": np.ones(300, bool)}})
    return values


def test_tracking_reset_is_environment_local():
    from eb.runners.bfm_ufo_policy import UFOBFMPolicy
    from eb.runners.bfm_ufo_replay import UFOExpertSlicer
    from eb.runners.bfm_ufo_agent import OBS_DIMS
    torch.set_num_threads(1)
    policy = UFOBFMPolicy({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16, "seed": 43,
                           "context_profile": "environment_tracking_v10", "rollout_expert_trajectories": True})
    policy.expert = UFOExpertSlicer(clips(), manifest_sha256="a" * 64)
    policy.online_transitions = 10240
    obs = {("states" if k == "state" else k): torch.zeros(4, v) for k, v in OBS_DIMS.items()}
    obs.update(episode_id=torch.zeros(4, 1, dtype=torch.long), step_id=torch.zeros(4, 1, dtype=torch.long),
               ufo_reset_reference=torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0]]))
    policy.predict_action_batch(env_obs=obs)
    assert hasattr(policy, "rollout_context"), "environment-local tracking state is missing"
    obs["step_id"] += 1
    policy.predict_action_batch(env_obs=obs)
    before = copy.deepcopy(policy.rollout_context.state_dict())
    index=int(policy.rollout_context.tracking_ids[0])
    obs["episode_id"][index] += 1
    obs["step_id"][index] = 0
    obs["ufo_reset_reference"][index, 2] = 2
    policy.predict_action_batch(env_obs=obs)
    after = policy.rollout_context.state_dict()
    torch.testing.assert_close(before["tracking_z"][1:], after["tracking_z"][1:], atol=0, rtol=0)
    others=[i for i in range(4) if i!=index]
    assert torch.equal(before["episode_ids"][others],after["episode_ids"][others])


def test_swing_foot_is_not_filtered_as_support():
    from eb.runners import rpo_ufo_contact
    assert hasattr(rpo_ufo_contact, "support_velocity_mask"), "support-point velocity filtering is missing"
    height = np.array([[.002, .2], [.002, .2], [.002, .002]])
    speed = np.array([[.1, -1.], [.2, 0.], [.1, .16]])
    assert rpo_ufo_contact.support_velocity_mask(height, speed).tolist() == [True, False, False]


def test_replay_arrays_restore_without_materializing_state_dict(tmp_path):
    from eb.runners.bfm_ufo_replay import UFOReplayBridge
    from eb.runners.bfm_ufo_agent import VELOCITY_AUX_NAMES
    from test_bfm_ufo_soft_velocity import velocity_batch
    from eb.runners.bfm_ufo_worker import rollout_transitions
    replay = UFOReplayBridge(capacity=16, aux_names=VELOCITY_AUX_NAMES)
    replay.append(rollout_transitions(velocity_batch(), aux_names=VELOCITY_AUX_NAMES))
    assert hasattr(replay, "save_arrays"), "streamed replay checkpoint is missing"
    state = replay.save_arrays(tmp_path)
    expected = replay.sample_numpy(7)
    restored = UFOReplayBridge(capacity=16, aux_names=VELOCITY_AUX_NAMES)
    restored.load_arrays(tmp_path, state)
    actual = restored.sample_numpy(7)
    np.testing.assert_array_equal(actual["action"], expected["action"])
    np.testing.assert_array_equal(actual["z"], expected["z"])
    assert restored.received == 8
    member = tmp_path / state["arrays"][0]["path"]
    member.write_bytes(member.read_bytes()[:-1])
    with pytest.raises(ValueError):
        restored.load_arrays(tmp_path, state)


def test_streamed_checkpoint_restores_next_native_update(tmp_path):
    from test_bfm_ufo_checkpoint import objects, receipt
    from eb.runners.bfm_ufo_policy import UFOBFMPolicy
    from eb.runners.bfm_ufo_checkpoint import save_ufo_checkpoint, load_ufo_checkpoint
    agent, replay = objects()
    agent.update(replay, 11264)
    metadata=receipt()
    metadata["execution"]["checkpoint_format"]="ufo_agent_state_v2"
    policy=UFOBFMPolicy({"hidden_dim":64,"hidden_layers":2,"batch_size":16})
    path=tmp_path/"global_step_1"
    saved=save_ufo_checkpoint(agent,replay,{"rank_0":policy.rollout_state_dict()},path,receipt=metadata)
    assert saved["format"] == "ufo_agent_state_v2"
    expected=agent.update(replay,12288)
    restored,buffers=objects()
    load_ufo_checkpoint(path,agent=restored,replay=buffers,expected={},device="cpu")
    actual=restored.update(buffers,12288)
    for key in expected:
        torch.testing.assert_close(actual[key],expected[key],atol=1e-6,rtol=1e-5)


def test_tracking_recipe_protects_topology_budget_and_independent_eval():
    from eb.domain.training_recipe import RECIPES, validate_update_target, recipe_eval_image_digest
    from eb.domain.rlinf_form import build_overrides
    from eb.rpo_bfm import codec_hash_for_execution, CODEC_V2_HASH
    name="rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_v10"
    assert name in RECIPES, "derived tracking/contact recipe is not registered"
    recipe=RECIPES[name]
    values=build_overrides(name,4,"tracking",[])
    assert "env.train.total_num_envs=6144" in values
    assert "actor.model.batch_size=2048" in values
    assert "actor.model.replay_capacity=4000000" in values
    assert "+actor.model.rollout_expert_trajectories=True" in values
    assert recipe.binding.checkpoint_format=="ufo_agent_state_v2"
    assert codec_hash_for_execution(recipe.binding.model_dump())==CODEC_V2_HASH
    validate_update_target(524,start=24,recipe_id=name)
    with pytest.raises(ValueError):
        validate_update_target(525,start=24,recipe_id=name)
    with pytest.raises(ValueError):
        recipe_eval_image_digest(name)


def test_completed_substep_rejects_nonfinite_contact_force():
    value=interval();q=torch.zeros(2,23)
    force=torch.zeros(2,2);force[0,0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):
        value.sample(0,q,q,q,q,force,torch.zeros(2,1,3))


def test_tracking_phase_sync_and_full_context_restore():
    from eb.runners.bfm_ufo_policy import UFOBFMPolicy, pool_state
    from eb.runners.bfm_ufo_replay import UFOExpertSlicer
    from eb.runners.bfm_ufo_agent import OBS_DIMS
    cfg={"hidden_dim":64,"hidden_layers":2,"batch_size":16,"seed":43,
         "context_profile":"environment_tracking_v10","rollout_expert_trajectories":True}
    policy=UFOBFMPolicy(cfg);policy.expert=UFOExpertSlicer(clips(),manifest_sha256="a"*64)
    policy.online_transitions=10240
    obs={("states" if k=="state" else k):torch.zeros(4,v) for k,v in OBS_DIMS.items()}
    obs.update(episode_id=torch.zeros(4,1,dtype=torch.long),step_id=torch.zeros(4,1,dtype=torch.long),
               ufo_reset_reference=torch.tensor([[0,0,2],[1,1,2],[0,0,2],[1,1,2]]))
    policy.predict_action_batch(env_obs=obs)
    cache=policy.rollout_context.tracking_z.clone()
    obs["step_id"][:]=249;policy.predict_action_batch(env_obs=obs)
    for env_id in policy.rollout_context.tracking_ids:
        torch.testing.assert_close(policy.z[env_id],cache[policy.rollout_context.cache_index[env_id],249])
    policy.apply_learner_packet({"model":policy.model.state_dict(),"pool":pool_state(policy.native),
                                 "version":5,"online_transitions":10240})
    policy.predict_action_batch(env_obs=obs)
    torch.testing.assert_close(policy.rollout_context.tracking_z,cache,atol=0,rtol=0)
    assert (policy.rollout_context.encoding_versions[policy.rollout_context.mask]==0).all()
    saved=policy.rollout_state_dict()
    restored=UFOBFMPolicy(cfg);restored.model.load_state_dict(policy.model.state_dict())
    restored.expert=UFOExpertSlicer(clips(),manifest_sha256="a"*64)
    restored.load_rollout_state_dict(saved)
    obs["step_id"][:]=250
    first=policy.predict_action_batch(env_obs=obs);second=restored.predict_action_batch(env_obs=obs)
    torch.testing.assert_close(first[0],second[0],atol=0,rtol=0)
    assert (restored.rollout_context.starts[restored.rollout_context.mask]==250).all()
    assert (restored.rollout_context.encoding_versions[restored.rollout_context.mask]==5).all()
    saved.pop('environment_context')
    with pytest.raises(ValueError,match='Unknown UFO rollout state'):
        restored.load_rollout_state_dict(saved)


def test_tracking_expert_rejects_ineligible_window():
    from eb.runners.bfm_ufo_replay import UFOExpertSlicer
    expert=UFOExpertSlicer(clips(),manifest_sha256='a'*64)
    with pytest.raises(ValueError):expert.sample_at(0,100,seq_length=250)


def test_rlinf_transports_tracking_and_completed_substeps():
    from rlinf.data.schema.embodied_types import EnvOutput
    reference=torch.tensor([[0,0,12],[1,1,24]])
    trace=torch.arange(342,dtype=torch.float32).reshape(2,171)
    mode=torch.tensor([[True],[False]])
    obs={'states':torch.zeros(2,52),'ufo_reset_reference':reference,'ufo_interval_trace':trace,'ufo_tracking_mode':mode}
    payload=EnvOutput(obs=obs,final_obs=obs,dones=torch.tensor([True,False])).to_dict()
    for field in ('obs','final_obs'):
        torch.testing.assert_close(payload[field]['ufo_reset_reference'],reference)
        torch.testing.assert_close(payload[field]['ufo_interval_trace'],trace)
        torch.testing.assert_close(payload[field]['ufo_tracking_mode'],mode)


def test_peak_event_preserves_velocity_direction_and_same_frame_feet():
    value=interval();q=torch.zeros(2,23);feet=torch.zeros(2,2,13)
    for index in range(8):
        dq=q.clone();dq[0,4]=-.9 if index==3 else .1
        feet[:,:,7]=index
        value.sample(index,q,dq,q,q,torch.ones(2,2),torch.zeros(2,1,3),foot_state=feet)
    event=value.trace(torch.tensor([1,1]))
    assert event.shape==(2,223)
    assert event[0,171+4]==pytest.approx(-.9)
    assert event[0,194+7]==3
    assert event[0,222]==4


def test_measured_kd_variant_keeps_provisional_recipe_immutable():
    from eb.domain.training_recipe import RECIPES,recipe_image_digest,recipe_resources
    from eb.domain.rlinf_form import build_overrides
    from eb.rpo_bfm import action_control_for_execution,codec_hash_for_execution,CODEC_V2_HASH
    original='rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_v10'
    selected='rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10'
    assert selected in RECIPES,'measured controller requires a derived recipe'
    assert recipe_image_digest(original)=='sha256:c2a876c23450b6d1cde3c4b89c889b860cb5831b406c6479949530b00d0ab8e2'
    assert recipe_resources(selected)['gpus']==4
    binding=RECIPES[selected].binding.model_dump()
    assert codec_hash_for_execution(binding)==CODEC_V2_HASH
    control=action_control_for_execution(binding)
    assert control['ankle_nominal_damping']==2.5
    assert 'ankle_filter_cutoff_hz' not in control
    assert '+actor.model.ankle_control_profile=kd25' in build_overrides(selected,4,'kd',[])


def test_measured_profile_rejects_environment_partition_mismatch():
    from eb.runners.bfm_ufo_policy import UFOBFMPolicy
    from eb.runners.bfm_ufo_replay import UFOExpertSlicer
    from eb.runners.bfm_ufo_agent import OBS_DIMS
    from eb.runners.bfm_ufo_rollout_context import tracking_mask
    policy=UFOBFMPolicy({'hidden_dim':64,'hidden_layers':2,'batch_size':16,'seed':43,
        'context_profile':'environment_tracking_v10','ankle_control_profile':'kd25'})
    policy.expert=UFOExpertSlicer(clips(),manifest_sha256='a'*64)
    obs={("states" if k=="state" else k):torch.zeros(4,v) for k,v in OBS_DIMS.items()}
    obs.update(episode_id=torch.zeros(4,1,dtype=torch.long),step_id=torch.zeros(4,1,dtype=torch.long),
        ufo_reset_reference=torch.tensor([[0,0,0],[1,1,0],[0,0,0],[1,1,0]]),
        ufo_tracking_mode=torch.as_tensor(~tracking_mask(4,43))[:,None])
    with pytest.raises(ValueError,match='Tracking partition differs'):
        policy.predict_action_batch(env_obs=obs)


def test_tracking_period_is_environment_local():
    from eb.runners.bfm_ufo_policy import UFOBFMPolicy
    from eb.runners.bfm_ufo_replay import UFOExpertSlicer
    from eb.runners.bfm_ufo_agent import OBS_DIMS
    policy=UFOBFMPolicy({'hidden_dim':64,'hidden_layers':2,'batch_size':16,'seed':43,
                         'context_profile':'environment_tracking_v10'})
    policy.expert=UFOExpertSlicer(clips(),manifest_sha256='a'*64)
    obs={("states" if k=="state" else k):torch.zeros(4,v) for k,v in OBS_DIMS.items()}
    obs.update(episode_id=torch.zeros(4,1,dtype=torch.long),step_id=torch.zeros(4,1,dtype=torch.long),
               ufo_reset_reference=torch.tensor([[0,0,0],[1,1,0],[0,0,0],[1,1,0]]))
    policy.predict_action_batch(env_obs=obs)
    context=policy.rollout_context;before=copy.deepcopy(context.state_dict())
    selected=int(context.tracking_ids[0]);others=context.tracking_ids[1:]
    obs['step_id'][:]=1;obs['step_id'][selected]=250
    policy.predict_action_batch(env_obs=obs)
    assert context.starts[selected]==250 and (context.starts[others]==0).all()
    torch.testing.assert_close(context.tracking_z[1:],before['tracking_z'][1:],atol=0,rtol=0)
    torch.testing.assert_close(context.references[others],before['references'][others])


@pytest.mark.parametrize('profile,expected',[(None,1051),('environment_tracking_v10',43)])
def test_image_rollout_seed_matches_environment_only_for_tracking(profile,expected):
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    from omegaconf import OmegaConf,open_dict
    source=Path('/workspace/rlinf/rlinf/workers/rollout/hf/huggingface_worker.py').read_text()
    module=ast.parse(source)
    cls=next(node for node in module.body if isinstance(node,ast.ClassDef) and node.name=='MultiStepRolloutWorker')
    init=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='init_worker')
    # Execute the real image's configuration prelude, stopping before model/GPU allocation.
    prelude=[]
    for node in init.body:
        if isinstance(node,ast.AnnAssign):break
        prelude.append(node)
    prelude.append(ast.Return(ast.Name('rollout_model_config',ast.Load())))
    fn=ast.FunctionDef(name='seed_config',args=init.args,body=prelude,decorator_list=[])
    namespace={'copy':copy,'open_dict':open_dict}
    exec(compile(ast.fix_missing_locations(ast.Module([fn],[])),'<image rollout seed>','exec'),namespace)
    cfg=OmegaConf.create({'actor':{'seed':42},'rollout':{'model':{'precision':'float32','model_path':''}}})
    model=OmegaConf.create({'policy_setup':'rpo-v2-bfm-ufo','context_profile':profile})
    result=namespace['seed_config'](SimpleNamespace(model_cfg=model,cfg=cfg,_rank=1))
    assert result.seed==expected


def test_physics_startup_records_actual_mass_and_partition():
    from eb.runners import rpo_ufo_contact
    assert hasattr(rpo_ufo_contact,'physics_startup_receipt'),'actual DR mass receipt is missing'
    value=rpo_ufo_contact.physics_startup_receipt(np.ones((2,24))*2,np.array([True,False]),seed=42)
    assert value['actual_total_mass_kg']==[48.,48.]
    assert value['body_weight_N']==pytest.approx([470.88,470.88])
    with pytest.raises(ValueError):
        rpo_ufo_contact.physics_startup_receipt(np.full((2,24),np.nan),np.array([True,False]),seed=42)


