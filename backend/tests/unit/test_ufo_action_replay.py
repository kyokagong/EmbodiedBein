import numpy as np
import pytest


def packet():
    n=2048
    return {'requested':np.zeros((48,n,23),np.float32),
        'applied':np.zeros((48,n,23),np.float32),'target':np.zeros((48,n,23),np.float32),
        'done':np.zeros((48,n),bool),'original_alive':np.ones((48,n),bool),
        'states':np.zeros((48,n,52),np.float32),'history_actor':np.zeros((48,n,300),np.float32),
        'reference':np.zeros((48,n,3),np.int64),'episode':np.zeros((48,n),np.int64),
        'step':np.zeros((48,n),np.int64),'initial_qpos':np.zeros((n,30),np.float32),
        'initial_qvel':np.zeros((n,29),np.float32),'interval':np.zeros((48,n,223),np.float32)}


def test_action_program_requires_bounded_complete_finite_history():
    from eb.runners.rpo_ufo_action_replay import validate_program
    good=packet(); validate_program(good,policy=False)
    for key,value in [('requested',np.zeros((47,2048,23))),('initial_qvel',np.zeros((2048,28)))]:
        bad={**good,key:value}
        with pytest.raises(ValueError): validate_program(bad,policy=False)
    bad={**good,'requested':good['requested'].copy()}; bad['requested'][0,0,0]=np.nan
    with pytest.raises(ValueError): validate_program(bad,policy=False)
    bad={**good,'original_alive':good['original_alive'].copy()}
    bad['original_alive'][1,0]=False
    with pytest.raises(ValueError): validate_program(bad,policy=False)
    with pytest.raises(ValueError): validate_program(good,policy=True)


def test_replay_binds_full_initial_state_without_zeroing_momentum():
    from eb.runners.rpo_ufo_action_replay import check_initial
    p=packet(); p['initial_qvel'][5,3]=2.5
    check_initial(p,p['initial_qpos'],p['initial_qvel'])
    with pytest.raises(ValueError,match='initial'): check_initial(p,p['initial_qpos'],np.zeros((2048,29)))


def test_population_history_is_same_run_and_selected_trace_agrees():
    from eb.runners.rpo_ufo_action_replay import validate_population
    a={'case_ids':np.array([1]),'original_alive':np.ones((384,1),bool)}
    for key,width in [('qpos_after',30),('qvel_after',29),('motor_torque',23),('contact_torque',23),('limit_torque',23)]:
        a[key]=np.zeros((384,1,width)); a['population_'+key]=np.zeros((384,2048,width))
    a['population_original_alive']=np.ones((384,2048),bool)
    a['population_contact_depth_after']=np.zeros((384,2048))
    a['contact_depth_after']=np.zeros((384,1))
    a['population_foot_force']=np.zeros((384,2048,2,3)); a['foot_force']=np.zeros((384,1,2,3))
    validate_population(a,384)
    a['population_qvel_after'][10,1,8]=.1
    with pytest.raises(ValueError,match='population'): validate_population(a,384)


def test_context_recording_keeps_actual_legacy_duplicates_and_unknown_reference():
    import torch
    from types import SimpleNamespace as NS
    from eb.runners.rpo_ufo_action_replay import context_snapshot
    p=NS(context_profile=None,version=600,native=NS(env_idx_with_expert_rollout=torch.tensor([1,1,3])))
    context,ids=context_snapshot(p,torch.tensor([0,2,4,6]))
    np.testing.assert_array_equal(ids,[1,1,3])
    np.testing.assert_array_equal(context[:,0],[0,2,0,1])
    np.testing.assert_array_equal(context[:,1],[0,2,4,6])
    np.testing.assert_array_equal(context[:,2:5],-np.ones((4,3)))
