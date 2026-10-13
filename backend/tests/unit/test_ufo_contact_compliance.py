import numpy as np
import pytest


def test_piecewise_penalty_continuity_and_extreme_scale():
    torch=pytest.importorskip('torch')
    from eb.runners.rpo_ufo_compliance import velocity_penalty
    r=torch.tensor([.5,.7,1.,2.,7.,1e3])
    actual=velocity_penalty(r)
    old=(r-.7).clamp_min(0).square()
    torch.testing.assert_close(actual[:3],old[:3])
    assert actual[3]>=5*old[3]
    assert torch.isfinite(actual).all()
    x=torch.tensor([1.-1e-5,1.,1.+1e-5],requires_grad=True)
    velocity_penalty(x).sum().backward()
    assert float(x.grad.max()-x.grad.min())<.001


def test_flexion_rejects_straight_knee_without_rewriting_reference():
    from eb.runners.rpo_ufo_compliance import flexion_mask
    q=np.zeros((3,23));q[:,11:13]=.3;q[1,11]=.149;q[2,7]=.101
    before=q.copy()
    assert flexion_mask(q).tolist()==[True,False,False]
    np.testing.assert_array_equal(q,before)
    assert flexion_mask(q,np.array([[True,True],[False,True],[False,True]])).all()
    q[0,0]=np.nan
    with pytest.raises(ValueError):flexion_mask(q)


def test_long_gate_requires_complete_windows_and_whole_segment_physics():
    from eb.runners.rpo_ufo_compliance import long_training_gate
    records=[{'step':i,'strict_shadow_length_sum':360.,'completed_episodes':10,'physics_max_rated_ratio':2.4,
        'first_terminated':0,'first_samples':10,'coverage_miss':0,'ground_outside':0,'context_errors':0}
        for i in range(25,525)]
    assert long_training_gate(records,start=24,target=524)['passed']
    records[0]['physics_max_rated_ratio']=2.5
    assert not long_training_gate(records,start=24,target=524)['passed']
    records[0]['physics_max_rated_ratio']=2.4
    assert not long_training_gate(records[:-1],start=24,target=524)['passed']
    records[-1]['strict_shadow_length_sum']=float('nan')
    assert not long_training_gate(records,start=24,target=524)['passed']


def test_gate_weights_completed_episodes_not_rollout_means():
    from eb.runners.rpo_ufo_compliance import long_training_gate
    rows=[{'step':i,'strict_shadow_length_sum':100.,'completed_episodes':1 if i%2 else 100,
        'physics_max_rated_ratio':2.,'first_terminated':0,'first_samples':1,
        'coverage_miss':0,'ground_outside':0,'context_errors':0} for i in range(25,525)]
    result=long_training_gate(rows,start=24,target=524)
    assert not result['passed']
    assert result['strict_shadow_last_windows']==pytest.approx([100/50.5]*2)


def test_new_recipe_rejects_small_segment_bypass_and_eval():
    from eb.domain.training_recipe import UFO_COMPLIANCE_RECIPE,validate_update_target,recipe_eval_image_digest
    validate_update_target(524,start=24,recipe_id=UFO_COMPLIANCE_RECIPE)
    validate_update_target(3524,start=524,recipe_id=UFO_COMPLIANCE_RECIPE)
    for start,target in ((524,1024),(500,1000),(1024,1524),(3524,4024),(24,525)):
        with pytest.raises(ValueError):validate_update_target(target,start=start,recipe_id=UFO_COMPLIANCE_RECIPE)
    with pytest.raises(ValueError):recipe_eval_image_digest(UFO_COMPLIANCE_RECIPE)


def test_long_gate_recalculates_hash_bound_records(tmp_path):
    import hashlib,json
    from eb.runners.rpo_ufo_compliance import long_training_gate,require_long_training_gate
    records=[{'step':i,'strict_shadow_length_sum':360.,'completed_episodes':10,
        'physics_max_rated_ratio':2.,'first_terminated':0,'first_samples':1,
        'coverage_miss':0,'ground_outside':0,'context_errors':0} for i in range(25,525)]
    def write(value):
        for name,body in (('training-gate.json',value),('training-gate-records.json',records)):
            (tmp_path/name).write_text(json.dumps(body))
        members=[{'path':name,'sha256':hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()}
                 for name in ('training-gate.json','training-gate-records.json')]
        (tmp_path/'resume-manifest.json').write_text(json.dumps({'step':524,'files':members}))
    value=long_training_gate(records,start=24,target=524);write(value)
    assert require_long_training_gate(tmp_path,step=524)['passed']
    records[0]['physics_max_rated_ratio']=3.
    write(value)
    with pytest.raises(ValueError,match='differs from actual'):require_long_training_gate(tmp_path,step=524)


def test_compliant_penalty_enters_completed_substep_accumulator():
    torch=pytest.importorskip('torch')
    from eb.runners.rpo_ufo_isaac import PhysicsInterval
    limits={'joint_pos':torch.tensor([[-1.,1.]]*23),'joint_vel':torch.ones(23),'applied_torque':torch.full((23,),27.)}
    value=PhysicsInterval(1,limits,device='cpu',compliance=True)
    q=torch.zeros(1,23);value.begin()
    for i in range(8):
        dq=q.clone()
        if i==3:dq[0,0]=2.
        value.sample(i,q,dq,q,q,torch.zeros(1,2),torch.zeros(1,1,3))
    assert value.velocity_penalty().item()==pytest.approx(9.69/8)
