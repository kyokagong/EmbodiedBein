import json
from types import SimpleNamespace as NS
import numpy as np
import pytest


def test_C2_ground_hook_runs_after_frozen_B0_setup_and_changes_one_field(monkeypatch):
    import mujoco
    from mjlab.terrains.terrain_entity import TerrainEntity
    from eb.runners.rpo_ufo_ground_stiffness import install
    original=TerrainEntity._import_ground_plane
    monkeypatch.setattr(TerrainEntity,'_import_ground_plane',original)
    env=NS(_spec=mujoco.MjSpec());original(env,'before');before=env._spec.geom('before')
    install();TerrainEntity._import_ground_plane(env,'after');after=env._spec.geom('after')
    assert np.array_equal(before.solref,[.02,1.]) and np.array_equal(after.solref,[.02,.1])
    assert np.array_equal(before.solimp,after.solimp) and np.array_equal(before.friction,after.friction)
    assert before.condim==after.condim
    model=env._spec.compile();assert np.array_equal(model.geom_solref[1],[.02,.1])


def test_C2_validator_propagates_ground_ratio_without_relaxing_default(monkeypatch,tmp_path):
    from eb.runners import rpo_ufo_ground_stiffness as ground,rpo_ufo_friction_impedance as friction
    calls=[]
    monkeypatch.setattr(friction,'validate_receipt',lambda *args,**kwargs:calls.append(kwargs))
    receipt={'ground_stiffness_factor':'C2'}
    ground.validate_receipt(receipt,{},tmp_path)
    assert calls==[{'ground_damping_ratio':.1}]
    receipt['ground_stiffness_factor']='unknown'
    with pytest.raises(ValueError,match='ground'):ground.validate_receipt(receipt,{},tmp_path)


def test_original_S2_validator_rejects_C2_header_without_explicit_ratio(monkeypatch,tmp_path):
    from eb.runners import rpo_ufo_friction_impedance as friction,rpo_ufo_solver_cone as cone
    monkeypatch.setattr(cone,'validate_receipt',lambda *args,**kwargs:None)
    receipt={'ground_stiffness_factor':'C2','solver_friction_factor':'S2','phases':[]}
    with pytest.raises(ValueError,match='ground'):friction.validate_receipt(receipt,{},tmp_path)
    friction.validate_receipt(receipt,{},tmp_path,ground_damping_ratio=.1)
