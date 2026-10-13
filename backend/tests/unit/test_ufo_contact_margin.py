from types import SimpleNamespace as NS
import json

import numpy as np
import pytest


def test_A1_only_changes_ground_margin_and_keeps_true_geometry(monkeypatch):
    import mujoco
    from mjlab.terrains.terrain_entity import TerrainEntity
    from eb.runners.rpo_ufo_contact_margin import install
    original = TerrainEntity._import_ground_plane
    monkeypatch.setattr(TerrainEntity, '_import_ground_plane', original)
    env = NS(_spec=mujoco.MjSpec())
    original(env, 'ground')
    before = env._spec.geom('ground')
    frozen = {k: np.asarray(getattr(before, k)).copy() for k in ('size','pos','quat','friction','solref','solimp')}
    env = NS(_spec=mujoco.MjSpec())
    install(); TerrainEntity._import_ground_plane(env, 'ground')
    after = env._spec.geom('ground')
    assert after.margin == .003 and after.gap == 0
    assert all(np.array_equal(getattr(after, k), v) for k, v in frozen.items())
    body = env._spec.worldbody.add_body(pos=[0,0,.009])
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.01,0,0])
    model = env._spec.compile(); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 1 and abs(data.contact[0].dist + .001) < 1e-12
    assert data.contact[0].includemargin == .003


def test_A1_validator_checks_actual_margin_metadata_and_contacts(monkeypatch, tmp_path):
    from eb.runners import rpo_ufo_contact_margin as margin, rpo_ufo_ground_stiffness as ground
    calls = []
    monkeypatch.setattr(ground, 'validate_receipt', lambda *args, **kwargs: calls.append(kwargs))
    receipt = {'contact_margin_factor':'A1', 'phases':[{'mode':'anchor','rank':0}]}
    (tmp_path/'actions-anchor-0.json').write_text(json.dumps({'actual_contact_options':{'geom_margin':[.003,0.], 'geom_gap':[0.,0.]}}))
    (tmp_path/'events-anchor-0.json').write_text(json.dumps([{'file':'contact.npz'}]))
    np.savez(tmp_path/'contact.npz', contact_geom=np.array([[0,1]]), contact_includemargin=np.array([.003]))
    margin.validate_receipt(receipt, {}, tmp_path)
    assert calls == [{'contact_margin_m':.003}]
    np.savez(tmp_path/'contact.npz', contact_geom=np.array([[0,1]]), contact_includemargin=np.array([.001]))
    with pytest.raises(ValueError, match='margin'):
        margin.validate_receipt(receipt, {}, tmp_path)
