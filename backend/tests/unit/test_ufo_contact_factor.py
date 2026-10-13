import numpy as np
import pytest


def test_ground_factor_changes_only_time_constant_and_preserves_contact_pair():
    import mujoco
    from eb.runners.rpo_ufo_contact_factor import configure_ground
    spec = mujoco.MjSpec.from_string('<mujoco><worldbody><geom name="terrain" type="plane" size="1 1 .01"/></worldbody></mujoco>')
    geom = spec.geom('terrain')
    before = spec.compile()
    configure_ground(geom, 'C1')
    after = spec.compile()
    np.testing.assert_allclose(after.geom_solref, [[.04, 1.]])
    for key in ('geom_solimp', 'geom_friction', 'geom_priority', 'geom_solmix', 'geom_contype', 'geom_conaffinity'):
        np.testing.assert_array_equal(getattr(before, key), getattr(after, key))
    with pytest.raises(ValueError):
        configure_ground(geom, 'B0')  # Fail if the presumed frozen defaults differ.
    with pytest.raises(ValueError):
        configure_ground(geom, 'unbounded')


def test_depth_summary_counts_overflow_and_conservative_quantile():
    from eb.runners.rpo_ufo_contact_factor import depth_summary
    hist = np.zeros(2001, dtype=int)
    hist[10] = 99; hist[2000] = 1
    result = depth_summary(hist, .023)
    assert result['count'] == 100
    assert result['p99_upper_m'] == pytest.approx(.00011)
    assert result['max_m'] == .023
    assert result['overflow_count'] == 1
    with pytest.raises(ValueError):
        depth_summary(hist, float('nan'))
    with pytest.raises(ValueError):
        depth_summary(hist, .001)
    with pytest.raises(ValueError):
        depth_summary(np.zeros(2001,dtype=int), .001)
    fractional = hist.astype(float); fractional[10] = .5
    with pytest.raises(ValueError):
        depth_summary(fractional, .023)


@pytest.mark.parametrize('candidate,expected', [('B0',.0105),('C1',.0205)])
def test_compiled_contact_mixes_the_single_factor(candidate, expected):
    import mujoco
    from eb.runners.rpo_ufo_contact_factor import configure_ground
    spec = mujoco.MjSpec.from_string('<mujoco><worldbody><geom name="terrain" type="plane" size="1 1 .01"/><body pos="0 0 .049"><freejoint/><geom type="box" size=".05 .05 .05" solref=".001 2"/></body></worldbody></mujoco>')
    configure_ground(spec.geom('terrain'), candidate)
    model = spec.compile(); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    np.testing.assert_allclose(data.contact[0].solref, [expected,1.5])


def test_receipt_rejects_self_consistent_but_false_first_step_count(tmp_path, monkeypatch):
    import json
    from eb.runners import rpo_ufo_contact_factor as factor, rpo_ufo_dynamics_diagnostics as base
    monkeypatch.setattr(base, 'validate_receipt', lambda *args, **kwargs: None)
    phases = [{'rank':r,'mode':m,'original_episode_peak_ratio':1.,'initial_survivors_48':2048}
        for r in range(3) for m in ('anchor','policy')]
    members = []
    for phase in phases:
        suffix = f"{phase['mode']}-{phase['rank']}"
        hist = np.zeros(2001,dtype=int); hist[10] = 20
        np.savez_compressed(tmp_path/f'contact-population-{suffix}.npz',peak_ratio=np.ones(2048),
            strict_length=np.full(2048,48),depth_histogram=hist,original_alive=np.ones(2048,dtype=bool))
        stats = {'candidate':'C1','completed_substeps':384,'ground_time_constant_s':.04,
            'effective_solref_min':[.0205,1.5],'effective_solref_max':[.0205,1.5],
            'depth':factor.depth_summary(hist,.0001),'strict_length_mean':48.,
            'peak_ratio_quantiles':[1.,1.,1.,1.],'first_step_terminated_count':0,
            'first_step_terminated_fraction':0.}
        (tmp_path/f'contact-population-{suffix}.json').write_text(json.dumps(stats))
        (tmp_path/f'population-{suffix}.json').write_text(json.dumps([[1.,2048]]*384))
        members += [{'file':f'contact-population-{suffix}.{extension}'} for extension in ('json','npz')]
    receipt = {'contact_factor':'C1','phases':phases,'files':members}
    factor.validate_receipt(receipt,{},tmp_path)
    path = tmp_path/'contact-population-anchor-0.json'
    stats = json.loads(path.read_text()); stats.update(first_step_terminated_count=1,
        first_step_terminated_fraction=1/2048)
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError,match='summary'):
        factor.validate_receipt(receipt,{},tmp_path)


def test_measurement_preserves_refresh_default_used_by_reset(tmp_path, monkeypatch):
    import inspect
    import rpo_ufo_mjlab as physics
    from mjlab.terrains.terrain_entity import TerrainEntity
    from eb.runners.rpo_ufo_contact_factor import install_measurement
    monkeypatch.setattr(physics,'RPOUFOMJLabPhysics',physics.RPOUFOMJLabPhysics)
    monkeypatch.setattr(TerrainEntity,'_import_ground_plane',TerrainEntity._import_ground_plane)
    default = inspect.signature(physics.RPOUFOMJLabPhysics._refresh).parameters['dt'].default
    install_measurement('B0',0,'anchor',tmp_path)
    assert inspect.signature(physics.RPOUFOMJLabPhysics._refresh).parameters['dt'].default == default


