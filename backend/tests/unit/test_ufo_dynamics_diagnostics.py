import numpy as np
import pytest


def test_mass_and_force_contributions_keep_contact_coupling_and_braking_sign():
    from eb.runners.rpo_ufo_dynamics_diagnostics import dense_mass_matrix, force_balance
    matrix = dense_mass_matrix(np.array([[2., 0.], [.5, 1.]]), np.array([-1, 0]))
    np.testing.assert_allclose(matrix, [[2., .5], [.5, 1.]])
    sparse = dense_mass_matrix(np.array([[2., 1., .5]]), np.array([-1, 0]))
    np.testing.assert_allclose(sparse, matrix)
    terms = {'actuator': np.array([0., -27.]), 'contact': np.array([40., 50.])}
    acceleration = np.linalg.solve(matrix, sum(terms.values()))
    result = force_balance(matrix, terms, np.zeros(2), acceleration * .0025, .0025, acceleration)
    assert result['delta_v_terms']['actuator'][1] < 0
    assert result['delta_v_terms']['contact'][1] > 0
    np.testing.assert_allclose(result['delta_v_predicted'], acceleration * .0025)
    assert result['force_residual_max_abs'] < 1e-10
    with pytest.raises(ValueError):
        dense_mass_matrix(np.array([[0., 0.], [0., 1.]]), np.array([-1, 0]))


def test_constraint_contact_and_limit_are_separate_and_sum_to_generalized_force():
    from eb.runners.rpo_ufo_dynamics_diagnostics import constraint_forces
    rows = constraint_forces(np.array([[1., 2.], [0., 1.], [1., 0.]]),
        np.array([3., 4., 5.]), np.array([6, 3, 0]))
    np.testing.assert_allclose(rows['contact'], [3., 6.])
    np.testing.assert_allclose(rows['limit'], [0., 4.])
    np.testing.assert_allclose(rows['other_constraint'], [5., 0.])
    np.testing.assert_allclose(sum(rows.values()), [8., 10.])


def test_privileged_audit_separates_expected_lift_and_localizes_velocity_mismatch():
    from eb.runners.rpo_ufo_dynamics_diagnostics import privileged_audit
    reference = np.zeros((2, 358))
    online = reference.copy(); online[:, 0] += [.01, .02]
    result = privileged_audit(reference, online, np.array([.01, .02]))
    assert result['unexpected_mismatch_rows'] == 0
    assert result['groups']['height']['max_abs_after_lift'] < 1e-10
    online[0, 214] = .1
    result = privileged_audit(reference, online, np.array([.01, .02]))
    assert result['unexpected_mismatch_rows'] == 1
    assert result['groups']['linear_velocity']['max_abs_after_lift'] == .1
    assert result['groups']['orientation']['max_abs_after_lift'] == 0


def test_dynamics_receipt_rejects_path_escape_and_nonzero_updates(tmp_path):
    from eb.runners.rpo_ufo_dynamics_diagnostics import validate_receipt, SCHEMA
    receipt = {'schema_version': SCHEMA, 'qualification': 'diagnostics_only',
        'baseline_step': 600, 'optimizer_updates': 1, 'dataset_manifest_sha256': 'frozen',
        'phases': [], 'files': []}
    with pytest.raises(ValueError, match='identity'):
        validate_receipt(receipt, {'manifest_sha256': 'frozen'}, tmp_path)
    receipt['optimizer_updates'] = 0
    receipt['phases'] = [{'rank': r, 'mode': m, 'seed': 42+r, 'steps':48, 'num_envs':2048,
        'initial_reference_sha256': str(r)*64, 'optimizer_updates':0, 'event_count':0}
        for r in range(3) for m in ('anchor','policy')]
    receipt['files'] = [{'file': '../escape.json', 'sha256': 'x', 'bytes': 1}]
    with pytest.raises(ValueError, match='evidence'):
        validate_receipt(receipt, {'manifest_sha256': 'frozen'}, tmp_path)


def test_dynamics_receipt_cannot_pass_with_only_unrelated_file(tmp_path):
    from eb.runners.rpo_ufo_dynamics_diagnostics import validate_receipt, digest, SCHEMA
    path = tmp_path / 'unrelated.json'; path.write_text('{}')
    receipt = {'schema_version': SCHEMA, 'qualification': 'diagnostics_only',
        'baseline_step':600, 'optimizer_updates':0, 'dataset_manifest_sha256':'frozen',
        'phases':[{'rank':r, 'mode':m, 'seed':42+r, 'steps':48, 'num_envs':2048,
            'initial_reference_sha256':str(r)*64, 'optimizer_updates':0, 'event_count':0}
            for r in range(3) for m in ('anchor','policy')],
        'files':[{'file':path.name, 'bytes':path.stat().st_size, 'sha256':digest(path)}]}
    with pytest.raises(ValueError, match='evidence'):
        validate_receipt(receipt, {'manifest_sha256':'frozen'}, tmp_path)


def test_braking_label_must_refer_to_fastest_joint():
    from eb.runners.rpo_ufo_dynamics_diagnostics import peak_joint_is_braking
    mask = peak_joint_is_braking(np.array([[2., -1.], [-2., 1.]]),
        np.array([[20., 1.], [20., 1.]]), np.array([8., 8.]))
    assert mask.tolist() == [False, True]


def test_audit_comparison_allows_roundoff_but_rejects_changed_measurement():
    from eb.runners.rpo_ufo_dynamics_diagnostics import same_audit_groups
    original = {'linear_velocity': {'columns':[214,286], 'mismatch_rows':1655,
        'max_abs_after_lift':.137, 'mean_abs_after_lift':.0002863948466256261}}
    altered = {'linear_velocity': dict(original['linear_velocity'])}
    altered['linear_velocity']['mean_abs_after_lift'] = .00028639487572945654
    assert same_audit_groups(original, altered)
    altered['linear_velocity']['mean_abs_after_lift'] = .01
    assert not same_audit_groups(original, altered)


def test_cpu_kinematics_uses_world_root_angular_velocity_and_link_origin():
    import mujoco
    from eb.runners.rpo_ufo_dynamics_diagnostics import kinematic_body_state
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="root"><freejoint name="floating"/><geom type="sphere" size=".1"/>
        <body name="child" pos="1 0 0"><joint name="hinge" axis="0 1 0"/>
          <geom type="sphere" size=".1"/></body></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    body = kinematic_body_state(model, data, np.array([0.,0.,1.]),
        np.array([np.sqrt(.5),0.,0.,np.sqrt(.5)]), np.array([1.,0.,0.]),
        np.array([0.,0.,2.]), np.array([0.]), np.array([3.]), [1], [1,2])
    np.testing.assert_allclose(body['body_lin_vel'][0], [1.,0.,0.], atol=1e-9)
    np.testing.assert_allclose(body['body_lin_vel'][1], [-1.,0.,0.], atol=1e-9)
    np.testing.assert_allclose(body['body_ang_vel'][1], [-3.,0.,2.], atol=1e-9)
