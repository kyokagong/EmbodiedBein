import xml.etree.ElementTree as ET
import pytest
from eb.rpo_bfm import (ENV_ID, ENV_V2_ID, CODEC_HASH, CODEC_V2_HASH, REACHABLE_RECIPE,
    RPO_LAB_JOINT_ORDER, JOINT_POSITION_MIN, JOINT_POSITION_MAX, spec,
    codec_hash_for_execution, action_control_for_execution, auxiliary_profile_for_execution)


def test_reachable_fingerprint_preserves_observation_definition():
    old, new = spec(), spec(ENV_V2_ID)
    assert old['interface_id'] == ENV_ID and old['action_scale'] == .25
    for key in ('default_joint_pos', 'state_layout', 'privileged_layout', 'history_frame', 'joint_order'):
        assert old[key] == new[key]
    assert CODEC_HASH != CODEC_V2_HASH
    execution = dict(recipe_id=REACHABLE_RECIPE)
    assert codec_hash_for_execution(execution) == CODEC_V2_HASH
    control = action_control_for_execution(execution)
    assert control['action_midpoint_rad'] != control['observation_default_joint_pos_rad']
    assert auxiliary_profile_for_execution(execution) == 'soft_velocity_v7'
    assert codec_hash_for_execution({}) == CODEC_HASH
    assert action_control_for_execution({}) is None
    with pytest.raises(ValueError): spec('unknown')


def test_frozen_bounds_match_measured_readonly_asset():
    import os
    path = os.environ.get('EB_TEST_RPO_URDF')
    if not path: pytest.skip('explicit readonly asset mount required')
    joints = {node.attrib['name']: node.find('limit').attrib for node in ET.parse(path).findall('joint')}
    assert tuple(float(joints[name]['lower']) for name in RPO_LAB_JOINT_ORDER) == JOINT_POSITION_MIN
    assert tuple(float(joints[name]['upper']) for name in RPO_LAB_JOINT_ORDER) == JOINT_POSITION_MAX
