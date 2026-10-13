import json
import numpy as np
import pytest
from eb.rpo_bfm import BODY_ORDER, RPO_LAB_JOINT_ORDER, CODEC_HASH, CODEC_V2_HASH
from eb.runners.rpo_ufo_motion import convert_robot_state, digest, validate_robot_state_manifest


def test_control_view_preserves_bytes_splits_and_ancestry(tmp_path, monkeypatch):
    asset = tmp_path / 'asset.json'; asset.write_text('{}')
    arrays = {key: np.zeros((12, width), np.float32) for key, width in
              {'root_pos': 3, 'root_quat': 4, 'q': 23, 'dq': 23, 'root_lin_vel_w': 3,
               'root_ang_vel_w': 3, 'states': 52, 'privileged_state': 358}.items()}
    arrays['root_quat'][:, 0] = 1
    arrays['q'][:, 17] = -.6
    arrays['valid'] = np.ones(12, bool)
    clips = []
    for split in ('train', 'validation', 'test'):
        path = tmp_path / (split + '.npz'); np.savez(path, **arrays)
        clips.append(dict(id=split, source_group=split, source='g1', split=split,
                          file=path.name, sha256=digest(path), fps=50, frames=12))
    parent = dict(schema_version='eb/rpo-robot-state/v1', asset_sha256=digest(asset),
        codec_sha256=CODEC_HASH, quaternion='wxyz', velocity_frame='world', velocity_point='root_link_origin',
        joint_order=list(RPO_LAB_JOINT_ORDER), body_order=list(BODY_ORDER),
        source_refs=[dict(sha256='b'*64, license='fixture')], parents=[dict(manifest_sha256='c'*64)], clips=clips)
    source = tmp_path / 'manifest.json'; source.write_text(json.dumps(parent))
    # Sampling-window qualification belongs to its existing tests, this fixture tests immutable derivation.
    monkeypatch.setattr('eb.runners.rpo_ufo_motion.load_robot_state', lambda *a, **kw: None)
    out = tmp_path / 'derived'
    derived = convert_robot_state(source, out, asset_manifest=asset, control_profile='reachable_v2')
    assert derived['clips'] == parent['clips'] and derived['source_refs'] == parent['source_refs']
    assert derived['parents'][0]['manifest_sha256'] == digest(source)
    assert derived['parents'][1:] == parent['parents']
    for clip in clips:
        assert (out / clip['file']).read_bytes() == (tmp_path / clip['file']).read_bytes()
    validate_robot_state_manifest(derived, asset_sha256=digest(asset), expected_codec_sha256=CODEC_V2_HASH)
    with pytest.raises(ValueError, match='codec'):
        validate_robot_state_manifest(derived, asset_sha256=digest(asset))
    derived['clips'][1]['source_group'] = 'train'
    with pytest.raises(ValueError, match='split'):
        validate_robot_state_manifest(derived, asset_sha256=digest(asset), expected_codec_sha256=CODEC_V2_HASH)
