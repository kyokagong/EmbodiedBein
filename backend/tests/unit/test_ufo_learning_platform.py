import json
from pathlib import Path
import pytest


@pytest.mark.parametrize('operation',['probe_rpo_ufo_learning_diagnostics_v1','probe_rpo_ufo_learning_diagnostics_v2','probe_rpo_ufo_learning_diagnostics_v4','probe_rpo_ufo_dynamics_diagnostics_v1'])
def test_learning_diagnostics_reserve_three_physics_samplers_without_updates(operation):
    from eb.capacity import estimate_job
    from eb.domain.job import JobKind, JobSpec
    result=estimate_job(JobSpec(kind=JobKind.ASSET_VALIDATE,
        embodiment_env_id='ee_rpo_v2_bfm_23dof_v2',
        params={'operation':operation},resources={'gpus':4}))
    assert result.basis['inputs'].get('num_envs') == 6144
    assert result.basis['inputs'].get('learner_count') == 0


def test_dynamics_evidence_finalizer_reserves_cpu_without_gpu_or_learner():
    from eb.capacity import estimate_job
    from eb.domain.job import JobKind, JobSpec
    result=estimate_job(JobSpec(kind=JobKind.ASSET_VALIDATE,
        embodiment_env_id='ee_rpo_v2_bfm_23dof_v2',
        params={'operation':'probe_rpo_ufo_dynamics_finalize_v2'}, resources={'gpus':0}))
    assert result.per_gpu_peak_gb == 0
    assert result.basis['inputs'].get('learner_count') == 0
    assert result.basis['inputs'].get('physics_integration_steps') == 0


def test_learning_receipt_rejects_modified_arrays(tmp_path):
    from eb.runners.rpo_ufo_learning_diagnostics import validate_receipt, digest, SCHEMA
    p=tmp_path/'arrays.npz'; p.write_bytes(b'original')
    receipt={'schema_version':SCHEMA,'qualification':'diagnostics_only',
        'dataset_manifest_sha256':'frozen',
        'baseline_step':600,'optimizer_updates':0,
        'phases':[{'rank':r,'mode':m,'seed':42+r,'steps':48,'num_envs':2048,
                   'initial_reference_sha256':str(r)} for r in range(3) for m in ('anchor','policy')],
        'files':[{'file':p.name,'bytes':p.stat().st_size,'sha256':digest(p)}]}
    validate_receipt(receipt,{'manifest_sha256':'frozen'},tmp_path)
    receipt['optimizer_updates']=1
    with pytest.raises(ValueError,match='identity'):
        validate_receipt(receipt,{'manifest_sha256':'frozen'},tmp_path)
    receipt['optimizer_updates']=0
    p.write_bytes(b'changed!')
    with pytest.raises(ValueError,match='evidence'):
        validate_receipt(receipt,{'manifest_sha256':'frozen'},tmp_path)


def test_value_only_receipt_requires_unchanged_frozen_state(tmp_path):
    from eb.runners import rpo_ufo_learning_diagnostics as m
    assert hasattr(m,'validate_value_receipt'), 'value-only diagnostic collection is missing'
    value={'state_hash_before':'a','state_hash_after':'a','optimizer_updates':0,
           'normalizers_updated':False,'training_rng_consumed':False}
    p=tmp_path/'reward-value-audit.json';p.write_text(json.dumps(value))
    receipt={'schema_version':'eb/rpo-learning-value/v1','qualification':'diagnostics_only',
        'baseline_step':600,'optimizer_updates':0,'dataset_manifest_sha256':'frozen',
        'files':[{'file':p.name,'bytes':p.stat().st_size,'sha256':m.digest(p)}]}
    m.validate_value_receipt(receipt,{'manifest_sha256':'frozen'},tmp_path)
    value['state_hash_after']='b';p.write_text(json.dumps(value))
    receipt['files']=[{'file':p.name,'bytes':p.stat().st_size,'sha256':m.digest(p)}]
    with pytest.raises(ValueError,match='state'):
        m.validate_value_receipt(receipt,{'manifest_sha256':'frozen'},tmp_path)
