import pytest


def test_legacy_checkpoint_weight_is_scoped_and_rejects_symlink(tmp_path):
    from eb.checkpoints import checkpoint_weights
    run = tmp_path / 'run_source'
    weight = run / 'output/exp/checkpoints/global_step_5800/actor/model_state_dict/full_weights.pt'
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b'weights')
    assert checkpoint_weights(run, 5800)['bytes'] == 7
    assert checkpoint_weights(run, 5800)['path'] == str(weight.resolve())
    with pytest.raises(ValueError):
        checkpoint_weights(run, 5900)
    weight.unlink()
    outside = tmp_path / 'outside.pt'; outside.write_bytes(b'weights')
    weight.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink|符号'):
        checkpoint_weights(run, 5800)


@pytest.mark.asyncio
async def test_failed_training_still_lists_weights_for_evaluation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from eb.api.routes_train import list_checkpoints
    from eb.store.models import TrainJob
    weight = tmp_path / 'runs/run_source/output/exp/checkpoints/global_step_5800/actor/model_state_dict/full_weights.pt'
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b'weights')
    row = SimpleNamespace(id='trn_source', job_id='job_source', status='failed', run_dir='')
    session = SimpleNamespace(get=AsyncMock(return_value=row),
        scalars=AsyncMock(return_value=[SimpleNamespace(id='run_source')]))
    monkeypatch.setattr('eb.api.routes_train._sync_run', AsyncMock())
    monkeypatch.setattr('eb.api.routes_train.get_settings', lambda: SimpleNamespace(data_root=tmp_path))
    result = await list_checkpoints('trn_source', session, SimpleNamespace(id='user'))
    assert result['items'][0]['step'] == 5800
    assert result['items'][0]['weights_only'] is True
    assert result['items'][0]['run_id'] == 'run_source'
