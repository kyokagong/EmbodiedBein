import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock
import pytest
from eb.model_worker import main
from eb.adapters.data_convert import DataConvertAdapter
from eb.domain.job import JobKind, JobSpec
from eb.model_sources import MODEL_SOURCES


def test_model_download_mounts_managed_model_directory(monkeypatch):
    monkeypatch.setenv('https_proxy', 'http://proxy.example:8080')
    spec = JobSpec(kind=JobKind.DATA_CONVERT, embodiment_env_id='ee_libero_franka_delta_v1',
                   params={'operation':'download_model','source_id':'dm05','repo_id':'Dexmal/DM05','revision':'a'*40,
                           'model_id':'mdl_0123456789abcdef'}, resources={'gpus':0})
    rendered = DataConvertAdapter().render(spec, 'run_test')
    assert '--model-dir' in rendered.command
    assert any(src.endswith('/models/mdl_0123456789abcdef') and src == dst and mode == 'rw'
               for src, dst, mode in rendered.mounts)
    assert rendered.env['https_proxy'] == 'http://proxy.example:8080'
    assert rendered.env['HF_ENDPOINT'] == 'https://huggingface.co'
    assert rendered.env['HF_HUB_DISABLE_XET'] == '1'
    assert rendered.command[-2:] == ['--source-id', 'dm05']


def test_exact_robotwin_model_sources_are_pinned_to_expected_artifacts():
    assert set(MODEL_SOURCES) >= {'lingbot-vla-2', 'gr00t-n1.7', 'galaxea-g0', 'dm05', 'xiaomi-robotics-1'}
    assert MODEL_SOURCES['galaxea-g0']['allow_patterns'] == (
        'G0_3B_base.pt', 'G0_3B_base_dataset_statistics.json')
    assert MODEL_SOURCES['xiaomi-robotics-1']['required_files'] == ('model_states.pt',)
    assert MODEL_SOURCES['dm05']['framework'] == 'opendm_dm05'


def test_pinned_model_download_includes_actual_rlinf_normalization_layout(tmp_path, monkeypatch):
    revision = 'a'*40
    files = [SimpleNamespace(rfilename='model.safetensors',size=7),
             SimpleNamespace(rfilename='physical-intelligence/libero/norm_stats.json',size=2)]
    def download(repo_id, **kwargs):
        assert kwargs['revision'] == revision and kwargs['repo_type'] == 'model'
        assert kwargs['token'] is False
        root = tmp_path/'model'
        (root/'physical-intelligence/libero').mkdir(parents=True)
        (root/'model.safetensors').write_bytes(b'weights')
        (root/'physical-intelligence/libero/norm_stats.json').write_text('{}')
    monkeypatch.setitem(sys.modules,'huggingface_hub', SimpleNamespace(
        HfApi=lambda:SimpleNamespace(model_info=lambda *a,**k:SimpleNamespace(sha=revision,siblings=files)),
        snapshot_download=download))
    monkeypatch.setattr('eb.model_worker.shutil.disk_usage', Mock(return_value=SimpleNamespace(free=100*1024**3)))
    monkeypatch.setattr(sys,'argv',['worker','--repo-id','RLinf/RLinf-Pi05-LIBERO-SFT','--revision',revision,'--output',str(tmp_path),'--source-id','pi05-libero'])
    main()
    receipt = json.loads((tmp_path/'model-download.json').read_text())
    assert receipt['status']=='complete' and len(receipt['files'])==2
    monkeypatch.setattr(sys,'argv',['worker','--repo-id','RLinf/RLinf-Pi05-LIBERO-SFT','--revision','main','--output',str(tmp_path),'--source-id','pi05-libero'])
    with pytest.raises(ValueError):
        main()


def test_model_download_resumes_after_transport_failure(tmp_path, monkeypatch):
    class RequestException(Exception):
        pass
    RequestException.__module__ = 'requests.exceptions'
    revision = 'a' * 40
    files = [SimpleNamespace(rfilename='model.safetensors', size=7),
             SimpleNamespace(rfilename='physical-intelligence/robotwin/norm_stats.json', size=2)]
    attempts = 0
    def download(repo_id, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RequestException('interrupted')
        root = Path(kwargs['local_dir'])
        (root/'physical-intelligence/robotwin').mkdir(parents=True, exist_ok=True)
        (root/'model.safetensors').write_bytes(b'weights')
        (root/'physical-intelligence/robotwin/norm_stats.json').write_text('{}')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda:SimpleNamespace(model_info=lambda *a, **k:SimpleNamespace(sha=revision, siblings=files)),
        snapshot_download=download))
    monkeypatch.setattr('eb.model_worker.shutil.disk_usage', Mock(return_value=SimpleNamespace(free=100*1024**3)))
    monkeypatch.setattr('eb.model_worker.time.sleep', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['worker', '--repo-id', 'RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle', '--revision', revision,
                                     '--output', str(tmp_path), '--source-id', 'pi05-robotwin'])
    main()
    assert attempts == 2


def test_large_model_download_uses_verified_ranges(tmp_path, monkeypatch):
    from eb.model_worker import _download_large_files
    source = b'abcdef'
    class Response:
        status_code = 206
        url = 'https://cdn.example/model'
        def __init__(self, content=b'x'):
            self.content = content
        def raise_for_status(self): pass
        def close(self): pass
    def get(url, headers, **kwargs):
        start, end = map(int, headers['Range'].removeprefix('bytes=').split('-'))
        return Response(source[start:end + 1])
    monkeypatch.setitem(sys.modules, 'requests', SimpleNamespace(get=get))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        hf_hub_url=lambda *a, **k:'https://hub.example/model'))
    item = SimpleNamespace(rfilename='model.safetensors', size=len(source))
    _download_large_files('RLinf/test', 'a' * 40, tmp_path, [item], chunk_size=2)
    assert (tmp_path/'model.safetensors').read_bytes() == source


@pytest.mark.asyncio
@pytest.mark.parametrize('receipt', [[], {'status':'complete','revision':'a'*40,'repo_id':'other/model','source_id':'pi05-libero','bytes':1}])
async def test_invalid_download_receipt_becomes_failure(tmp_path, monkeypatch, receipt):
    from eb.api.routes_model import reconcile_model
    monkeypatch.setattr('eb.settings.get_settings', lambda:SimpleNamespace(data_root=tmp_path))
    root = tmp_path/'runs/run_1/outputs'
    root.mkdir(parents=True)
    (root/'model-download.json').write_text(json.dumps(receipt))
    row = SimpleNamespace(kind='downloading', metrics_at_checkpoint={'download':{
        'job_id':'job_1','repo_id':'RLinf/RLinf-Pi05-LIBERO-SFT','source_id':'pi05-libero','revision':'a'*40}})
    session = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(id='job_1',status='succeeded',error=None)),
        scalar=AsyncMock(return_value=SimpleNamespace(id='run_1')),commit=AsyncMock())
    await reconcile_model(row,session)
    assert row.kind == 'download_failed'


@pytest.mark.asyncio
@pytest.mark.parametrize('damage', ['none', 'missing', 'truncated', 'empty_list', 'duplicate', 'escape', 'symlink', 'total', 'no_norm'])
async def test_download_receipt_checks_actual_model_files(tmp_path, monkeypatch, damage):
    from eb.api.routes_model import reconcile_model
    monkeypatch.setattr('eb.settings.get_settings', lambda:SimpleNamespace(data_root=tmp_path))
    root = tmp_path/'runs/run_1/outputs'
    model = root/'model'
    (model/'physical-intelligence/libero').mkdir(parents=True)
    weights = model/'model.safetensors'
    weights.write_bytes(b'weights')
    norm = 'physical-intelligence/libero/norm_stats.json'
    (model/norm).write_text('{}')
    files = [{'path':'model.safetensors','bytes':7,'sha256':'a'*64},
             {'path':norm,'bytes':2,'sha256':'b'*64}]
    receipt = {'status':'complete','revision':'a'*40,'repo_id':'RLinf/RLinf-Pi05-LIBERO-SFT','source_id':'pi05-libero',
               'model_path':'model','bytes':9,'files':files}
    if damage == 'missing': weights.unlink()
    if damage == 'truncated': weights.write_bytes(b'w')
    if damage == 'empty_list': receipt['files'] = []
    if damage == 'duplicate': files.append(dict(files[0]))
    if damage == 'escape': files[0]['path'] = '../model/model.safetensors'
    if damage == 'symlink':
        weights.rename(model/'weights')
        weights.symlink_to('weights')
    if damage == 'total': receipt['bytes'] = 10
    if damage == 'no_norm':
        files.pop()
        receipt['bytes'] = 7
    (root/'model-download.json').write_text(json.dumps(receipt))
    row = SimpleNamespace(kind='downloading', metrics_at_checkpoint={'download':{
        'job_id':'job_1','repo_id':'RLinf/RLinf-Pi05-LIBERO-SFT','source_id':'pi05-libero','revision':'a'*40}})
    session = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(id='job_1',status='succeeded',error=None)),
        scalar=AsyncMock(return_value=SimpleNamespace(id='run_1')),commit=AsyncMock())
    await reconcile_model(row,session)
    assert row.kind == ('base' if damage == 'none' else 'download_failed')
