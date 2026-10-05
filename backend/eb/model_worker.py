"""Pinned public model acquisition, executed only in an ebd task container."""
import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _download_large_files(repo_id, revision, root, items, chunk_size=16 * 1024**2):
    if not items:
        return
    import requests
    from huggingface_hub import hf_hub_url

    opened, tasks = {}, []
    try:
        for item in items:
            path = root / item.rfilename
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file() and path.stat().st_size == item.size:
                continue
            probe = requests.get(hf_hub_url(repo_id, item.rfilename, revision=revision),
                                 headers={'Range':'bytes=0-0'}, stream=True, timeout=30)
            probe.raise_for_status()
            url = probe.url
            probe.close()
            part = path.with_suffix(path.suffix + '.part')
            fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o644)
            os.ftruncate(fd, item.size)
            opened[item.rfilename] = (fd, part, path, url)
            tasks.extend((item.rfilename, start, min(start + chunk_size, item.size) - 1)
                         for start in range(0, item.size, chunk_size))

        def fetch(task):
            name, start, end = task
            fd, _, _, url = opened[name]
            for attempt in range(20):
                try:
                    response = requests.get(url, headers={'Range':f'bytes={start}-{end}'}, timeout=45)
                    response.raise_for_status()
                    expected = end - start + 1
                    if response.status_code != 206 or len(response.content) != expected:
                        raise IOError('invalid ranged model response')
                    os.pwrite(fd, response.content, start)
                    return
                except Exception:
                    if attempt == 19:
                        raise
                    time.sleep(1)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch, tasks))
        for fd, part, path, _ in opened.values():
            os.fsync(fd)
            os.close(fd)
            os.replace(part, path)
        opened.clear()
    finally:
        for fd, *_ in opened.values():
            os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-dir')
    parser.add_argument('--source-id', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', args.repo_id) or not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        raise ValueError('public repo id and immutable revision required')
    from huggingface_hub import HfApi, snapshot_download
    from eb.model_sources import MODEL_SOURCES
    source = MODEL_SOURCES.get(args.source_id)
    if source is None or source['repo_id'] != args.repo_id:
        raise ValueError('model source is not trusted')
    print(f'Resolving public model {args.repo_id}@{args.revision}', flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(args.repo_id, revision=args.revision, files_metadata=True, token=False)
    patterns = source['allow_patterns']
    selected = [f for f in info.siblings if any(fnmatch.fnmatch(f.rfilename, p) for p in patterns)]
    names = {f.rfilename for f in selected}
    missing = set(source['required_files']) - names
    if missing:
        raise ValueError('model source is missing required files: ' + ', '.join(sorted(missing)))
    if any(f.size is None for f in selected):
        raise ValueError('unknown download size')
    size = sum(f.size for f in selected)
    root = Path(args.model_dir) if args.model_dir else output / 'model'
    if size > 80 * 1024**3 or shutil.disk_usage(root.parent).free < size + 20 * 1024**3:
        raise ValueError('model download disk budget exceeded')
    print(f'Downloading {len(selected)} files ({size} bytes)', flush=True)
    large = [item for item in selected if item.size >= 64 * 1024**2]
    _download_large_files(args.repo_id, args.revision, root, large)
    remaining = [item for item in selected if item not in large]
    if remaining:
        for attempt in range(200):
            try:
                snapshot_download(args.repo_id, revision=args.revision, repo_type='model', token=False,
                                  local_dir=str(root), allow_patterns=[f.rfilename for f in remaining])
                break
            except Exception as exc:
                if not type(exc).__module__.startswith(('requests.', 'urllib3.')):
                    raise
                if attempt == 199:
                    raise
                print(f'Download interrupted ({type(exc).__name__}); resuming in 2s '
                      f'[{attempt + 1}/200]', flush=True)
                time.sleep(2)
    shutil.rmtree(root / '.cache', ignore_errors=True)
    files = []
    print('Download finished; verifying file sizes and SHA256', flush=True)
    for item in selected:
        path = root / item.rfilename
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or path.stat().st_size != item.size:
            raise ValueError('invalid downloaded model member')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        files.append({'path':item.rfilename,'bytes':item.size,'sha256':digest})
    manifest = {'status':'complete','source_id':args.source_id,'repo_id':args.repo_id,'revision':info.sha,
                'framework':source['framework'],'model_path':str(root) if args.model_dir else 'model','files':files,'bytes':size,
                'training_history':f'upstream checkpoint {args.repo_id}; further SFT is continued fine-tuning'}
    temp = output / 'model-download.tmp'
    temp.write_text(json.dumps(manifest))
    temp.replace(output / 'model-download.json')
    print(f'Model ready: {size} bytes, revision {info.sha}', flush=True)


if __name__ == '__main__':
    main()
