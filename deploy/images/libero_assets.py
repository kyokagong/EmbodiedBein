"""Build-time pinned LIBERO assets with bounded request rate and resumable cache."""
import hashlib
import json
from pathlib import Path
import shutil
import time

import requests
from huggingface_hub import configure_http_backend, snapshot_download

revision = "3ba78404b48e8c70fa4ac9782d5aac8e5b46d55f"
root = Path("/workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/libero/libero/assets")


def session():
    result = requests.Session()
    # Each redirect also invokes response hooks. A single download thread stays
    # below the mirror's burst limit, including metadata and redirect requests.
    result.hooks["response"].append(lambda response, *args, **kwargs: time.sleep(0.6))
    return result


configure_http_backend(backend_factory=session)
snapshot = Path(snapshot_download(repo_id="RLinf/LIBERO-assets", repo_type="dataset",
    revision=revision, max_workers=1))
shutil.copytree(snapshot, root, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache"))
files = []
for path in sorted(root.rglob("*")):
    if path.is_file():
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append({"path":str(path.relative_to(root)),"bytes":path.stat().st_size,"sha256":digest})
assert len(files) > 100, "Incomplete LIBERO asset bundle"
tokenizer = Path("/opt/eb-openpi-cache/big_vision/paligemma_tokenizer.model")
tokenizer_hash = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
assert tokenizer_hash == "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
Path("/opt/eb-runtime-assets.json").write_text(json.dumps({"libero_assets_repo":"RLinf/LIBERO-assets",
    "libero_assets_revision":revision,"files":files,"tokenizer_sha256":tokenizer_hash},indent=2))
print(f"Installed {len(files)} pinned LIBERO asset files")
