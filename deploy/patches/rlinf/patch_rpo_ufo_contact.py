"""Freeze owned contact diagnostics/control sources; native UFO objectives are unchanged."""
import hashlib
import json
from pathlib import Path

registry=Path('/workspace/rlinf/rlinf/envs/sim/isaaclab/__init__.py')
s=registry.read_text();old='    "RPO-UFO-Isaac-RSI-reachable-v7": RPOUFOIsaacEnv,'
new='    "RPO-UFO-Isaac-RSI-contact-v8": RPOUFOIsaacEnv,'
assert s.count(old)==1 and new not in s
registry.write_text(s.replace(old,old+'\n'+new))
path=Path('/opt/ufo-auxiliary-source-files.json');hashes=json.loads(path.read_text())
hashes['rpo_ufo_contact.py']='';hashes['rpo_ufo_contact_probe.py']=''
for name in hashes:
    p=Path('/workspace/tests/rpo_ufo_tracking_diagnostic.py') if name=='rpo_ufo_tracking_diagnostic.py' else Path('/workspace/eb_runner')/name
    hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
path.write_text(json.dumps(hashes,indent=2))
path=Path('/opt/ufo-source-receipt.json');receipt=json.loads(path.read_text())
receipt.update(contact_profile='contact_stable_v8',
    auxiliary_source_sha256='sha256:'+hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    contact_patch_hash='sha256:'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
path.write_text(json.dumps(receipt,indent=2))
