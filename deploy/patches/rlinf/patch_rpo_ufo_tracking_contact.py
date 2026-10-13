"""Owned derived context/physics/checkpoint bridge; pinned UFO losses stay intact."""
import hashlib
import json
import os
from pathlib import Path

registry=Path('/workspace/rlinf/rlinf/envs/sim/isaaclab/__init__.py')
source=registry.read_text()
old='    "RPO-UFO-Isaac-RSI-ground-v9": RPOUFOIsaacEnv,'
new='    "RPO-UFO-Isaac-RSI-tracking-contact-v10": RPOUFOIsaacEnv,'
assert source.count(old)==1 and new not in source
registry.write_text(source.replace(old,old+'\n'+new))
schema=Path('/workspace/rlinf/rlinf/data/schema/embodied_types.py')
source=schema.read_text()
old='"ufo_action_trace", "episode_id", "step_id")'
assert source.count(old)==1
schema.write_text(source.replace(old,'"ufo_action_trace", "ufo_reset_reference", "ufo_interval_trace", "ufo_tracking_mode", "episode_id", "step_id")'))
worker=Path('/workspace/rlinf/rlinf/workers/rollout/hf/huggingface_worker.py')
source=worker.read_text()
old='                rollout_model_config.seed = int(self.cfg.actor.seed) + 1009 * self._rank\n'
assert source.count(old)==1
worker.write_text(source.replace(old,old+
    '            if rollout_model_config.get("context_profile") == "environment_tracking_v10":\n'
    '                rollout_model_config.seed = int(self.cfg.actor.seed) + self._rank\n'))
path=Path('/opt/ufo-auxiliary-source-files.json')
hashes=json.loads(path.read_text())
hashes.update(bfm_ufo_rollout_context='',rpo_ufo_tracking_contact_probe='')
if os.environ.get('EB_UFO_BUILD_DYNAMICS_PROFILE')=='contact_compliance_v11':
    hashes.update({'rpo_ufo_compliance.py':'','rpo_ufo_compliance_probe.py':''})
for key in ('bfm_ufo_rollout_context','rpo_ufo_tracking_contact_probe'):
    hashes[key+'.py']=hashes.pop(key)
for name in hashes:
    file=(Path('/workspace/tests/rpo_ufo_tracking_diagnostic.py') if name=='rpo_ufo_tracking_diagnostic.py'
          else Path('/workspace/eb_runner')/name)
    hashes[name]=hashlib.sha256(file.read_bytes()).hexdigest()
path.write_text(json.dumps(hashes,indent=2))
path=Path('/opt/ufo-source-receipt.json')
receipt=json.loads(path.read_text())
receipt.update(context_profile='environment_tracking_v10',
    auxiliary_source_sha256='sha256:'+hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    tracking_contact_patch_hash='sha256:'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
receipt['action_control_source_sha256']='sha256:'+hashlib.sha256(Path('/workspace/eb_runner/rpo_bfm_codec.py').read_bytes()).hexdigest()
if os.environ.get('EB_UFO_BUILD_ANKLE_CONTROL')=='kd25':
    receipt['ankle_control_profile']='kd25'
if os.environ.get('EB_UFO_BUILD_DYNAMICS_PROFILE')=='contact_compliance_v11':
    receipt['dynamics_profile']='contact_compliance_v11'
path.write_text(json.dumps(receipt,indent=2))
