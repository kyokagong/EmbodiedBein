"""Incremental hooks on the frozen single-card image; never write the external repository."""
import hashlib
import json
from pathlib import Path


def patch(root):
    from patch_rpo_ufo import replace
    replace(root, "rlinf/workers/rollout/hf/huggingface_worker.py",
        "            rollout_model_config.model_path = self.cfg.rollout.model.model_path\n",
        "            rollout_model_config.model_path = self.cfg.rollout.model.model_path\n"
        "            if rollout_model_config.get('policy_setup') == 'rpo-v2-bfm-ufo':\n"
        "                rollout_model_config.seed = int(self.cfg.actor.seed) + self._rank\n"
        "                rollout_model_config.sampler_rank = self._rank\n")
    replace(root, "rlinf/workers/rollout/hf/huggingface_worker.py",
        "    def ufo_rollout_state(self):\n",
        "    def ufo_sync_receipt(self):\n"
        "        return self.hf_model.sync_receipt()\n\n"
        "    def ufo_rollout_state(self):\n")
    replace(root, "rlinf/runners/embodied_runner.py",
        "        actor_handle.wait()\n        rollout_handle.wait()\n",
        "        actor_handle.wait()\n        rollout_handle.wait()\n"
        "        if self.cfg.actor.model.get('topology_id') == 'one_learner_three_samplers':\n"
        "            receipts = self.rollout.ufo_sync_receipt().wait()\n"
        "            self.actor.verify_sampler_sync(receipts).wait()\n")
    receipt_path = Path("/opt/ufo-source-receipt.json")
    receipt = json.loads(receipt_path.read_text())
    receipt["topology_patch_hash"] = "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    patch(Path("/workspace/rlinf"))
