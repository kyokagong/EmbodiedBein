"""Image-local UFO hook installation; no loss implementation or external writes."""
from pathlib import Path
import sys


def replace(root: Path, relative: str, old: str, new: str):
    path = root / relative
    source = path.read_text()
    if source.count(old) != 1:
        raise ValueError(f"UFO patch expected one site: {relative}: {old!r}")
    path.write_text(source.replace(old, new))


def patch(root: Path):
    replace(root, "examples/embodiment/train_embodied_agent.py",
        '    if cfg.actor.model.get("policy_setup") == "rpo-v2-bfm-zero":\n',
        '    if cfg.actor.model.get("policy_setup") == "rpo-v2-bfm-ufo":\n'
        '        from bfm_ufo_worker import UFOActorWorker\n'
        '        actor_worker_cls = UFOActorWorker\n'
        '    elif cfg.actor.model.get("policy_setup") == "rpo-v2-bfm-zero":\n')
    replace(root, "rlinf/models/embodiment/mlp_policy/__init__.py",
        '    iql_config = cfg.get("iql_config", None)\n',
        '    if cfg.get("policy_setup") == "rpo-v2-bfm-ufo":\n'
        '        from bfm_ufo_policy import UFOBFMPolicy\n'
        '        return UFOBFMPolicy(cfg)\n'
        '    iql_config = cfg.get("iql_config", None)\n')
    replace(root, "rlinf/envs/sim/isaaclab/__init__.py", 'from rpo_bfm_extension import RPOBFMEnv\n',
        'from rpo_bfm_extension import RPOBFMEnv\nfrom rpo_ufo_isaac import RPOUFOIsaacEnv\n')
    replace(root, "rlinf/envs/sim/isaaclab/__init__.py", '    "RPO-BFM-Pretrain-v1": RPOBFMEnv,\n',
        '    "RPO-BFM-Pretrain-v1": RPOBFMEnv,\n    "RPO-UFO-Isaac-RSI-v1": RPOUFOIsaacEnv,\n')
    replace(root, "rlinf/data/schema/embodied_types.py", '"bfm_aux_rewards", "episode_id", "step_id")',
        '"bfm_aux_rewards", "ufo_aux_rewards", "episode_id", "step_id")')
    replace(root, "rlinf/config.py",
        '    assert cfg.actor.training_backend in SUPPORTED_TRAINING_BACKENDS, (\n',
        '    native_ufo = cfg.actor.model.get("policy_setup") == "rpo-v2-bfm-ufo"\n'
        '    if native_ufo:\n'
        '        from bfm_ufo_worker import validate_ufo_runtime_config\n'
        '        validate_ufo_runtime_config(cfg)\n'
        '    assert cfg.actor.training_backend in SUPPORTED_TRAINING_BACKENDS or (native_ufo and cfg.actor.training_backend == "ufo_agent"), (\n')
    replace(root, "rlinf/workers/rollout/hf/huggingface_worker.py",
        '            if rollout_model_config.get("policy_setup") == "rpo-v2-bfm-zero":\n',
        '            if rollout_model_config.get("policy_setup") in {"rpo-v2-bfm-zero", "rpo-v2-bfm-ufo"}:\n')
    replace(root, "rlinf/workers/rollout/hf/huggingface_worker.py",
        '        if not self.weight_syncer.receiver_initialized():\n',
        '        if self.cfg.actor.model.get("policy_setup") == "rpo-v2-bfm-ufo":\n'
        '            packet = await recv_func()\n'
        '            self.hf_model.apply_learner_packet(packet)\n'
        '            self.version = packet["version"]\n'
        '            return\n'
        '        if not self.weight_syncer.receiver_initialized():\n')
    replace(root, "rlinf/workers/rollout/hf/huggingface_worker.py",
        '    @Worker.timer("sync_model_from_actor")\n',
        '    def ufo_rollout_state(self):\n'
        '        return self.hf_model.rollout_state_dict()\n\n'
        '    def load_ufo_rollout_state(self, states):\n'
        '        self.hf_model.load_rollout_state_dict(states[f"rank_{self._rank}"], reset_episode=True)\n\n'
        '    @Worker.timer("sync_model_from_actor")\n')
    replace(root, "rlinf/runners/embodied_runner.py", '        self.actor.load_checkpoint(actor_checkpoint_path).wait()\n',
        '        if self.cfg.actor.model.get("policy_setup") == "rpo-v2-bfm-ufo":\n'
        '            self.actor.load_checkpoint(resume_dir).wait()\n'
        '            progress = self.actor.progress().wait()[0]\n'
        '            self.global_step = progress["sampling_rounds"]\n'
        '            states = self.actor.restored_rollout_states().wait()[0]\n'
        '            self.rollout.load_ufo_rollout_state(states).wait()\n'
        '        else:\n'
        '            self.actor.load_checkpoint(actor_checkpoint_path).wait()\n')
    replace(root, "rlinf/runners/embodied_runner.py", '        actor_checkpoint_path = os.path.join(resume_dir, "actor")\n',
        '        actor_checkpoint_path = resume_dir if self.cfg.actor.model.get("policy_setup") == "rpo-v2-bfm-ufo" else os.path.join(resume_dir, "actor")\n')
    replace(root, "rlinf/runners/embodied_runner.py", '    def _save_checkpoint(self):\n',
        '    def _save_checkpoint(self):\n'
        '        if self.cfg.actor.model.get("policy_setup") == "rpo-v2-bfm-ufo":\n'
        '            states = self.rollout.ufo_rollout_state().wait()\n'
        '            self.actor.set_rollout_states(states).wait()\n'
        '            progress = self.actor.progress().wait()[0]\n'
        '            updates = progress["learner_joint_updates"]\n'
        '            path = os.path.join(self.cfg.runner.logger.log_path, self.cfg.runner.logger.experiment_name, f"checkpoints/global_step_{updates}")\n'
        '            self.actor.save_checkpoint(path).wait()\n'
        '            return\n')
    replace(root, "rlinf/runners/embodied_runner.py", '        time_metrics = self.timer.consume_durations()\n',
        '        if self.cfg.actor.model.get("policy_setup") == "rpo-v2-bfm-ufo":\n'
        '            step = int(self._aggregate_numeric_metrics(actor_training_metrics)["ufo/learner_joint_updates"])\n'
        '        time_metrics = self.timer.consume_durations()\n')
    replace(root, "rlinf/runners/embodied_runner.py", '        self._finish_run()\n\n    def run_pipeline(self):\n',
        '            if self.cfg.actor.model.get("policy_setup") == "rpo-v2-bfm-ufo":\n'
        '                progress = self.actor.progress().wait()[0]\n'
        '                if progress["target_reached"] or time.time() - start_time >= 3300:\n'
        '                    break\n'
        '        if self.cfg.actor.model.get("policy_setup") == "rpo-v2-bfm-ufo":\n'
        '            self._save_checkpoint()\n'
        '            if not self.actor.progress().wait()[0]["target_reached"]:\n'
        '                self._finish_run()\n'
        '                raise RuntimeError("UFO sampling/wall budget ended before joint-update target")\n'
        '        self._finish_run()\n\n    def run_pipeline(self):\n')


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
