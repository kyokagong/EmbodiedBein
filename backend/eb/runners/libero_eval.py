"""Evaluation-only dispatch: no actor worker or optimizer is created."""
from rlinf.config import validate_cfg
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from libero_env import LiberoEnvWorker as EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def evaluate(cfg):
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    rollout = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=placement.get_strategy("rollout"))
    env = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=placement.get_strategy("env"))
    runner = EmbodiedEvalRunner(cfg, rollout, env)
    runner.init_workers()
    metrics = runner.evaluate()
    runner.metric_logger.log(step=0, data={f"eval/{key}": value for key, value in metrics.items()})
    runner.metric_logger.finish()
    return {key: value.item() if hasattr(value, "item") else value for key, value in metrics.items()}
