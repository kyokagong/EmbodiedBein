"""Validate the real eval config without launching workers or loading a model."""
from hydra import compose, initialize_config_dir
from omegaconf.errors import ConfigAttributeError
from eb.domain.rlinf_form import build_overrides
from rlinf.config import validate_cfg, validate_embodied_cfg
from rlinf_entry import prepare_eval_model

for suite in ("spatial", "object", "goal", "10"):
    with initialize_config_dir(version_base="1.1", config_dir="/workspace/rlinf/examples/embodiment/config"):
        cfg = compose(config_name=f"libero_{suite}_ppo_openpi_pi05",
            overrides=build_overrides(f"libero_{suite}_pi05_eval", 4, "eval-config-probe", []))
    try:
        validate_embodied_cfg(cfg)
    except ConfigAttributeError as error:
        assert "model_type" in str(error)
    else:
        raise AssertionError("Expected the original incomplete rollout config to fail")
    rollout_path = cfg.rollout.model.model_path
    prepare_eval_model(cfg)
    assert cfg.rollout.model.model_path == rollout_path
    assert cfg.rollout.model.openpi.config_name == "pi05_libero"
    validated = validate_cfg(cfg)
    assert validated.runner.only_eval
    assert validated.rollout.model.model_type == "openpi"
    print("PASS actual validate_cfg", suite)

from rlinf.utils.runner_utils import check_progress
assert "runner.save_interval=40" in build_overrides("libero_spatial_pi05_ppo", 4, "save-cadence", [])
assert not check_progress(1, 2, -1, 40, 1.0)[1]
assert check_progress(2, 2, -1, 40, 1.0)[1]
assert check_progress(40, 1000, -1, 40, 1.0)[1]
print("PASS native periodic and final checkpoint decisions")

import ray
ray.shutdown()
