"""Check a real published SFT checkpoint against the PPO architecture without GPU allocation."""
import torch
from hydra import compose, initialize_config_dir
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.openpi_action_model import OpenPi0Config, OpenPi0ForRLActionPrediction

with initialize_config_dir(version_base="1.1", config_dir="/workspace/rlinf/examples/embodiment/config"):
    cfg = compose(config_name="libero_spatial_ppo_openpi_pi05")
config = OpenPi0Config(**get_openpi_config("pi05_libero").model.__dict__)
config.__dict__.update(cfg.actor.model.openpi)
with torch.device("meta"):
    model = OpenPi0ForRLActionPrediction(config)
expected = model.state_dict()
actual = torch.load("/model/model_state_dict/full_weights.pt", map_location="cpu", mmap=True, weights_only=True)
missing = sorted(set(expected) - set(actual))
unexpected = sorted(set(actual) - set(expected))
mismatched = [key for key in expected.keys() & actual.keys() if expected[key].shape != actual[key].shape]
assert missing and all(key.startswith("value_head.") for key in missing), missing
assert not unexpected, unexpected
assert not mismatched, mismatched
print("PASS: actual SFT checkpoint matches PPO policy; only new critic is absent", len(actual), missing)
