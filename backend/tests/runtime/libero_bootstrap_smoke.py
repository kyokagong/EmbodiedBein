"""Exercise upstream EnvWorker timeout rewards with its missing import repaired."""
from types import SimpleNamespace
from inspect import unwrap
import torch
from omegaconf import OmegaConf
from rlinf.data.schema.embodied_types import EnvOutput
from rlinf.workers.env.env_worker import EnvWorker
from libero_env import install_env_compat

worker = SimpleNamespace(cfg=OmegaConf.create({"algorithm": {"gamma": 0.9}}))
output = EnvOutput(obs={}, rewards=torch.zeros(2, 4),
    truncations=torch.tensor([[False, True, True, False], [True, False, True, False]]),
    valid_step_mask=torch.tensor([[True, True, True, True], [False, True, True, True]]))
values = torch.tensor([[2.0], [3.0]])
compute = unwrap(EnvWorker.compute_bootstrap_rewards)
try:
    compute(worker, output, values, None)
except NameError as error:
    assert "add_time_limit_bootstrap_rewards" in str(error)
else:
    raise AssertionError("Expected original missing helper import to reproduce")
install_env_compat()
actual = compute(worker, output, values, None)
torch.testing.assert_close(actual, torch.tensor([[0, 1.8, 0, 0], [0, 0, 2.7, 0]]))
assert not output.rewards.any()
print("PASS: actual EnvWorker adds bootstrap only at the first valid timeout, preserving original rewards")
