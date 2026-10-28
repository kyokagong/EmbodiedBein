"""Small CPU checks for the task-only numerical state verifier."""

import pytest
import torch
from safetensors.torch import save_file

from eb.runners.rlinf_pbfm_runtime import OPTIMIZERS, verify_native_state


class Agent:
    def __init__(self):
        self._model = torch.nn.Module()
        self._model.register_buffer("target_state", torch.ones(1))
        self._model.register_buffer("normalizer_state", torch.ones(1))
        self.optimizers = {}
        for name in sorted(OPTIMIZERS):
            parameter = torch.nn.Parameter(torch.ones(1))
            self._model.register_parameter(name, parameter)
            optimizer = torch.optim.Adam([parameter])
            parameter.square().sum().backward()
            optimizer.step()
            self.optimizers[name] = optimizer

    @property
    def optimizer_dict(self):
        return {key: value.state_dict() for key, value in self.optimizers.items()}


def save(agent, root):
    (root / "model").mkdir()
    save_file(agent._model.state_dict(), str(root / "model/model.safetensors"))
    torch.save(agent.optimizer_dict, root / "optimizers.pth")


def test_full_optimizer_roundtrip_and_update_counts(tmp_path):
    agent = Agent()
    save(agent, tmp_path)
    report = verify_native_state(agent, tmp_path, expected_updates=1)
    assert report["strict_state_verified"] is True
    assert len(report["optimizer_names"]) == 8
    with pytest.raises(ValueError, match="step"):
        verify_native_state(agent, tmp_path, expected_updates=2)


@pytest.mark.parametrize(
    "failure",
    ["missing_optimizer", "missing_model_key", "missing_parameter_state", "wrong_saved_state", "nonfinite"],
)
def test_invalid_state_cannot_complete(tmp_path, failure):
    agent = Agent()
    save(agent, tmp_path)
    if failure == "missing_optimizer":
        agent.optimizers.pop("actor_optimizer")
    if failure == "missing_model_key":
        state = agent._model.state_dict()
        state.pop("target_state")
        save_file(state, str(tmp_path / "model/model.safetensors"))
    if failure == "missing_parameter_state":
        agent.optimizers["actor_optimizer"].state.clear()
    if failure == "wrong_saved_state":
        state = agent.optimizer_dict
        state["actor_optimizer"]["param_groups"][0]["lr"] = 1.0
        torch.save(state, tmp_path / "optimizers.pth")
    if failure == "nonfinite":
        agent._model.normalizer_state.fill_(float("nan"))
    with pytest.raises((ValueError, RuntimeError)):
        verify_native_state(agent, tmp_path, expected_updates=1)
