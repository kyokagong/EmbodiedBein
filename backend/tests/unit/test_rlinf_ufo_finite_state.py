import copy
import io
import re

import pytest

from eb.runners import rlinf_ufo_runtime as runtime


@pytest.fixture
def make_agent():
    torch = pytest.importorskip("torch")
    native = pytest.importorskip("ufo.agents.common.execution")

    class ToyAgent(native.AgentExecutionMixin):
        def __init__(self):
            self._model = torch.nn.Module()
            for name in ("_actor", "_forward_map", "_backward_map", "_critic", "_target_critic"):
                setattr(self._model, name, torch.nn.Linear(2, 2))
            self._model._target_critic.requires_grad_(False)
            self._model._obs_normalizer = torch.nn.Module()
            self._model._obs_normalizer.register_parameter("scale", torch.nn.Parameter(torch.ones(2)))
            self._model._obs_normalizer.register_buffer("running_mean", torch.zeros(2))
            self.optimizers = {
                f"{name}_optimizer": torch.optim.Adam(getattr(self._model, f"_{name}").parameters())
                for name in ("actor", "critic")
            }
            for name, optimizer in self.optimizers.items():
                module = getattr(self._model, "_" + name.removesuffix("_optimizer"))
                sum(parameter.square().sum() for parameter in module.parameters()).backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    return ToyAgent


def check(agent):
    return runtime.verify_full_native_state_finite(
        agent, rank=2, local_time=18432, global_time=73728, optimizer_steps=112
    )


def reload_on_cpu(agent, make_agent):
    import torch

    stream = io.BytesIO()
    torch.save({"model": agent._model.state_dict(), "optimizers": agent.optimizer_dict}, stream)
    stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=True)
    loaded = make_agent()
    loaded._model.load_state_dict(saved["model"])
    for name, optimizer in loaded.optimizers.items():
        optimizer.load_state_dict(saved["optimizers"][name])
    return loaded


def test_full_state_check_accepts_finite_state_without_mutation(make_agent):
    import torch

    agent = make_agent()
    model_before = copy.deepcopy(agent._model.state_dict())
    optimizer_before = copy.deepcopy(agent.optimizer_dict)
    assert check(agent) is None
    assert all(torch.equal(value, model_before[name]) for name, value in agent._model.state_dict().items())
    for name, optimizer in agent.optimizer_dict.items():
        for index, state in optimizer["state"].items():
            assert all(torch.equal(value, optimizer_before[name]["state"][index][key])
                       for key, value in state.items())
        assert optimizer["param_groups"] == optimizer_before[name]["param_groups"]


@pytest.mark.parametrize("phase", ("post_train", "cpu_readback"))
@pytest.mark.parametrize("component, label", (
    ("critic", "model.parameters._critic.weight"),
    ("target", "model.parameters._target_critic.weight"),
    ("normalizer_parameter", "model.parameters._obs_normalizer.scale"),
    ("normalizer_buffer", "model.buffers._obs_normalizer.running_mean"),
))
def test_full_state_check_rejects_hidden_model_nonfinite(make_agent, phase, component, label):
    import torch

    agent = make_agent()
    values = {
        "critic": agent._model._critic.weight,
        "target": agent._model._target_critic.weight,
        "normalizer_parameter": agent._model._obs_normalizer.scale,
        "normalizer_buffer": agent._model._obs_normalizer.running_mean,
    }
    with torch.no_grad():
        values[component].reshape(-1)[0] = float("nan")
    if phase == "cpu_readback":
        agent = reload_on_cpu(agent, make_agent)
    # The old actor/FB fingerprint does not cover these values.
    runtime.parameter_snapshot(agent._model)
    with pytest.raises(FloatingPointError, match=re.escape(label)) as failure:
        check(agent)
    assert "rank=2" in str(failure.value)
    assert "optimizer_steps=112" in str(failure.value)


@pytest.mark.parametrize("phase", ("post_train", "cpu_readback"))
@pytest.mark.parametrize("component, label", (
    ("moment", "agent.optimizer_dict.critic_optimizer.state.0.exp_avg"),
    ("hyperparameter", "agent.optimizer_dict.critic_optimizer.param_groups[0].betas[0]"),
))
def test_full_state_check_rejects_nonfinite_nested_optimizer_state(make_agent, phase, component, label):
    agent = make_agent()
    optimizer = agent.optimizers["critic_optimizer"]
    if component == "moment":
        state = optimizer.state[next(iter(optimizer.state))]
        state["exp_avg"].reshape(-1)[0] = float("inf")
    else:
        optimizer.param_groups[0]["betas"] = (float("nan"), 0.999)
    if phase == "cpu_readback":
        agent = reload_on_cpu(agent, make_agent)
    runtime.parameter_snapshot(agent._model)
    with pytest.raises(FloatingPointError, match=re.escape(label)):
        check(agent)
