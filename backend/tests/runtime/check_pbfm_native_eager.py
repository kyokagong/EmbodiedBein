"""CPU boundary check using Warp's real public API and MJLab's real dispatch."""

import os
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402
import warp as wp  # noqa: E402
from mjlab.sim import Simulation  # noqa: E402

from eb.runners import rlinf_pbfm_runtime as runtime  # noqa: E402
from eb.runners import rlinf_pbfm_worker as worker  # noqa: E402


def main():
    wp.init()
    assert all(d.is_cpu for d in wp.get_devices())
    device = NS(is_cuda=True, is_mempool_supported=True, is_mempool_enabled=True,
                default_allocator=object(), mempool_allocator=object())
    device.current_allocator = device.mempool_allocator
    with patch.object(wp._src.context.runtime, "get_device", return_value=device):
        assert runtime.configure_simulation_execution({"identity": {"num_envs_per_rank": 1024}}) is False
        assert device.is_mempool_enabled and device.current_allocator is device.mempool_allocator
        assert runtime.configure_simulation_execution({"identity": {"num_envs_per_rank": 2048}}) is True
        assert not device.is_mempool_enabled and device.current_allocator is device.default_allocator

        # The actual Worker must switch before config/env build, including the
        # probe early-return path whose real environment has only two lanes.
        wp.set_mempool_enabled("cuda:0", True)
        token = object()

        def config(*args):
            assert not device.is_mempool_enabled
            return NS(model_dump=lambda **kwargs: {})

        subject = NS(used=False, plan={"identity": {"num_envs_per_rank": 2048}},
                     describe_runtime=lambda: {"rank": 0, "world_size": 1},
                     _probe=lambda cfg, identity: token)
        with (patch.object(worker, "build_native_config", side_effect=config),
              patch.object(worker, "write_json"), patch.object(torch.cuda, "set_device"),
              patch.object(torch.distributed, "is_initialized", return_value=False)):
            assert worker.PbfmTrainWorker.run_segment(subject, {}, probe=True) is token

        sim = Simulation.__new__(Simulation)
        sim.wp_device, sim.num_envs, sim.cfg = device, 2, NS(nconmax=512, njmax=4096)
        with patch.object(wp, "get_cuda_driver_version", return_value=(13, 0)):
            sim.use_cuda_graph = sim._should_use_cuda_graph()
        sim.create_graph()
        env = NS(base_env=NS(mjlab_env=NS(sim=sim)))
        evidence = runtime.simulation_execution_evidence(env, eager=True)
        assert evidence["mode"] == "native_eager" and not evidence["mempool_enabled"]
        assert not evidence["use_cuda_graph"] and not any(evidence["graphs"].values())
        sim.step_graph = object()
        try:
            runtime.simulation_execution_evidence(env, eager=True)
        except ValueError:
            pass
        else:
            raise AssertionError("E2048 graph state was accepted")
        sim.step_graph = None

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        weights = [(p, p.clone()) for p in model.parameters()]
        states = {id(v): v.clone() for state in optimizer.state.values() for v in state.values()}
        rng = torch.get_rng_state().clone()
        calls, snapshots = [], []
        action, result = object(), (object(), object(), object(), object(), object())
        error = RuntimeError("original native step error")

        def step(value, *, to_numpy):
            calls.append("step")
            assert value is action and to_numpy is False
            if len([v for v in calls if v == "step"]) == 2:
                raise error
            return result

        env.step = step
        trainer = NS(train_env=env, _priority_eval_env=env)
        with (
            patch.multiple(torch.cuda, synchronize=Mock(side_effect=lambda: calls.append("sync")),
                           empty_cache=Mock(side_effect=lambda: calls.append("empty"))),
            patch("gc.collect", side_effect=lambda: calls.append("gc")),
            patch.object(runtime, "write_json"),
            patch.object(runtime, "record_cuda_memory", side_effect=lambda *a, **k: snapshots.append((a, k))),
        ):
            runtime.install_simulation_memory_handoff(trainer, 0, [], {"main": evidence})
            assert env.step(action, to_numpy=False) is result
            try:
                env.step(action, to_numpy=False)
            except RuntimeError as exc:
                assert exc is error
            else:
                raise AssertionError("native step error was swallowed")
        assert calls == ["sync", "gc", "empty", "step"] * 2
        assert [args[1] for args, _ in snapshots] == ["before_sample", "after_sample_cleanup"] * 2
        assert [details["sample"] for _, details in snapshots] == [1, 1, 2, 2]
        for param, before in weights:
            assert torch.equal(param, before)
        for state in optimizer.state.values():
            for value in state.values():
                assert torch.equal(value, states[id(value)])
        assert torch.equal(torch.get_rng_state(), rng)

        # A resumed/disabled priority evaluator is observed without constructing
        # one; a synchronization error prevents sampling and keeps its identity.
        env.step, trainer._priority_eval_env = step, None
        execution = {"main": evidence}
        calls.clear()
        with (
            patch.multiple(torch.cuda, synchronize=Mock(side_effect=lambda: calls.append("sync")),
                           empty_cache=Mock(side_effect=lambda: calls.append("empty"))),
            patch("gc.collect", side_effect=lambda: calls.append("gc")),
            patch.object(runtime, "write_json"), patch.object(runtime, "record_cuda_memory"),
        ):
            runtime.install_simulation_memory_handoff(trainer, 0, [], execution)
            assert env.step(action, to_numpy=False) is result
            assert execution["priority_eval"] == {"available": False}
            before_failure = list(calls)
            with patch.object(torch.cuda, "synchronize", side_effect=error):
                try:
                    env.step(action, to_numpy=False)
                except RuntimeError as exc:
                    assert exc is error
                else:
                    raise AssertionError("synchronize error was swallowed")
            assert calls == before_failure
    print("NATIVE_EAGER_PUBLIC_API_GRAPH_STATE_AND_SAMPLING_HANDOFF_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
