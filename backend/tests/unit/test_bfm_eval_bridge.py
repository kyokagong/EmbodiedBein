import numpy as np
import pytest


def test_bfm_eval_packs_native_history_without_flat_observations():
    from eb.runners.rpo_bfm_eval import pack_observation
    obs={key:np.full((1,width),i,np.float32) for i,(key,width) in enumerate(
        (("states",52),("last_action",23),("history_actor",300),("privileged_state",358)))}
    result=pack_observation(obs)
    assert result["policy"].shape==(1,375) and result["critic"].shape==(1,733)
    assert np.array_equal(result["policy"][:,75:],obs["history_actor"])
    assert np.array_equal(result["critic"][:,:375],result["policy"])
    with pytest.raises(ValueError):pack_observation({**obs,"states":np.zeros((1,780))})
    from eb.runners.rpo_flat_artifact import _numpy
    assert np.array_equal(_numpy(result["policy"]),result["policy"])


def test_bfm_eval_uses_current_command_and_explicit_latent():
    from eb.runners.rpo_bfm_eval import CommandActorSession
    class Graph:
        def __init__(self,command=False):self.command=command;self.inputs=None
        def get_inputs(self):return [type("Input",(),{"name":"observation"})()]
        def run(self,outputs,inputs):
            self.inputs=inputs
            return [np.full((1,256),1,np.float32) if self.command else np.zeros((1,23),np.float32)]
    actor,command=Graph(),Graph(True)
    session=CommandActorSession(actor,command)
    session.command=lambda:np.array([[.3,-.1,.2]],np.float32)
    obs=np.zeros((1,375),np.float32)
    action=session.run(None,{"observation":obs})[0]
    assert action.shape==(1,23)
    assert np.array_equal(command.inputs["command"],session.command())
    assert np.array_equal(actor.inputs["z"],session.last_latent)
    session.command=lambda:np.array([[2.,0.,0.]],np.float32)
    with pytest.raises(ValueError):session.run(None,{"observation":obs})


def test_standard_sensor_noise_uses_native_units_and_original_physical_amplitudes():
    from types import SimpleNamespace
    from eb.runners.rpo_bfm_eval import sensor_noise_bounds
    bounds=sensor_noise_bounds(SimpleNamespace(joint_pos=.01,joint_vel=1.5,projected_gravity=.05,ang_vel=.2))
    assert bounds.shape==(52,)
    assert np.allclose(bounds[:23],.01) and np.allclose(bounds[23:46],1.5)
    assert np.allclose(bounds[46:],.05)


def test_flat_mesh_diagnostic_records_absent_terrain_indices(monkeypatch):
    import sys
    from types import SimpleNamespace
    from eb.runners import rpo_flat_artifact as evaluator
    monkeypatch.setitem(sys.modules,"rpo_wbc_artifact",SimpleNamespace(actuator_delay_steps=lambda _:0))
    monkeypatch.setattr(evaluator,"_state",lambda _, index=0: {"root_pos_w":np.zeros(3)})
    data=SimpleNamespace(**{key:np.zeros((1,23)) for key in
        ("joint_stiffness","joint_damping","joint_armature","joint_friction_coeff")})
    view=SimpleNamespace(**{key:lambda:np.zeros((1,24)) for key in ("get_masses","get_coms","get_material_properties")})
    env=SimpleNamespace(scene=SimpleNamespace(terrain=SimpleNamespace(),env_origins=np.zeros((1,3))),
        robot=SimpleNamespace(data=data,root_physx_view=view,actuators={}),
        command_generator=SimpleNamespace(command=np.zeros((1,3))))
    row=evaluator._initial_state(env,60000,{})
    assert row["terrain_level"] is None and row["terrain_column"] is None
