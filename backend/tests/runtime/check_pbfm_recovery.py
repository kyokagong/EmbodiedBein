"""CPU-only native counter and invalid-frame capture checks; no GPU work."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS

import torch

from check_pbfm_2048_accounting import native_config, run_native_loop
from eb.runners.pbfm_source import prepare_pbfm_source

source = Path('/tmp/eb-pbfm-accounting-source')
prepare_pbfm_source(Path('/opt/pbfm-upstream'), source)
sys.path.insert(0, str(source))
from eb.runners import pbfm_numerics as numerics
from pbfm.envs.mjlab import observations
from pbfm.terrains.terrain_observation import observations_from_clearances

with tempfile.TemporaryDirectory(prefix='eb-recovery-cpu-') as temp:
    root=Path(temp)
    cfg=native_config(root,num_envs=2048,world=4,endpoint=51290112)
    cfg=cfg.model_copy(update={'checkpoint_every_steps':5120000})
    saves=[]
    times,final,logs=run_native_loop(root,cfg,start_updates=19984,start_local=2578432,saves_out=saves)
    assert len(times)==80016 and times[0]==2580480 and len(logs)==5001
    assert final['optimizer_steps']==100000 and final['global_time']==51290112
    assert [s['optimizer_steps'] for s in saves]==[*range(29968,100000,10000),100000]
    print(json.dumps({'native_remaining_updates':len(times),'final_counts':{k:final[k] for k in
        ('optimizer_steps','local_time','global_time')},'periodic_updates':[s['optimizer_steps'] for s in saves]}))
    heights=torch.tensor([[[1.,1.]],[[float('nan'),1.]]])
    class Core:
        num_envs=2
        robot_root_states=torch.zeros(2,13)
        body_pos=torch.zeros(2,24,3)
        dof_pos=torch.zeros(2,23)
        dof_vel=torch.zeros(2,23)
        _training_terrain_global_step=10313728
    import mujoco
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    sensor=NS(data=NS(heights=heights),_cached_frame_pos=torch.zeros(2,1,3))
    core=Core()
    core.mjlab_env=NS(scene=NS(sensors={'terrain_height':sensor}),sim=NS(mj_model=model,
        data=NS(qpos=torch.zeros(2,30),qvel=torch.zeros(2,29),qacc=torch.zeros(2,29))))
    original=observations._terrain_observations
    observations._terrain_observations=lambda core:observations_from_clearances(heights[:,0],0,clip=.5)
    numerics.OUTPUT_ROOT=root
    numerics.install_numerics_capture(1)
    try:
        observations._terrain_observations(core)
    except RuntimeError as error:
        assert 'non-finite' in str(error)
    else:
        raise AssertionError('numeric failure must remain a failure')
    evidence=json.loads((root/'outputs/numerics/rank-1/invalid-frame.json').read_text())
    assert evidence['nonfinite_entries']['physics_qpos']==0
    assert evidence['nonfinite_entries']['heights']==1
    assert evidence['termination_changed'] is False and evidence['observation_sanitized'] is False
    assert (root/'outputs/numerics/rank-1/scene.mjb').is_file()
    observations._terrain_observations=original
    print('INVALID_FRAME_PRESERVED_WITHOUT_SANITIZING_OR_CHANGING_TERMINATION')
