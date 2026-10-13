"""S2 changes friction impedance relative to the captured elliptic S1 baseline."""
import json


def install():
    import rpo_ufo_mjlab as physics
    original=physics.make_config

    def make_config(*args,**kwargs):
        config=original(*args,**kwargs)
        if config.sim.mujoco.cone!='elliptic' or config.sim.mujoco.impratio!=1.:
            raise ValueError('S2 requires the original S1 impratio')
        config.sim.mujoco.impratio=.1
        return config

    physics.make_config=make_config


def validate_receipt(receipt,validation,output, *, ground_damping_ratio=1.):
    try:
        from eb.runners.rpo_ufo_solver_cone import validate_receipt as cone_validate
    except ImportError:
        from rpo_ufo_solver_cone import validate_receipt as cone_validate
    if (ground_damping_ratio not in (1.,.1) or receipt.get('ground_stiffness_factor')!=
            ('C2' if ground_damping_ratio==.1 else None)):
        raise ValueError('Frozen ground stiffness differs from impratio stage')
    cone_validate(receipt,validation,output,ground_damping_ratio=ground_damping_ratio)
    if receipt.get('solver_friction_factor')!='S2' or receipt.get('control_damping_factor'):
        raise ValueError('Frozen impratio factor differs')
    expected={'cone':1,'impratio':.1,'tolerance':1e-8,'ls_iterations':50,'ls_tolerance':.01}
    for phase in receipt['phases']:
        data=json.loads((output/f"actions-{phase['mode']}-{phase['rank']}.json").read_text())
        if data.get('actual_solver_options')!=expected:
            raise ValueError('Actual impratio or original solver options differ')
