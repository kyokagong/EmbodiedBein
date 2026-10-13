"""S1 changes only the contact friction cone of the frozen I1 replay."""
import json


def install():
    import rpo_ufo_mjlab as physics
    original = physics.make_config

    def make_config(*args, **kwargs):
        config = original(*args, **kwargs)
        if config.sim.mujoco.cone != 'pyramidal':
            raise ValueError('S1 requires the original pyramidal cone')
        config.sim.mujoco.cone = 'elliptic'
        return config

    physics.make_config = make_config


def validate_receipt(receipt, validation, output, *, ground_damping_ratio=1.):
    try:
        from eb.runners.rpo_ufo_action_replay import validate_receipt as action_validate
    except ImportError:
        from rpo_ufo_action_replay import validate_receipt as action_validate
    action_validate(receipt, validation, output,ground_damping_ratio=ground_damping_ratio)
    if (receipt.get('solver_cone_factor') != 'S1'
            or receipt.get('integration_factor') != 'I1'
            or receipt.get('action_program_stage') != 'replay'):
        raise ValueError('Frozen solver cone stage differs')
    models = json.loads((output / 'startup-models.json').read_text())
    if any(models[f"{p['mode']}-{p['rank']}"]['compiled_solver']['cone'] != 1
           for p in receipt['phases']):
        raise ValueError('Compiled solver cone differs from S1')
