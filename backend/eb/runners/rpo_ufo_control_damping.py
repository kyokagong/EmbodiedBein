"""D1 advances braking without changing the frozen motor limits."""
import json


def install():
    import rpo_ufo_mjlab as physics
    original=physics.actuator_parameters

    def parameters(name):
        values=original(name)
        if 'ankle' in name:
            if values!=(40.,2.5,27.,8.):
                raise ValueError('D1 requires the original ankle damping and limits')
            return 40.,10.,27.,8.
        return values

    physics.actuator_parameters=parameters


def validate_receipt(receipt, validation, output):
    try:
        from eb.runners.rpo_ufo_action_replay import validate_receipt as action_validate
        from eb.rpo_bfm import RPO_LAB_JOINT_ORDER
    except ImportError:
        from rpo_ufo_action_replay import validate_receipt as action_validate
        from rpo_bfm import RPO_LAB_JOINT_ORDER
    action_validate(receipt,validation,output)
    if (receipt.get('control_damping_factor')!='D1' or receipt.get('solver_cone_factor')
            or receipt.get('integration_factor')!='I1' or receipt.get('action_program_stage')!='replay'):
        raise ValueError('Frozen control damping stage differs')
    models=json.loads((output/'startup-models.json').read_text())
    for p in receipt['phases']:
        model=models[f"{p['mode']}-{p['rank']}"];params=model['actuator_parameters']
        if (set(params)!=set(RPO_LAB_JOINT_ORDER) or model['compiled_solver']['cone']!=0
                or any(params[n]!=[40.,10.,27.,8.] for n in RPO_LAB_JOINT_ORDER if 'ankle' in n)):
            raise ValueError('Compiled control damping or original cone differs')
