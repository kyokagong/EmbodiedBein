"""C2 changes only the ground damping ratio after the original ground setup."""
import numpy as np


def install():
    from mjlab.terrains.terrain_entity import TerrainEntity
    original=TerrainEntity._import_ground_plane

    def ground(self,name):
        original(self,name)
        geom=self._spec.geom(name)
        if not np.array_equal(geom.solref,[.02,1.]):
            raise ValueError('C2 requires the original ground solref')
        geom.solref[1]=.1

    TerrainEntity._import_ground_plane=ground


def validate_receipt(receipt,validation,output, *, contact_margin_m=0.):
    if receipt.get('ground_stiffness_factor')!='C2':
        raise ValueError('Frozen ground stiffness factor differs')
    if (contact_margin_m not in (0., .003) or receipt.get('contact_margin_factor') !=
            ('A1' if contact_margin_m == .003 else None)):
        raise ValueError('Frozen contact margin differs from ground stiffness stage')
    try:
        from eb.runners.rpo_ufo_friction_impedance import validate_receipt as friction_validate
    except ImportError:
        from rpo_ufo_friction_impedance import validate_receipt as friction_validate
    friction_validate(receipt,validation,output,ground_damping_ratio=.1)
