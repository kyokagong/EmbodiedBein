"""A1 ground contact band, with the original signed geometry depth preserved."""
import json

import numpy as np


def install():
    from mjlab.terrains.terrain_entity import TerrainEntity
    original = TerrainEntity._import_ground_plane

    def ground(self, name):
        original(self, name)
        geom = self._spec.geom(name)
        if geom.margin != 0 or geom.gap != 0:
            raise ValueError('A1 requires the original zero margin/gap')
        geom.margin = .003

    TerrainEntity._import_ground_plane = ground


def validate_receipt(receipt, validation, output):
    if receipt.get('contact_margin_factor') != 'A1':
        raise ValueError('Frozen contact margin factor differs')
    try:
        from eb.runners.rpo_ufo_ground_stiffness import validate_receipt as ground_validate
    except ImportError:
        from rpo_ufo_ground_stiffness import validate_receipt as ground_validate
    ground_validate(receipt, validation, output, contact_margin_m=.003)
    for phase in receipt['phases']:
        suffix = f"{phase['mode']}-{phase['rank']}"
        actual = json.loads((output/f'actions-{suffix}.json').read_text())['actual_contact_options']
        margins, gaps = actual['geom_margin'], actual['geom_gap']
        if len(margins) < 2 or margins != [.003] + [0.]*(len(margins)-1) or gaps != [0.]*len(margins):
            raise ValueError('Actual A1 ground margin/gap differs')
        for event in json.loads((output/f'events-{suffix}.json').read_text()):
            with np.load(output/event['file']) as arrays:
                pairs = arrays['contact_geom']
                included = arrays['contact_includemargin']
                if pairs.ndim != 2 or pairs.shape[1] != 2 or included.shape != (len(pairs),):
                    raise ValueError('Included contact margin shape differs')
                expected = np.where((pairs == 0).any(-1), .003, 0.)
                if not np.allclose(included, expected, atol=1e-7, rtol=0):
                    raise ValueError('Actual included contact margin differs')
