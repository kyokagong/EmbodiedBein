"""One frozen ground-contact factor; reuse the original integrated-force phases."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

OPERATION = 'probe_rpo_ufo_contact_factor_v1'
TIME_CONSTANT = {'B0': .02, 'C1': .04}


def configure_ground(geom, candidate):
    if candidate not in TIME_CONSTANT or not np.array_equal(geom.solref, [.02, 1.]):
        raise ValueError('Ground factor requires the frozen ground defaults and B0/C1')
    geom.solref[0] = TIME_CONSTANT[candidate]


def depth_summary(histogram, maximum):
    histogram = np.asarray(histogram)
    if (histogram.shape != (2001,) or (histogram < 0).any() or (histogram != np.floor(histogram)).any()
            or not np.isfinite(histogram).all() or not np.isfinite(maximum) or maximum < 0):
        raise ValueError('Invalid contact depth evidence')
    count = int(histogram.sum())
    occupied = np.flatnonzero(histogram)
    if ((not count and maximum != 0.) or (count and
            (maximum < occupied[-1]*1e-5-1e-7 or
             (occupied[-1] < 2000 and maximum > (occupied[-1]+1)*1e-5+1e-7)))):
        raise ValueError('Contact depth maximum differs from histogram')
    index = int(np.searchsorted(histogram.cumsum(), np.ceil(count*.99))) if count else 0
    return {'count': count, 'p99_upper_m': min(maximum, (index+1)*1e-5) if index < 2000 else maximum,
        'max_m': float(maximum), 'overflow_count': int(histogram[-1])}


def install_measurement(candidate, rank, mode, output, *, substeps=8):
    import torch
    import warp as wp
    from mjlab.terrains.terrain_entity import TerrainEntity
    import rpo_ufo_mjlab as physics
    from rpo_ufo_dynamics_diagnostics import write_json

    build_plane = TerrainEntity._import_ground_plane

    def ground(self, name):
        build_plane(self, name)
        configure_ground(self._spec.geom(name), candidate)

    TerrainEntity._import_ground_plane = ground
    original = physics.RPOUFOMJLabPhysics

    class MeasuredPhysics(original):
        def _refresh(self, dt=.0025):
            result = super()._refresh(dt)
            # Forward refresh gives contact distances at the integrated qpos.
            if hasattr(self, 'population') and len(self.population) > getattr(self, '_measured_samples', 0):
                if not hasattr(self, '_depth_hist'):
                    self._depth_hist = torch.zeros(2001, dtype=torch.int64, device=self.device())
                    self._depth_max = 0.; self._solref_min = np.full(2, np.inf); self._solref_max = np.zeros(2)
                    self._original_peak = torch.zeros(self.num_envs, device=self.device())
                    self._strict_first = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device())
                    self._first_terminated = 0
                self._measured_samples = len(self.population)
                data = self.lab.sim.wp_data
                ratio = (wp.to_torch(data.qvel)[:, self.dof_ids].abs()/self.limits['joint_vel']).amax(-1)
                self._original_peak = torch.maximum(self._original_peak, ratio.masked_fill(~self.alive, 0.))
                count = int(data.nacon.numpy()[0])
                worlds = wp.to_torch(data.contact.worldid)[:count]
                mask = self.alive[worlds]
                depth = (-wp.to_torch(data.contact.dist)[:count][mask]).clamp_min(0.)
                if len(depth):
                    bins = (depth/1e-5).long().clamp_max(2000)
                    self._depth_hist += torch.bincount(bins, minlength=2001)
                    self._depth_max = max(self._depth_max, float(depth.max()))
                    geom = wp.to_torch(data.contact.geom)[:count]
                    ground_mask = mask & (geom == 0).any(-1)
                    refs = wp.to_torch(data.contact.solref)[:count][ground_mask]
                    if len(refs):
                        self._solref_min = np.minimum(self._solref_min, refs.amin(0).cpu().numpy())
                        self._solref_max = np.maximum(self._solref_max, refs.amax(0).cpu().numpy())
            return result

        def step(self, actions):
            result = super().step(actions)
            first = self.alive & (self._strict_first == 0) & (self.interval.first_violation > 0)
            self._strict_first[first] = self.interval.first_violation[first]
            if self._measured_samples == substeps:
                self._first_terminated = int((result[2] | result[3]).sum())
            return result

        def close(self):
            if hasattr(self, '_depth_hist'):
                peaks = self._original_peak.cpu().numpy()
                lengths = torch.where(self._strict_first > 0, self._strict_first, 48).cpu().numpy()
                hist = self._depth_hist.cpu().numpy()
                suffix = f'{mode}-{rank}'
                np.savez_compressed(output/f'contact-population-{suffix}.npz',
                    peak_ratio=peaks, strict_length=lengths, depth_histogram=hist,
                    original_alive=self.alive.cpu().numpy())
                write_json(output/f'contact-population-{suffix}.json', {
                    'candidate': candidate, 'ground_time_constant_s': TIME_CONSTANT[candidate],
                    'completed_substeps': self._measured_samples,
                    'depth': depth_summary(hist, self._depth_max),
                    'effective_solref_min': self._solref_min, 'effective_solref_max': self._solref_max,
                    'peak_ratio_quantiles': np.quantile(peaks, [.5,.95,.99,1.]),
                    'strict_length_mean': float(lengths.mean()),
                    'first_step_terminated_count': self._first_terminated,
                    'first_step_terminated_fraction': self._first_terminated/self.num_envs})
            super().close()

    physics.RPOUFOMJLabPhysics = MeasuredPhysics


def validate_receipt(receipt, validation, output, *, substeps=8, physics_dt=.0025, ground_damping_ratio=1.):
    try:
        from eb.runners.rpo_ufo_dynamics_diagnostics import validate_receipt as base_validate
    except ImportError:
        from rpo_ufo_dynamics_diagnostics import validate_receipt as base_validate
    base_validate(receipt, validation, output, substeps=substeps, physics_dt=physics_dt)
    candidate = receipt.get('contact_factor')
    if candidate not in TIME_CONSTANT:
        raise ValueError('Contact factor identity differs')
    members = {row['file'] for row in receipt['files']}
    for phase in receipt['phases']:
        suffix = f"{phase['mode']}-{phase['rank']}"
        required = {f'contact-population-{suffix}.npz', f'contact-population-{suffix}.json'}
        if not required <= members:
            raise ValueError('Contact population evidence missing')
        stats = json.loads((output/f'contact-population-{suffix}.json').read_text())
        expected = [(TIME_CONSTANT[candidate]+.001)/2, (ground_damping_ratio+2.)/2]
        if (stats['candidate'] != candidate or stats['completed_substeps'] != 48*substeps
                or stats['ground_time_constant_s'] != TIME_CONSTANT[candidate]
                or not np.allclose(stats['effective_solref_min'], expected, atol=1e-7, rtol=0)
                or not np.allclose(stats['effective_solref_max'], expected, atol=1e-7, rtol=0)):
            raise ValueError('Effective contact factor was not applied')
        with np.load(output/f'contact-population-{suffix}.npz', allow_pickle=False) as arrays:
            peaks, lengths = arrays['peak_ratio'], arrays['strict_length']
            alive = arrays['original_alive']
            population = np.asarray(json.loads((output/f'population-{suffix}.json').read_text()))
            depth = depth_summary(arrays['depth_histogram'], stats['depth']['max_m'])
            if (peaks.shape != (2048,) or lengths.shape != (2048,) or not np.isfinite(peaks).all()
                    or (peaks < 0).any() or (lengths < 1).any() or (lengths > 48).any()
                    or depth != stats['depth'] or not np.isclose(lengths.mean(), stats['strict_length_mean'])
                    or not np.allclose(np.quantile(peaks,[.5,.95,.99,1.]), stats['peak_ratio_quantiles'])
                    or not np.isclose(peaks.max(), phase['original_episode_peak_ratio'])
                    or alive.shape != (2048,) or alive.dtype != bool
                    or int(alive.sum()) != phase['initial_survivors_48']
                    or not 0 <= stats['first_step_terminated_count'] <= 2048
                    or stats['first_step_terminated_count'] != 2048-int(population[substeps,1])
                    or stats['first_step_terminated_fraction'] != stats['first_step_terminated_count']/2048):
                raise ValueError('Contact population summary differs')


def main():
    from rpo_ufo_dynamics_diagnostics import phase, digest, write_json, SCHEMA
    candidate = os.environ['EB_UFO_CONTACT_FACTOR']
    if candidate not in TIME_CONSTANT:
        raise ValueError('Unknown contact candidate')
    output = Path('/outputs'); output.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) == 3:
        rank, mode = int(sys.argv[1]), sys.argv[2]
        if rank not in range(3) or mode not in ('anchor','policy'):
            raise ValueError('Unknown contact phase')
        import rpo_ufo_learning_diagnostics as learning
        admitted = learning.visible_devices()
        os.environ['CUDA_VISIBLE_DEVICES'] = admitted[rank+1]
        # phase repeats selection; retain the four-card mapping after masking.
        learning.visible_devices = lambda: admitted
        install_measurement(candidate, rank, mode, output)
        write_json(output/f'phase-{mode}-{rank}.json', phase(rank, mode, output)); return
    if sys.argv[1:]:
        raise ValueError('Unbounded contact diagnostic arguments')
    for mode in ('anchor','policy'):
        children = [subprocess.Popen([sys.executable, __file__, str(r), mode]) for r in range(3)]
        results = [child.wait() for child in children]
        if any(results):
            raise RuntimeError(f'Contact factor {candidate}/{mode} failed: {results}')
    receipt = {'schema_version': SCHEMA, 'qualification':'diagnostics_only', 'baseline_step':600,
        'optimizer_updates':0, 'contact_factor':candidate,
        'phases':[json.loads((output/f'phase-{m}-{r}.json').read_text())
            for m in ('anchor','policy') for r in range(3)],
        'dataset_manifest_sha256':digest(Path(os.environ['EB_UFO_MOTION_ROOT'])/'manifest.json'),
        'files':[{'file':p.name, 'bytes':p.stat().st_size, 'sha256':digest(p)}
            for p in sorted(output.iterdir()) if p.is_file()]}
    validate_receipt(receipt, {'manifest_sha256':receipt['dataset_manifest_sha256']}, output)
    write_json(output/'dynamics-diagnostics.json', receipt)


if __name__ == '__main__':
    main()
