"""Reproduce bootstrap-only LIBERO metadata through the actual trajectory builder."""
import torch
from rlinf.data.schema.embodied_types import ChunkStepResult, convert_trajectories_to_batch
from rlinf.data.schema.embodied_trajectory_builder import EmbodiedTrajectoryBuilder
from rlinf.utils.nested_dict_process import process_nested_dict_for_train

builder = EmbodiedTrajectoryBuilder()
flags = torch.zeros(1, 10, dtype=torch.bool)
builder.append_step_result(ChunkStepResult(
    dones=flags, terminations=flags, truncations=flags,
    valid_step_mask=torch.ones_like(flags), prev_values=torch.zeros(1, 1)))
for step in range(20):
    builder.append_step_result(ChunkStepResult(
        dones=flags, terminations=flags, truncations=flags,
        rewards=torch.full((1, 10), float(step)),
        prev_values=torch.zeros(1, 1), prev_logprobs=torch.zeros(1, 1),
        versions=torch.zeros(1, 1)))
batch = convert_trajectories_to_batch([builder.to_trajectory()])
assert batch['valid_step_mask'].shape == (1, 1, 10)
shuffle = torch.arange(19, -1, -1)
try:
    process_nested_dict_for_train(batch, shuffle)
except IndexError:
    pass
else:
    raise AssertionError('Expected original bootstrap-only metadata shuffle failure')
from libero_ppo import discard_bootstrap_only_validity
discard_bootstrap_only_validity(batch)
actual = process_nested_dict_for_train(batch, shuffle)
assert actual['rewards'].shape == (20, 10)
torch.testing.assert_close(actual['rewards'][:, 0], shuffle.float())
assert 'valid_step_mask' not in batch
for mask in (torch.zeros(1, 1, 10, dtype=torch.bool), torch.ones(20, 1, 10, dtype=torch.bool)):
    candidate = {'valid_step_mask': mask, 'prev_logprobs': batch['prev_logprobs']}
    discard_bootstrap_only_validity(candidate)
    assert candidate['valid_step_mask'] is mask
print('PASS: real trajectory builder reproduces sparse mask; owned boundary preserves shuffled rewards')
