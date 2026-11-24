# Frozen P-BFM RPO adaptation

`upstream-manifest.json` freezes the complete read-only P-BFM input at
`c6eacbdb49d6c050814b6f457bb0cb30a02a16ca`. `replacements.json` contains exact,
single-match source edits; drift and repeated application fail before import.
The output receipt indexes the source, every patch, and every adapted file.

RPO dynamics and joint semantics were copied from the read-only UFO tree at
`bedef7f5898d20cf9a237aa44f1c4a23f93c937a`, including its local RPO integration:

| Input | SHA-256 |
| --- | --- |
| `configs/robots/rpo_23dof.yaml` | `451f73eaa8e5ebbb3c23d6eb8361e692a8ac9fcbfa679820eb6109558e69e1c2` |
| `ufo/data/robots/rpo/rpo_23dof.xml` | `353481c28d4227d8b19ce42a6775c4069df3e0fe9eb11435696bf6a430b0c17a` |

The managed XML relocates the mesh directory to the separately frozen
`/robot-assets/rpo_description/meshes` mount and names the existing two foot
collision meshes. It adds no colliders. RPO sole scans and virtual-edge samples
use a conservative union of the actual collision mesh bounds in the ankle body
frames; the effective configuration records those bounds. G1 retains its native geometry.
The managed YAML adds the RPO Hydra
recipe and explicit physics/control overrides. The Hydra robot package carries
RPO bodies, limits, gains, motion and contact semantics, with zero extensions.
The patches preserve G1 validation and reject unknown robot names, carry
`action_rescale`, and preserve explicit motion extensions. They do not change
the native agent losses, optimizer order, Replay or distributed implementation.

The virtual camera in `eb.runners.pbfm_source.CAMERA_CONTRACT` mounts on the RPO
torso at `[0.08, 0, 0.10]`, pitched down 15 degrees. It is `simulation_only`.
Its placement is a simulation candidate requiring actual depth/render evidence;
this package makes no physical camera calibration or hardware claim. Crop,
depth quantity, delay, history, normalization and installation randomization
retain `optical_z_maxcrop18_reset_v2`. Source reset code fills reset environments
with the current camera frame.

P-BFM code retains its upstream CC BY-NC 4.0 notice. RPO assets and motion data
retain their own provenance and license requirements; the code license does not
qualify data or grant delivery rights. No pretrained G1/UFO weights are included.
