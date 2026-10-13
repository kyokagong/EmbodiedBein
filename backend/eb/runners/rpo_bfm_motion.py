"""Bounded observation-only motion loader. Pickle is restricted to pinned conversion."""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

try:
    from eb.rpo_bfm import CODEC_HASH, MOTION_CONVERTER_REVISION, LOCAL_POOL_CONVERTER_REVISION, REFERENCE_POOL_CONVERTER_REVISION, validate_motion_manifest
except ModuleNotFoundError:
    from rpo_bfm import CODEC_HASH, MOTION_CONVERTER_REVISION, LOCAL_POOL_CONVERTER_REVISION, REFERENCE_POOL_CONVERTER_REVISION, validate_motion_manifest


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class MotionView:
    def __init__(self, clips, manifest_sha256, *, seed):
        self.clips = clips
        self.manifest_sha256 = manifest_sha256
        self.rng = np.random.default_rng(seed)

    def sample_sequences(self, batch_size: int, *, length: int = 8) -> dict:
        if length < 1 or batch_size < length or batch_size % length:
            raise ValueError("BFM expert batch must be divisible by continuous sequence length")
        starts = []
        for index, (_, arrays) in enumerate(self.clips):
            valid = arrays["valid"]
            if len(valid) <= length:
                continue
            count = np.convolve(valid.astype(np.int32), np.ones(length+1, np.int32), mode="valid")
            starts.extend((index, int(start)) for start in np.flatnonzero(count == length+1))
        if not starts:
            raise ValueError("BFM motion has no valid continuous expert windows")
        choice = self.rng.integers(len(starts), size=batch_size // length)
        result = {key: [] for key in ("states", "privileged_state", "next_states",
                                      "next_privileged_state", "sample_ids")}
        for item in choice:
            index, start = starts[int(item)]
            arrays = self.clips[index][1]
            for key in ("states", "privileged_state"):
                result[key].append(arrays[key][start:start+length])
                result["next_"+key].append(arrays[key][start+1:start+length+1])
            result["sample_ids"].append(np.stack((np.full(length, index), np.arange(start, start+length)), -1))
        return {key: np.concatenate(value).copy() for key, value in result.items()}

    def state_dict(self):
        return {"schema_version":"eb/bfm-motion-sampler/v1", "manifest_sha256":self.manifest_sha256,
                "rng":copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        if state.get("manifest_sha256") != self.manifest_sha256 or state.get("schema_version") != "eb/bfm-motion-sampler/v1":
            raise ValueError("BFM expert sampler checkpoint belongs to another motion view")
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])

    def reference(self, limit=4096):
        if not 8 <= limit <= 400000:
            raise ValueError("BFM reference size outside allowed range")
        arrays = {key: np.concatenate([clip[key][clip["valid"]] for _, clip in self.clips])
                  for key in ("states", "privileged_state", "reward_features")}
        # Fixed train-only reference distribution, independent of sampling RNG and evaluation.
        indices = np.linspace(0, len(arrays["states"])-1, min(limit, len(arrays["states"]))).round().astype(int)
        return {key: value[indices].copy() for key, value in arrays.items()}


def load_motion_view(root: Path, *, asset_sha256: str, seed: int = 0) -> MotionView:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if root.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > 1024*1024:
        raise ValueError("BFM motion manifest path is invalid")
    root = root.resolve()
    manifest = json.loads(manifest_path.read_text())
    validate_motion_manifest(manifest, asset_sha256=asset_sha256)
    clips = []
    for item in manifest["clips"]:
        path = root / item["file"]
        if (path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file()
                or path.stat().st_size > 256*1024*1024 or digest(path) != item["sha256"]):
            raise ValueError("BFM motion file path, size or hash mismatch")
        with zipfile.ZipFile(path) as archive:
            if sum(member.file_size for member in archive.infolist()) > 256*1024*1024:
                raise ValueError("BFM expanded motion clip exceeds memory budget")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in ("states", "privileged_state", "reward_features", "valid")}
        frames = item["frames"]
        for key, width in (("states", 52), ("privileged_state", 358), ("reward_features", 6)):
            value = arrays[key]
            if value.shape != (frames, width) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"BFM motion {key} shape/dtype/finite check failed")
        if arrays["valid"].shape != (frames,) or arrays["valid"].dtype != np.bool_:
            raise ValueError("BFM motion valid mask is invalid")
        if item["split"] == "train":
            clips.append((item["id"], arrays))
    sampler = MotionView(clips, digest(manifest_path), seed=seed)
    original = sampler.state_dict()
    sampler.sample_sequences(8)
    sampler.load_state_dict(original)
    return sampler


OFFICIAL_SOURCES = {
    "A1-_Stand_stageii.pkl": "189eabf1622f468ed1264c938e8941b2ae181eb295c62e73722e70db40039c5f",
    "36_11.pkl": "44d04c28be7ffbe516ffa23875cdbcd3767c04972c9fcdbe04aa345f3b80c4b7",
}


def source_profile(profile: str):
    if profile == "smoke":
        return {name:{"sha256":sha, "loader":"pickle"} for name, sha in OFFICIAL_SOURCES.items()}, 500, MOTION_CONVERTER_REVISION
    if profile in {"local_pool", "reference_pool"}:
        revision = LOCAL_POOL_CONVERTER_REVISION if profile == "local_pool" else REFERENCE_POOL_CONVERTER_REVISION
        return json.loads(Path(__file__).with_name("rpo_bfm_local_sources.json").read_text()), None, revision
    raise ValueError("BFM source profile is unsupported")


def reference_group(clip_id):
    # Unknown original ancestry: keep the whole pool together, never claim independent holdouts.
    return "official-rpo-unverified-ancestry"


def reference_mask(q, dq, height, upright, limits):
    return (((q >= limits[:, 0]-1e-4) & (q <= limits[:, 1]+1e-4)).all(-1)
            & (np.abs(dq) <= 50).all(-1) & (height >= .5) & (upright >= np.cos(.8)))


def geometry_screen(model, data, q, roots, rotations, body_pos, joint_names, body_names):
    import mujoco

    addresses = model.jnt_qposadr[[model.joint(n).id for n in joint_names]]
    bids = np.array([model.body(n).id for n in body_names])
    free = int(model.jnt_qposadr[model.joint("floating_base_joint").id])
    # The official MJCF disables self collisions; enable them for this geometry check only.
    model.geom_conaffinity[model.geom_contype != 0] = 1
    body_masks = np.zeros(model.nbody, dtype=model.body_conaffinity.dtype)
    np.bitwise_or.at(body_masks, model.geom_bodyid, model.geom_conaffinity)
    model.body_conaffinity[:] = body_masks
    parents = model.body_parentid
    nearby = {(a, int(parents[a])) for a in range(1, model.nbody)}
    nearby |= {(a, int(parents[parents[a]])) for a in range(1, model.nbody)}
    depth = np.zeros((len(q), 2))
    max_fk = 0.
    for i in range(len(q)):
        data.qpos[:] = model.qpos0
        data.qpos[free:free+3] = roots[i]
        data.qpos[free+3:free+7] = rotations[i]
        data.qpos[addresses] = q[i]
        mujoco.mj_forward(model, data)
        max_fk = max(max_fk, float(np.linalg.norm(data.xpos[bids]-body_pos[i], axis=-1).max()))
        for contact in data.contact:
            a, b = map(int, model.geom_bodyid[[contact.geom1, contact.geom2]])
            if a == b or (a != 0 and b != 0 and ((a, b) in nearby or (b, a) in nearby)):
                continue
            kind = 0 if a == 0 or b == 0 else 1
            depth[i, kind] = max(depth[i, kind], -float(contact.dist))
    if max_fk > .002:
        raise ValueError("Independent RPO FK differs from the training articulation by over 2mm")
    return (depth <= .02).all(-1), {
        "method":"mujoco_collision_mesh_fk_no_control_rollout", "penetration_limit_m":.02,
        "max_ground_penetration_m":float(depth[:, 0].max()),
        "max_nonadjacent_self_penetration_m":float(depth[:, 1].max()),
        "ground_rejected_frames":int((depth[:, 0] > .02).sum()),
        "self_rejected_frames":int((depth[:, 1] > .02).sum()),
        "max_body_position_error_m":max_fk,
        "unchecked":["adjacent_and_two_hop_links", "exact_triangle_mesh_intersection", "reference_contact_timing"],
    }


def resample_times(frames, fps, cap):
    if frames < 2 or not np.isfinite(fps) or not 10 <= fps <= 500:
        raise ValueError("BFM original motion length/FPS is invalid")
    count = int((frames-1)/fps*50) + 1
    return np.arange(min(count, cap) if cap is not None else count) / 50


def lab_quaternions_xyzw(quat):
    quat = np.asarray(quat)
    if (quat.ndim != 2 or quat.shape[1] != 4 or not np.isfinite(quat).all()
            or np.any(np.abs(np.linalg.norm(quat, axis=-1)-1) > .01)):
        raise ValueError("BFM lab quaternion shape, norm or finite check failed")
    # gmr_to_lab.py already converts its output to wxyz; SciPy expects xyzw.
    return quat[:, [1, 2, 3, 0]]


def verify_source_files(source: Path, *, profile="smoke") -> None:
    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("BFM source directory is invalid")
    root = source.resolve()
    for name, item in source_profile(profile)[0].items():
        path = root / name
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
                or path.stat().st_size > 128*1024*1024 or digest(path) != item["sha256"]):
            raise ValueError(f"BFM pinned source hash/path/size mismatch: {name}")


def load_g1_reference(source, asset_sha, *, frames=100):
    try:
        from eb.rpo_bfm import RPO_LAB_JOINT_ORDER,BODY_ORDER
    except ModuleNotFoundError:
        from rpo_bfm import RPO_LAB_JOINT_ORDER,BODY_ORDER
    path, cal_path = source/"rpo_g1_walk.npz", source/"r2r-calibration.json"
    receipt_path=source/"retarget-receipt.json"
    for file in (path,cal_path,receipt_path):
        if file.is_symlink() or not file.is_file() or file.stat().st_size > 1024*1024:
            raise ValueError("G1 codec input exceeds bounded file/path budget")
    receipt=json.loads(receipt_path.read_text())
    calibration=json.loads(cal_path.read_text())
    if (receipt.get("schema_version") != "eb/rpo-g1-r2r-trial/v1" or receipt.get("status") != "succeeded"
            or frames not in (100,400) or receipt.get("asset_sha256") != asset_sha or receipt.get("frames") != frames or receipt.get("fps") != 50
            or receipt.get("source_use") != "local_noncommercial_no_redistribution"
            or receipt.get("calibration_verified") is not True or calibration.get("verified") is not True
            or receipt.get("calibration_applied_to_solver") is not True
            or calibration.get("method") != "hhtools_constrained_chain_alignment_soma_offsets_v1"
            or calibration.get("asset_sha256") != asset_sha
            or calibration.get("source_export_sha256") != receipt.get("source_export_sha256")
            or digest(path) != receipt.get("trajectory_sha256") or digest(cal_path) != receipt.get("calibration_sha256")):
        raise ValueError("G1 codec source/calibration/asset binding mismatch")
    with zipfile.ZipFile(path) as archive:
        if sum(member.file_size for member in archive.infolist()) > 1024*1024:
            raise ValueError("G1 expanded reference exceeds bounded trial budget")
    with np.load(path,allow_pickle=False) as data:
        if (tuple(data["joint_names"]) != RPO_LAB_JOINT_ORDER or tuple(data["body_names"]) != BODY_ORDER
                or data["fps"] != 50 or not np.allclose(data["timestamps"],np.arange(frames)/50)):
            raise ValueError("G1 codec order/frequency/timestamps mismatch")
        raw={dst:data[src].copy() for src,dst in (("q","dof_pos"),("root_pos","root_pos"),("root_quat","root_rot"),("valid","valid"))}
    for key,shape in (("dof_pos",(frames,23)),("root_pos",(frames,3)),("root_rot",(frames,4))):
        if raw[key].shape != shape or not np.isfinite(raw[key]).all():
            raise ValueError("G1 codec source shape/finite mismatch")
    if (raw["valid"].dtype != np.bool_ or raw["valid"].shape != (frames,)
            or not np.allclose(np.linalg.norm(raw["root_rot"],axis=1),1,atol=1e-4)):
        raise ValueError("G1 codec mask/quaternion mismatch")
    raw["fps"]=50
    return raw,receipt


def load_g1_pool(source,asset_sha):
    import re
    receipt_path,selection_path=source/"pool-receipt.json",source/"selection.json"
    for path in (receipt_path,selection_path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size>1024*1024:
            raise ValueError("G1 pool receipt/selection path exceeds bounded budget")
    receipt=json.loads(receipt_path.read_text())
    rows=json.loads(selection_path.read_text())
    if (receipt.get("schema_version")!="eb/rpo-g1-reference-pool/v1" or receipt.get("status")!="succeeded"
            or receipt.get("asset_sha256")!=asset_sha or receipt.get("selection_sha256")!=digest(selection_path)
            or receipt.get("source_use")!="local_noncommercial_no_redistribution"
            or receipt.get("split_policy")!="take_name_all_subjects_together"
            or not isinstance(rows,list) or not 1<=len(rows)<=60
            or rows!=[{key:clip[key] for key in ("id","source_group","original_group","split")} for clip in receipt["clips"]]):
        raise ValueError("G1 pool frozen sources or split selection differs")
    raw,metadata={},{}
    for clip in receipt["clips"]:
        name=clip["id"]
        if (not re.fullmatch(r"[A-Za-z0-9_]+",name) or clip["directory"]!=f"clips/{name}"
                or clip["split"] not in {"train","validation","test"}):
            raise ValueError("G1 pool clip path/split is invalid")
        directory=source/"clips"/name
        if directory.is_symlink() or (source/"clips").is_symlink():
            raise ValueError("G1 pool forbids linked clip directories")
        motion,meta=load_g1_reference(directory,asset_sha,frames=400)
        if digest(directory/"retarget-receipt.json")!=clip["retarget_sha256"] or meta["source_group"]!=clip["source_group"]:
            raise ValueError("G1 pool member identity/group differs")
        key=f"g1_{name}.npz"
        raw[key],metadata[key]=motion,{**meta,"split":clip["split"]}
    return raw,metadata,receipt


def convert_official(source: Path, output: Path, asset_manifest: Path, report_path: Path | None = None, *, profile="smoke", official_view:Path|None=None) -> dict:
    """Trusted hash-pinned pickle → actual Isaac FK → immutable observation-only view.

    Invoke only in an isolated, offline task container, never in the API process.
    Body rotations are read from the actual Isaac articulation, including URDF import transforms.
    """
    g1_metadata=None
    pool_metadata=None
    if profile == "mixed_pool":
        from rpo_bfm import MIXED_POOL_CONVERTER_REVISION,REFERENCE_POOL_CONVERTER_REVISION
        pool_raw,pool_clips,pool_metadata=load_g1_pool(source,digest(asset_manifest))
        if official_view is None:
            raise ValueError("Mixed prior requires the frozen official RPO reference view")
        official_manifest=json.loads((official_view/"manifest.json").read_text())
        if official_manifest.get("converter_revision")!=REFERENCE_POOL_CONVERTER_REVISION:
            raise ValueError("Mixed prior needs geometry-checked official RPO v5 references")
        load_motion_view(official_view,asset_sha256=digest(asset_manifest))
        sources={name:{"sha256":meta["trajectory_sha256"],"loader":"g1_reference"} for name,meta in pool_clips.items()}
        cap,revision=400,MIXED_POOL_CONVERTER_REVISION
    elif profile == "g1_reference":
        try:
            from eb.rpo_bfm import G1_CONVERTER_REVISION
        except ModuleNotFoundError:
            from rpo_bfm import G1_CONVERTER_REVISION
        g1_raw,g1_metadata=load_g1_reference(source,digest(asset_manifest))
        sources,cap,revision={"rpo_g1_walk.npz":{"sha256":g1_metadata["trajectory_sha256"],"loader":"g1_reference"}},100,G1_CONVERTER_REVISION
    else:
        verify_source_files(source, profile=profile)
        sources, cap, revision = source_profile(profile)
    if output.exists():
        raise ValueError("BFM conversion refuses to overwrite an existing view")
    if asset_manifest.is_symlink() or not asset_manifest.is_file():
        raise ValueError("BFM conversion requires frozen RPO asset manifest")
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    env = None
    try:
        import pickle
        import torch
        import gymnasium as gym
        from scipy.spatial.transform import Rotation, Slerp
        from rpo_bfm import ACTION_GAIN, DEFAULT_Q, BODY_ORDER, RPO_LAB_JOINT_ORDER
        from rpo_bfm_codec import encode_state, encode_privileged, quaternion_matrix, heading_inverse
        from rpo_bfm_extension import GYM_ID, RPOBFMEnvCfg, register_task

        register_task()
        cfg = RPOBFMEnvCfg()
        cfg.scene.num_envs = 32
        cfg.seed = 42
        env = gym.make(GYM_ID, cfg=cfg, render_mode=None).unwrapped
        env.reset()
        robot = env.robot
        if profile in {"reference_pool","g1_reference","mixed_pool"}:
            import mujoco
            geometry_model = mujoco.MjModel.from_xml_path(str(asset_manifest.parent / "rpo_description/mjcf/rpo.xml"))
            geometry_data = mujoco.MjData(geometry_model)
        staging = output.with_name(output.name + ".staging")
        staging.mkdir(parents=True, exist_ok=False)
        clips, refs = [], []
        for name, item in sources.items():
            if pool_metadata:
                g1_raw,g1_metadata=pool_raw[name],pool_clips[name]
            source_hash = item["sha256"]
            # All source bytes verified before any executable deserialization.
            if item["loader"] == "g1_reference":
                raw=g1_raw
            elif item["loader"] == "joblib":
                import joblib
                raw = joblib.load(source / name)
            else:
                with (source / name).open("rb") as stream:
                    raw = pickle.load(stream)
            fps = float(raw["fps"])
            q = np.asarray(raw["dof_pos"], dtype=np.float64)
            pos = np.asarray(raw["root_pos"], dtype=np.float64)
            quat = np.asarray(raw["root_rot"], dtype=np.float64)  # Isaac Lab output wxyz.
            if (q.ndim != 2 or q.shape[1] != 23 or pos.shape != (len(q), 3)
                    or quat.shape != (len(q), 4) or not 10 <= fps <= 500
                    or not all(np.isfinite(a).all() for a in (q, pos, quat))
                    or np.any(np.linalg.norm(quat, axis=1) < .5)):
                raise ValueError("BFM original motion shape/finite/quaternion/FPS check failed")
            old_t = np.arange(len(q)) / fps
            new_t = resample_times(len(q), fps, cap)
            if len(new_t) < 9:
                raise ValueError("BFM original motion has fewer than nine frames")
            joints = np.stack([np.interp(new_t, old_t, column) for column in q.T], axis=1).astype(np.float32)
            roots = np.stack([np.interp(new_t, old_t, column) for column in pos.T], axis=1).astype(np.float32)
            rotations = Slerp(old_t, Rotation.from_quat(lab_quaternions_xyzw(quat)))(new_t).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
            body_positions, body_rotations = [], []
            for start in range(0, len(joints), env.num_envs):
                count = min(env.num_envs, len(joints)-start)
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] = env.scene.env_origins
                root_state[:count, :3] += torch.from_numpy(roots[start:start+count]).to(env.device)
                root_state[:count, 3:7] = torch.from_numpy(rotations[start:start+count]).to(env.device)
                root_state[:, 7:13] = 0
                target = robot.data.default_joint_pos.clone()
                target[:count] = torch.from_numpy(joints[start:start+count]).to(env.device)
                robot.write_root_state_to_sim(root_state)
                robot.write_joint_state_to_sim(target, torch.zeros_like(target))
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(env.step_dt)
                body_positions.append((robot.data.body_pos_w[:count]-env.scene.env_origins[:count, None]).cpu().numpy().copy())
                body_rotations.append(robot.data.body_quat_w[:count].cpu().numpy().copy())
            body_pos = np.concatenate(body_positions).astype(np.float32)
            body_quat = np.concatenate(body_rotations).astype(np.float32)
            body_lin_vel = np.gradient(body_pos, .02, axis=0).astype(np.float32)
            body_ang_vel = np.empty_like(body_pos)
            for body in range(24):
                rot = Rotation.from_quat(body_quat[:, body, [1, 2, 3, 0]])
                delta = (rot[1:] * rot[:-1].inv()).as_rotvec() / .02
                body_ang_vel[0, body] = delta[0]
                body_ang_vel[-1, body] = delta[-1]
                body_ang_vel[1:-1, body] = (delta[:-1]+delta[1:])/2
            dq = np.gradient(joints, .02, axis=0).astype(np.float32)
            tensors = [torch.from_numpy(a) for a in (body_pos, body_quat, body_lin_vel, body_ang_vel)]
            root_matrix = quaternion_matrix(tensors[1][:, 0])
            gravity = (root_matrix.transpose(-1, -2) @ torch.tensor([0., 0., -1.]).expand(len(joints), 3)[..., None]).squeeze(-1)
            omega = (root_matrix.transpose(-1, -2) @ tensors[3][:, 0, :, None]).squeeze(-1)
            states = encode_state(torch.from_numpy(joints), torch.from_numpy(dq), gravity, omega).numpy().astype(np.float32)
            privileged = encode_privileged(*tensors).numpy().astype(np.float32)
            local_velocity = (heading_inverse(tensors[1][:, 0]) @ tensors[2][:, 0, :, None]).squeeze(-1).numpy()
            pose_cost = np.mean(states[:, :23]**2, axis=1)
            reward_features = np.column_stack((local_velocity[:, :2], body_ang_vel[:, 0, 2],
                                               -gravity[:, 2].numpy(), body_pos[:, 0, 2], pose_cost)).astype(np.float32)
            limits = robot.data.joint_pos_limits[0].cpu().numpy()
            valid = ((joints >= limits[:, 0] - 1e-4) & (joints <= limits[:, 1] + 1e-4)).all(-1)
            valid &= (np.abs(dq) <= 50).all(-1) & (body_pos[:, 0, 2] >= .5)
            if profile == "local_pool":
                valid &= (np.abs((joints-np.asarray(DEFAULT_Q))/(.25*np.asarray(ACTION_GAIN))) <= 1+1e-6).all(-1)
                valid &= -gravity[:, 2].numpy() >= np.cos(.80)
            quality = {}
            if profile in {"reference_pool","g1_reference","mixed_pool"}:
                valid = reference_mask(joints, dq, body_pos[:, 0, 2], -gravity[:, 2].numpy(), limits)
                geometry_valid, quality = geometry_screen(geometry_model, geometry_data, joints, roots,
                    rotations, body_pos, RPO_LAB_JOINT_ORDER, BODY_ORDER)
                quality["before_geometry_valid_frames"] = int(valid.sum())
                valid &= geometry_valid
                if g1_metadata:
                    valid &= raw["valid"]
                if pool_metadata:
                    temporal=(np.linalg.norm(body_lin_vel[:,0],axis=-1)<=8)&(np.linalg.norm(body_ang_vel[:,0],axis=-1)<=20)
                    quality["temporal_rejected_frames"]=int((~temporal).sum())
                    valid &= temporal
                    feet=[BODY_ORDER.index(side+"_ankle_roll_link") for side in ("left","right")]
                    low=body_pos[:,feet,2]<.12
                    speed=np.linalg.norm(body_lin_vel[:,feet,:2],axis=-1)
                    quality["low_foot_origin_moving_fraction"]=float(((speed>.3)&low).sum()/max(int(low.sum()),1))
                    quality["foot_contact_status"]="link_origin_proxy_only_no_verified_contact_labels"
                quality["continuous_9frame_starts"] = int((np.convolve(valid.astype(int), np.ones(9, int), mode="valid") == 9).sum())
            # Keep validity masks; never clip joints to manufacture good references.
            file = staging / (Path(name).stem + ".npz")
            np.savez(file, states=states, privileged_state=privileged, reward_features=reward_features,
                     valid=valid.astype(bool), root_pos=roots, root_quat=rotations, q=joints, dq=dq,
                     body_pos=body_pos, body_quat=body_quat, body_lin_vel=body_lin_vel, body_ang_vel=body_ang_vel,
                     timestamps=new_t.astype(np.float64))
            clips.append({"id":Path(name).stem, "file":file.name, "sha256":digest(file), "fps":50,
                          "frames":len(joints), "split":g1_metadata.get("split","train") if g1_metadata else "train",
                          "source_group":g1_metadata["source_group"] if g1_metadata else reference_group(Path(name).stem) if profile == "reference_pool" else Path(name).stem,
                          "valid_frames":int(valid.sum()), "source_fps":fps,
                          "reference_quality":quality,
                          "source_license":"CC-BY-NC-ND-4.0; local noncommercial no redistribution" if g1_metadata else "BSD-3-Clause; original motion rights not independently verified"})
            refs.append({"uri":f"https://github.com/Roboparty/robolab/blob/6b1c3d9988497c8961dcba77892de32edc1770e1/data/motions/rpo_lab/{name}",
                         "revision":"6b1c3d9988497c8961dcba77892de32edc1770e1", "sha256":source_hash,
                         "license":"BSD-3-Clause"})
        if g1_metadata:
            refs=[{**ref,"uri":ref["url"]} for ref in g1_metadata["source_refs"]]
        if pool_metadata:
            for clip in official_manifest["clips"]:
                new_file="rpo_"+clip["file"]
                with np.load(official_view/clip["file"],allow_pickle=False) as data:
                    arrays={key:data[key].copy() for key in data.files}
                temporal=(np.linalg.norm(arrays["body_lin_vel"][:,0],axis=-1)<=8)&(np.linalg.norm(arrays["body_ang_vel"][:,0],axis=-1)<=20)
                rejected=int((arrays["valid"]&~temporal).sum())
                arrays["valid"] &= temporal
                quality={**clip["reference_quality"],"temporal_rejected_frames":rejected,
                    "continuous_9frame_starts":int((np.convolve(arrays["valid"].astype(int),np.ones(9,int),mode="valid")==9).sum())}
                np.savez(staging/new_file,**arrays)
                clips.append({**clip,"id":"rpo_"+clip["id"],"file":new_file,"split":"train",
                    "sha256":digest(staging/new_file),"valid_frames":int(arrays["valid"].sum()),"reference_quality":quality})
            refs.extend(official_manifest["source_refs"])
        manifest = {"schema_version":"eb/rpo-bfm-motion/v1", "purpose":"behavior_prior", "usage":"train",
                    "joint_order":list(RPO_LAB_JOINT_ORDER), "body_order":list(BODY_ORDER),
                    "codec_sha256":CODEC_HASH, "asset_sha256":digest(asset_manifest),
                    "qualification":"reference_candidate" if profile in {"reference_pool","g1_reference"} else "diagnostic", "dynamic_replay_verified":False,
                    "source_rights":"repository_license_only", "source_refs":refs, "clips":clips,
                    "converter_revision":revision, "velocity_method":"central_finite_difference_50hz",
                    "body_velocity_point":"link_origin", "source_frame_cap":cap,
                    "source_quaternion":"wxyz", "output_quaternion":"wxyz"}
        if profile in {"reference_pool","g1_reference"}:
            manifest.update(reference_recipe={"pd_target_range_required":False,
                "joint_limits":"training_isaac_urdf", "standing_height_min_m":.5,
                "standing_tilt_max_rad":.8, "joint_speed_max_radps":50,
                "split_policy":"all_unverified_ancestors_train_no_independent_holdout"},
                unchecked=["upstream_motion_rights", "independent_source_holdouts", "reference_contact_timing", "controller_evaluation"])
        if g1_metadata:
            manifest.update(source_rights="local_research_use_confirmed_no_redistribution",
                source_use=g1_metadata["source_use"],calibration_verified=True,
                calibration_applied_to_solver=True,
                calibration_sha256=g1_metadata["calibration_sha256"],
                source_retarget_sha256=digest(source/("pool-receipt.json" if pool_metadata else "retarget-receipt.json")),
                unchecked=["physical_hand_tip_and_sole_alignment","independent_source_holdouts","reference_contact_timing","controller_evaluation"])
        if pool_metadata:
            manifest.update(qualification="local_research_reference",source_pool_sha256=digest(source/"pool-receipt.json"),
                selection_sha256=pool_metadata["selection_sha256"],official_manifest_sha256=digest(official_view/"manifest.json"),
                reference_recipe={"split_policy":"take_name_all_subjects_together_official_unknown_ancestors_train_only",
                    "min_train_groups":5,"min_train_frames":2000,"min_validation_groups":1,"min_test_groups":1,
                    "pd_target_range_required":False,"root_speed_limit_mps":8,"root_angular_speed_limit_radps":20},
                unchecked=["official_original_motion_rights","physical_hand_tip_and_sole_alignment",
                    "reference_contact_timing","controller_evaluation"])
        (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
        validation = validate_motion_manifest(manifest, asset_sha256=digest(asset_manifest))
        sampler = load_motion_view(staging, asset_sha256=digest(asset_manifest))
        sampler.sample_sequences(16)
        staging.rename(output)
        receipt = {"schema_version":"eb/rpo-bfm-motion-conversion/v1", "status":"succeeded",
                   "validation":{**validation, "manifest_sha256":digest(output / "manifest.json")},
                   "clips":len(clips), "frames":sum(c["frames"] for c in clips), "source_refs":refs}
        (output.parent / "rpo-bfm-motion-receipt.json").write_text(json.dumps(receipt, indent=2))
        if report_path is not None:
            files = [output / "manifest.json", *(output / clip["file"] for clip in clips)]
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps({
                "format":"rpo_bfm_motion", "clips_count":len(clips),
                "frames_count":receipt["frames"], "total_bytes":sum(path.stat().st_size for path in files),
                "source_refs":refs,
                "validation":{**receipt["validation"], "storage_ref":str(output.resolve())},
            }, sort_keys=True))
        print(json.dumps(receipt), flush=True)
        return receipt
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--official-view",type=Path)
    parser.add_argument("--profile", choices=("smoke", "local_pool", "reference_pool", "g1_reference","mixed_pool"), default="smoke")
    args = parser.parse_args()
    convert_official(args.source, args.output, args.asset_manifest, args.report, profile=args.profile,official_view=args.official_view)
