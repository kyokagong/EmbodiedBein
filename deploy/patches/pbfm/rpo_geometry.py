"""Conservative RPO foot envelopes from the actual frozen collision meshes."""

from __future__ import annotations

import mujoco
import numpy as np


def rpo_foot_geometry(xml_path: str, feet: list[str]) -> dict:
    model = mujoco.MjModel.from_xml_path(xml_path)
    bounds, names = [], []
    for foot in feet:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot)
        if body < 0:
            raise ValueError("RPO foot body is missing")
        vertices = []
        for geom in np.flatnonzero(model.geom_bodyid == body):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom))
            if not name or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError("RPO foot collision must be an explicitly named frozen mesh")
            mesh = model.geom_dataid[geom]
            offset, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, model.geom_quat[geom])
            points = model.mesh_vert[offset : offset + count] @ matrix.reshape(3, 3).T + model.geom_pos[geom]
            vertices.append(points)
            names.append(name)
        if not vertices:
            raise ValueError("RPO foot has no physical collision geometry")
        points = np.concatenate(vertices)
        bounds.append({"body": foot, "minimum": points.min(0).tolist(), "maximum": points.max(0).tolist()})
    if len(bounds) != 2 or len(set(names)) != len(names):
        raise ValueError("RPO requires two feet with unique physical collision geometry")
    low = np.min([row["minimum"] for row in bounds], axis=0)
    high = np.max([row["maximum"] for row in bounds], axis=0)
    if not np.isfinite([low, high]).all() or (high <= low).any():
        raise ValueError("RPO collision mesh bounds are invalid")
    center = (low + high) / 2
    center[2] = low[2]
    size = np.floor((high - low)[:2] / 0.01) * 0.01
    if (size <= 0).any():
        raise ValueError("RPO foot is smaller than the declared sole grid resolution")
    points = np.stack(
        np.meshgrid(
            np.linspace(low[0], high[0], 10),
            np.linspace(low[1], high[1], 5),
            np.linspace(low[2], high[2], 2),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    return {
        "schema_version": "eb/pbfm-rpo-foot-geometry/v1",
        "bound_strategy": "union_of_collision_mesh_aabbs_in_ankle_body_frames",
        "feet": bounds,
        "collision_geom_names": names,
        "sole_grid_size": size.tolist(),
        "sole_center": center.tolist(),
        "sole_resolution": 0.01,
        "edge_points": points.tolist(),
    }
