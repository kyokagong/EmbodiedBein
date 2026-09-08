from eb.domain.urdf import iter_mesh_filenames, missing_joints, next_clone_id, parse_urdf_joints, resolve_mesh_rel


def test_next_clone_id_increments_existing_version():
    assert next_clone_id("ee_bhl_biped_12dof_v1", set()) == "ee_bhl_biped_12dof_v2"
    taken = {"ee_bhl_biped_12dof_v2", "ee_bhl_biped_12dof_v3"}
    assert next_clone_id("ee_bhl_biped_12dof_v1", taken) == "ee_bhl_biped_12dof_v4"
    assert next_clone_id("ee_bhl_biped_12dof_v2", taken) == "ee_bhl_biped_12dof_v4"


def test_resolve_mesh_rel_package_parent():
    rel = resolve_mesh_rel(
        "source/berkeley_humanoid_lite_assets/data/robots/berkeley_humanoid/berkeley_humanoid_lite/urdf/berkeley_humanoid_lite.urdf",
        "package://../meshes/base_visual.stl",
    )
    assert rel.endswith("berkeley_humanoid_lite/meshes/base_visual.stl")
    assert ".." not in rel.split("/")


def test_bhl_seed_urdf_meshes_resolve_on_disk():
    from pathlib import Path

    from eb.domain.embodiment import BHL_ASSET_INDEX
    from eb.domain.urdf import collect_urdf_joints

    root = Path("/data/kk/Berkeley-Humanoid-Lite")
    urdf = root / BHL_ASSET_INDEX[0]["path"]
    if not urdf.is_file():
        return
    names, rel = collect_urdf_joints(root, [str(BHL_ASSET_INDEX[0]["path"])])
    assert names
    assert rel
    text = (root / rel).read_text(errors="replace")
    missing = [fn for fn in iter_mesh_filenames(text) if not (root / resolve_mesh_rel(rel, fn)).is_file()]
    assert missing == []


def test_iter_mesh_filenames_reads_package_hrefs():
    names = iter_mesh_filenames(
        '<mesh filename="package://../meshes/base_visual.stl"/>'
        "<mesh filename='leg_left_hip_roll_visual.stl'/>"
    )
    assert names == ["package://../meshes/base_visual.stl", "leg_left_hip_roll_visual.stl"]


def test_parse_urdf_and_subset():
    urdf = """
    <robot>
      <joint name="leg_left_hip_roll_joint" type="revolute">
        <parent link="base"/>
        <child link="left_hip"/>
      </joint>
      <joint name="arm_left_shoulder_joint" type="revolute">
        <parent link="base"/>
        <child link="left_arm"/>
      </joint>
    </robot>
    """
    names = [j["name"] for j in parse_urdf_joints(urdf)]
    assert "leg_left_hip_roll_joint" in names
    assert missing_joints(["leg_left_hip_roll_joint"], names) == []
    assert missing_joints(["ghost_joint"], names) == ["ghost_joint"]
