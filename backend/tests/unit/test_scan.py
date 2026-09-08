from pathlib import Path

from eb.domain.scan import scan_embodiment
from eb.domain.urdf import parse_urdf_joint_tree


def test_parse_urdf_joint_tree_reads_limits():
    urdf = """
    <robot>
      <joint name="leg_pitch_joint" type="revolute">
        <parent link="base_link"/>
        <child link="leg_pitch_link"/>
        <origin xyz="0.0 -0.08 -0.15" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.5708" upper="1.5708" effort="80.0" velocity="12.0"/>
      </joint>
    </robot>
    """
    joints = parse_urdf_joint_tree(urdf)
    assert joints[0]["name"] == "leg_pitch_joint"
    assert joints[0]["type"] == "revolute"
    assert joints[0]["parent"] == "base_link"
    assert abs(joints[0]["limit"]["lower"] + 1.5708) < 1e-6
    assert abs(joints[0]["limit"]["upper"] - 1.5708) < 1e-6


def test_scan_finds_urdfs_and_gym_ids(tmp_path: Path):
    (tmp_path / "robot.urdf").write_text(
        '<robot><joint name="j1" type="revolute"><parent link="a"/><child link="b"/></joint></robot>'
    )
    (tmp_path / "env").mkdir()
    (tmp_path / "env" / "isaaclab_demo.yaml").write_text("id: demo\n")
    (tmp_path / "register.py").write_text('gym.register(id="Isaac-Demo-v0", entry_point="x:Y")\n')
    result = scan_embodiment(tmp_path)
    assert result["status"] == "ready"
    assert any(v["path"].endswith("robot.urdf") for v in result["urdf_variants"])
    assert "Isaac-Demo-v0" in result["gym_ids"]
    assert any("isaaclab_demo.yaml" in p for p in result["rlinf_envs"])
