import json
from pathlib import Path

from eb.agent import tools as _tools  # noqa: F401
from eb.harness.toolbus import ToolBus

SNAPSHOT = Path(__file__).parent / "matrix_snapshot.json"


def test_matrix_snapshot():
    bus = ToolBus()
    matrix = bus.permission_matrix()
    if not SNAPSHOT.exists():
        SNAPSHOT.write_text(json.dumps(matrix, indent=2, sort_keys=True))
    expected = json.loads(SNAPSHOT.read_text())
    names = {r["name"] for r in matrix}
    assert names == {r["name"] for r in expected}
    for row in matrix:
        if row["name"] in {"deploy.release", "contract.create", "contract.modify", "gate.override"}:
            assert row["effect"] == "forbidden"
