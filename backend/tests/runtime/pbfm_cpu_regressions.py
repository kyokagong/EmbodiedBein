"""Run the native source regressions in the exact task image, without GPUs."""

import json
import os
import sys
from pathlib import Path

import pytest

from eb.runners.pbfm_source import prepare_pbfm_source


def main():
    root = Path("/tmp/eb-pbfm-cpu-regressions")
    adaptation = prepare_pbfm_source(Path("/sources/pbfm"), root)
    sys.path.insert(0, str(root))
    os.chdir(root)
    print(
        json.dumps({"scope": "cpu_source_regressions", "patch_sha256": adaptation["patch_sha256"]}),
        flush=True,
    )
    return pytest.main(
        [
            "-q",
            "-rs",
            "-p",
            "no:cacheprovider",
            "tests/test_robot_config_training.py",
            "tests/test_motor_action_delay.py",
            "tests/test_depth_maxcrop_profile.py",
            "tests/test_sparse_successors.py",
            "tests/test_reset_transitions.py",
            str(Path(__file__).with_name("test_pbfm_state.py")),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
