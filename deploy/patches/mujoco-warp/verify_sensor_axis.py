"""CPU-only compile and numerical checks for the frozen UNKNOWN axis correction."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import warp as wp
from mujoco_warp._src import sensor
from mujoco_warp._src.types import ObjType


@wp.kernel
def frame_axes(
    xmat: wp.array2d[wp.mat33],
    ximat: wp.array2d[wp.mat33],
    geom: wp.array2d[wp.mat33],
    site: wp.array2d[wp.mat33],
    camera: wp.array2d[wp.mat33],
    objtypes: wp.array[int],
    out: wp.array2d[wp.vec3],
):
    case, axis = wp.tid()
    out[case, axis] = sensor._frame_axis(
        xmat, ximat, geom, site, camera, 0, 0, objtypes[case], -1, 0, axis
    )


def main():
    wp.init()
    wp.load_module(module=sensor.__name__, device="cpu")
    matrices = []
    for angle in (0.1, 0.3, 0.5, 0.7, 0.9):
        cosine, sine = np.cos(angle), np.sin(angle)
        matrices.append(np.array([
            [cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1],
        ], dtype=np.float32))
    arrays = [wp.array(value.reshape(1, 1, 3, 3), dtype=wp.mat33, device="cpu") for value in matrices]
    kinds = [ObjType.BODY, ObjType.XBODY, ObjType.GEOM, ObjType.SITE, ObjType.CAMERA]
    objtypes = wp.array([int(kind.value) for kind in kinds] + [-1], dtype=int, device="cpu")
    out = wp.zeros((6, 3), dtype=wp.vec3, device="cpu")
    wp.launch(frame_axes, (6, 3), inputs=[*arrays, objtypes, out], device="cpu")
    expected = np.stack([
        matrices[1].T, matrices[0].T, matrices[2].T, matrices[3].T, matrices[4].T,
        np.eye(3, dtype=np.float32),
    ])
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-7, rtol=0)
    print("SENSOR_CPU_COMPILE_AND_18_FRAME_AXES_VERIFIED")


if __name__ == "__main__":
    main()
