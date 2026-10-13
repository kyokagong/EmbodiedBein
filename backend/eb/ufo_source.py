"""Measured agent-only source identity; refresh only with a verified recipe image."""

SOURCE_RECEIPT = {
    "packages": {
        "gymnasium": "1.2.0",
        "pydantic": "2.11.5",
        "python": "3.11.16",
        "safetensors": "0.6.2",
        "tensordict": "0.11.0",
        "torch": "2.7.0+cu128",
    },
    "patch_hash": "sha256:f99aa821d9ccdf436f91010a7b175209a2bda0f32f40afe8e0a6b49ddaf7d5b5",
    "rlinf": {
        "base_patch_hash": "sha256:df5e404b711161d7068894890d1a90bdcf61f064c8c9bfea60210899ecf787a5",
        "commit": "018e6a9e0da60f96eeda8b11008543911027d414",
        "content_hash": "sha256:dbaa8bed56be412f6d012ee34030822941945786cede6bf9bc3de0fec5445d48",
        "license": "Apache-2.0",
    },
    "schema_version": "eb/ufo-image-source/v1",
    "scope": "agent-only; Python3.10 package metadata bypassed by tested source import",
    "ufo": {
        "commit": "0b89378de02f77712072d983bc43ec2a46455dc0",
        "content_hash": "sha256:2191cf26c77262f7ef2e3525ca23878a8f08c46fd02a20b7776a791f8f5e7582",
        "license": "CC-BY-NC-4.0",
        "scope": "agent Python source; binary assets excluded",
        "tree": "3a6b8b0d6a04287f37c1d7834e51561a8e014fc9",
    },
}

FOUR_GPU_SOURCE_RECEIPT = {**SOURCE_RECEIPT,
    "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23"}
SAFE_SOURCE_RECEIPT = {**FOUR_GPU_SOURCE_RECEIPT,
    "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272"}
STABLE_SOURCE_RECEIPT = {**SAFE_SOURCE_RECEIPT,
    "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77"}
SLEW_SOURCE_RECEIPT = {**STABLE_SOURCE_RECEIPT,
    "action_control_source_sha256": "sha256:1b0bbc0a7a350581ef1939faf93a3e27d824bde62107ecfe4e9ff988bbc088fb",
    "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"}
VELOCITY_SOURCE_RECEIPT = {**SLEW_SOURCE_RECEIPT,
    "auxiliary_profile": "soft_velocity_v5",
    "auxiliary_source_sha256": "sha256:58c495c65bd5831b001832ddc1a73cf0ef79c9437907d2d084c048eeb5a45c33",
    "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a"}
# The evaluator parent does not apply the two training-only Gym/rollout alias patches.
VELOCITY_EVAL_SOURCE_RECEIPT = {key: value for key, value in VELOCITY_SOURCE_RECEIPT.items()
    if key not in {"learning_profile_patch_hash", "rollout_profile_patch_hash"}}
# Measured from the immutable F producer/evaluator; the owned sources agree.
SCALE_SOURCE_RECEIPT = {**VELOCITY_SOURCE_RECEIPT,
    "auxiliary_source_sha256": "sha256:fdc746045470489225ed92f4334043858c3c45875b65e9b9ec13623d128d7dc7",
    "scaling_profile": "env6144_batch2048_v6",
    "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20"}
SCALE_EVAL_SOURCE_RECEIPT = {key: value for key, value in SCALE_SOURCE_RECEIPT.items()
    if key not in {"learning_profile_patch_hash", "rollout_profile_patch_hash"}}
REACHABLE_SOURCE_RECEIPT = {**SCALE_SOURCE_RECEIPT,
    "auxiliary_profile": "soft_velocity_v7", "control_profile": "reachable_v2",
    "optimizer_profile": "stabilized_v7",
    "auxiliary_source_sha256": "sha256:4612307d49eacdfab44e7bea5981daa1c4634b0ead9e6b51626f6603f88e6938",
    "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
    "ufo": {**SOURCE_RECEIPT["ufo"],
        "upstream_content_hash": SOURCE_RECEIPT["ufo"]["content_hash"],
        "content_hash": "sha256:2ec3a9015dda0fd4cfe6861e6ebb35fbf55c78bf43c22544a82ab2368a222e84",
        "scope": "pinned native agent plus measured v7 regularization/reward bounds"}}
CONTACT_SOURCE_RECEIPT = {**REACHABLE_SOURCE_RECEIPT, "contact_profile": "contact_stable_v8",
    "auxiliary_source_sha256": "sha256:46f51b0dc291a70d002afeef2f198eda2133c137b5b8658cb4996d05a241760d",
    "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814"}

GROUND_SOURCE_RECEIPT = {**CONTACT_SOURCE_RECEIPT, "ground_profile": "wide_plane_v9",
    "auxiliary_source_sha256": "sha256:8359d881e7a3523a740b10fb01f4ee53e70a56ecaeb0f31ece3d0fb2e94c03d5", "ground_patch_hash": "sha256:554a18609c9b8a420811fa6b1ded908a4bdf181e52e8d513c8772de6084eff59"}

TRACKING_CONTACT_SOURCE_RECEIPT = {**GROUND_SOURCE_RECEIPT,"context_profile":"environment_tracking_v10",
    "auxiliary_source_sha256":"sha256:8a70883d9df2c2e857a012a2668fd2271c2168ef18630e7f04e3dd8fc245f297",
    "tracking_contact_patch_hash":"sha256:c86825c7962d1078a8ab575db1e53198c35818324e366b7626496d569bd9675f",
    "action_control_source_sha256":"sha256:0cad3bac9e3efb3f1f37d6cac873f789548143b9cd597ed90a93a3045339cd24"}
TRACKING_CONTACT_KD_SOURCE_RECEIPT = {**TRACKING_CONTACT_SOURCE_RECEIPT,"ankle_control_profile":"kd25",
    "auxiliary_source_sha256":"sha256:dc57a21a994cb639d3833b5fa3cb0eefa8630721795227eae2ad0d7aa4e20445",
    "tracking_contact_patch_hash":"sha256:a0db175687641801ae1a6b755ad5dae2258a0af16cc6db71da3ebcaa6f43e99d"}
COMPLIANCE_SOURCE_RECEIPT = {**TRACKING_CONTACT_KD_SOURCE_RECEIPT,
'auxiliary_source_sha256': 'sha256:ce37467a97a287c6c63825b53b69906d16ee3a79c375914ef7049105cd1a5249',
 'dynamics_profile': 'contact_compliance_v11',
 'tracking_contact_patch_hash': 'sha256:d9b075574bf38e19cd74b39db52edf92573a019ba24d06e08fb82861f07dbf15'}

MJLAB_SOURCE_RECEIPT = {'action_control_profile_patch_hash': 'sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d',
 'action_control_source_sha256': 'sha256:0cad3bac9e3efb3f1f37d6cac873f789548143b9cd597ed90a93a3045339cd24',
 'auxiliary_profile': 'soft_velocity_v7',
 'auxiliary_profile_patch_hash': 'sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a',
 'auxiliary_source_sha256': 'sha256:0bdd6b97862fe21dea301b957f55ad0c25b62e07a489675efb7eb4705b7e05f2',
 'contact_patch_hash': 'sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814',
 'contact_profile': 'native_mjcf_v12',
 'context_profile': 'environment_tracking_v10',
 'control_profile': 'reachable_v2',
 'ground_patch_hash': 'sha256:554a18609c9b8a420811fa6b1ded908a4bdf181e52e8d513c8772de6084eff59',
 'learning_profile_patch_hash': 'sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272',
 'mjlab_patch_hash': 'sha256:3b06c0e0883c415c910fee686422f7c25827b047ec37528083ddf87b0a6a8eb4',
 'optimizer_profile': 'stabilized_v7',
 'packages': {'gymnasium': '1.2.0',
              'mjlab': '1.4.0',
              'mujoco': '3.8.0',
              'mujoco-warp': '3.8.0.3',
              'pydantic': '2.11.5',
              'python': '3.11.16',
              'safetensors': '0.6.2',
              'tensordict': '0.11.0',
              'torch': '2.7.0+cu128',
              'warp-lang': '1.12.0'},
 'patch_hash': 'sha256:f99aa821d9ccdf436f91010a7b175209a2bda0f32f40afe8e0a6b49ddaf7d5b5',
 'physics_profile': 'native_mjcf_v12',
 'rlinf': {'base_patch_hash': 'sha256:df5e404b711161d7068894890d1a90bdcf61f064c8c9bfea60210899ecf787a5',
           'commit': '018e6a9e0da60f96eeda8b11008543911027d414',
           'content_hash': 'sha256:dbaa8bed56be412f6d012ee34030822941945786cede6bf9bc3de0fec5445d48',
           'license': 'Apache-2.0'},
 'rollout_profile_patch_hash': 'sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77',
 'scaling_profile': 'env6144_batch2048_v6',
 'scaling_profile_patch_hash': 'sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20',
 'schema_version': 'eb/ufo-image-source/v1',
 'scope': 'native UFO agent + platform RPO MJCF/MJLab bridge; tested '
          'Python3.11 import',
 'simulator': 'mjlab',
 'stabilization_patch_hash': 'sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1',
 'topology_patch_hash': 'sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23',
 'tracking_contact_patch_hash': 'sha256:d9b075574bf38e19cd74b39db52edf92573a019ba24d06e08fb82861f07dbf15',
 'ufo': {'commit': '0b89378de02f77712072d983bc43ec2a46455dc0',
         'content_hash': 'sha256:2ec3a9015dda0fd4cfe6861e6ebb35fbf55c78bf43c22544a82ab2368a222e84',
         'license': 'CC-BY-NC-4.0',
         'scope': 'pinned native agent plus measured v7 regularization/reward '
                  'bounds',
         'tree': '3a6b8b0d6a04287f37c1d7834e51561a8e014fc9',
         'upstream_content_hash': 'sha256:2191cf26c77262f7ef2e3525ca23878a8f08c46fd02a20b7776a791f8f5e7582'}}

MJLAB_VIABILITY_SOURCE_RECEIPT = {'action_control_profile_patch_hash': 'sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d',
 'action_control_source_sha256': 'sha256:0cad3bac9e3efb3f1f37d6cac873f789548143b9cd597ed90a93a3045339cd24',
 'auxiliary_profile': 'soft_velocity_v7',
 'auxiliary_profile_patch_hash': 'sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a',
 'auxiliary_source_sha256': 'sha256:6cbb53ec59105284b00f72f3b752fdae8e098361ef5a5959936aefb34ba75a37',
 'contact_patch_hash': 'sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814',
 'contact_profile': 'native_mjcf_viability_v13',
 'context_profile': 'environment_tracking_v10',
 'control_profile': 'reachable_v2',
 'ground_patch_hash': 'sha256:554a18609c9b8a420811fa6b1ded908a4bdf181e52e8d513c8772de6084eff59',
 'learning_profile_patch_hash': 'sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272',
 'mjlab_patch_hash': 'sha256:3b06c0e0883c415c910fee686422f7c25827b047ec37528083ddf87b0a6a8eb4',
 'optimizer_profile': 'stabilized_v7',
 'packages': {'gymnasium': '1.2.0',
              'mjlab': '1.4.0',
              'mujoco': '3.8.0',
              'mujoco-warp': '3.8.0.3',
              'pydantic': '2.11.5',
              'python': '3.11.16',
              'safetensors': '0.6.2',
              'tensordict': '0.11.0',
              'torch': '2.7.0+cu128',
              'warp-lang': '1.12.0'},
 'patch_hash': 'sha256:f99aa821d9ccdf436f91010a7b175209a2bda0f32f40afe8e0a6b49ddaf7d5b5',
 'physics_profile': 'native_mjcf_viability_v13',
 'rlinf': {'base_patch_hash': 'sha256:df5e404b711161d7068894890d1a90bdcf61f064c8c9bfea60210899ecf787a5',
           'commit': '018e6a9e0da60f96eeda8b11008543911027d414',
           'content_hash': 'sha256:dbaa8bed56be412f6d012ee34030822941945786cede6bf9bc3de0fec5445d48',
           'license': 'Apache-2.0'},
 'rollout_profile_patch_hash': 'sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77',
 'scaling_profile': 'env6144_batch2048_v6',
 'scaling_profile_patch_hash': 'sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20',
 'schema_version': 'eb/ufo-image-source/v1',
 'scope': 'native UFO agent + RPO MJLab dynamic RSI and training-only position '
          'tolerance',
 'simulator': 'mjlab',
 'stabilization_patch_hash': 'sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1',
 'topology_patch_hash': 'sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23',
 'tracking_contact_patch_hash': 'sha256:d9b075574bf38e19cd74b39db52edf92573a019ba24d06e08fb82861f07dbf15',
 'ufo': {'commit': '0b89378de02f77712072d983bc43ec2a46455dc0',
         'content_hash': 'sha256:2ec3a9015dda0fd4cfe6861e6ebb35fbf55c78bf43c22544a82ab2368a222e84',
         'license': 'CC-BY-NC-4.0',
         'scope': 'pinned native agent plus measured v7 regularization/reward '
                  'bounds',
         'tree': '3a6b8b0d6a04287f37c1d7834e51561a8e014fc9',
         'upstream_content_hash': 'sha256:2191cf26c77262f7ef2e3525ca23878a8f08c46fd02a20b7776a791f8f5e7582'},
 'viability_patch_hash': 'sha256:46cd010c4c1c1353bdd943b4dbbb4b0aed44794ba1e4a29d400d9416ee9dec0d'}
