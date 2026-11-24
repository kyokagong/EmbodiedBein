"""Target-image CPU regression for native robot ${len:...} interpolation."""

from omegaconf import OmegaConf
from omegaconf.errors import UnsupportedInterpolationType


def main():
    OmegaConf.clear_resolver("len")
    config = OmegaConf.create({
        "robot": {"key_bodies": ["base", "left_foot", "right_foot"],
                  "num_key_bodies": "${len:${robot.key_bodies}}"},
    })
    try:
        OmegaConf.to_container(config, resolve=True)
    except UnsupportedInterpolationType:
        print("REPRODUCED_NATIVE_LEN_RESOLVER_MISSING", flush=True)
    else:
        raise AssertionError("unregistered native len resolver should fail")

    from eb.runners.rlinf_pbfm_runtime import resolved_environment_config

    resolved = resolved_environment_config(config)
    assert resolved["robot"]["num_key_bodies"] == 3
    assert resolved["robot"]["key_bodies"] == ["base", "left_foot", "right_foot"]
    OmegaConf.register_new_resolver("len", lambda value: 99, replace=True)
    assert resolved_environment_config(config)["robot"]["num_key_bodies"] == 99
    OmegaConf.clear_resolver("len")
    unknown = OmegaConf.create({"invalid": "${unknown_pbfm_resolver:3}"})
    try:
        resolved_environment_config(unknown)
    except UnsupportedInterpolationType:
        pass
    else:
        raise AssertionError("unknown resolvers must remain a hard error")
    print("STRICT_NATIVE_LEN_RESOLUTION_AND_UNKNOWN_REJECTION_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
