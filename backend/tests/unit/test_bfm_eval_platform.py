import hashlib
import json
from types import SimpleNamespace
import pytest


def test_bfm_eval_freezes_native_sources_and_independent_case_plan(tmp_path,monkeypatch):
    from eb.adapters.rlinf import RlinfAdapter
    from eb.domain.job import JobKind,JobSpec
    from eb.rpo_bfm import ENV_ID,TASK_ID,EVAL_TEMPLATE,EVAL_SUITE,CODEC_HASH,eval_suite_spec
    checkpoint=tmp_path/"models/runs/run_abcdef"
    (checkpoint/"model_state_dict").mkdir(parents=True)
    (checkpoint/"model_state_dict/full_weights.pt").write_bytes(b"weights")
    config=tmp_path/"runs/run_abcdef/output/resolved-config.yaml"
    config.parent.mkdir(parents=True);config.write_text("actor: {}")
    motion=tmp_path/"datasets/ds_abcdef/content"
    motion.mkdir(parents=True);(motion/"manifest.json").write_text("{}")
    assets=tmp_path/"roboparty/rpo-v2"
    assets.mkdir(parents=True);(assets/"asset-manifest.json").write_text("{}")
    sha=hashlib.sha256(b"{}").hexdigest()
    settings=SimpleNamespace(data_root=tmp_path,assets_root=tmp_path,models_root=tmp_path/"models",
        datasets_root=tmp_path/"datasets",allowed_path_prefixes=[])
    monkeypatch.setattr("eb.settings.get_settings",lambda:settings)
    monkeypatch.setattr("eb.adapters.rlinf.get_settings",lambda:settings)
    monkeypatch.setattr("eb.adapters.isaaclab_rslrl.get_settings",lambda:settings)
    plan=dict(schema_version="eb/rpo-bfm-eval-plan/v1",suite_id=EVAL_SUITE,suite_spec=eval_suite_spec(),
        codec_sha256=CODEC_HASH,profile="smoke",diagnostic_commands=True,record_video=True,
        source_run_id="run_abcdef",asset_sha256=sha,motion_manifest_sha256=sha,
        source_weights_sha256="a"*64,source_config_sha256="b"*64,image_digest="sha256:"+"c"*64,
        seeds=list(range(60000,60004)))
    plan["hash"]="sha256:"+hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    spec=JobSpec(kind=JobKind.EVAL_SIM,embodiment_env_id=ENV_ID,task_id=TASK_ID,image=plan["image_digest"],
        inputs=dict(base_model=str(checkpoint),datasets=[dict(path=str(motion),validation=dict(manifest_sha256=sha))]),
        params=dict(template_id=EVAL_TEMPLATE,rpo_eval_plan=plan,seeds=plan["seeds"],record_video=True),resources=dict(gpus=1))
    rendered=RlinfAdapter().render(spec,"run_test")
    assert rendered.image==plan["image_digest"]
    assert rendered.command[1].endswith("rpo_bfm_eval.py") and "export" not in rendered.command
    assert "--diagnostic-commands" in rendered.command and "--record-video" in rendered.command
    assert rendered.command[rendered.command.index("--output")+1]=="/workspace/output"
    assert all(mode=="ro" for _,dest,mode in rendered.mounts if dest!="/workspace/output")
    for key,value in (("seeds",[42]),("motion_manifest_sha256","d"*64),("suite_spec",{})):
        bad=spec.model_copy(deep=True)
        bad.params["rpo_eval_plan"][key]=value
        with pytest.raises(ValueError):RlinfAdapter().render(bad,"run_changed")
    from eb.rpo_bfm import EVAL_BATCH_SUITE,batch_suite_spec
    parallel=spec.model_copy(deep=True)
    batch_plan=parallel.params["rpo_eval_plan"]
    batch_plan.update(suite_id=EVAL_BATCH_SUITE,suite_spec=batch_suite_spec(),
        parallelism=dict(gpus=4,envs_per_gpu=4,protocol="independent_cases_shared_terrain_v1"))
    batch_plan["hash"]="sha256:"+hashlib.sha256(json.dumps({k:v for k,v in batch_plan.items() if k!="hash"},sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    parallel.resources.gpus=4
    rendered=RlinfAdapter().render(parallel,"run_parallel")
    assert rendered.command[1].endswith("rpo_bfm_parallel_eval.py")
    assert rendered.command[rendered.command.index("--eval-gpus")+1]=="4"
    assert rendered.command[rendered.command.index("--envs-per-gpu")+1]=="4"
    parallel.resources.gpus=1
    with pytest.raises(ValueError):RlinfAdapter().render(parallel,"run_mismatch")
    from eb.rpo_bfm import VALIDATION_SUITE, eval_case_seeds, frozen_eval_suite_spec
    parallel.resources.gpus=4
    batch_plan.update(suite_id=VALIDATION_SUITE, profile="validation", diagnostic_commands=False,
        suite_spec=frozen_eval_suite_spec(VALIDATION_SUITE, "validation"), seeds=eval_case_seeds(VALIDATION_SUITE, "validation"))
    parallel.params["seeds"] = batch_plan["seeds"]
    batch_plan["hash"]="sha256:"+hashlib.sha256(json.dumps({k:v for k,v in batch_plan.items() if k!="hash"},sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    rendered=RlinfAdapter().render(parallel,"run_validation")
    assert "--diagnostic-commands" not in rendered.command
    assert rendered.command[rendered.command.index("--seeds")+1] == ",".join(map(str, range(61000, 61016)))


def test_bfm_validation_and_acceptance_have_disjoint_frozen_seeds():
    from eb.rpo_bfm import EVAL_BATCH_SUITE, VALIDATION_SUITE, ACCEPTANCE_SUITE, eval_case_seeds, frozen_eval_suite_spec
    validation = eval_case_seeds(VALIDATION_SUITE, "validation")
    acceptance = eval_case_seeds(ACCEPTANCE_SUITE, "acceptance")
    assert validation == list(range(61000, 61016))
    assert acceptance == list(range(62000, 62100))
    assert set(validation).isdisjoint(acceptance)
    assert set(validation + acceptance).isdisjoint(range(50000, 50100))
    assert frozen_eval_suite_spec(VALIDATION_SUITE, "validation")["evaluation_role"] == "validation"
    assert frozen_eval_suite_spec(VALIDATION_SUITE, "validation")["required_video"] is True
    for suite, profile in ((VALIDATION_SUITE, "standard"), (ACCEPTANCE_SUITE, "validation"), (EVAL_BATCH_SUITE, "acceptance")):
        with pytest.raises(ValueError): eval_case_seeds(suite, profile)


def test_repair_acceptance_has_new_seeds_and_identical_physical_criteria():
    from eb.rpo_bfm import UFO_ACCEPTANCE_SUITE, UFO_REPAIR_ACCEPTANCE_SUITE, UFO_VALIDATION_SUITE, eval_case_seeds, frozen_eval_suite_spec
    new = eval_case_seeds(UFO_REPAIR_ACCEPTANCE_SUITE, "acceptance")
    assert new == list(range(66000, 66100))
    assert set(new).isdisjoint(eval_case_seeds(UFO_ACCEPTANCE_SUITE, "acceptance"))
    assert set(new).isdisjoint(eval_case_seeds(UFO_VALIDATION_SUITE, "validation"))
    assert frozen_eval_suite_spec(UFO_REPAIR_ACCEPTANCE_SUITE, "acceptance") == frozen_eval_suite_spec(UFO_ACCEPTANCE_SUITE, "acceptance")
    with pytest.raises(ValueError):
        eval_case_seeds(UFO_REPAIR_ACCEPTANCE_SUITE, "validation")


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["validation", "acceptance"])
async def test_new_bfm_evaluations_require_videos_at_api_and_collector(tmp_path, profile):
    from unittest.mock import AsyncMock
    from eb.api.routes_eval import RpoEvalIn, _prepare_rpo
    from eb.rpo_bfm import VALIDATION_SUITE, ACCEPTANCE_SUITE
    from eb.rpo_flat import validate_flat_eval
    session = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(card={"schema_version":"eb/rpo-bfm-model/v1"})))
    with pytest.raises(ValueError, match="videos"):
        await _prepare_rpo(RpoEvalIn(model_version_id="mv_test", profile=profile), session)
    with pytest.raises(ValueError, match="videos"):
        validate_flat_eval(tmp_path, dict(suite_id=VALIDATION_SUITE if profile=="validation" else ACCEPTANCE_SUITE,
                                         profile=profile, record_video=False), "run_test", bfm=True)


def test_native_full_actor_evidence_has_a_bounded_export_budget(tmp_path):
    from eb.domain.training_recipe import RECIPES, UFO_STABLE_RECIPE
    from eb.onnx_io import sha256_file
    from eb.rpo_bfm import CODEC_HASH, UFO_VALIDATION_SUITE, eval_case_seeds, frozen_eval_suite_spec
    from eb.rpo_flat import ROBOLAB_REVISION, validate_flat_eval

    def evidence(name, size=1):
        path = tmp_path / "exported" / name
        path.parent.mkdir(exist_ok=True)
        with path.open("wb") as file:
            file.truncate(size)
        return {"path": name, "sha256": sha256_file(path)}

    suite, profile = UFO_VALIDATION_SUITE, "validation"
    plan = dict(schema_version="eb/rpo-bfm-eval-plan/v1", suite_id=suite, profile=profile,
        suite_spec=frozen_eval_suite_spec(suite, profile), seeds=eval_case_seeds(suite, profile),
        record_video=True, checkpoint_step=24, execution=RECIPES[UFO_STABLE_RECIPE].binding.model_dump())
    plan["hash"] = "sha256:" + hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    command = {**evidence("command-to-latent.onnx"), "reference_split": "train", "command_dim": 3, "latent_dim": 256}
    onnx = {**evidence("policy.onnx", 128 * 1024 * 1024), "obs": 375, "act": 23,
            "max_joint_target_error_rad": 0, "latent": 256, "normalization": "embedded"}
    exported = dict(schema_version="eb/rpo-bfm-export/v1", status="verified", onnx=onnx,
        codec_sha256=CODEC_HASH, command_latent=command, execution=plan["execution"],
        learner_joint_updates=24, checkpoint_format="ufo_agent_state_v1",
        golden=evidence("golden-traces.npz"), prompts=evidence("prompts.npz"))
    receipt = dict(schema_version="eb/rpo-bfm-eval/v1", run_id="run_test", plan_hash=plan["hash"],
        seeds=plan["seeds"], backend="isaac", criterion="velocity_commands", required_steps=1000,
        official_revision=ROBOLAB_REVISION, record_video=True, velocity_frame="yaw_aligned",
        yaw_rate_frame="world_z", onnx=onnx, command_latent=command, codec_sha256=CODEC_HASH)
    (tmp_path / "exported/export-manifest.json").write_text(json.dumps(exported))
    (tmp_path / "eval-manifest.json").write_text(json.dumps(receipt))
    (tmp_path / "eval-cases.jsonl").write_text("")
    # Reaching the next receipt check proves the full actor passed size and SHA validation.
    with pytest.raises(ValueError, match="case seeds differ"):
        validate_flat_eval(tmp_path, plan, "run_test", bfm=True)
    with (tmp_path / "exported/policy.onnx").open("wb") as file:
        file.truncate(513 * 1024 * 1024)
    with pytest.raises(ValueError, match="evidence unavailable or unsafe: exported/policy.onnx"):
        validate_flat_eval(tmp_path, plan, "run_test", bfm=True)
