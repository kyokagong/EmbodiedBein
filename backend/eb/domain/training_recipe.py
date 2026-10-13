"""Approved training identities and budgets; no runtime or filesystem access."""

import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from eb.domain.job import JobKind, JobSpec
from eb.ufo_source import FOUR_GPU_SOURCE_RECEIPT, SOURCE_RECEIPT

UFO_ISAAC_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_v1"
UFO_FOUR_GPU_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_v1"
UFO_SAFE_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_safe_v2"
UFO_STABLE_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_stable_v3"
UFO_SLEW_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_v4"
UFO_VELOCITY_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_velocity_v5"
UFO_SCALE_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_scale6144_v6"
UFO_REACHABLE_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7"
UFO_GROUND_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9"
UFO_TRACKING_CONTACT_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_v10"
UFO_LEARNING_DIAGNOSTIC_IMAGE_DIGEST = "sha256:d34c0411f598e657f26786c01f56f468ebe901c89e5087a0ba11c2b4d55dc10a"
UFO_LEARNING_DIAGNOSTIC_V2_IMAGE_DIGEST = "sha256:a48697422731461c66308f5b2b4b3ababbeb28b8f030954fa98637ba018f2ca9"
UFO_LEARNING_VALUE_V3_IMAGE_DIGEST = "sha256:3c0ffc0f41a9a40c79a624e501f60e5befb5527224e26fa063fc599c1b00ff8b"
UFO_LEARNING_DIAGNOSTIC_V4_IMAGE_DIGEST = "sha256:b4d976ea9163a6f57b033572cb3fb78225ec9f480e6b40e2afd3cae0ddf0e455"
UFO_DYNAMICS_DIAGNOSTIC_IMAGE_DIGEST = "sha256:399a9a9ef0f03a36725c6bb35b7c2df8fc8a0346f1360e3d35dab239b21f7a98"
UFO_DYNAMICS_FINALIZER_IMAGE_DIGEST = "sha256:66b8b7ea6665ee9d5713c020ffc1c5185829cbfc5929ef4bf98271d1588ab894"
UFO_CONTACT_FACTOR_IMAGE_DIGEST = "sha256:2abedffa36778c6cf4d8ef4b07293d9db5ebd35210f0105e501008d8e3809e22"
UFO_INTEGRATION_FACTOR_IMAGE_DIGEST = "sha256:564e13b45162d06561f9ddf105c65730a18735a2db8f47f7ba5ce0f786251b9d"
UFO_ACTION_REPLAY_IMAGE_DIGEST = "sha256:67085ed631fa00c912c0f36841d21a5d143c05438834f3bbf10928ca507e598a"
UFO_SOLVER_CONE_IMAGE_DIGEST = "sha256:b3b5846081daad2ff5060d3229279665a4294af3988e1af03f0afd7c9b807657"
UFO_CONTROL_DAMPING_IMAGE_DIGEST = "sha256:9d326d40074efaa5b77cb77aaa2e32017205944297b85c38d20b0e5ef45ca4b3"
UFO_FRICTION_IMPEDANCE_IMAGE_DIGEST = "sha256:e8fae50c4c2d5cd7b35d36cb826b2c009344144e25f75bcc245892e204fc13fa"
UFO_GROUND_STIFFNESS_IMAGE_DIGEST = "sha256:ffe631ac28292746b1e883f6e9b7d9e66a16e874cd623df47665fcf49d04432d"
UFO_CONTACT_MARGIN_IMAGE_DIGEST = "sha256:91659e6d06263baf49619c4aa0ae737e2450776459b769b0cdb5f068a1aec397"
UFO_TRACKING_CONTACT_KD_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10"
UFO_COMPLIANCE_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_compliance_v11"
UFO_MJLAB_RECIPE = "rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_v12"
UFO_MJLAB_VIABILITY_RECIPE = "rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_viability_v13"
UFO_MJLAB_VIABILITY_IMAGE_DIGEST = 'sha256:e23281518d388565f4b1162dc9d2f21947543b2fd0a014b0288365be2788cee2'
UFO_MJLAB_IMAGE_DIGEST = 'sha256:071d5aacb6660f0bb48144fa2c6c0af34a8cfff9ef90819909dd4d2b4b63c986'
UFO_COMPLIANCE_IMAGE_DIGEST = "sha256:6a709e3a277f19675e7ef464967e37f2ab77eae56388fc8dc3b4f083a62223ff"
UFO_TRACKING_CONTACT_KD_IMAGE_DIGEST = "sha256:1970d0dc894b0a80c5dc27bd6a5b94c8d8463c52f3ec5c3e8108dda0b1b073f9"
UFO_TRACKING_CONTACT_IMAGE_DIGEST = "sha256:c2a876c23450b6d1cde3c4b89c889b860cb5831b406c6479949530b00d0ab8e2"
UFO_GROUND_IMAGE_DIGEST = "sha256:1034a824a3782202b914275344f6c0985d0f1721fb423e5620b5a964dc16672e"
UFO_CONTACT_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8"
UFO_CONTACT_V8_IMAGE_DIGEST = "sha256:288c9bc690f0b0cefce4529b9ae07d3d0c9163b201d08d228161b7818685b5e2"
# Updated after measuring the owned v7 image; never used for admission before that.
UFO_REACHABLE_IMAGE_DIGEST = "sha256:2359c60a3f40ef9f6a47b59d9c02194d1aa14afa603789960c5807e988277fb7"
# Replaced only after building and measuring the new immutable images.
UFO_SCALE_IMAGE_DIGEST = "sha256:e643aa62117fc1b5c70195d7b589225bf42621843ddf5218bd1c99895e04ffd7"
UFO_SCALE_EVAL_IMAGE_DIGEST = "sha256:56bdc6720926895843f5426c76dd937abaa24d77a2c8be30c8810c0ae5172161"
UFO_VELOCITY_IMAGE_DIGEST = "sha256:e1382226cd5f0bd3795ee6a187f62555ba77f591e342c862a64bf1825a463dee"
UFO_VELOCITY_EVAL_IMAGE_DIGEST = "sha256:3094f020480ce099ff281ac76c4812bd47985b7d4a489a15a0550c49e6e9fbca"
UFO_SLEW_IMAGE_DIGEST = "sha256:88aee59f8d0f62d9e99affd231b34511f4624a9203bf7f9205c5f2f4008c1fe4"
UFO_SLEW_EVAL_IMAGE_DIGEST = "sha256:9129bc326a1e1d68122bc630dff29e362066c2fdc4f0285eda7f61d211072d99"
UFO_STABLE_IMAGE_DIGEST = "sha256:fac4b89f2821c072d86d6b648afdee98e179d74615a13847df238eb083e6c241"
UFO_STABLE_EVAL_IMAGE_DIGEST = "sha256:86cdf16e1368c8780a190192418c39cef77737a0e72bdd63ba3147575657fa4d"
UFO_CONTACT_IMAGE_DIGEST = "sha256:2f3450683e7d3f71e85b17f0fd494b2a5df5781cdd72483d1e882ca45ead4d91"
UFO_SAFE_IMAGE_DIGEST = "sha256:adbca4ec9cf899e18da848b7e2a47d2df50d1f8e74e9015655b6deac193b0f99"
UFO_FOUR_GPU_IMAGE_DIGEST = "sha256:60fbb44a17c3247f7ed83104466e4a652dd06d8712b22ffb1ea202ea6d03cf2b"
UFO_FOUR_GPU_EVAL_IMAGE_DIGEST = "sha256:c2b36fe55f84880c28382e37801e4689abbfaed04ee507b0d4cbf78ae2d22563"
UFO_TRACKING_IMAGE_DIGEST = "sha256:64c9803986b1a5c836d58fee5017d6a67e8695e022849ed46bb31ace167a4acd"
# Measured task image; refresh after rebuilding the pinned recipe, before capability promotion.
UFO_ISAAC_IMAGE_DIGEST = "sha256:5abec37ade5f55fd043f9aceff9fc8b32ed7ff422b08c80879857bbd745af015"


class ExecutionBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = "eb/training-binding/v1"
    recipe_id: str
    algorithm_id: str
    provider_id: str
    runtime_id: str
    simulator_id: str
    progress_unit: str
    checkpoint_format: str
    topology_id: str


@dataclass(frozen=True)
class TrainingRecipe:
    binding: ExecutionBinding
    image: str
    status: str
    embodiment_env_id: str
    task_id: str


RECIPES = {
    UFO_ISAAC_RECIPE: TrainingRecipe(
        binding=ExecutionBinding(
            recipe_id=UFO_ISAAC_RECIPE,
            algorithm_id="bfm_fb_cpr_aux",
            provider_id="ufo",
            runtime_id="rlinf",
            simulator_id="isaaclab",
            progress_unit="learner_joint_update",
            checkpoint_format="ufo_agent_state_v1",
            topology_id="single_learner",
        ),
        image="eb/rlinf:rpo-bfm-ufo-agent-v1",
        # 16→24 restore, validation and held-out media: see 2026-10-10 GPU ledger.
        status="smoke_passed",
        embodiment_env_id="ee_rpo_v2_bfm_23dof_v1",
        task_id="tsk_rpo_bfm_pretrain",
    ),
    UFO_FOUR_GPU_RECIPE: TrainingRecipe(
        binding=ExecutionBinding(
            recipe_id=UFO_FOUR_GPU_RECIPE,
            algorithm_id="bfm_fb_cpr_aux",
            provider_id="ufo",
            runtime_id="rlinf",
            simulator_id="isaaclab",
            progress_unit="learner_joint_update",
            checkpoint_format="ufo_agent_state_v1",
            topology_id="one_learner_three_samplers",
        ),
        image="eb/rlinf:rpo-bfm-ufo-agent-4gpu-v1",
        # Four-card 16→24 restore and 4×1 case equivalence/media verified in the GPU ledger.
        status="smoke_passed",
        embodiment_env_id="ee_rpo_v2_bfm_23dof_v1",
        task_id="tsk_rpo_bfm_pretrain",
    ),
}

RECIPES[UFO_SAFE_RECIPE] = TrainingRecipe(
    binding=RECIPES[UFO_FOUR_GPU_RECIPE].binding.model_copy(update={"recipe_id": UFO_SAFE_RECIPE}),
    # Safe 16→24 restore, true terminal traces, validation/held-out media and cancel verified.
    image="eb/rlinf:rpo-bfm-ufo-agent-4gpu-safe-v2", status="smoke_passed",
    embodiment_env_id="ee_rpo_v2_bfm_23dof_v1", task_id="tsk_rpo_bfm_pretrain",
)
RECIPES[UFO_STABLE_RECIPE] = TrainingRecipe(
    binding=RECIPES[UFO_FOUR_GPU_RECIPE].binding.model_copy(update={"recipe_id": UFO_STABLE_RECIPE}),
    image="eb/rlinf:rpo-bfm-ufo-agent-4gpu-stable-v3", status="smoke_passed",
    embodiment_env_id="ee_rpo_v2_bfm_23dof_v1", task_id="tsk_rpo_bfm_pretrain",
)
RECIPES[UFO_SLEW_RECIPE] = TrainingRecipe(
    binding=RECIPES[UFO_STABLE_RECIPE].binding.model_copy(update={"recipe_id": UFO_SLEW_RECIPE}),
    # Four-GPU16→24 native restore, controller traces, independent media and cancel verified.
    image="eb/rlinf:rpo-bfm-ufo-agent-4gpu-slew-v4", status="smoke_passed",
    embodiment_env_id="ee_rpo_v2_bfm_23dof_v1", task_id="tsk_rpo_bfm_pretrain",
)
RECIPES[UFO_VELOCITY_RECIPE] = TrainingRecipe(
    binding=RECIPES[UFO_SLEW_RECIPE].binding.model_copy(update={"recipe_id": UFO_VELOCITY_RECIPE}),
    # Immutable source/census, native nine-aux CPU restore/export verified; GPU qualification pending.
    image="eb/rlinf:rpo-bfm-ufo-agent-4gpu-slew-velocity-v5", status="dependency_ready",
    embodiment_env_id="ee_rpo_v2_bfm_23dof_v1", task_id="tsk_rpo_bfm_pretrain",
)
RECIPES[UFO_SCALE_RECIPE] = TrainingRecipe(
    binding=RECIPES[UFO_VELOCITY_RECIPE].binding.model_copy(update={"recipe_id": UFO_SCALE_RECIPE}),
    # Real 16→24, six-Adam/hash restore, 16-case/held-out media and GPU cancel verified.
    image="eb/rlinf:rpo-bfm-ufo-agent-4gpu-scale6144-v6", status="smoke_passed",
    embodiment_env_id="ee_rpo_v2_bfm_23dof_v1", task_id="tsk_rpo_bfm_pretrain",
)
RECIPES[UFO_REACHABLE_RECIPE] = TrainingRecipe(
    binding=RECIPES[UFO_SCALE_RECIPE].binding.model_copy(update={"recipe_id": UFO_REACHABLE_RECIPE}),
    # Real four-card 16→24/six-Adam/hash restore verified; user deferred independent evaluation.
    image="eb/rlinf:rpo-bfm-ufo-agent-4gpu-reachable-v7", status="smoke_passed",
    embodiment_env_id="ee_rpo_v2_bfm_23dof_v2", task_id="tsk_rpo_bfm_pretrain",
)


RECIPES[UFO_CONTACT_RECIPE] = TrainingRecipe(
    binding=RECIPES[UFO_REACHABLE_RECIPE].binding.model_copy(update={"recipe_id": UFO_CONTACT_RECIPE}),
    image="eb/rlinf:rpo-bfm-ufo-agent-4gpu-contact-v8-poststep1", status="smoke_passed",
    embodiment_env_id="ee_rpo_v2_bfm_23dof_v2", task_id="tsk_rpo_bfm_pretrain",
)


RECIPES[UFO_GROUND_RECIPE] = TrainingRecipe(
    binding=RECIPES[UFO_CONTACT_RECIPE].binding.model_copy(update={"recipe_id": UFO_GROUND_RECIPE}),
    # Real four-card 16→24, six-Adam/hash restore and ground coverage; no independent evaluation.
    image="eb/rlinf:rpo-bfm-ufo-agent-4gpu-ground-v9", status="smoke_passed",
    embodiment_env_id="ee_rpo_v2_bfm_23dof_v2", task_id="tsk_rpo_bfm_pretrain",
)

RECIPES[UFO_TRACKING_CONTACT_RECIPE] = TrainingRecipe(
    binding=RECIPES[UFO_GROUND_RECIPE].binding.model_copy(update={
        "recipe_id":UFO_TRACKING_CONTACT_RECIPE,"checkpoint_format":"ufo_agent_state_v2"}),
    # Diagnostic/provisional image retains inherited divergent rollout seeds; use measured KD recipe.
    image="eb/rlinf:rpo-bfm-ufo-tracking-contact-v10", status="blocked",
    embodiment_env_id="ee_rpo_v2_bfm_23dof_v2",task_id="tsk_rpo_bfm_pretrain")
RECIPES[UFO_TRACKING_CONTACT_KD_RECIPE] = TrainingRecipe(
    binding=RECIPES[UFO_TRACKING_CONTACT_RECIPE].binding.model_copy(update={"recipe_id":UFO_TRACKING_CONTACT_KD_RECIPE}),
    # Real 6144-env 16→24 restore, six Adams/file hashes/partition/mean8 reward verified; no independent eval.
    image="eb/rlinf:rpo-bfm-ufo-tracking-contact-kd-v10",status="smoke_passed",
    embodiment_env_id="ee_rpo_v2_bfm_23dof_v2",task_id="tsk_rpo_bfm_pretrain")

RECIPES[UFO_COMPLIANCE_RECIPE] = TrainingRecipe(
    binding=RECIPES[UFO_TRACKING_CONTACT_KD_RECIPE].binding.model_copy(update={'recipe_id':UFO_COMPLIANCE_RECIPE}),
    image='eb/rlinf:rpo-bfm-ufo-compliance-v11',status='smoke_passed',
    embodiment_env_id='ee_rpo_v2_bfm_23dof_v2',task_id='tsk_rpo_bfm_pretrain')

RECIPES[UFO_MJLAB_RECIPE] = TrainingRecipe(
    binding=RECIPES[UFO_TRACKING_CONTACT_KD_RECIPE].binding.model_copy(update={
        "recipe_id": UFO_MJLAB_RECIPE, "simulator_id": "mjlab"}),
    image="eb/rlinf:rpo-bfm-ufo-mjlab-v12", status="smoke_passed",
    embodiment_env_id="ee_rpo_v2_bfm_23dof_v2", task_id="tsk_rpo_bfm_pretrain")

RECIPES[UFO_MJLAB_VIABILITY_RECIPE] = TrainingRecipe(
    binding=RECIPES[UFO_MJLAB_RECIPE].binding.model_copy(update={'recipe_id': UFO_MJLAB_VIABILITY_RECIPE}),
    image='eb/rlinf:rpo-bfm-ufo-mjlab-viability-v13', status='smoke_passed',
    embodiment_env_id='ee_rpo_v2_bfm_23dof_v2', task_id='tsk_rpo_bfm_pretrain')


def four_gpu_recipe(recipe_id):
    return RECIPES[recipe_id].binding.topology_id == "one_learner_three_samplers"


def recipe_image_digest(recipe_id):
    if recipe_id == UFO_MJLAB_VIABILITY_RECIPE:
        return UFO_MJLAB_VIABILITY_IMAGE_DIGEST
    if recipe_id == UFO_MJLAB_RECIPE:
        return UFO_MJLAB_IMAGE_DIGEST
    if recipe_id == UFO_COMPLIANCE_RECIPE:
        return UFO_COMPLIANCE_IMAGE_DIGEST
    if recipe_id == UFO_TRACKING_CONTACT_KD_RECIPE:
        return UFO_TRACKING_CONTACT_KD_IMAGE_DIGEST
    if recipe_id == UFO_TRACKING_CONTACT_RECIPE:
        return UFO_TRACKING_CONTACT_IMAGE_DIGEST
    if recipe_id == UFO_GROUND_RECIPE:
        return UFO_GROUND_IMAGE_DIGEST
    if recipe_id == UFO_CONTACT_RECIPE:
        return UFO_CONTACT_V8_IMAGE_DIGEST
    if recipe_id == UFO_REACHABLE_RECIPE:
        return UFO_REACHABLE_IMAGE_DIGEST
    if recipe_id == UFO_SCALE_RECIPE:
        return UFO_SCALE_IMAGE_DIGEST
    if recipe_id == UFO_VELOCITY_RECIPE:
        return UFO_VELOCITY_IMAGE_DIGEST
    if recipe_id == UFO_SLEW_RECIPE:
        return UFO_SLEW_IMAGE_DIGEST
    if recipe_id == UFO_STABLE_RECIPE:
        return UFO_STABLE_IMAGE_DIGEST
    if recipe_id == UFO_SAFE_RECIPE:
        return UFO_SAFE_IMAGE_DIGEST
    return UFO_FOUR_GPU_IMAGE_DIGEST if recipe_id == UFO_FOUR_GPU_RECIPE else UFO_ISAAC_IMAGE_DIGEST


def recipe_eval_image_digest(recipe_id):
    if recipe_id == UFO_MJLAB_VIABILITY_RECIPE:
        raise ValueError('MJLab independent evaluation requires qualifying training evidence')
    if recipe_id == UFO_MJLAB_RECIPE:
        raise ValueError("MJLab independent evaluation is not qualified; training evidence required")
    if recipe_id in {UFO_COMPLIANCE_RECIPE,UFO_TRACKING_CONTACT_RECIPE,UFO_TRACKING_CONTACT_KD_RECIPE}:
        raise ValueError("Tracking/contact independent evaluation requires qualifying training evidence")
    if recipe_id == UFO_GROUND_RECIPE:
        raise ValueError("Ground v9 independent evaluation is not qualified; training evidence required")
    if recipe_id == UFO_CONTACT_RECIPE:
        raise ValueError("Contact v8 independent evaluation is not qualified; training evidence required")
    if recipe_id == UFO_REACHABLE_RECIPE:
        raise ValueError("Reachable v7 independent evaluation is not qualified; training evidence required")
    if recipe_id == UFO_SCALE_RECIPE:
        return UFO_SCALE_EVAL_IMAGE_DIGEST
    if recipe_id == UFO_VELOCITY_RECIPE:
        return UFO_VELOCITY_EVAL_IMAGE_DIGEST
    if recipe_id == UFO_SLEW_RECIPE:
        return UFO_SLEW_EVAL_IMAGE_DIGEST
    if recipe_id == UFO_STABLE_RECIPE:
        return UFO_STABLE_EVAL_IMAGE_DIGEST
    return UFO_FOUR_GPU_EVAL_IMAGE_DIGEST if four_gpu_recipe(recipe_id) else UFO_ISAAC_IMAGE_DIGEST


def recipe_source_receipt(recipe_id):
    if recipe_id == UFO_MJLAB_VIABILITY_RECIPE:
        from eb.ufo_source import MJLAB_VIABILITY_SOURCE_RECEIPT
        return MJLAB_VIABILITY_SOURCE_RECEIPT
    if recipe_id == UFO_MJLAB_RECIPE:
        from eb.ufo_source import MJLAB_SOURCE_RECEIPT
        return MJLAB_SOURCE_RECEIPT
    if recipe_id == UFO_COMPLIANCE_RECIPE:
        from eb.ufo_source import COMPLIANCE_SOURCE_RECEIPT
        return COMPLIANCE_SOURCE_RECEIPT
    if recipe_id == UFO_TRACKING_CONTACT_KD_RECIPE:
        from eb.ufo_source import TRACKING_CONTACT_KD_SOURCE_RECEIPT
        return TRACKING_CONTACT_KD_SOURCE_RECEIPT
    if recipe_id == UFO_TRACKING_CONTACT_RECIPE:
        from eb.ufo_source import TRACKING_CONTACT_SOURCE_RECEIPT
        return TRACKING_CONTACT_SOURCE_RECEIPT
    if recipe_id == UFO_GROUND_RECIPE:
        from eb.ufo_source import GROUND_SOURCE_RECEIPT
        return GROUND_SOURCE_RECEIPT
    if recipe_id == UFO_CONTACT_RECIPE:
        from eb.ufo_source import CONTACT_SOURCE_RECEIPT
        return CONTACT_SOURCE_RECEIPT
    if recipe_id == UFO_REACHABLE_RECIPE:
        from eb.ufo_source import REACHABLE_SOURCE_RECEIPT
        return REACHABLE_SOURCE_RECEIPT
    if recipe_id == UFO_SCALE_RECIPE:
        from eb.ufo_source import SCALE_SOURCE_RECEIPT
        return SCALE_SOURCE_RECEIPT
    if recipe_id == UFO_VELOCITY_RECIPE:
        from eb.ufo_source import VELOCITY_SOURCE_RECEIPT
        return VELOCITY_SOURCE_RECEIPT
    if recipe_id == UFO_SLEW_RECIPE:
        from eb.ufo_source import SLEW_SOURCE_RECEIPT
        return SLEW_SOURCE_RECEIPT
    if recipe_id == UFO_STABLE_RECIPE:
        from eb.ufo_source import STABLE_SOURCE_RECEIPT
        return STABLE_SOURCE_RECEIPT
    if recipe_id == UFO_SAFE_RECIPE:
        from eb.ufo_source import SAFE_SOURCE_RECEIPT
        return SAFE_SOURCE_RECEIPT
    return FOUR_GPU_SOURCE_RECEIPT if four_gpu_recipe(recipe_id) else SOURCE_RECEIPT


def recipe_resources(recipe_id):
    four = four_gpu_recipe(recipe_id)
    return {
        "gpus": 4 if four else 1,
        "gpu_mode": "exclusive",
        "cpus": 24 if four else 8,
        "memory_gb": 96 if recipe_id in {UFO_MJLAB_VIABILITY_RECIPE,UFO_MJLAB_RECIPE,UFO_TRACKING_CONTACT_RECIPE,UFO_COMPLIANCE_RECIPE,UFO_TRACKING_CONTACT_KD_RECIPE, UFO_STABLE_RECIPE, UFO_SLEW_RECIPE, UFO_VELOCITY_RECIPE, UFO_SCALE_RECIPE, UFO_REACHABLE_RECIPE, UFO_CONTACT_RECIPE, UFO_GROUND_RECIPE} else 64,
        "shm_gb": 32,
        "devices": [],
    }


def recipe_profile_id(recipe_id):
    if recipe_id == UFO_MJLAB_VIABILITY_RECIPE:
        return 'rpo_bfm_ufo_mjlab_4gpu_rsi_viability_v13'
    if recipe_id == UFO_MJLAB_RECIPE:
        return "rpo_bfm_ufo_mjlab_4gpu_rsi_v12"
    if recipe_id == UFO_COMPLIANCE_RECIPE:
        return 'rpo_bfm_ufo_isaac_4gpu_rsi_compliance_v11'
    if recipe_id == UFO_TRACKING_CONTACT_KD_RECIPE:
        return "rpo_bfm_ufo_isaac_4gpu_rsi_tracking_contact_kd_v10"
    if recipe_id == UFO_TRACKING_CONTACT_RECIPE:
        return "rpo_bfm_ufo_isaac_4gpu_rsi_tracking_contact_v10"
    if recipe_id == UFO_GROUND_RECIPE:
        return "rpo_bfm_ufo_isaac_4gpu_rsi_ground_v9"
    if recipe_id == UFO_CONTACT_RECIPE:
        return "rpo_bfm_ufo_isaac_4gpu_rsi_contact_v8"
    if recipe_id == UFO_REACHABLE_RECIPE:
        return "rpo_bfm_ufo_isaac_4gpu_rsi_reachable_v7"
    if recipe_id == UFO_SCALE_RECIPE:
        return "rpo_bfm_ufo_isaac_4gpu_rsi_scale6144_v6"
    if recipe_id == UFO_VELOCITY_RECIPE:
        return "rpo_bfm_ufo_isaac_4gpu_rsi_slew_velocity_v5"
    if recipe_id == UFO_SLEW_RECIPE:
        return "rpo_bfm_ufo_isaac_4gpu_rsi_slew_v4"
    if recipe_id == UFO_STABLE_RECIPE:
        return "rpo_bfm_ufo_isaac_4gpu_rsi_stable_v3"
    if recipe_id == UFO_SAFE_RECIPE:
        return "rpo_bfm_ufo_isaac_4gpu_rsi_safe_v2"
    return "rpo_bfm_ufo_isaac_4gpu_rsi_v1" if recipe_id == UFO_FOUR_GPU_RECIPE else "rpo_bfm_ufo_isaac_rsi_v1"


def execution_binding(spec: JobSpec) -> ExecutionBinding | None:
    raw = spec.params.get("execution")
    if raw is None:
        if spec.params.get("template_id") in RECIPES:
            raise ValueError("UFO recipe requires execution binding")
        return None
    binding = ExecutionBinding.model_validate(raw)
    recipe = RECIPES.get(binding.recipe_id)
    if recipe is None or binding != recipe.binding:
        raise ValueError("Unknown or conflicting execution binding")
    if spec.params.get("stack", binding.runtime_id) != binding.runtime_id:
        raise ValueError("stack conflicts with execution runtime_id")
    if spec.params.get("template_id") != binding.recipe_id:
        raise ValueError("template_id conflicts with execution recipe_id")
    return binding


def recipe_timeout_h(recipe_id, purpose):
    if recipe_id == UFO_COMPLIANCE_RECIPE and purpose=='training':
        return 4
    if recipe_id in {UFO_MJLAB_VIABILITY_RECIPE,UFO_MJLAB_RECIPE,UFO_TRACKING_CONTACT_RECIPE,UFO_COMPLIANCE_RECIPE,UFO_TRACKING_CONTACT_KD_RECIPE, UFO_REACHABLE_RECIPE, UFO_CONTACT_RECIPE, UFO_GROUND_RECIPE} and purpose == "training":
        return 2
    return 6 if recipe_id == UFO_SCALE_RECIPE and purpose == "training" else 1


def validate_update_target(target: int | None, *, start: int = 0, recipe_id=None) -> None:
    if type(target) is not int or type(start) is not int or start < 0:
        raise ValueError("target_learner_updates and checkpoint step must be integers")
    if recipe_id == UFO_MJLAB_VIABILITY_RECIPE:
        if (start, target) not in {(0,100), (100,600)}:
            raise ValueError('MJLab v13 permits only fresh 100 updates and complete-state 100→600 observation')
        return
    maximum = 500 if recipe_id in {UFO_MJLAB_VIABILITY_RECIPE,UFO_MJLAB_RECIPE,UFO_TRACKING_CONTACT_RECIPE,UFO_COMPLIANCE_RECIPE,UFO_TRACKING_CONTACT_KD_RECIPE, UFO_REACHABLE_RECIPE, UFO_CONTACT_RECIPE, UFO_GROUND_RECIPE} else 5000 if recipe_id == UFO_SCALE_RECIPE else 1000
    if recipe_id == UFO_MJLAB_RECIPE and target > 24 and (start, target) != (24, 524):
        raise ValueError("MJLab v12 permits only the frozen 24→524 training segment")
    if recipe_id==UFO_COMPLIANCE_RECIPE:
        if target>524 and (start,target)!=(524,3524):
            raise ValueError('Compliant v11 permits only the gated 524→3524 continuation')
        maximum=3000 if start==524 and target==3524 else 500
    if not start < target <= start + maximum:
        raise ValueError(f"target_learner_updates must advance by 1–{maximum} joint updates")


def validate_ufo_overrides(overrides: list[str]) -> None:
    for item in overrides:
        key, separator, value = item.strip().partition("=")
        if key != "actor.seed" or not separator:
            raise ValueError("UFO recipe protects learning fields; only actor.seed may change")
        if not re.fullmatch(r"[0-9]{1,10}", value) or int(value) > 2**32 - 1:
            raise ValueError("UFO actor.seed must be an integer in 0..4294967295, without Hydra resolvers")


def validate_training_execution(spec: JobSpec) -> ExecutionBinding | None:
    binding = execution_binding(spec)
    if binding is None:
        if (
            spec.params.get("target_learner_updates") is not None
            or spec.params.get("purpose", "training") != "training"
        ):
            raise ValueError("Learner update target and qualification require a UFO recipe")
        return None
    recipe = RECIPES[binding.recipe_id]
    if (
        spec.kind != JobKind.TRAIN_RL
        or spec.embodiment_env_id != recipe.embodiment_env_id
        or spec.task_id != recipe.task_id
        or spec.image not in (recipe.image, recipe_image_digest(binding.recipe_id))
    ):
        raise ValueError("UFO kind/interface/task/image differs from approved recipe")
    if spec.runtime or spec.inputs.base_model or spec.params.get("warm_start"):
        raise ValueError("UFO runtime and model initialization are owned by the recipe")
    purpose = spec.params.get("purpose", "training")
    target = spec.params.get("target_learner_updates")
    initialization = (
        binding.recipe_id != UFO_MJLAB_VIABILITY_RECIPE
        and
        purpose == "qualification"
        and type(target) is int
        and target == 0
        and not spec.inputs.resume_from
        and "resume_step" not in spec.params
    )
    if not initialization:
        validate_update_target(target, start=spec.params.get("resume_step", 0), recipe_id=binding.recipe_id)
    validate_ufo_overrides(spec.overrides)
    if purpose == "qualification":
        if recipe.status not in {"dependency_ready", "smoke_passed", "validated"}:
            raise ValueError("UFO qualification requires dependency_ready evidence")
        if (
            spec.params["target_learner_updates"] > (100 if binding.recipe_id == UFO_MJLAB_VIABILITY_RECIPE else 24)
            or spec.resources.model_dump() != recipe_resources(binding.recipe_id)
            or not 0 < spec.policy.timeout_h <= 1
            or spec.policy.max_retries != 0
        ):
            raise ValueError("UFO qualification exceeds frozen update/resource/wall-clock budget")
    elif purpose == "training":
        if binding.recipe_id == UFO_MJLAB_VIABILITY_RECIPE and (
                not spec.inputs.resume_from or (spec.params.get('resume_step',0),target) != (100,600)):
            raise ValueError('MJLab v13 requires complete-state resume for 100→600')
        if binding.recipe_id == UFO_MJLAB_RECIPE and (
                not spec.inputs.resume_from or (spec.params.get("resume_step", 0), target) != (24, 524)):
            raise ValueError("MJLab pretraining requires complete-state resume for the frozen 24→524 segment")
        if recipe.status not in {"smoke_passed", "validated"}:
            raise ValueError("UFO training requires smoke_passed evidence")
        if spec.resources.model_dump() != recipe_resources(binding.recipe_id):
            raise ValueError("UFO topology requires frozen learner/sampler resources")
        if binding.recipe_id == UFO_SCALE_RECIPE and (not 0 < spec.policy.timeout_h <= 6 or spec.policy.max_retries != 0):
            raise ValueError("UFO scaled training exceeds frozen six-hour/no-retry budget")
        if binding.recipe_id==UFO_COMPLIANCE_RECIPE:
            if (spec.params.get('resume_step',0),target) not in {(24,524),(524,3524)}:
                raise ValueError('Compliant v11 training requires the frozen 24→524 or gated 524→3524 segment')
            long=target-spec.params.get('resume_step',0)>500
            if (not 0<spec.policy.timeout_h<=(4 if long else 2) or spec.policy.max_retries!=0
                    or (long and not spec.inputs.resume_from)):
                raise ValueError('Compliant training exceeds its frozen gated segment budget')
        if binding.recipe_id in {UFO_MJLAB_VIABILITY_RECIPE,UFO_MJLAB_RECIPE,UFO_TRACKING_CONTACT_RECIPE,UFO_TRACKING_CONTACT_KD_RECIPE, UFO_REACHABLE_RECIPE, UFO_CONTACT_RECIPE, UFO_GROUND_RECIPE} and (not 0 < spec.policy.timeout_h <= 2 or spec.policy.max_retries != 0):
            raise ValueError("UFO reachable training exceeds frozen two-hour/no-retry budget")
    else:
        raise ValueError("Unknown UFO purpose")
    if spec.params.get("source_receipt") != recipe_source_receipt(binding.recipe_id):
        raise ValueError("UFO source receipt differs from approved measured image")
    return binding
