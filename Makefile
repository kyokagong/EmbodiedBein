COMPOSE := docker compose -f deploy/docker-compose.yml --env-file deploy/.env
export COMPOSE_PROJECT_NAME := eb
# Builds use content snapshots and image digests, without Git provenance probes.
export BUILDX_GIT_INFO := false
export BUILDX_GIT_LABELS := false
export BUILDX_GIT_CHECK_DIRTY := false
RLINF_CONTENT_HASH = $(shell python3 deploy/images/code_snapshot_hash.py /data/kk/RLinf)
RLINF_PATCH_SHA256 = sha256:$(shell sha256sum deploy/patches/rlinf/robotwin-single-policy-eval.patch | cut -d' ' -f1)
RLINF_BFM_PATCH_SHA256 = sha256:$(shell sha256sum deploy/patches/rlinf/patch_rpo_bfm.py | cut -d' ' -f1)
.DEFAULT_GOAL := env

.PHONY: images-rlinf-ufo-agent
.PHONY: images-rlinf-ufo-mjlab
.PHONY: images-rlinf-ufo-mjlab-viability
images-rlinf-ufo-mjlab-viability:
	test "$$(docker image inspect eb/rlinf:rpo-bfm-ufo-mjlab-v12 --format '{{.Id}}')" = "sha256:071d5aacb6660f0bb48144fa2c6c0af34a8cfff9ef90819909dd4d2b4b63c986"
	docker build -t eb/rlinf:rpo-bfm-ufo-mjlab-viability-v13 -f deploy/images/rlinf-ufo-mjlab-viability.Dockerfile .
images-rlinf-ufo-mjlab:
	test "$$(docker image inspect eb/rlinf:rpo-bfm-ufo-compliance-v11 --format '{{.Id}}')" = "sha256:6a709e3a277f19675e7ef464967e37f2ab77eae56388fc8dc3b4f083a62223ff"
	docker build -t eb/rlinf:rpo-bfm-ufo-mjlab-v12 -f deploy/images/rlinf-ufo-mjlab.Dockerfile .
.PHONY: images-rlinf-ufo-four-gpu
.PHONY: images-rlinf-ufo-safe
.PHONY: images-rlinf-ufo-stable images-rlinf-ufo-stable-eval images-rlinf-ufo-contact
.PHONY: images-rlinf-ufo-slew images-rlinf-ufo-slew-eval
.PHONY: images-rlinf-ufo-velocity images-rlinf-ufo-velocity-eval images-rlinf-ufo-scaling images-rlinf-ufo-scaling-eval
.PHONY: images-rlinf-ufo-reachable images-rlinf-ufo-contact-v8
images-rlinf-ufo-contact-v8:
	test "$$(docker image inspect eb/rlinf:rpo-bfm-ufo-agent-4gpu-reachable-v7 --format '{{.Id}}')" = "sha256:2359c60a3f40ef9f6a47b59d9c02194d1aa14afa603789960c5807e988277fb7"
	docker build -t eb/rlinf:rpo-bfm-ufo-agent-4gpu-contact-v8-poststep1 -f deploy/images/rlinf-ufo-contact-v8.Dockerfile .
images-rlinf-ufo-reachable:
	test "$$(docker image inspect eb/rlinf:rpo-bfm-ufo-agent-4gpu-scale6144-v6 --format '{{.Id}}')" = "sha256:e643aa62117fc1b5c70195d7b589225bf42621843ddf5218bd1c99895e04ffd7"
	docker build -t eb/rlinf:rpo-bfm-ufo-agent-4gpu-reachable-v7 -f deploy/images/rlinf-ufo-reachable.Dockerfile .

images-rlinf-ufo-scaling:
	test "$$(docker image inspect eb/rlinf:rpo-bfm-ufo-agent-4gpu-slew-velocity-v5 --format '{{.Id}}')" = "sha256:e1382226cd5f0bd3795ee6a187f62555ba77f591e342c862a64bf1825a463dee"
	docker build -t eb/rlinf:rpo-bfm-ufo-agent-4gpu-scale6144-v6 -f deploy/images/rlinf-ufo-scaling.Dockerfile .

images-rlinf-ufo-scaling-eval:
	test "$$(docker image inspect eb/rlinf:rpo-ufo-slew-velocity-eval-v5 --format '{{.Id}}')" = "sha256:3094f020480ce099ff281ac76c4812bd47985b7d4a489a15a0550c49e6e9fbca"
	docker build -t eb/rlinf:rpo-ufo-scale6144-eval-v6 -f deploy/images/rlinf-ufo-scaling-eval.Dockerfile .

images-rlinf-ufo-velocity:
	test "$$(docker image inspect eb/rlinf:rpo-bfm-ufo-agent-4gpu-slew-v4 --format '{{.Id}}')" = "sha256:88aee59f8d0f62d9e99affd231b34511f4624a9203bf7f9205c5f2f4008c1fe4"
	docker build -t eb/rlinf:rpo-bfm-ufo-agent-4gpu-slew-velocity-v5 -f deploy/images/rlinf-ufo-velocity.Dockerfile .

images-rlinf-ufo-velocity-eval:
	test "$$(docker image inspect eb/rlinf:rpo-ufo-slew-eval-v4 --format '{{.Id}}')" = "sha256:9129bc326a1e1d68122bc630dff29e362066c2fdc4f0285eda7f61d211072d99"
	docker build -t eb/rlinf:rpo-ufo-slew-velocity-eval-v5 -f deploy/images/rlinf-ufo-velocity-eval.Dockerfile .

images-rlinf-ufo-slew:
	test "$$(docker image inspect eb/rlinf:rpo-bfm-ufo-agent-4gpu-stable-v3 --format '{{.Id}}')" = "sha256:fac4b89f2821c072d86d6b648afdee98e179d74615a13847df238eb083e6c241"
	docker build -t eb/rlinf:rpo-bfm-ufo-agent-4gpu-slew-v4 -f deploy/images/rlinf-ufo-slew.Dockerfile .

images-rlinf-ufo-slew-eval:
	test "$$(docker image inspect eb/rlinf:rpo-ufo-stable-eval-v3 --format '{{.Id}}')" = "sha256:86cdf16e1368c8780a190192418c39cef77737a0e72bdd63ba3147575657fa4d"
	docker build -t eb/rlinf:rpo-ufo-slew-eval-v4 -f deploy/images/rlinf-ufo-slew-eval.Dockerfile .

images-rlinf-ufo-contact:
	test "$$(docker image inspect eb/rlinf:rpo-ufo-stable-eval-v3 --format '{{.Id}}')" = "sha256:86cdf16e1368c8780a190192418c39cef77737a0e72bdd63ba3147575657fa4d"
	docker build -t eb/rlinf:rpo-ufo-contact-v1 -f deploy/images/rlinf-ufo-contact.Dockerfile .

images-rlinf-ufo-stable:
	test "$$(docker image inspect --format '{{.Id}}' eb/rlinf:rpo-bfm-ufo-agent-4gpu-safe-v2)" = sha256:adbca4ec9cf899e18da848b7e2a47d2df50d1f8e74e9015655b6deac193b0f99
	docker build -t eb/rlinf:rpo-bfm-ufo-agent-4gpu-stable-v3 -f deploy/images/rlinf-ufo-stable.Dockerfile .

images-rlinf-ufo-stable-eval:
	test "$$(docker image inspect --format '{{.Id}}' eb/rlinf:rpo-ufo-eval-v1)" = sha256:c2b36fe55f84880c28382e37801e4689abbfaed04ee507b0d4cbf78ae2d22563
	docker build -t eb/rlinf:rpo-ufo-stable-eval-v3 -f deploy/images/rlinf-ufo-stable-eval.Dockerfile .

.PHONY: images-rlinf-ufo-tracking
.PHONY: images-rlinf-ufo-eval
images-rlinf-ufo-eval:
	test "$$(docker image inspect --format '{{.Id}}' eb/rlinf:rpo-bfm-ufo-agent-4gpu-v1)" = sha256:60fbb44a17c3247f7ed83104466e4a652dd06d8712b22ffb1ea202ea6d03cf2b
	docker build -t eb/rlinf:rpo-ufo-eval-v1 -f deploy/images/rlinf-ufo-eval.Dockerfile .

images-rlinf-ufo-tracking:
	test "$$(docker image inspect --format '{{.Id}}' eb/rlinf:rpo-bfm-ufo-agent-4gpu-v1)" = sha256:60fbb44a17c3247f7ed83104466e4a652dd06d8712b22ffb1ea202ea6d03cf2b
	docker build -t eb/rlinf:rpo-ufo-tracking-v2 -f deploy/images/rlinf-ufo-tracking.Dockerfile .

images-rlinf-ufo-four-gpu:
	test "$$(docker image inspect --format '{{.Id}}' eb/rlinf:rpo-bfm-ufo-agent-v1)" = sha256:5abec37ade5f55fd043f9aceff9fc8b32ed7ff422b08c80879857bbd745af015
	docker build -t eb/rlinf:rpo-bfm-ufo-agent-4gpu-v1 -f deploy/images/rlinf-ufo-four-gpu.Dockerfile .

images-rlinf-ufo-safe:
	test "$$(docker image inspect --format '{{.Id}}' eb/rlinf:rpo-bfm-ufo-agent-4gpu-v1)" = sha256:60fbb44a17c3247f7ed83104466e4a652dd06d8712b22ffb1ea202ea6d03cf2b
	docker build -t eb/rlinf:rpo-bfm-ufo-agent-4gpu-safe-v2 -f deploy/images/rlinf-ufo-safe.Dockerfile .

images-rlinf-ufo-agent:
	docker build --build-context ufo-source=deploy/.sources/ufo-0b89378de02f77712072d983bc43ec2a46455dc0 \
	  --build-arg RLINF_REVISION="$$(git -C /data/kk/RLinf rev-parse HEAD)" \
	  --build-arg RLINF_CONTENT_HASH="$$(docker run --rm --network none -v "$(CURDIR)/deploy/images:/src:ro" -v /data/kk/RLinf:/rlinf:ro eb/test:local python /src/code_snapshot_hash.py /rlinf)" \
	  -t eb/rlinf:rpo-bfm-ufo-agent-v1 -f deploy/images/rlinf-ufo-agent.Dockerfile .

# Host ports from deploy/.env (compose still has :-defaults if a key is missing).
env_value = $(or $(shell sed -n 's/^$(1)=//p' deploy/.env 2>/dev/null),$(2))
EB_PORT_API := $(call env_value,EB_PORT_API,8000)
EB_PORT_VITE := $(call env_value,EB_PORT_VITE,5173)
EB_HOST_STATE_DIR := $(call env_value,EB_HOST_STATE_DIR,/data/kk/embodiedbein)
EB_HOST_MODELS_DIR := $(call env_value,EB_HOST_MODELS_DIR,/data/kk/models)
EB_HOST_ASSETS_DIR := $(call env_value,EB_HOST_ASSETS_DIR,/data/kk/assets)
EB_HOST_DATA_DIR := $(call env_value,EB_HOST_DATA_DIR,/data/kk/data)
export HF_ENDPOINT := $(call env_value,EB_HF_ENDPOINT,https://huggingface.co)

.PHONY: up down ps logs restart-api migrate psql backup images images-rlinf \
	dev-web test lint seed env smoke-job

env:
	@test -f deploy/.env || cp deploy/.env.example deploy/.env
	@mkdir -p "$(EB_HOST_STATE_DIR)" "$(EB_HOST_MODELS_DIR)" "$(EB_HOST_DATA_DIR)" "$(EB_HOST_ASSETS_DIR)" /run/eb backup deploy/secrets

up: env
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f --tail=200 api ebd

restart-api:
	$(COMPOSE) up -d --build api

migrate:
	$(COMPOSE) run --rm migrate

psql:
	$(COMPOSE) exec postgres psql -U eb -d eb

backup:
	@mkdir -p backup
	$(COMPOSE) exec -T postgres pg_dump -U eb eb | gzip > backup/eb-$$(date +%F).sql.gz

dev-web:
	cd frontend && npm ci && EB_PORT_API=$(EB_PORT_API) npm run dev -- --port $(EB_PORT_VITE) --strictPort

test:
	docker build -f deploy/test.Dockerfile -t eb/test:local .
	docker run --rm -v "$(CURDIR):/workspace:ro" -v /data/kk:/data/kk:ro eb/test:local
	docker run --rm -v "$(CURDIR)/rust:/workspace:ro" -v eb-cargo-cache:/usr/local/cargo -e CARGO_TARGET_DIR=/tmp/target -w /workspace rust:1.83-bookworm cargo test --workspace

lint:
	cd backend && uv run ruff check eb tests
	cd frontend && npx vue-tsc --noEmit

images:
	@docker image inspect isaac-lab-template >/dev/null 2>&1 || { echo "先有 isaac-lab-template"; exit 1; }
	docker tag $$(docker inspect isaac-lab-template --format '{{.Image}}') eb/isaac-lab:2.3.2
	docker build -t eb/isaac-lab:2.3.2 -f deploy/images/isaac.Dockerfile --build-arg BASE=isaac-lab-template .
	$(MAKE) images-rlinf
	docker build -t eb/mujoco:latest -f deploy/images/mujoco.Dockerfile .
	docker build -t eb/tools:latest -f deploy/images/tools.Dockerfile .

images-rlinf:
	@docker image inspect rlinf-berkeley-openpi:local >/dev/null 2>&1 || { echo "缺少基础镜像 rlinf-berkeley-openpi:local"; exit 1; }
	docker build --build-context openpi=/data/kk/openpi --build-arg BASE=eb/rlinf:0.4.0-emb --build-arg HF_ENDPOINT -t eb/rlinf:libero-pi05 -f deploy/images/rlinf.Dockerfile .
	docker build --build-context rlinf=/data/kk/RLinf \
	  --build-arg RLINF_CONTENT_HASH=$(RLINF_CONTENT_HASH) \
	  --build-arg RLINF_PATCH_SHA256=$(RLINF_PATCH_SHA256) \
	  -t eb/rlinf:robotwin-pi05 -f deploy/images/rlinf-robotwin.Dockerfile .
	docker build --build-context rlinf=/data/kk/RLinf \
	  --build-arg RLINF_CONTENT_HASH=$(RLINF_CONTENT_HASH) \
	  -t eb/rlinf:rpo-v2-wbc -f deploy/images/rlinf-rpo.Dockerfile .
	docker build \
	  --build-arg RLINF_CONTENT_HASH=$(RLINF_CONTENT_HASH) \
	  --build-arg RLINF_PATCH_SHA256=$(RLINF_BFM_PATCH_SHA256) \
	  -t eb/rlinf:rpo-v2-bfm-zero -f deploy/images/rlinf-bfm.Dockerfile .
	docker build --build-context gr00t=/data/kk/models/sources/Isaac-GR00T-N1.7 \
	  -t eb/rlinf:robotwin-gr00t-n1.7 -f deploy/images/rlinf-gr00t.Dockerfile .
	docker build --build-context xr1=/data/kk/models/sources/Xiaomi-Robotics-1 \
	  -t eb/rlinf:robotwin-xiaomi-xr1 -f deploy/images/rlinf-xr1.Dockerfile .
	docker build --build-context lingbot=/data/kk/models/sources/lingbot-vla-v2 \
	  --build-context utils3d=/data/kk/models/sources/utils3d-3fab839f0be9931dac7c8488eb0e1600c236e183 \
	  -t eb/rlinf:robotwin-lingbot-vla2 -f deploy/images/rlinf-lingbot.Dockerfile .
	docker build --build-context opendm=/data/kk/models/sources/opendm \
	  -t eb/rlinf:robotwin-dm05 -f deploy/images/rlinf-dm05.Dockerfile .

.PHONY: images-bfm-reference images-bfm-g1 images-g1-retarget
images-bfm-g1:
	docker build --build-arg RLINF_CONTENT_HASH=$(RLINF_CONTENT_HASH) \
	  --build-arg RLINF_PATCH_SHA256=$(RLINF_BFM_PATCH_SHA256) \
	  -t eb/rlinf:rpo-v2-bfm-g1-v3 -f deploy/images/rlinf-bfm.Dockerfile .

.PHONY: images-bfm-four-gpu images-bfm-staged-walking images-bfm-staged-eval
images-bfm-staged-eval:
	docker build --build-arg RLINF_CONTENT_HASH=$(RLINF_CONTENT_HASH) \
	  --build-arg RLINF_PATCH_SHA256=$(RLINF_BFM_PATCH_SHA256) \
	  -t eb/rlinf:rpo-v2-bfm-walking-eval-v11 -f deploy/images/rlinf-bfm.Dockerfile .

images-bfm-staged-walking:
	docker build --build-arg RLINF_CONTENT_HASH=$(RLINF_CONTENT_HASH) \
	  --build-arg RLINF_PATCH_SHA256=$(RLINF_BFM_PATCH_SHA256) \
	  -t eb/rlinf:rpo-v2-bfm-walking-v10 -f deploy/images/rlinf-bfm.Dockerfile .

images-bfm-four-gpu:
	docker build --build-arg RLINF_CONTENT_HASH=$(RLINF_CONTENT_HASH) \
	  --build-arg RLINF_PATCH_SHA256=$(RLINF_BFM_PATCH_SHA256) \
	  -t eb/rlinf:rpo-v2-bfm-four-gpu-v6 -f deploy/images/rlinf-bfm.Dockerfile .

.PHONY: images-bfm-eval images-bfm-eval-parallel
images-bfm-eval-parallel:
	docker build --build-arg RLINF_CONTENT_HASH=$(RLINF_CONTENT_HASH) \
	  --build-arg RLINF_PATCH_SHA256=$(RLINF_BFM_PATCH_SHA256) \
	  -t eb/rlinf:rpo-v2-bfm-parallel-eval-v8 -f deploy/images/rlinf-bfm.Dockerfile .

images-bfm-eval:
	docker build --build-arg RLINF_CONTENT_HASH=$(RLINF_CONTENT_HASH) \
	  --build-arg RLINF_PATCH_SHA256=$(RLINF_BFM_PATCH_SHA256) \
	  -t eb/rlinf:rpo-v2-bfm-eval-v2 -f deploy/images/rlinf-bfm.Dockerfile .

images-bfm-reference:
	docker build --build-arg RLINF_CONTENT_HASH=$(RLINF_CONTENT_HASH) \
	  --build-arg RLINF_PATCH_SHA256=$(RLINF_BFM_PATCH_SHA256) \
	  -t eb/rlinf:rpo-v2-bfm-reference-v5 -f deploy/images/rlinf-bfm.Dockerfile .

images-g1-retarget:
	docker build -t eb/tools:rpo-g1-r2r-v7 -f deploy/images/rpo-g1-r2r.Dockerfile .

smoke-job:
	@echo "Submit biped smoke via API (requires make up and eb-dev-key)."
	curl -sS http://127.0.0.1:$(EB_PORT_API)/api/v1/jobs \
	  -H 'Authorization: Bearer eb-dev-key' \
	  -H 'Content-Type: application/json' \
	  -d '{"spec":{"kind":"train.rl","name":"biped-smoke","embodiment_env_id":"ee_bhl_biped_12dof_v1","task_id":"tsk_bhl_velocity_biped","params":{"task":"Velocity-Berkeley-Humanoid-Lite-Biped-v0","num_envs":64,"max_iterations":5,"seed":0},"resources":{"gpus":1,"shm_gb":2}}}'
	@echo

.PHONY: images-rlinf-ufo-ground-v9
images-rlinf-ufo-ground-v9:
	test "$$(docker image inspect eb/rlinf:rpo-bfm-ufo-agent-4gpu-contact-v8-poststep1 --format '{{.Id}}')" = "sha256:288c9bc690f0b0cefce4529b9ae07d3d0c9163b201d08d228161b7818685b5e2"
	docker build -t eb/rlinf:rpo-bfm-ufo-agent-4gpu-ground-v9 -f deploy/images/rlinf-ufo-ground-v9.Dockerfile .

.PHONY: images-rlinf-ufo-tracking-contact
images-rlinf-ufo-tracking-contact:
	test "$$(docker image inspect eb/rlinf:rpo-bfm-ufo-ground-v9-flat --format '{{.Id}}')" = "sha256:c91f37b08dd2f623df4823c7b6f6394802909c8b20de45b429ff0dd03d15611f"
	docker build -t eb/rlinf:rpo-bfm-ufo-tracking-contact-v10 -f deploy/images/rlinf-ufo-tracking-contact.Dockerfile .

.PHONY: images-rlinf-ufo-tracking-contact-kd
images-rlinf-ufo-tracking-contact-kd:
	test "$$(docker image inspect eb/rlinf:rpo-bfm-ufo-ground-v9-flat --format '{{.Id}}')" = "sha256:c91f37b08dd2f623df4823c7b6f6394802909c8b20de45b429ff0dd03d15611f"
	docker build -t eb/rlinf:rpo-bfm-ufo-tracking-contact-kd-v10 -f deploy/images/rlinf-ufo-tracking-contact-kd.Dockerfile .

.PHONY: images-rlinf-ufo-compliance
images-rlinf-ufo-compliance:
	docker build -t eb/rlinf:rpo-bfm-ufo-compliance-v11 -f deploy/images/rlinf-ufo-compliance.Dockerfile .
