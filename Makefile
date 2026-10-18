COMPOSE := docker compose -f deploy/docker-compose.yml --env-file deploy/.env
export COMPOSE_PROJECT_NAME := eb
# Builds use content snapshots and image digests, without Git provenance probes.
export BUILDX_GIT_INFO := false
export BUILDX_GIT_LABELS := false
export BUILDX_GIT_CHECK_DIRTY := false
RLINF_CONTENT_HASH = $(shell python3 deploy/images/code_snapshot_hash.py /data/kk/RLinf)
RLINF_PATCH_SHA256 = sha256:$(shell sha256sum deploy/patches/rlinf/robotwin-single-policy-eval.patch | cut -d' ' -f1)

.DEFAULT_GOAL := env
.PHONY: images-ufo-native-deploy
images-ufo-native-deploy:
	test "$(docker image inspect ufo-rpo:rpo-23dof-20261016 --format '{{.Id}}')" = "sha256:8733fd6e6ed852b4687d43e5499a5bd4f6dec58b12e5ff23a3bc92ff233f46fd"
	docker build -t eb/ufo-native:rpo-deploy-v1 -f deploy/images/ufo-native-deploy.Dockerfile .

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
	docker build --build-context gr00t=/data/kk/models/sources/Isaac-GR00T-N1.7 \
	  -t eb/rlinf:robotwin-gr00t-n1.7 -f deploy/images/rlinf-gr00t.Dockerfile .
	docker build --build-context xr1=/data/kk/models/sources/Xiaomi-Robotics-1 \
	  -t eb/rlinf:robotwin-xiaomi-xr1 -f deploy/images/rlinf-xr1.Dockerfile .
	docker build --build-context lingbot=/data/kk/models/sources/lingbot-vla-v2 \
	  --build-context utils3d=/data/kk/models/sources/utils3d-3fab839f0be9931dac7c8488eb0e1600c236e183 \
	  -t eb/rlinf:robotwin-lingbot-vla2 -f deploy/images/rlinf-lingbot.Dockerfile .
	docker build --build-context opendm=/data/kk/models/sources/opendm \
	  -t eb/rlinf:robotwin-dm05 -f deploy/images/rlinf-dm05.Dockerfile .

smoke-job:
	@echo "Submit biped smoke via API (requires make up and eb-dev-key)."
	curl -sS http://127.0.0.1:$(EB_PORT_API)/api/v1/jobs \
	  -H 'Authorization: Bearer eb-dev-key' \
	  -H 'Content-Type: application/json' \
	  -d '{"spec":{"kind":"train.rl","name":"biped-smoke","embodiment_env_id":"ee_bhl_biped_12dof_v1","task_id":"tsk_bhl_velocity_biped","params":{"task":"Velocity-Berkeley-Humanoid-Lite-Biped-v0","num_envs":64,"max_iterations":5,"seed":0},"resources":{"gpus":1,"shm_gb":2}}}'
	@echo
