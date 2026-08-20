COMPOSE := docker compose -f deploy/docker-compose.yml --env-file deploy/.env
export COMPOSE_PROJECT_NAME := eb

.PHONY: up down ps logs restart-api migrate psql backup images \
	dev-web test lint seed env smoke-job

env:
	@test -f deploy/.env || cp deploy/.env.example deploy/.env
	@mkdir -p /var/lib/eb /run/eb backup deploy/secrets

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
	cd frontend && npm ci && npm run dev

test:
	cd backend && uv run pytest -q
	cd rust && cargo test --workspace --offline 2>/dev/null || cargo test --workspace

lint:
	cd backend && uv run ruff check eb tests
	cd frontend && npx vue-tsc --noEmit

images:
	@docker image inspect isaac-lab-template >/dev/null 2>&1 || { echo "先有 isaac-lab-template"; exit 1; }
	docker tag $$(docker inspect isaac-lab-template --format '{{.Image}}') eb/isaac-lab:2.3.2
	docker build -t eb/isaac-lab:2.3.2 -f deploy/images/isaac.Dockerfile --build-arg BASE=isaac-lab-template .
	@docker image inspect rlinf-berkeley-openpi >/dev/null 2>&1 || docker image inspect rlinf/rlinf:agentic-rlinf0.2-maniskill_libero >/dev/null 2>&1 || { echo "缺少本机 RLinf 镜像，跳过 eb/rlinf 标签"; true; }
	-docker tag rlinf-berkeley-openpi:local eb/rlinf:0.4.0-emb
	docker build -t eb/mujoco:latest -f deploy/images/mujoco.Dockerfile .
	docker build -t eb/tools:latest -f deploy/images/tools.Dockerfile .

smoke-job:
	@echo "Submit biped smoke via API (requires make up and eb-dev-key)."
	curl -sS http://127.0.0.1:8000/api/v1/jobs \
	  -H 'Authorization: Bearer eb-dev-key' \
	  -H 'Content-Type: application/json' \
	  -d '{"spec":{"kind":"train.rl","name":"biped-smoke","contract_id":"ec_bhl_biped_12dof_v1","params":{"task":"Velocity-Berkeley-Humanoid-Lite-Biped-v0","num_envs":64,"max_iterations":5,"seed":0},"resources":{"gpus":1,"shm_gb":2}}}'
	@echo
