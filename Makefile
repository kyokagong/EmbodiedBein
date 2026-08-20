COMPOSE := docker compose -f deploy/docker-compose.yml --env-file deploy/.env
export COMPOSE_PROJECT_NAME := eb

.PHONY: up down ps logs restart-api migrate psql backup images \
	dev-web test lint seed env

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
	docker build -t eb/isaac-lab:2.3.2 -f deploy/images/isaac.Dockerfile .
	docker build -t eb/rlinf:0.4.0-emb -f deploy/images/rlinf.Dockerfile .
	docker build -t eb/mujoco:latest -f deploy/images/mujoco.Dockerfile .
	docker build -t eb/tools:latest -f deploy/images/tools.Dockerfile .
