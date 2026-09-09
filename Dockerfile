FROM ghcr.io/astral-sh/uv:0.12.9 AS uv


FROM python:3.12-slim AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY api ./api
COPY data ./data
RUN uv sync --locked --no-dev

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD [".venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"]
CMD [".venv/bin/python", "-m", "api"]


FROM node:22-slim AS frontend

WORKDIR /app
RUN chown node:node /app
USER node

COPY --chown=node:node frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY --chown=node:node frontend ./

ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["node", "-e", "fetch('http://127.0.0.1:3000').then(r => { if (!r.ok) process.exit(1) }).catch(() => process.exit(1))"]
CMD ["npx", "wrangler", "dev", "--config", "dist/server/wrangler.json", "--ip", "0.0.0.0", "--port", "3000"]


# `docker build .` and Render produce the API image by default.
FROM api AS final
