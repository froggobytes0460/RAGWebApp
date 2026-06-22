FROM node:22-alpine@sha256:ab07539e0988b63558ff621f5fbe1077054c39d9809112974fb79993949d41cd AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install --omit=dev
COPY frontend/ .
RUN npm run build

FROM nginx:alpine@sha256:20316569d8f81a160065d7d2a5eeffc7ca97d79022462ee255fd23fa103a6b5c AS nginx
COPY --from=frontend-build /app/frontend/dist /usr/share/nginx/html
COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY nginx/certs/server.crt /etc/nginx/certs/server.crt
COPY nginx/certs/server.key /etc/nginx/certs/server.key
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf -k https://localhost/api/health || exit 1

FROM python:3.13-slim@sha256:c33f0bc4364a6881bed1ec0cc2665e6c53c87a43e774aaeab88e6f17af105e4f AS api
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && rm -rf /var/lib/apt/lists/*
RUN groupadd --system appgroup && useradd --system --gid appgroup --no-create-home appuser
COPY --chown=appuser:appgroup pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --no-cache && chown -R appuser:appgroup /app/.venv
COPY --chown=appuser:appgroup backend/ ./backend/
RUN mkdir -p /app/logs && chown appuser:appgroup /app/logs
ENV UV_NO_CACHE=1
ENV PYTHONPATH=/app
ENV HF_HOME=/tmp/hf_cache
ENV FASTEMBED_CACHE_PATH=/tmp/fastembed_cache
ENV XDG_CACHE_HOME=/tmp/cache
USER appuser
HEALTHCHECK --interval=15s --timeout=15s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request, sys; r = urllib.request.urlopen('http://localhost:8000/api/health', timeout=10); sys.exit(0 if r.status == 200 else 1)"
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "backend.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
