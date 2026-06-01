# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY backend_proxy ./backend_proxy

RUN pip install --upgrade pip \
 && pip install '.[perf,metrics]'

# Default config path; override via env or bind-mount.
ENV BACKEND_PROXY_CONFIG=/etc/backend-proxy/config.yaml

EXPOSE 9099
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9099/healthz', timeout=2).status==200 else 1)"

ENTRYPOINT ["python", "-m", "backend_proxy", "run"]
CMD ["--config", "/etc/backend-proxy/config.yaml", "--host", "0.0.0.0", "--port", "9099"]
