# floodfusion — the locked environment in a container.
#   docker build -t floodfusion .
#   docker run --rm floodfusion                       # the offline test suite
#   docker run --rm --gpus all -v "$PWD/data:/app/data" -v "$PWD/runs:/app/runs" floodfusion make demo
# The PyPI torch wheels bundle the CUDA runtime: --gpus all is all a GPU run needs
# (with the NVIDIA container toolkit on the host).
# Base images pinned by digest (Dependabot proposes updates).
FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f

COPY --from=ghcr.io/astral-sh/uv:0.8@sha256:1d31be550ff927957472b2a491dc3de1ea9b5c2d319a9cea5b6a48021e2990a6 /uv /uvx /bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# Dependencies first (a cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra dev --no-install-project
COPY src ./src
COPY configs ./configs
COPY tests ./tests
COPY Makefile ./
RUN uv sync --frozen --extra dev

RUN useradd --create-home --uid 1000 floodfusion && chown -R floodfusion /app
USER floodfusion
CMD ["pytest", "-m", "not network and not gpu", "-q", "-p", "no:cacheprovider"]
