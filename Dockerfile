FROM python:3.12-slim

# LightGBM needs OpenMP — on Linux this is libgomp1
# (the same role libomp played on macOS in Step 1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv itself inside the container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /code

# Copy ONLY dependency files first (better layer caching --
# if app code changes but dependencies don't, this layer is reused)
COPY pyproject.toml uv.lock ./

# Install dependencies into the image's environment (not a .venv we'd need to activate)
RUN uv sync --frozen --no-dev

# Now copy the application code and trained models
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

# Run via uv, WITHOUT --reload (reload is a dev-only feature)
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]