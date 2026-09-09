### Build frontend (Vite) in a Node stage ###
FROM node:18-alpine AS node_builder
WORKDIR /work/frontend

# copy only what is needed for install first to leverage caching
COPY frontend/package.json frontend/package-lock.json* ./
# Install dependencies. Do not skip optional packages — rollup may require
# an optional native binary during the Vite build, so `--no-optional` breaks
# production builds on some platforms. Use `--no-audit` and prefer offline
# caching for speed.
RUN npm install --no-audit --prefer-offline
COPY frontend/ ./
RUN npm run build


### Final Python runtime image ###
FROM python:3.12-slim

# System dependencies needed for insightface (cmake/build), onnxruntime, and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application sources
COPY . .

# Copy built frontend from the node builder stage
COPY --from=node_builder /work/frontend/dist ./frontend/dist

# Render sets $PORT at runtime — bind to it, not a hardcoded port
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
