FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-distutils python3-pip \
    ca-certificates git curl \
    libglib2.0-0 libgl1 ffmpeg \
    libsdl2-2.0-0 libsdl2-image-2.0-0 libsdl2-mixer-2.0-0 libsdl2-ttf-2.0-0 \
    libfreetype6 libportmidi0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /workspace

COPY pyproject.toml /workspace/pyproject.toml

RUN python - <<'PY'
import subprocess, sys, tomllib
from pathlib import Path

pyproject = Path('pyproject.toml')
if not pyproject.exists():
    raise SystemExit('pyproject.toml not found')

data = tomllib.loads(pyproject.read_text())
deps = data.get('project', {}).get('dependencies', [])

# Ensure CUDA-enabled PyTorch
base_deps = [d for d in deps if not d.startswith('torch==')]

# Install CUDA torch from the official index
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
])
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install',
    '--index-url', 'https://download.pytorch.org/whl/cu121',
    'torch==2.1.0+cu121'
])

extras = [
    'gymnasium[atari]',
    'gymnasium[accept-rom-license]',
    'opencv-python',
    'matplotlib',
]

subprocess.check_call([
    sys.executable, '-m', 'pip', 'install',
    *base_deps, *extras
])
PY

COPY . /workspace

ENV PYTHONUNBUFFERED=1
ENV SDL_VIDEODRIVER=dummy
