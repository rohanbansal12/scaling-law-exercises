#!/usr/bin/env bash
set -euo pipefail

NODE_MAJOR="${NODE_MAJOR:-22}"
NODE_VERSION="${NODE_VERSION:-}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch}"

log() {
  printf '\n==> %s\n' "$*"
}

ensure_path_line() {
  local line="$1"
  local file="$HOME/.bashrc"
  touch "$file"
  if ! grep -Fqx "$line" "$file"; then
    printf '\n%s\n' "$line" >> "$file"
  fi
}

log "Updating apt metadata and installing base tools"
sudo apt-get update
sudo apt-get install -y curl ca-certificates git tar xz-utils build-essential

if [ -z "$NODE_VERSION" ]; then
  log "Resolving latest Node.js ${NODE_MAJOR}.x release"
  TMP_NODE_INDEX="$(mktemp)"
  curl -fsSL https://nodejs.org/dist/index.tab -o "$TMP_NODE_INDEX"
  NODE_VERSION="$(
    awk -v major="v${NODE_MAJOR}." 'NR > 1 && $1 ~ "^" major { sub(/^v/, "", $1); print $1; exit }' "$TMP_NODE_INDEX"
  )"
  rm -f "$TMP_NODE_INDEX"
fi
if [ -z "$NODE_VERSION" ]; then
  echo "Could not resolve a Node.js ${NODE_MAJOR}.x release." >&2
  exit 1
fi

log "Installing Node.js ${NODE_VERSION} under ~/.local"
case "$(uname -m)" in
  x86_64|amd64) NODE_ARCH="linux-x64" ;;
  aarch64|arm64) NODE_ARCH="linux-arm64" ;;
  *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

NODE_PARENT="$HOME/.local"
NODE_DIR="$NODE_PARENT/node-v${NODE_VERSION}"
NODE_CURRENT="$NODE_PARENT/node-v22"
NODE_TARBALL="node-v${NODE_VERSION}-${NODE_ARCH}.tar.xz"
NODE_URL="https://nodejs.org/dist/v${NODE_VERSION}/${NODE_TARBALL}"

mkdir -p "$NODE_PARENT"
if [ ! -x "$NODE_DIR/bin/node" ]; then
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT
  curl -fsSL "$NODE_URL" -o "$TMP_DIR/$NODE_TARBALL"
  tar -xJf "$TMP_DIR/$NODE_TARBALL" -C "$NODE_PARENT"
  rm -rf "$NODE_DIR"
  mv "$NODE_PARENT/node-v${NODE_VERSION}-${NODE_ARCH}" "$NODE_DIR"
fi
ln -sfn "$NODE_DIR" "$NODE_CURRENT"

ensure_path_line 'export PATH="$HOME/.local/node-v22/bin:$HOME/.npm-global/bin:$HOME/.local/bin:$PATH"'
export PATH="$HOME/.local/node-v22/bin:$HOME/.npm-global/bin:$HOME/.local/bin:$PATH"

log "Configuring npm global installs under ~/.npm-global"
mkdir -p "$HOME/.npm-global"
npm config set prefix "$HOME/.npm-global"

log "Installing Codex CLI"
npm install -g @openai/codex@latest --include=optional

log "Installing uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
ensure_path_line 'export PATH="$HOME/.local/bin:$PATH"'

if [ -d "$PROJECT_DIR" ]; then
  log "Using existing project directory at $PROJECT_DIR"
elif [ -e "$PROJECT_DIR" ]; then
  echo "$PROJECT_DIR exists but is not a directory." >&2
  exit 1
else
  log "Creating project directory at $PROJECT_DIR"
  mkdir -p "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

log "Installing Python ${PYTHON_VERSION} with uv"
uv python install "$PYTHON_VERSION"

log "Initializing uv project"
if [ ! -f pyproject.toml ]; then
  uv init --bare --python "$PYTHON_VERSION"
fi

log "Creating virtual environment"
uv venv --python "$PYTHON_VERSION"

log "Installing Python packages for TPU JAX work"
uv add "jax[tpu]" numpy jupyter

log "Verifying installed tools"
node --version
npm --version
codex --version || true
uv --version
uv run python - <<'PY'
import jax
import numpy as np
print("python ok")
print("numpy", np.__version__)
print("jax", jax.__version__)
print("jax devices", jax.devices())
PY

cat <<EOF

Bootstrap complete.

Project directory:
  $PROJECT_DIR

To activate the environment:
  cd "$PROJECT_DIR"
  source .venv/bin/activate

If this is your first shell after running the script, reload PATH with:
  source ~/.bashrc

Then start Codex with:
  codex
EOF
