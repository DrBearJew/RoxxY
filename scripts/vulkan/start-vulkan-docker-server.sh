#!/usr/bin/env bash
set -euo pipefail

# Build/run the Vulkan llama-server Docker target with the host GPU exposed.
# Usage:
#   scripts/vulkan/start-vulkan-docker-server.sh /path/to/model.gguf [extra llama-server args...]
#   scripts/vulkan/start-vulkan-docker-server.sh --list-devices
# Env:
#   PORT=8080 CTX_SIZE=4096 IMAGE=llama-cpp-vulkan-server:local BUILD=auto|1|0 VULKAN_DEVICE=Vulkan0

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

IMAGE="${IMAGE:-llama-cpp-vulkan-server:local}"
BUILD="${BUILD:-auto}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-4096}"
VULKAN_DEVICE="${VULKAN_DEVICE:-Vulkan0}"
VOLUME_OPTS="${VOLUME_OPTS:-ro}"

if [[ "$BUILD" == "1" ]] || { [[ "$BUILD" == "auto" ]] && ! docker image inspect "$IMAGE" >/dev/null 2>&1; }; then
  docker build -t "$IMAGE" --target server -f .devops/vulkan.Dockerfile .
fi

device_args=()
for dev in /dev/dri/renderD* /dev/dri/card*; do
  [[ -e "$dev" ]] && device_args+=(--device "$dev:$dev")
done
if [[ ${#device_args[@]} -eq 0 ]]; then
  echo "No /dev/dri Vulkan device nodes found. Does the host see the GPU?" >&2
  exit 1
fi

group_args=()
for group in render video; do
  gid="$(getent group "$group" 2>/dev/null | cut -d: -f3 || true)"
  [[ -n "$gid" ]] && group_args+=(--group-add "$gid")
done

if [[ "${1:-}" == "--list-devices" ]]; then
  docker run --rm -it "${device_args[@]}" "${group_args[@]}" "$IMAGE" --list-devices
  exit 0
fi

model="${1:-}"
if [[ -z "$model" ]]; then
  echo "Usage: $0 /path/to/model.gguf [extra llama-server args...]" >&2
  echo "       $0 --list-devices" >&2
  exit 2
fi
shift

model_abs="$(realpath "$model")"
model_dir="$(dirname "$model_abs")"
model_file="$(basename "$model_abs")"

exec docker run --rm -it \
  "${device_args[@]}" \
  "${group_args[@]}" \
  -p "${PORT}:${PORT}" \
  -v "${model_dir}:/models:${VOLUME_OPTS}" \
  "$IMAGE" \
  -m "/models/${model_file}" \
  --host 0.0.0.0 --port "$PORT" \
  --device "$VULKAN_DEVICE" \
  --ctx-size "$CTX_SIZE" \
  --flash-attn on \
  --no-webui \
  "$@"
