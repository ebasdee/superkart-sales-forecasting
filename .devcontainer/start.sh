#!/usr/bin/env bash
# SuperKart - runs automatically every time the Codespace starts (postStartCommand in devcontainer.json).
# Manual use after a code change (e.g. after git pull):  bash .devcontainer/start.sh --rebuild
set -u
cd "$(dirname "$0")/.."                       # repository root
REBUILD="${1:-}"
log() { echo "[superkart-start] $*"; }

# 1. Wait for the Docker daemon
for _ in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
docker info >/dev/null 2>&1 || { log "Docker is not available - stopping."; exit 1; }

# 2. Issue 1 fix: the leftover iptables-legacy FORWARD DROP policy blocks container-to-container traffic.
#    It resets on every Codespace restart, so it is re-applied here.
sudo iptables-legacy -P FORWARD ACCEPT 2>/dev/null && log "iptables-legacy FORWARD policy set to ACCEPT"

# 3. Network, images and containers (built only when missing, or when --rebuild is given)
docker network create superkart-net >/dev/null 2>&1 || true
for app in backend frontend; do
  port=$([ "$app" = backend ] && echo 7860 || echo 8501)
  if [ "$REBUILD" = "--rebuild" ] || ! docker image inspect "superkart-$app" >/dev/null 2>&1; then
    log "Building image superkart-$app"
    docker build -t "superkart-$app" "./$app" || { log "Build failed for $app"; exit 1; }
    docker rm -f "$app" >/dev/null 2>&1 || true
  fi
  if docker container inspect "$app" >/dev/null 2>&1; then
    docker start "$app" >/dev/null && log "Started existing container $app"
  else
    # Planning assistant: the Codespaces secret ANTHROPIC_API_KEY is passed into the containers
    docker run -d --name "$app" --network superkart-net --restart unless-stopped \
      -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" -p "$port:$port" "superkart-$app" >/dev/null && log "Created container $app"
  fi
done

# 4. Wait until both apps answer locally
for port in 7860 8501; do
  for _ in $(seq 1 60); do curl -s -o /dev/null "http://localhost:$port" && break; sleep 2; done
done

# 5. Issue 2 fix: refresh the public port tunnels (a stale tunnel returns an empty 404 after a restart).
#    First wait until Codespaces has registered both forwarded ports, then switch them private -> public.
if [ -n "${CODESPACE_NAME:-}" ]; then
  for _ in $(seq 1 40); do
    ports=$(gh codespace ports -c "$CODESPACE_NAME" --json sourcePort -q '.[].sourcePort' 2>/dev/null)
    echo "$ports" | grep -qx 7860 && echo "$ports" | grep -qx 8501 && break
    sleep 3
  done
  for _ in $(seq 1 10); do
    gh codespace ports visibility 7860:private 8501:private -c "$CODESPACE_NAME" >/dev/null 2>&1 \
      && gh codespace ports visibility 7860:public 8501:public -c "$CODESPACE_NAME" >/dev/null 2>&1 \
      && { log "Ports 7860 and 8501 are public"; break; }
    sleep 6
  done
  log "App: https://$CODESPACE_NAME-8501.app.github.dev"
  log "API: https://$CODESPACE_NAME-7860.app.github.dev"
fi
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
