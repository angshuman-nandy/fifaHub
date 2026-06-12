#!/usr/bin/env bash
# Steps 3-6 of instructions.md: local verification, GitHub repo, HF Space, post-deploy checks.
# Run this once .env has real values (not REPLACE_ME).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "ERROR: .env not found" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

fail=0
for var in HF_TOKEN ANTHROPIC_API_KEY GH_REPO HF_SPACE; do
  val="${!var:-}"
  if [ -z "$val" ] || [ "$val" = "REPLACE_ME" ]; then
    echo "ERROR: $var is not set in .env" >&2
    fail=1
  fi
done
for repo_var in GH_REPO HF_SPACE; do
  val="${!repo_var:-}"
  if [ -n "$val" ] && ! [[ "$val" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
    echo "ERROR: $repo_var must be in 'owner/name' format (got '$val')" >&2
    fail=1
  fi
done
[ "$fail" = "1" ] && exit 1

for cmd in git python3 pip; do
  command -v "$cmd" >/dev/null || { echo "ERROR: $cmd not found" >&2; exit 1; }
done

PY=.venv/bin/python3
PIP=.venv/bin/pip
[ -x "$PY" ] || { python3 -m venv .venv; }

echo "==> Step 3: local verification (real ANTHROPIC_API_KEY)"
"$PIP" install -q -r requirements.txt
.venv/bin/uvicorn app:app --port 7860 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
sleep 2
curl -sf localhost:7860/healthz >/dev/null
curl -sf localhost:7860/ | grep -q "PITCH"
curl -sf localhost:7860/api/claude -H 'content-type: application/json' \
  -d '{"model":"claude-sonnet-4-6","max_tokens":50,"messages":[{"role":"user","content":"Say OK"}]}' \
  | grep -q '"text"'
kill $SERVER_PID 2>/dev/null || true
trap - EXIT
echo "    local checks passed"

echo "==> Step 4: push to existing GitHub repo"
if [ ! -d .git ]; then
  git init -b main
fi
git add -A
git commit -m "WC26 hub: FastAPI + static frontend for HF Spaces" || echo "    (nothing new to commit)"

git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/${GH_REPO}.git"
if GIT_TERMINAL_PROMPT=0 git push -u origin main; then
  echo "    pushed to https://github.com/$GH_REPO"
else
  echo "    WARNING: push to GitHub failed (no cached credentials for https). Push manually, e.g.:" >&2
  echo "      git push -u origin main" >&2
fi
echo "    NOTE: .github/workflows/sync-to-hf.yml needs an 'HF_TOKEN' repo secret and an 'HF_SPACE'" >&2
echo "    repo variable (Settings > Secrets and variables > Actions) for future auto-sync on push." >&2

echo "==> Step 5: Hugging Face Space"
"$PIP" install -q huggingface_hub
"$PY" - <<'PYEOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
space = os.environ["HF_SPACE"]
api.create_repo(repo_id=space, repo_type="space", space_sdk="docker", exist_ok=True)
api.add_space_secret(repo_id=space, key="ANTHROPIC_API_KEY", value=os.environ["ANTHROPIC_API_KEY"])
PYEOF

echo "==> Trigger first deploy"
git push --force "https://oauth2:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE}" main

echo "==> Step 6: post-deploy verification (waiting for build, up to 10 min)"
"$PY" - <<'PYEOF'
import os, sys, time
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
space = os.environ["HF_SPACE"]
deadline = time.time() + 600
last = None
while time.time() < deadline:
    stage = api.get_space_runtime(space).stage
    if stage != last:
        print("    stage:", stage)
        last = stage
    if stage == "RUNNING":
        sys.exit(0)
    if stage == "BUILD_ERROR":
        print(api.get_space_runtime(space))
        sys.exit(1)
    time.sleep(15)
print("    timed out waiting for RUNNING")
sys.exit(2)
PYEOF

OWNER_NAME=$(echo "$HF_SPACE" | tr '[:upper:]' '[:lower:]' | tr '/' '-')
URL="https://${OWNER_NAME}.hf.space"
curl -sf "$URL/healthz" >/dev/null
curl -sf "$URL/" | grep -q "PITCH"
curl -sf "$URL/api/claude" -H 'content-type: application/json' \
  -d '{"model":"claude-sonnet-4-6","max_tokens":50,"messages":[{"role":"user","content":"Say OK"}]}' \
  | grep -q '"text"'

echo
echo "==> Done"
echo "Space URL: $URL"
echo "Repo URL:  https://github.com/$GH_REPO"
