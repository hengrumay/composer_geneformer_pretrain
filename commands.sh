set -euo pipefail

REPO_DIR="$(pwd)"
echo ">>> Repo dir: ${REPO_DIR}"
if command -v git >/dev/null 2>&1; then
  echo ">>> Repo git: $(git rev-parse --short HEAD 2>/dev/null || echo '<unknown>')"
fi

echo ">>> Skipping Geneformer pip install (not required for training; avoids heavy deps like anndata/scanpy/ray)"

echo ">>> Installing repo dependencies"
echo ">>> Runtime versions (pre-install)"
python - <<'PY'
import sys
def v(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "<unknown>")
    except Exception as e:
        return f"<not importable: {e}>"

print("python:", sys.version.split()[0])
print("torch:", v("torch"))
print("torchvision:", v("torchvision"))
print("mlflow:", v("mlflow"))
PY

python - <<'PY'
from importlib.metadata import version, PackageNotFoundError

# Constrain only Databricks-controlled packages you really don't want pip to downgrade.
# Do NOT constrain torch/torchvision here (DBR uses local build suffixes).
names = ["mlflow", "databricks-sdk"]

lines = []
for n in names:
    try:
        v = version(n)
    except PackageNotFoundError:
        continue
    lines.append(f"{n}=={v}")

path = "/tmp/pip_constraints.txt"
with open(path, "w") as f:
    f.write("\n".join(lines) + ("\n" if lines else ""))
print("Wrote constraints:", path)
for ln in lines:
    print("  ", ln)
PY

echo ">>> Installing repo requirements (with constraints)"
python -m pip install -r requirements.txt -c /tmp/pip_constraints.txt --upgrade-strategy only-if-needed

# If you keep composer==0.32.1, pip check will always complain on DBR torch 2.7.1/torchvision 0.22.1.
# But import may still work. We'll allow it and not fail the job on pip check noise.
echo ">>> Installing composer (no-deps to avoid torch/torchvision downgrades)"
python -m pip install --no-deps "composer==0.32.1"

echo ">>> Verifying composer import"
python - <<'PY'
import composer
import torch, torchvision
print("composer:", getattr(composer, "__version__", "<unknown>"))
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
PY

echo ">>> pip check (non-fatal on Databricks)"
python -m pip check || true

# Create working directory (config can override)
mkdir -p /pretrain/temp

echo ">>> Starting training"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING="${MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING:-true}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Multi-node parameters (provided by serverless/scg-cli)
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-}"   # do NOT default; must be provided by platform for multi-node

# Auto-detect GPUs per node unless NPROC_PER_NODE is explicitly set.
if [ -z "${NPROC_PER_NODE:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    NPROC_PER_NODE="1"
  fi
fi

# If we are doing multi-worker A10 torchrun, force 1 proc per node.
if [ "${NNODES}" != "1" ]; then
  NPROC_PER_NODE="1"
fi

echo ">>> Using NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE}"

echo ">>> Sanity check (per node): torch cuda + env"
python - <<'PY'
import os, torch, socket
print("host:", socket.gethostname())
print("cuda:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
for k in ("NNODES","WORLD_SIZE","RANK","LOCAL_RANK","NODE_RANK","MASTER_ADDR","MASTER_PORT"):
    print(f"env {k}:", os.getenv(k))
PY

if [ "${NNODES}" != "1" ]; then
  if [ -z "${MASTER_ADDR}" ] || [ -z "${MASTER_PORT}" ] || [ -z "${NODE_RANK}" ]; then
    echo "ERROR: Multi-node run requested but MASTER_ADDR/MASTER_PORT/NODE_RANK not set."
    env | egrep '^(NNODES|NODE_RANK|MASTER_ADDR|MASTER_PORT|RANK|WORLD_SIZE|LOCAL_RANK)=' || true
    exit 2
  fi

  # IMPORTANT: do not fail immediately; rank0 may not be listening yet.
  echo ">>> Rendezvous probe: retry connect to MASTER_ADDR:MASTER_PORT (up to 180s)"
  python - <<'PY'
import os, socket, time
host = os.environ["MASTER_ADDR"]
port = int(os.environ["MASTER_PORT"])
ip = socket.gethostbyname(host)
print("resolved:", host, "->", ip, "port", port)
deadline = time.time() + 180
last = None
while time.time() < deadline:
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect((ip, port))
        print("connect OK")
        raise SystemExit(0)
    except Exception as e:
        last = e
        time.sleep(2)
    finally:
        try: s.close()
        except Exception: pass
print("connect still failing after retries:", last)
# Do NOT hard-fail here; torchrun itself may still succeed depending on timing.
print("continuing into torchrun anyway...")
PY

  echo ">>> Multi-node torchrun launch"
  echo ">>> Hostname: $(hostname)"
  echo ">>> Env (filtered):"
  env | egrep '^(NNODES|NODE_RANK|MASTER_ADDR|MASTER_PORT|RANK|WORLD_SIZE|LOCAL_RANK)=' || true

  TORCHRUN_EXTRA_ARGS=()
  if torchrun --help 2>/dev/null | grep -q -- '--tee'; then
    TORCHRUN_EXTRA_ARGS+=(--tee 3)
  fi
  if torchrun --help 2>/dev/null | grep -q -- '--log_dir'; then
    TORCHRUN_EXTRA_ARGS+=(--log_dir /tmp/torchrun_logs)
  fi

  torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${TORCHRUN_EXTRA_ARGS[@]}" \
    train.py parameters_sgcli.yaml

else
  # Single-node path
  if [ "${NPROC_PER_NODE}" = "1" ]; then
    echo ">>> Single-node single-process run (no torchrun): python train.py"
    python train.py parameters_sgcli.yaml
  else
    echo ">>> Single-node torchrun (--standalone)"
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train.py parameters_sgcli.yaml
  fi
fi