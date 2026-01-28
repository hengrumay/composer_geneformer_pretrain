set -euo pipefail

REPO_DIR="$(pwd)"
echo ">>> Repo dir: ${REPO_DIR}"

echo ">>> Skipping Geneformer pip install (not required for training; avoids heavy deps like anndata/scanpy/ray)"

echo ">>> Installing repo dependencies"
# IMPORTANT: Do not let pip resolve dependencies here; it can downgrade torch and break the
# Databricks runtime (especially multi-node torchrun). We only ensure our top-level pkgs exist.
python -m pip install --no-deps -r requirements.txt

# Ensure Composer is available on Python 3.12.
# Some older MosaicML releases may not have py312 wheels; prefer `composer` if available.
echo ">>> Ensuring composer is importable (py312-safe)"
set +e
python - <<'PY'
import sys
try:
    import composer  # noqa: F401
    print("composer: already importable")
    sys.exit(0)
except Exception as e:
    print("composer import failed:", repr(e))
    sys.exit(1)
PY
COMPOSER_OK=$?
set -e
if [ "${COMPOSER_OK}" != "0" ]; then
  echo ">>> Installing Composer (attempt 1): pip install --no-deps composer"
  python -m pip install --no-deps composer || true
  set +e
  python - <<'PY'
import sys
try:
    import composer  # noqa: F401
    print("composer: import OK after installing 'composer'")
    sys.exit(0)
except Exception as e:
    print("composer still not importable:", repr(e))
    sys.exit(1)
PY
  COMPOSER_OK=$?
  set -e
fi
if [ "${COMPOSER_OK}" != "0" ]; then
  echo ">>> Installing Composer (attempt 2): pip install --no-deps mosaicml"
  python -m pip install --no-deps mosaicml || true
  set +e
  python - <<'PY'
import sys
try:
    import composer  # noqa: F401
    print("composer: import OK after installing 'mosaicml'")
    sys.exit(0)
except Exception as e:
    print("composer still not importable:", repr(e))
    sys.exit(1)
PY
  COMPOSER_OK=$?
  set -e

  # If mosaicml installed but is missing its CLI runtime module ("mcli"), install mosaicml-cli.
  if [ "${COMPOSER_OK}" != "0" ]; then
    echo ">>> Installing missing mosaicml runtime dep (attempt 2a): pip install --no-deps mosaicml-cli"
    python -m pip install --no-deps mosaicml-cli || true
    set +e
    python - <<'PY'
import sys
try:
    import composer  # noqa: F401
    print("composer: import OK after installing 'mosaicml-cli'")
    sys.exit(0)
except Exception as e:
    print("composer still not importable:", repr(e))
    sys.exit(1)
PY
    COMPOSER_OK=$?
    set -e
  fi
fi
if [ "${COMPOSER_OK}" != "0" ]; then
  echo "ERROR: 'composer' module is still not importable after install attempts."
  echo "Tried: pip install --no-deps composer  (then)  pip install --no-deps mosaicml  (then)  pip install --no-deps mosaicml-cli"
  exit 12
fi

# Create working directory (config can override)
mkdir -p /pretrain/temp

echo ">>> Starting training"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Multi-node support: set NNODES>1 in workload.yaml env_variables.
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29400}"

# -----------------------------------------------------------------------------
# Distributed toggle + parameters from parameters_sgcli.yaml
# -----------------------------------------------------------------------------
# You can still override everything via environment variables, but by default we
# read `distributed:` config from parameters_sgcli.yaml so you can switch modes
# without editing workload.yaml.
if command -v python3 >/dev/null 2>&1 && [ -f "parameters_sgcli.yaml" ]; then
  eval "$(
    python3 - <<'PY'
import shlex, sys
try:
    import yaml
except Exception as e:
    print(f"echo 'WARNING: cannot import pyyaml to read parameters_sgcli.yaml: {e}'", file=sys.stderr)
    raise SystemExit(0)

cfg = yaml.safe_load(open("parameters_sgcli.yaml")) or {}
dcfg = (cfg.get("distributed") or {})
enabled = bool(dcfg.get("enabled", False))
mode = (dcfg.get("mode") or "").strip() or None

# Defaults if not enabled
if not enabled:
    mode = None

def export(name: str, value):
    if value is None:
        return
    print(f"export {name}={shlex.quote(str(value))}")

export("DIST_CFG_ENABLED", "1" if enabled else "0")
export("DIST_CFG_MODE", mode or "")

torchrun_cfg = dcfg.get("torchrun") or {}
sr_cfg = dcfg.get("serverless_gpu") or {}

# torchrun overrides
nppn = torchrun_cfg.get("nproc_per_node", None)
if isinstance(nppn, str) and nppn.strip().lower() == "auto":
    nppn = None
export("DIST_TORCHRUN_NPROC_PER_NODE", nppn)

# serverless_gpu overrides
export("DIST_SERVERLESS_GPU_GPUS", sr_cfg.get("gpus", None))
export("DIST_SERVERLESS_GPU_GPU_TYPE", sr_cfg.get("gpu_type", None))
export("DIST_SERVERLESS_GPU_REMOTE", sr_cfg.get("remote", None))
export("DIST_SERVERLESS_GPU_MANUAL_INIT_PROCESS_GROUP", sr_cfg.get("manual_init_process_group", None))
export("DIST_SERVERLESS_GPU_DDP_BACKEND", sr_cfg.get("ddp_backend", None))
PY
  )"
fi

# Auto-detect GPUs per node unless NPROC_PER_NODE is explicitly set.
if [ -z "${NPROC_PER_NODE:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    # Fallback: assume 1 GPU
    NPROC_PER_NODE="1"
  fi
fi
echo ">>> Using NPROC_PER_NODE=${NPROC_PER_NODE}"

# Apply optional distributed config from YAML (unless explicitly overridden by env).
# - DIST_CFG_MODE can be: "none" | "torchrun" | "serverless_gpu"
DIST_CFG_MODE="${DIST_CFG_MODE:-}"
if [ -n "${DIST_CFG_MODE}" ] && [ -z "${DISTRIBUTED_MODE:-}" ]; then
  export DISTRIBUTED_MODE="${DIST_CFG_MODE}"
fi
DISTRIBUTED_MODE="${DISTRIBUTED_MODE:-}"

# Optional override: torchrun nproc_per_node from YAML
if [ -n "${DIST_TORCHRUN_NPROC_PER_NODE:-}" ] && [ -z "${NPROC_PER_NODE_EXPLICIT:-}" ]; then
  export NPROC_PER_NODE="${DIST_TORCHRUN_NPROC_PER_NODE}"
  export NPROC_PER_NODE_EXPLICIT=1
  echo ">>> Overriding NPROC_PER_NODE from parameters_sgcli.yaml: ${NPROC_PER_NODE}"
fi

USE_SERVERLESS_GPU_DISTRIBUTED="${USE_SERVERLESS_GPU_DISTRIBUTED:-0}"
if [ "${DISTRIBUTED_MODE}" = "serverless_gpu" ]; then
  USE_SERVERLESS_GPU_DISTRIBUTED=1
fi

if [ "${USE_SERVERLESS_GPU_DISTRIBUTED}" = "1" ] || [ "${USE_SERVERLESS_GPU_DISTRIBUTED}" = "true" ]; then
  if [ "${NNODES}" != "1" ]; then
    echo "ERROR: USE_SERVERLESS_GPU_DISTRIBUTED is currently supported only for NNODES=1."
    exit 2
  fi

  # These env vars are consumed by train.py to configure serverless_gpu.launcher.distributed(...)
  export USE_SERVERLESS_GPU_DISTRIBUTED=1
  export SERVERLESS_GPU_GPUS="${SERVERLESS_GPU_GPUS:-${DIST_SERVERLESS_GPU_GPUS:-${NPROC_PER_NODE}}}"
  export SERVERLESS_GPU_GPU_TYPE="${SERVERLESS_GPU_GPU_TYPE:-${DIST_SERVERLESS_GPU_GPU_TYPE:-${GPU_TYPE:-}}}"
  export SERVERLESS_GPU_REMOTE="${SERVERLESS_GPU_REMOTE:-${DIST_SERVERLESS_GPU_REMOTE:-false}}"
  export SERVERLESS_GPU_MANUAL_INIT_PROCESS_GROUP="${SERVERLESS_GPU_MANUAL_INIT_PROCESS_GROUP:-${DIST_SERVERLESS_GPU_MANUAL_INIT_PROCESS_GROUP:-1}}"
  export SERVERLESS_GPU_DDP_BACKEND="${SERVERLESS_GPU_DDP_BACKEND:-${DIST_SERVERLESS_GPU_DDP_BACKEND:-nccl}}"

  echo ">>> Serverless GPU @distributed enabled"
  echo ">>> SERVERLESS_GPU_GPUS=${SERVERLESS_GPU_GPUS}"
  echo ">>> SERVERLESS_GPU_GPU_TYPE=${SERVERLESS_GPU_GPU_TYPE:-<unset>}"
  echo ">>> SERVERLESS_GPU_REMOTE=${SERVERLESS_GPU_REMOTE}"
  echo ">>> SERVERLESS_GPU_MANUAL_INIT_PROCESS_GROUP=${SERVERLESS_GPU_MANUAL_INIT_PROCESS_GROUP}"
  echo ">>> SERVERLESS_GPU_DDP_BACKEND=${SERVERLESS_GPU_DDP_BACKEND}"

  # Do NOT use torchrun here; the serverless_gpu launcher will create the distributed workers.
  python train.py parameters_sgcli.yaml
  exit 0
fi

echo ">>> Sanity check (per node): torch cuda + env"
python - <<'PY'
import os
import torch

print("cuda:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
for k in ("NNODES", "WORLD_SIZE", "RANK", "LOCAL_RANK", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT", "RDZV_ID"):
    print(f"env {k}:", os.getenv(k))
PY

if [ "${NNODES}" != "1" ]; then
  if [ -z "${MASTER_ADDR}" ] || [ -z "${NODE_RANK}" ]; then
    echo "ERROR: Multi-node run requested (NNODES=${NNODES}) but MASTER_ADDR/NODE_RANK not set."
    echo "Expected sgcli/runtime to provide: MASTER_ADDR, MASTER_PORT, NODE_RANK (and optionally NNODES)."
    echo "--- Environment (filtered) ---"
    env | egrep '^(NNODES|NODE_RANK|MASTER_ADDR|MASTER_PORT|RANK|WORLD_SIZE|LOCAL_RANK)=' || true
    exit 2
  fi

  # Serverless note: multi-host torchrun can be blocked by serverless networking policies.
  # We probe basic reachability to fail fast (instead of "hanging" at rendezvous).
  echo ">>> Rendezvous probe: DNS + TCP connect to MASTER_ADDR:MASTER_PORT"
  python - <<'PY'
import os, socket, sys
host = os.environ.get("MASTER_ADDR")
port = int(os.environ.get("MASTER_PORT", "0") or "0")
print("MASTER_ADDR:", host)
print("MASTER_PORT:", port)
if not host or not port:
    print("probe: missing MASTER_ADDR/MASTER_PORT; skipping")
    raise SystemExit(0)
try:
    ip = socket.gethostbyname(host)
    print("resolved:", host, "->", ip)
except Exception as e:
    print("resolve FAIL:", e)
    raise SystemExit(0)

s = socket.socket()
s.settimeout(3)
try:
    s.connect((ip, port))
    print("connect OK")
except Exception as e:
    print("connect FAIL:", e)
    print("NOTE: If connect fails, multi-node torchrun rendezvous will hang/fail on Serverless.")
    print("      Recommended: use distributed.mode=serverless_gpu (Databricks-managed launcher) instead of torchrun.")
    # Fail fast so we don't burn time "hanging" in torchrun rendezvous.
    raise SystemExit(42)
finally:
    try:
        s.close()
    except Exception:
        pass
PY

  # Resolve MASTER_ADDR to an IP if possible (helps avoid per-node DNS quirks).
  if command -v getent >/dev/null 2>&1; then
    MASTER_ADDR_IP="$(getent hosts "${MASTER_ADDR}" | awk '{print $1}' | head -n 1 || true)"
    if [ -n "${MASTER_ADDR_IP}" ]; then
      MASTER_ADDR="${MASTER_ADDR_IP}"
    fi
  fi

  echo ">>> Multi-node torchrun: NNODES=${NNODES} NODE_RANK=${NODE_RANK} RDZV=${MASTER_ADDR}:${MASTER_PORT}"
  echo ">>> Hostname: $(hostname)"
  echo ">>> Env (filtered):"
  env | egrep '^(NNODES|NODE_RANK|MASTER_ADDR|MASTER_PORT|RANK|WORLD_SIZE|LOCAL_RANK)=' || true

  RDZV_ID="${RDZV_ID:-geneformer-sgcli-mmt-test}"
  echo ">>> RDZV_ID=${RDZV_ID}"

  # Extra torchrun logging (only if supported by this torch version).
  TORCHRUN_EXTRA_ARGS=()
  if torchrun --help 2>/dev/null | grep -q -- '--tee'; then
    TORCHRUN_EXTRA_ARGS+=(--tee 3)
  fi
  if torchrun --help 2>/dev/null | grep -q -- '--log_dir'; then
    TORCHRUN_EXTRA_ARGS+=(--log_dir /tmp/torchrun_logs)
  fi
  if [ "${#TORCHRUN_EXTRA_ARGS[@]}" -gt 0 ]; then
    echo ">>> torchrun extra args: ${TORCHRUN_EXTRA_ARGS[*]}"
  fi

  echo ">>> torchrun: waiting for all nodes to rendezvous (if one node died, others will wait until timeout)"
  torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --rdzv_backend=c10d \
    --rdzv_id="${RDZV_ID}" \
    --rdzv_conf join_timeout=1800,timeout=1800,read_timeout=1800 \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    "${TORCHRUN_EXTRA_ARGS[@]}" \
    train.py parameters_sgcli.yaml
else
  if [ "${NPROC_PER_NODE}" = "1" ]; then
    # For single-GPU smoke tests, avoid torchrun/distributed rendezvous entirely.
    # Some environments can segfault inside torch.distributed even with 1 process.
    echo ">>> Single-node single-process run (no torchrun): python train.py"
    python train.py parameters_sgcli.yaml
  else
    echo ">>> Single-node torchrun (--standalone)"
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train.py parameters_sgcli.yaml
  fi
fi

#sh download_dataset.sh
##################################################
##the following code is only needed to copy the dataset from s3 to locally create streaming dataset
#echo ">>> Configuring aws"
#cd /
#apt update
#apt install unzip
#curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
#unzip awscliv2.zip
#sudo ./aws/install

#echo ">>> Copying data from s3.. might take few mins"
#cd /composer_geneformer_pretrain
#mkdir /Geneformer/data/dataset -p
#aws s3 cp s3://srijit-nair-sandbox-bucket/geneformer/data/token_dictionary.pkl /Geneformer/data/token_dictionary.pkl
#aws s3 cp --recursive s3://srijit-nair-sandbox-bucket/geneformer/data/dataset /Geneformer/data/dataset
#mkdir /Geneformer/data -p
#aws s3 cp --quiet --recursive s3://srijit-nair-sandbox-bucket/geneformer/data /Geneformer/data
#echo "done"
#python create_mds.py
