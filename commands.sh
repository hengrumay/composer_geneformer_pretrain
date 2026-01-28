set -euo pipefail

REPO_DIR="$(pwd)"
echo ">>> Repo dir: ${REPO_DIR}"

echo ">>> Skipping Geneformer pip install (not required for training; avoids heavy deps like anndata/scanpy/ray)"

echo ">>> Installing repo dependencies"
# IMPORTANT: Do not let pip resolve dependencies here; it can downgrade torch and break the
# Databricks runtime (especially multi-node torchrun). We only ensure our top-level pkgs exist.
python -m pip install --no-deps -r requirements.txt

# Create working directory (config can override)
mkdir -p /pretrain/temp

echo ">>> Starting training (single-node torchrun)"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

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

# Multi-node support (A10 multi-GPU == multi-node): set NNODES>1 in workload.yaml env_variables.
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29400}"

if [ "${NNODES}" != "1" ]; then
  if [ -z "${MASTER_ADDR}" ] || [ -z "${NODE_RANK}" ]; then
    echo "ERROR: Multi-node run requested (NNODES=${NNODES}) but MASTER_ADDR/NODE_RANK not set."
    echo "Expected sgcli/runtime to provide: MASTER_ADDR, MASTER_PORT, NODE_RANK (and optionally NNODES)."
    echo "--- Environment (filtered) ---"
    env | egrep '^(NNODES|NODE_RANK|MASTER_ADDR|MASTER_PORT|RANK|WORLD_SIZE|LOCAL_RANK)=' || true
    exit 2
  fi

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

  torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --rdzv_backend=c10d \
    --rdzv_conf timeout=900 \
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


