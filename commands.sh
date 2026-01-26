set -euo pipefail

REPO_DIR="$(pwd)"
echo ">>> Repo dir: ${REPO_DIR}"

echo ">>> Installing Geneformer (python package)"
# Use bash explicitly; /bin/sh may be dash and does not support `set -o pipefail`.
bash geneformer_prep.sh

echo ">>> Installing repo dependencies"
python -m pip install -r requirements.txt

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
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29400}"

if [ "${NNODES}" != "1" ]; then
  echo ">>> Multi-node torchrun: NNODES=${NNODES} NODE_RANK=${NODE_RANK} RDZV=${MASTER_ADDR}:${MASTER_PORT}"
  torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    train.py parameters_sgcli_smoke.yaml
else
  echo ">>> Single-node torchrun (--standalone)"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train.py parameters_sgcli_smoke.yaml
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


