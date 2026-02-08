# PyTorch 2.8 with MosaicML Composer - Setup Notes
Date: 2026-02-08

---

## The Challenge

MosaicML Composer 0.32.1 has two conflicting constraints:
- `torch<2.7.1` - prevents torch 2.8
- `mlflow>=2.14.1,<3.0` (via `[mlflow]` extra) - prevents mlflow 3.6+

---

## Solution 1: PyTorch 2.8 + MLflow <3.0 (Branch: `mmt_test_torch28`)

Use `mosaicml[mlflow]==0.32.1` and force-reinstall torch 2.8 afterwards.

**`dependencies.yaml`**:
```yaml
version: "4"
dependencies:
  - mosaicml[mlflow]==0.32.1
  - mosaicml-streaming==0.13.0
  - transformers==4.44.0
  - datasets==2.21.0
  - boto3
  - omegaconf
  - tqdm
```

**`commands.sh`**:
```bash
pip install -r requirements.txt

# Uninstall and reinstall torch 2.8
pip uninstall -y torch torchvision torchaudio
pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

**Result**: torch 2.8.0, mlflow ~2.x

---

## Solution 2: PyTorch 2.8 + MLflow >=3.6 (Branch: `mmt_test_torch28_mlflow36`)

Use `mosaicml==0.32.1` **WITHOUT** the `[mlflow]` extra, install mlflow separately.

**`dependencies.yaml`**:
```yaml
version: "4"
dependencies:
  - mosaicml==0.32.1          # NO [mlflow] extra!
  - mlflow>=3.6.0             # Install separately
  - mosaicml-streaming==0.13.0
  - transformers==4.44.0
  - datasets==2.21.0
  - boto3
  - omegaconf
  - tqdm
```

**`commands.sh`**:
```bash
pip install -r requirements.txt

# Uninstall and reinstall torch 2.8
pip uninstall -y torch torchvision torchaudio
pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126

# Verify versions
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import mlflow; print(f'MLflow: {mlflow.__version__}')"
```

**Result**: torch 2.8.0, mlflow 3.9.0

**Caveat**: Some Composer mlflow integration features may not work correctly (Composer 0.32.1 was tested against mlflow <3.0).

---

## Important Notes

### CUDA Version
- Use `cu126` index URL to match Databricks environment (not `cu128`)
- Databricks shows `torch 2.7.0+cu126` by default

### torch Installation Order
1. `pip install -r requirements.txt` → installs torch 2.7 via mosaicml
2. `pip uninstall -y torch torchvision torchaudio` → removes 2.7
3. `pip install torch==2.8.0...` → installs 2.8

---

## Fixes Applied

1. **Deprecated Environment Variable** (`train.yaml`)
   - Changed `NCCL_ASYNC_ERROR_HANDLING` → `TORCH_NCCL_ASYNC_ERROR_HANDLING`

2. **Invalid cd Command** (`commands.sh`)
   - Removed `cd /composer_geneformer_pretrain` (path doesn't exist)
   - Script already runs from correct directory via `train.yaml` command

3. **Removed `fsdp_config`** (`train.py`)
   - Commented out - deprecated in Composer 0.32+

---

## Multi-Node H100 (16 GPUs)

- Requires NCCL environment variables for multi-node communication
- If you see "8/16 clients joined" timeout, it's an infrastructure networking issue, not code
- H100 8 GPUs (single node) is a reliable fallback

### NCCL Environment Variables (in `train.yaml`):
```yaml
environment:
  env_variables:
    NCCL_DEBUG: "INFO"
    NCCL_SOCKET_IFNAME: "eth"
    NCCL_IB_DISABLE: "1"
    TORCH_NCCL_ASYNC_ERROR_HANDLING: "1"
    NCCL_TIMEOUT: "1800"
    TORCH_DISTRIBUTED_DEBUG: "DETAIL"
    TORCH_NCCL_BLOCKING_WAIT: "1"
    TORCH_DIST_INIT_BARRIER_TIMEOUT: "1800"
```

---

## Branches

| Branch | torch | mlflow | Status |
|--------|-------|--------|--------|
| `mmt_test_torch27` | 2.7 | <3.0 | Baseline |
| `mmt_test_torch28` | 2.8 | <3.0 | Working |
| `mmt_test_torch28_mlflow36` | 2.8 | >=3.6 | Testing |

---

## Background: Why the Original Sequence Didn't Work

The originally tested sequence:
```
%pip install mlflow>=3.6.0 mosaicml==0.32.1
%pip install mosaicml[mlflow]==0.32.1        # ← This downgrades mlflow!
%pip install mosaicml[nlp,streaming]
%pip install mosaicml-streaming==0.13.0
%pip install --force-reinstall torch==2.8.0 ...
```

**Problem**: Step 2 (`mosaicml[mlflow]==0.32.1`) has dependency `mlflow>=2.14.1,<3.0`, so pip **downgrades** mlflow from 3.6+ back to ~2.x.

**Solution**: Don't use the `[mlflow]` extra. Install `mosaicml==0.32.1` and `mlflow>=3.6.0` as separate packages.
