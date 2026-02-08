# PyTorch 2.8 with MosaicML Composer - Setup Notes
Date: 2026-02-08

## Recent Fixes Applied

1. **Deprecated Environment Variable** (`train.yaml`)
   - Changed `NCCL_ASYNC_ERROR_HANDLING` → `TORCH_NCCL_ASYNC_ERROR_HANDLING`

2. **Invalid cd Command** (`commands.sh`)
   - Removed `cd /composer_geneformer_pretrain` (path doesn't exist)
   - Script already runs from correct directory via `train.yaml` command

3. **Removed `fsdp_config`** (`train.py`)
   - Commented out - deprecated in Composer 0.32+

---

## PyTorch 2.8 Requirements

**The core challenge**: MosaicML Composer 0.32.1 pins PyTorch <2.8, so you must force-reinstall after Composer installs.

**Working Setup** (tested on A10 8 GPUs and H100 16 GPUs):

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

**`commands.sh`** (force-reinstall PyTorch 2.8 after Composer):
```bash
pip install -r requirements.txt

pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

---

## MLflow Limitation

- `mosaicml[mlflow]==0.32.1` requires `mlflow>=2.14.1,<3.0`
- **Cannot use mlflow>=3.6.0** with this Composer version
- If you install mlflow>=3.6.0 first, `mosaicml[mlflow]` will downgrade it

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

## Branch: mmt_test_torch28

This branch contains the PyTorch 2.8 configuration.
For PyTorch 2.7, see branch: mmt_test_torch27


###############################
i do want to revisit the mlflow -- previously when others tried 
```
%pip install mlflow>=3.6.0 mosaicml==0.32.1
%pip install mosaicml[mlflow]==0.32.1
%pip install mosaicml[nlp,streaming]
%pip install mosaicml-streaming==0.13.0
%pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
``` 

this sequence was used ... but mosaicml[mlflow]==0.32.1 conflicts with  mlflow>=3.6.0 -- > did installing mosaicml[mlflow]==0.32.1 afters overwrite the prior mlflow>=3.6.0?


Good question. Yes, step 2 would overwrite the mlflow>=3.6.0 from step 1.
Here's what happens in that sequence:
pip install mlflow>=3.6.0 mosaicml==0.32.1 → Installs mlflow 3.6+ and mosaicml (without mlflow extra)
pip install mosaicml[mlflow]==0.32.1 → This has dependency mlflow>=2.14.1,<3.0, so pip downgrades mlflow to ~2.x
Steps 3-5 don't touch mlflow
Result: You end up with mlflow <3.0, not >=3.6.0.
The sequence is effectively misleading - the mlflow>=3.6.0 install in step 1 gets undone by step 2. The only way to keep mlflow>=3.6.0 is to:
Install mosaicml==0.32.1 without the [mlflow] extra
Then install mlflow>=3.6.0 separately
Accept that some mlflow integration features in Composer might not work correctly (since Composer 0.32.1 was tested against mlflow <3.0)
Bottom line: With mosaicml[mlflow]==0.32.1, you're locked to mlflow <3.0. If you need mlflow>=3.6.0, you'd have to:
Use mosaicml==0.32.1 (no mlflow extra) and manually install mlflow>=3.6.0
Or wait for a newer Composer release that supports mlflow 3.x

>>> test the non-mlflow-extra approach on a branch?