# Geneformer SGCLI smoke test

This folder is a minimal `sgcli` config to run the `hengrumay/composer_geneformer_pretrain` repo on Databricks Serverless GPU.

## Quick map for `workload.yaml` runs
- `workload.yaml`: sgcli job spec (local snapshot). Uses `requirements.yaml` and runs `sgcli_entrypoint.sh`.
- `requirements.yaml`: minimal bootstrap (pip + mlflow); repo deps installed later.
- `sgcli_entrypoint.sh`: finds the extracted repo and calls `commands.sh`.
- `commands.sh`: installs `requirements.txt` with constraints, installs `composer` (no-deps), launches `torchrun ... train.py parameters_sgcli.yaml`.
- `parameters_sgcli.yaml`: training config (paths, hparams, save/eval intervals, best checkpoint).
- `train.py`: main training script (distributed setup, checkpoints, best checkpoint).
- `cfgutils.py`: builders for loggers/callbacks/algorithms used by `train.py`.
- `mcli/__init__.py`: stub so Composer imports succeed without installing `mosaicml-cli`.
- `requirements.txt`: repo dependencies installed by `commands.sh`.

Legacy / not used by `workload.yaml` (safe to ignore unless you explicitly run them):
- `train_wo_yaml.py`, `train.yaml`, `parameters.yaml`, `geneformer_prep.sh` (install is skipped), older helper scripts.

## Recommended workflow

Test locally first (fast iteration), then switch to running from GitHub.

### 1) Local folder test (repo snapshot)

`workload.yaml` is set up to run from a **local repo snapshot** (your local folder is tarred and uploaded).

```bash
cd /Users/may.merkletan/Documents/Projects/_backups/db_backups/SGC_testing/sgc_cli_test/geneformer_sgc_cli_test
sgcli run -f workload.yaml
```

#### Important: keep the snapshot small

`repo_snapshot` snapshots the **git repo** that `local_repo_path` belongs to. Since this folder lives inside the
larger `db_backups` repo, the safest way to avoid uploading the whole parent repo is to make this folder its **own**
small git repo (nested), then snapshot that:

```bash
cd /Users/may.merkletan/Documents/Projects/_backups/db_backups/SGC_testing/sgc_cli_test/geneformer_sgc_cli_test
git init
git add -A
git commit -m "geneformer sgcli snapshot"
```

Then rerun:

```bash
sgcli run -f workload.yaml
```

### 2) GitHub test (after local is stable)

Use `workload_git.yaml` to run from GitHub:

```bash
cd /Users/may.merkletan/Documents/Projects/_backups/db_backups/SGC_testing/sgc_cli_test/geneformer_sgc_cli_test
sgcli run -f workload_git.yaml
```

## Promote working local snapshot changes to the GitHub branch

Once the local snapshot run is stable, copy the working code into the GitHub repo branch and run from `workload_git.yaml`.

### Files to copy into the repo branch root

From this folder (`geneformer_sgc_cli_test/`), copy these into the root of your GitHub branch (`hengrumay/composer_geneformer_pretrain`):

- `commands.sh`
- `train.py`
- `parameters_sgcli.yaml`
- `geneformer_prep.sh`
- (optional) `README.md`

### Commit + push to the branch

```bash
git clone git@github.com:hengrumay/composer_geneformer_pretrain.git
cd composer_geneformer_pretrain
git checkout mmt_sgc_cli_test

# copy files in, then:
git add commands.sh train.py geneformer_prep.sh parameters_sgcli.yaml README.md
git commit -m "Make sgcli run use UC Volumes + torchrun launcher"
git push
```

### Run via GitHub

```bash
cd /Users/may.merkletan/Documents/Projects/_backups/db_backups/SGC_testing/sgc_cli_test/geneformer_sgc_cli_test
sgcli run -f workload_git.yaml
```

## How it works

- `requirements.yaml` is intentionally minimal; `commands.sh` installs repo deps at runtime.
- The training data is expected to already exist on Databricks Volumes (e.g. `/Volumes/main/...`).

## H100 note

On this `sgcli` wheel:
- `gpu_type: h100_80gb` means **8 GPUs per node**, so `compute.gpus` must be a **multiple of 8**.

## Run

```bash
cd /Users/may.merkletan/Documents/Projects/_backups/db_backups/SGC_testing/sgc_cli_test/geneformer_sgc_cli_test

# ensure sgcli is installed in your active env, then:
sgcli run -f workload.yaml
```

## If you need to refactor `train.py` in the GitHub repo

Yes—clone locally, make a branch, push, then point `workload.yaml` at that branch.

```bash
git clone git@github.com:hengrumay/composer_geneformer_pretrain.git
cd composer_geneformer_pretrain
git checkout mmt_sgc_cli_test
git checkout -b mmt_sgc_cli_test_refactor1

# edit train.py ...

git add train.py
git commit -m "Refactor train.py for sgcli/H100"
git push -u origin mmt_sgc_cli_test_refactor1
```

Then update `geneformer_sgc_cli_test/workload_git.yaml`:
- `git_branch: mmt_sgc_cli_test_refactor1`

## What you must confirm

- **Entrypoint script**:
  - Local snapshot mode: `workload.yaml` runs `sgcli_entrypoint.sh` which then runs `commands.sh`.
  - GitHub mode: `workload_git.yaml` runs `commands.sh` from the repo branch.
- **Dataset paths**: your repo configs/scripts should reference the Volume dataset paths you provided.

