#!/usr/bin/env bash
set -euo pipefail

# Install Geneformer as a python package without requiring apt/git-lfs.
# Note: the HF repo may use git-lfs for some artifacts, but the python package itself is sufficient
# for imports used by this training script.
python -m pip install --upgrade pip
# Install Geneformer without pulling its (very large) transitive dependency set.
# We pin the required deps in requirements.txt already; avoiding `--deps` prevents
# breaking core Databricks runtime packages (mlflow/databricks-sdk/packaging).
python -m pip install --no-deps "git+https://huggingface.co/ctheodoris/Geneformer@b07f4b1e8893a0923a8fde223fe3b5a60b976d99"

# -----------------------------------------------------------------------------
# Fallback (original approach) — uncomment if you hit issues with the HF pip install
# (e.g., git-lfs pointers needed at install time).
# -----------------------------------------------------------------------------
# # install git-lfs , pre-req for geneformer clone
# curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
# apt-get install git-lfs
# git lfs install
#
# # install geneformer
# cd /
# git clone https://huggingface.co/ctheodoris/Geneformer
# cd Geneformer
# git checkout b07f4b1e8893a0923a8fde223fe3b5a60b976d99
# pip install .

#Download training data and converting to streaming dataset
#commenting since we already have it in s3
#sh ./download_dataset.sh 
#python  create_mds.py
