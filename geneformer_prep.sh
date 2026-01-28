#!/usr/bin/env bash
set -euo pipefail

# Install is intentionally disabled: training does not require the geneformer python package,
# and installing it pulls heavy deps (anndata/scanpy/ray) that can destabilize Serverless.
echo ">>> geneformer_prep.sh: skipping Geneformer install"

# -----------------------------------------------------------------------------
# Fallback (original approach) — uncomment if you hit issues with the HF pip install
# (e.g., git-lfs pointers needed at install time).
# -----------------------------------------------------------------------------
# python -m pip install --upgrade pip
# python -m pip install "git+https://huggingface.co/ctheodoris/Geneformer@b07f4b1e8893a0923a8fde223fe3b5a60b976d99"
#
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
#
# # Download training data and converting to streaming dataset
# sh ./download_dataset.sh
# python create_mds.py
