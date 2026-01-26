set -euo pipefail

# sgcli repo_snapshot extracts the tarball under $HOME/<directory_name>
REPO_DIR="$HOME/geneformer_sgc_cli_test"
echo ">>> Using repo snapshot dir: ${REPO_DIR}"
cd "${REPO_DIR}"

echo ">>> Listing repo root:"
ls -la | head

echo ">>> Running commands.sh"
bash ./commands.sh

