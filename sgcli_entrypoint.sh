set -euo pipefail

# sgcli repo_snapshot extracts the tarball under $HOME/<directory_name>, but the name is not stable.
# Find the extracted repo by locating commands.sh under $HOME.
if [ -d "$HOME/geneformer_sgc_cli_test" ]; then
  REPO_DIR="$HOME/geneformer_sgc_cli_test"
else
  REPO_DIR="$(find "$HOME" -maxdepth 2 -type f -name commands.sh -print -quit | xargs -r dirname)"
fi

if [ -z "${REPO_DIR:-}" ] || [ ! -d "${REPO_DIR}" ]; then
  echo "ERROR: Could not locate extracted repo (commands.sh) under \$HOME"
  find "$HOME" -maxdepth 3 -type f -name commands.sh | head -n 20 || true
  exit 1
fi

echo ">>> Using repo snapshot dir: ${REPO_DIR}"
cd "${REPO_DIR}"

echo ">>> Listing repo root:"
ls -la | head

echo ">>> Running commands.sh"
bash ./commands.sh

