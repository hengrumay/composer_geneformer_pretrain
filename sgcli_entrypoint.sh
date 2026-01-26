set -euo pipefail

# sgcli repo_snapshot extracts the tarball under $HOME/<directory_name>, but the
# directory name depends on the local_repo_path basename. Auto-detect robustly.
REPO_DIR="${SGCLI_REPO_DIR:-}"

if [ -z "${REPO_DIR}" ]; then
  # Prefer current working directory if it already contains commands.sh
  if [ -f "./commands.sh" ]; then
    REPO_DIR="$(pwd)"
  else
    # Find the first commands.sh under $HOME (common in repo root)
    FOUND_COMMANDS_SH="$(find "${HOME}" -maxdepth 3 -type f -name "commands.sh" -print -quit || true)"
    if [ -n "${FOUND_COMMANDS_SH}" ]; then
      REPO_DIR="$(dirname "${FOUND_COMMANDS_SH}")"
    fi
  fi
fi

if [ -z "${REPO_DIR}" ] || [ ! -d "${REPO_DIR}" ]; then
  echo "ERROR: Could not locate extracted repo directory."
  echo "HOME=${HOME}"
  echo "PWD=$(pwd)"
  echo "Top-level HOME contents:"
  ls -la "${HOME}" | head
  exit 1
fi

echo ">>> Using repo snapshot dir: ${REPO_DIR}"
cd "${REPO_DIR}"

echo ">>> Listing repo root:"
ls -la | head

echo ">>> Running commands.sh"
bash ./commands.sh

