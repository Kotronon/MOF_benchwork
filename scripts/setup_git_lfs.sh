#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v git-lfs >/dev/null 2>&1; then
  if [[ "$(uname -s)" != "Darwin" ]]; then
    printf 'Git LFS is missing. Install it with your OS package manager, then rerun this script.\n' >&2
    exit 1
  fi
  if ! command -v brew >/dev/null 2>&1; then
    printf 'Git LFS is missing and Homebrew is unavailable. Install Git LFS first: https://git-lfs.com/\n' >&2
    exit 1
  fi
  brew install git-lfs
fi

git lfs install

for archive in external/umof_data.zip external/umof_models.zip; do
  filter="$(git check-attr filter -- "${archive}" | awk '{print $3}')"
  if [[ "${filter}" != "lfs" ]]; then
    printf 'Expected Git LFS tracking for %s, found filter=%s\n' "${archive}" "${filter}" >&2
    exit 1
  fi
done

printf 'Git LFS is ready for the uMOF archives.\n'
git lfs ls-files
printf '\nIf these archives are already committed as regular Git blobs, migrate that local commit before pushing:\n'
printf '  git lfs migrate import --include="external/umof_data.zip,external/umof_models.zip" --include-ref=refs/heads/%s\n' "$(git branch --show-current)"