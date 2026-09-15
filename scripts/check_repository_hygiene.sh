#!/usr/bin/env bash
set -euo pipefail

tracked_forbidden="$({
  git ls-files 'ai_module/debug/*'
  git ls-files '*.tar.gz' '*.zip' '.disk_check.tmp' '*.bak_*'
  git ls-files 'use'
} | sed '/^$/d' | sort -u)"

if [[ -n "${tracked_forbidden}" ]]; then
  echo "Forbidden generated or binary artifacts are tracked:" >&2
  echo "${tracked_forbidden}" >&2
  exit 1
fi
