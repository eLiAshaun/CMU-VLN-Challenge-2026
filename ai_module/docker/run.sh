#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if (($# == 0)); then
  set -- q1
fi
exec "$ROOT_DIR/demo/run_real_question.sh" "$@"
