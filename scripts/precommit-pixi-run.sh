#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?task required}"
shift || true

if ! command -v pixi >/dev/null 2>&1; then
  export PIXI_HOME="${HOME}/.pixi"
  export PATH="${PIXI_HOME}/bin:${PATH}"
  if ! command -v pixi >/dev/null 2>&1; then
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="${HOME}/.pixi/bin:${PATH}"
  fi
fi

exec pixi run "${TASK}" "$@"
