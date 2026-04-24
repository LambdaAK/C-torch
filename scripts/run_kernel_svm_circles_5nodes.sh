#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="$ROOT/python"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
EPOCHS="${EPOCHS:-20}"

PYTHONPATH="$PYTHONPATH" python3 "$ROOT/examples/python/train_kernel_svm_circles_distributed.py" \
  --rank 0 \
  --world-size 5 \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  --epochs "$EPOCHS" &

sleep 2

for rank in 1 2 3 4; do
  PYTHONPATH="$PYTHONPATH" python3 "$ROOT/examples/python/train_kernel_svm_circles_distributed.py" \
    --rank "$rank" \
    --world-size 5 \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    --epochs "$EPOCHS" &
done

wait
