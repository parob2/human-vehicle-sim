#!/usr/bin/env bash
# Full PIE RF pipeline: extract all videos → merge → augment → train → eval
# Resume-safe: re-run this script any time; it skips completed videos and only
# trains after all 52 clips are extracted.
set -euo pipefail

export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."
LOG="pie/features_cache/pipeline_full.log"
exec >> "$LOG" 2>&1

echo ""
echo "=== PIE pipeline resume $(date -Is) ==="

python3 pie/run_extraction_pipeline.py --upgrade-joints --extract

PENDING=$(python3 pie/run_extraction_pipeline.py --pending-count 2>/dev/null | tail -1)
echo "Videos still pending extraction: ${PENDING}"

if [[ "${PENDING}" -gt 0 ]]; then
  echo "Extraction not finished yet (~${PENDING} videos left)."
  echo "Re-run: bash pie/run_full_pipeline.sh"
  echo "Monitor: tail -f pie/features_cache/pipeline_full.log"
  exit 0
fi

echo "=== All videos extracted — merge + augment $(date -Is) ==="
python3 pie/run_extraction_pipeline.py --merge --augment

echo "=== Training RF (9-feature heading) $(date -Is) ==="
python3 pie/train_rf_pie_heading.py

echo "=== Evaluating $(date -Is) ==="
python3 pie/evaluate_pie_heading.py

echo "=== Pipeline finished $(date -Is) ==="
python3 pie/run_extraction_pipeline.py --status
