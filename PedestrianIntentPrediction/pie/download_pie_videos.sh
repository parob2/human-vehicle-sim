#!/usr/bin/env bash
# Download PIE video clips from YorkU (approx. 74 GB for all sets).
set -euo pipefail

PIE_ROOT="${PIE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../PIE_dataset" && pwd)}"
BASE_URL="https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips"
SETS=("${@:-set01 set02 set03 set04 set05 set06}")

for set_id in "${SETS[@]}"; do
  dest="${PIE_ROOT}/PIE_clips/${set_id}"
  mkdir -p "${dest}"
  echo "=== ${set_id} ==="
  for clip in "${dest}"/*.mp4; do
    [[ -f "${clip}" ]] || continue
    echo "  skip existing $(basename "${clip}")"
  done
  # Discover clips from annotation filenames when set folder is empty.
  mapfile -t videos < <(ls "${PIE_ROOT}/annotations/${set_id}"/video_*_annt.xml 2>/dev/null \
    | sed -E 's|.*/(video_[0-9]+)_annt.xml|\1|' | sort -u)
  for vid in "${videos[@]}"; do
    out="${dest}/${vid}.mp4"
    if [[ -f "${out}" ]]; then
      echo "  exists ${set_id}/${vid}.mp4"
      continue
    fi
    echo "  downloading ${set_id}/${vid}.mp4 ..."
    wget -c "${BASE_URL}/${set_id}/${vid}.mp4" -O "${out}"
  done
done

echo "Done. Clips under ${PIE_ROOT}/PIE_clips/"
