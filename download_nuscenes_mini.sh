#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-/data/sets/nuscenes}"
mkdir -p "$TARGET"
cd /tmp

if [[ ! -f v1.0-mini.tgz ]]; then
  wget https://www.nuscenes.org/data/v1.0-mini.tgz
fi

# RunPod network volumes cannot restore the archive creator's user/group IDs.
tar --no-same-owner -xf v1.0-mini.tgz -C "$TARGET"
echo "nuScenes mini extracted to: $TARGET"
echo "Expected metadata directory: $TARGET/v1.0-mini"
