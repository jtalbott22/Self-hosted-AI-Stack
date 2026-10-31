#!/usr/bin/env bash
# Fetch the LAM Gaussian-splat web renderer into avatar/html/lib/ and, if
# given, copy an exported avatar bundle into avatar/html/avatars/.
#
# Usage (from the repository root):
#   bash avatar/scripts/setup-web.sh [path/to/avatar.zip ...]
set -euo pipefail

VERSION="${LAM_RENDERER_VERSION:-0.0.8}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LIB="$HERE/html/lib"
AVATARS="$HERE/html/avatars"
mkdir -p "$LIB" "$AVATARS"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading gaussian-splat-renderer-for-lam@$VERSION"
curl -fsSL "https://registry.npmjs.org/gaussian-splat-renderer-for-lam/-/gaussian-splat-renderer-for-lam-$VERSION.tgz" \
  | tar xz -C "$TMP"
cp "$TMP/package/build/gaussian-splat-renderer-for-lam.module.js" "$LIB/lam-renderer.js"

# The published module imports a helper from greggman.github.io. Drop that
# line so the page has no third-party runtime dependency.
sed -i '/greggman.github.io/d' "$LIB/lam-renderer.js"

imports="$(grep -c '^import' "$LIB/lam-renderer.js" || true)"
if [ "$imports" != "0" ]; then
  echo "warning: lam-renderer.js still has $imports import line(s); check it by hand" >&2
fi
echo "Renderer: $LIB/lam-renderer.js"

for zip in "$@"; do
  cp "$zip" "$AVATARS/"
  echo "Avatar:   $AVATARS/$(basename "$zip")"
done
