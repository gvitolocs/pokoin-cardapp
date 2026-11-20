#!/bin/bash
# Usage: bash server-rs/scripts/install-card-refresh.sh   (bash, not sh: it uses [[ ]] and pipefail)
# Build an immutable tools revision and atomically select it. Not run by the daily job.
[ -n "${BASH_VERSION:-}" ] || { echo "install-card-refresh.sh needs bash: run 'bash $0'" >&2; exit 1; }
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
cargo build --release -p card-images -p index-refresh
revision=$(git rev-parse --short HEAD)
root=$HOME/.local/lib/card-refresh
dest=$root/$revision
[[ ! -e $dest ]] || { echo "$dest already exists" >&2; exit 1; }
mkdir -p "$root"
tmp=$(mktemp -d "$root/.install-XXXXXX")
trap 'rm -rf -- "$tmp"' EXIT
cp target/release/card-images target/release/index-refresh "$tmp/"
cp scripts/*.sh scripts/*.py "$tmp/"
chmod +x "$tmp"/*.sh "$tmp"/*.py
mv "$tmp" "$dest"
ln -s "$revision" "$root/.current-$$"
mv -Tf "$root/.current-$$" "$root/current"
echo "installed: $dest"
