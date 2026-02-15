#!/bin/bash
# =============================================================
# sync-wiki.sh — docs/wiki/ の内容を GitHub Wiki に同期する
# =============================================================
#
# 使い方:
#   ./scripts/sync-wiki.sh
#
# 前提条件:
#   - GitHub への push 権限があること
#   - git がインストールされていること
#
# 動作:
#   1. GitHub Wiki リポジトリをクローン
#   2. docs/wiki/ の全 .md ファイルをコピー
#   3. 変更があれば commit & push
# =============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WIKI_SOURCE="${REPO_ROOT}/docs/wiki"
WIKI_TMPDIR=$(mktemp -d)
WIKI_REPO="git@github.com:Taiyou/DeSciTokyo.wiki.git"

echo "=== DeSciTokyo Wiki Sync ==="
echo "Source: ${WIKI_SOURCE}"
echo "Target: ${WIKI_REPO}"
echo ""

# Clone wiki repo
echo "Cloning wiki repository..."
git clone "${WIKI_REPO}" "${WIKI_TMPDIR}" 2>/dev/null || {
    echo "Initializing new wiki repository..."
    cd "${WIKI_TMPDIR}"
    git init -b master
    git remote add origin "${WIKI_REPO}"
}

# Copy all wiki pages
echo "Copying wiki pages..."
cp "${WIKI_SOURCE}"/*.md "${WIKI_TMPDIR}/"

# Copy images if they exist
if [ -d "${WIKI_SOURCE}/images" ]; then
    echo "Copying images..."
    mkdir -p "${WIKI_TMPDIR}/images"
    cp "${WIKI_SOURCE}"/images/*.png "${WIKI_TMPDIR}/images/" 2>/dev/null || true
    echo "  Copied $(ls "${WIKI_TMPDIR}/images/" 2>/dev/null | wc -l) images"
fi

cd "${WIKI_TMPDIR}"

# Check for changes
if git diff --quiet HEAD 2>/dev/null && [ -z "$(git status --porcelain)" ]; then
    echo "No changes to sync."
    rm -rf "${WIKI_TMPDIR}"
    exit 0
fi

# Commit and push
git add -A
git commit -m "Sync wiki from docs/wiki/ $(date +%Y-%m-%d)"
echo "Pushing to GitHub Wiki..."
git push origin master

echo ""
echo "=== Wiki synced successfully! ==="
echo "View at: https://github.com/Taiyou/DeSciTokyo/wiki"

# Cleanup
rm -rf "${WIKI_TMPDIR}"
