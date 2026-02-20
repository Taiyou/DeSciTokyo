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
#   2. docs/wiki/ の全 .md ファイルと画像をコピー
#   3. 変更があれば commit & push
# =============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WIKI_SOURCE="${REPO_ROOT}/docs/wiki"
WIKI_TMPDIR=$(mktemp -d)

# Support both SSH and HTTPS
WIKI_REPO_SSH="git@github.com:Taiyou/DeSciTokyo.wiki.git"
WIKI_REPO_HTTPS="https://github.com/Taiyou/DeSciTokyo.wiki.git"

echo "=== DeSciTokyo Wiki Sync ==="
echo "Source: ${WIKI_SOURCE}"
echo ""

# Try SSH first, fall back to HTTPS
echo "Cloning wiki repository..."
if git clone "${WIKI_REPO_SSH}" "${WIKI_TMPDIR}" 2>/dev/null; then
    echo "  Cloned via SSH"
elif git clone "${WIKI_REPO_HTTPS}" "${WIKI_TMPDIR}" 2>/dev/null; then
    echo "  Cloned via HTTPS"
else
    echo "  Initializing new wiki repository..."
    cd "${WIKI_TMPDIR}"
    git init -b master
    git remote add origin "${WIKI_REPO_SSH}"
fi

# Remove old content (except .git)
echo "Cleaning old content..."
find "${WIKI_TMPDIR}" -maxdepth 1 -not -name '.git' -not -name '.' -exec rm -rf {} + 2>/dev/null || true

# Copy all wiki pages
echo "Copying wiki pages..."
cp "${WIKI_SOURCE}"/*.md "${WIKI_TMPDIR}/"
echo "  Copied $(ls "${WIKI_TMPDIR}"/*.md | wc -l) pages"

# Copy images if they exist
if [ -d "${WIKI_SOURCE}/images" ]; then
    echo "Copying images..."
    mkdir -p "${WIKI_TMPDIR}/images"
    cp "${WIKI_SOURCE}"/images/*.png "${WIKI_TMPDIR}/images/" 2>/dev/null || true
    echo "  Copied $(ls "${WIKI_TMPDIR}/images/" 2>/dev/null | wc -l) images"
fi

cd "${WIKI_TMPDIR}"

# Check for changes
git add -A
if git diff --cached --quiet 2>/dev/null; then
    echo ""
    echo "No changes to sync."
    rm -rf "${WIKI_TMPDIR}"
    exit 0
fi

# Show summary
echo ""
echo "Changes:"
git diff --cached --stat

# Commit and push
git commit -m "Sync wiki from docs/wiki/ $(date +%Y-%m-%d)"
echo ""
echo "Pushing to GitHub Wiki..."
git push origin master --force

echo ""
echo "=== Wiki synced successfully! ==="
echo "View at: https://github.com/Taiyou/DeSciTokyo/wiki"

# Cleanup
rm -rf "${WIKI_TMPDIR}"
