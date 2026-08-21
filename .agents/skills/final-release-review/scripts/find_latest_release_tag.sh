#!/usr/bin/env bash
set -euo pipefail

remote="${1:-origin}"
pattern="${2:-v*}"

git fetch "$remote" --tags --prune --quiet

remote_tags=$(
  git ls-remote --tags --refs "$remote" "$pattern" |
    sed -E 's#^[^[:space:]]+[[:space:]]+refs/tags/##'
)

latest_tag=$(
  git tag -l "$pattern" --sort=-v:refname |
    while IFS= read -r tag; do
      if grep -Fqx "$tag" <<<"$remote_tags"; then
        printf '%s\n' "$tag"
        break
      fi
    done
)

if [[ -z "$latest_tag" ]]; then
  echo "No tags found matching pattern '$pattern' after fetching from $remote." >&2
  exit 1
fi

echo "$latest_tag"
