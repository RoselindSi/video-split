#!/usr/bin/env bash
# Refuse to let an API key reach the repository. Run before any commit that
# touches the auditor; wire it into .git/hooks/pre-commit to make it automatic.
#
# It matches the SHAPE of an Ark key rather than any particular value: the
# whole point is that no key is ever written down here, including inside the
# guard that looks for keys.
set -u
pat='ark-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
hits=$(git grep -InE "$pat" -- . 2>/dev/null)
if [ -n "$hits" ]; then
  echo "!! an API key is present in tracked files. Remove it and rotate the key --"
  echo "   deleting it in a later commit does not remove it from history."
  echo "$hits"
  exit 1
fi
staged=$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null)
if [ -n "$staged" ]; then
  bad=$(echo "$staged" | xargs -r grep -lInE "$pat" 2>/dev/null)
  if [ -n "$bad" ]; then
    echo "!! staged file(s) contain an API key: $bad"
    exit 1
  fi
fi
echo "no API key found in tracked or staged files"
