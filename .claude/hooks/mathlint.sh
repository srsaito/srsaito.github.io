#!/usr/bin/env bash
# PostToolUse Write|Edit: warn (non-blocking) when a _notes/_posts markdown file
# contains raw $...$ inline math with _, *, or | — kramdown mangles it (table-split bug).
# See CLAUDE.md "Math & tables". Exit 2 surfaces the warning to Claude without reverting the edit.
input=$(cat)
f=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
case "$f" in
  *_notes/*.md|*_posts/*.md) ;;
  *) exit 0 ;;
esac
[ -f "$f" ] || exit 0

report=$(python3 - "$f" <<'PY'
import re, sys
t = open(sys.argv[1], encoding="utf-8").read()
t = re.sub(r"```.*?```", "", t, flags=re.S)      # drop fenced code
t = re.sub(r"\$\$.*?\$\$", "", t, flags=re.S)     # drop display math
bad = [m.group(0) for m in re.finditer(r"(?<!\$)\$(?!\$)([^\$\n]+?)\$(?!\$)", t)
       if any(c in m.group(1) for c in "_*|")]
if bad:
    for b in bad[:5]:
        print(b)
PY
)

if [ -n "$report" ]; then
  {
    echo "math-lint: $f has raw inline \$...\$ math containing _, *, or | — kramdown will mangle it."
    echo "Use \\(...\\) delimiters (and \\mid for |). Run it through obsidian_to_jekyll.py or fix these:"
    printf '%s\n' "$report"
  } >&2
  exit 2
fi
exit 0
