#!/usr/bin/env bash
# PostToolUse hook: auto-format a Rust file right after Claude edits/writes it.
# Reads the hook JSON on stdin, extracts the edited path, and runs rustfmt if it's *.rs.
set -euo pipefail

file="$(node -e 'let d="";process.stdin.on("data",c=>d+=c).on("end",()=>{try{const j=JSON.parse(d);const i=j.tool_input||{};process.stdout.write(i.file_path||i.filePath||"")}catch{}})' 2>/dev/null || true)"

case "$file" in
  *.rs)
    if [ -f "$file" ]; then
      rustfmt --edition 2024 "$file" >/dev/null 2>&1 || true
    fi
    ;;
esac
exit 0
