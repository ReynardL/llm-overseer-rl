#!/usr/bin/env bash
# Usage (inside WSL): overseer_call.sh <prompt_file> <schema_file> <out_json>
# One isolated headless call: pinned model and effort, no tools, no MCP servers, empty cwd, CLI login.
# Writes the result object to <out_json> and the whole stream (its system/init record lists tools and MCP servers)
# to <out_json minus .json>.stream.jsonl.
# CLAUDE_BIN=/mnt/c/.../claude.exe uses the Windows binary via interop; OVERSEER_TMP picks where the empty cwd goes
# (use a /mnt/c path for the Windows binary so its cwd is not a \\wsl$ UNC path).
set -euo pipefail
prompt=$(cat "$1"); pf=$(realpath "$1"); schema=$(cat "$2"); out=$(realpath -m "$3")
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
export CLAUDE_CODE_DISABLE_CLAUDE_MDS=1 CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 \
       CLAUDE_CODE_DISABLE_BUNDLED_SKILLS=1 CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL=1 DISABLE_AUTOUPDATER=1
# Forward the isolation vars to a Windows binary.
export WSLENV="${WSLENV:+$WSLENV:}CLAUDE_CODE_DISABLE_CLAUDE_MDS:CLAUDE_CODE_DISABLE_AUTO_MEMORY:CLAUDE_CODE_DISABLE_BUNDLED_SKILLS:CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL:DISABLE_AUTOUPDATER"
work=$(mktemp -d -p "${OVERSEER_TMP:-/tmp}")
# --strict-mcp-config with an empty config: no MCP servers load. The file sits beside the empty cwd.
mcp="${OVERSEER_TMP:-/tmp}/empty_mcp.json"; echo '{"mcpServers": {}}' > "$mcp"
case "${CLAUDE_BIN:-claude}" in *.exe) mcp=$(wslpath -w "$mcp") ;; esac
stream="${out%.json}.stream.jsonl"
cd "$work"
start=$(date +%s)
# --setting-sources project,local: user settings are not loaded.
# OVERSEER_STDIN=1 pipes the prompt on stdin: a Windows command line holds 32K characters.
flags=(--model claude-fable-5-2 --effort xhigh --output-format stream-json --verbose --json-schema "$schema"
       --setting-sources project,local --strict-mcp-config --mcp-config "$mcp"
       --tools "" --disable-slash-commands --no-session-persistence)
if [ "${OVERSEER_STDIN:-0}" = 1 ]; then
  "${CLAUDE_BIN:-claude}" -p "${flags[@]}" < "$pf" > "$stream" || rc=$?
else
  "${CLAUDE_BIN:-claude}" -p "$prompt" "${flags[@]}" > "$stream" || rc=$?
fi
python3 -c 'import json, sys  # the last result record: the same object --output-format json gives
r = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip().startswith("{")]
r = [x for x in r if x.get("type") == "result"]
open(sys.argv[2], "w").write(json.dumps(r[-1]) if r else "")' "$stream" "$out" || true
echo "exit=${rc:-0} secs=$(( $(date +%s) - start )) cwd=$work"
cd / && rmdir "$work"
