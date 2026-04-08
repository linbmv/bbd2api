# Testing

## 2026-04-08
- `python -m py_compile server.py`
  - Result: pass
- `python - <<EOF` importing `server` and printing `_claude_ping()` / `_claude_message_start(...)`
  - Result: pass
  - Verified output includes both `event:` and `data:` lines.
- `git diff -- server.py`
  - Result: inspected
  - Verified `/v1/messages` streaming now uses Claude SSE helper emitters in both normal and buffered-tool paths.
