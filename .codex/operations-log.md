# Operations Log

## 2026-04-08
- Investigated `/v1/messages` empty-response behavior in `server.py`.
- Identified that Claude-style SSE responses were emitted as `data:` only, without `event:` fields.
- Identified that the buffered tool path delayed the first valid SSE frame, causing clients to appear stalled or empty.
- Added Claude SSE helper emitters and switched `/v1/messages` streaming paths to use event+data frames.
- Added an immediate Claude SSE prelude for buffered tool responses to avoid blank/stalled client behavior.
