import json
import sqlite3
import time
import uuid
from pathlib import Path
from threading import Lock


class ConversationStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    latest_thread_id TEXT,
                    latest_key_id TEXT,
                    latest_assistant_id TEXT,
                    last_history_hash TEXT,
                    last_checkpoint TEXT,
                    status TEXT NOT NULL DEFAULT 'active'
                );

                CREATE TABLE IF NOT EXISTS history_refs (
                    history_hash TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS thread_bindings (
                    thread_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    key_id TEXT,
                    assistant_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tool_calls (
                    tool_use_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    thread_id TEXT,
                    key_id TEXT,
                    dedupe_key TEXT,
                    name TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    result_json TEXT,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_history_refs_conversation_id
                    ON history_refs(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_thread_bindings_conversation_id
                    ON thread_bindings(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_checkpoints_conversation_id
                    ON checkpoints(conversation_id, id DESC);
                CREATE INDEX IF NOT EXISTS idx_tool_calls_conversation_id
                    ON tool_calls(conversation_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_tool_calls_dedupe
                    ON tool_calls(conversation_id, dedupe_key, status);
                """
            )

    def touch_conversation(self, conversation_id: str, status: str = "active") -> str:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (conversation_id, created_at, updated_at, status)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    status = excluded.status
                """,
                (conversation_id, now, now, status),
            )
        return conversation_id

    def new_conversation(self) -> str:
        return self.touch_conversation(uuid.uuid4().hex)

    def resolve_conversation_id(self, explicit_id: str | None, history_hashes: list[str], tool_use_ids: list[str]) -> str:
        if explicit_id:
            return self.touch_conversation(explicit_id)

        with self._connect() as conn:
            for tool_use_id in tool_use_ids:
                row = conn.execute(
                    "SELECT conversation_id FROM tool_calls WHERE tool_use_id = ?",
                    (tool_use_id,),
                ).fetchone()
                if row:
                    return self.touch_conversation(row["conversation_id"])

            for history_hash in history_hashes:
                if not history_hash:
                    continue
                row = conn.execute(
                    "SELECT conversation_id FROM history_refs WHERE history_hash = ?",
                    (history_hash,),
                ).fetchone()
                if row:
                    return self.touch_conversation(row["conversation_id"])

        return self.new_conversation()

    def remember_history_hashes(self, conversation_id: str, history_hashes: list[str]):
        now = time.time()
        hashes = [h for h in history_hashes if h]
        if not hashes:
            return
        with self._lock, self._connect() as conn:
            for history_hash in hashes:
                conn.execute(
                    """
                    INSERT INTO history_refs (history_hash, conversation_id, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(history_hash) DO UPDATE SET
                        conversation_id = excluded.conversation_id
                    """,
                    (history_hash, conversation_id, now),
                )
            conn.execute(
                "UPDATE conversations SET updated_at = ?, last_history_hash = ? WHERE conversation_id = ?",
                (now, hashes[-1], conversation_id),
            )

    def bind_thread(self, conversation_id: str, thread_id: str, key_id: str | None, assistant_id: str | None):
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO thread_bindings (thread_id, conversation_id, key_id, assistant_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    conversation_id = excluded.conversation_id,
                    key_id = excluded.key_id,
                    assistant_id = excluded.assistant_id,
                    updated_at = excluded.updated_at
                """,
                (thread_id, conversation_id, key_id, assistant_id, now, now),
            )
            conn.execute(
                """
                UPDATE conversations
                SET updated_at = ?, latest_thread_id = ?, latest_key_id = ?, latest_assistant_id = ?
                WHERE conversation_id = ?
                """,
                (now, thread_id, key_id, assistant_id, conversation_id),
            )

    def get_thread_binding(self, thread_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT conversation_id, key_id, assistant_id FROM thread_bindings WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_latest_binding(self, conversation_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT latest_thread_id AS thread_id, latest_key_id AS key_id, latest_assistant_id AS assistant_id
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
        if not row or not row["thread_id"]:
            return None
        return dict(row)

    def record_checkpoint(self, conversation_id: str, kind: str, payload: dict | None = None):
        payload = payload or {}
        now = time.time()
        payload_json = json.dumps(payload, ensure_ascii=False)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO checkpoints (conversation_id, kind, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, kind, payload_json, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ?, last_checkpoint = ? WHERE conversation_id = ?",
                (now, kind, conversation_id),
            )

    def get_recent_checkpoints(self, conversation_id: str, limit: int = 10) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT kind, payload_json, created_at
                FROM checkpoints
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [
            {
                "kind": row["kind"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def record_tool_calls(self, conversation_id: str, thread_id: str, key_id: str | None, calls: list[dict]):
        now = time.time()
        with self._lock, self._connect() as conn:
            for call in calls:
                conn.execute(
                    """
                    INSERT INTO tool_calls (
                        tool_use_id, conversation_id, thread_id, key_id, dedupe_key, name,
                        input_json, result_json, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'emitted', ?, ?)
                    ON CONFLICT(tool_use_id) DO UPDATE SET
                        conversation_id = excluded.conversation_id,
                        thread_id = excluded.thread_id,
                        key_id = excluded.key_id,
                        dedupe_key = excluded.dedupe_key,
                        name = excluded.name,
                        input_json = excluded.input_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        call["id"],
                        conversation_id,
                        thread_id,
                        key_id,
                        call.get("dedupe_key"),
                        call["name"],
                        json.dumps(call.get("input", {}), ensure_ascii=False),
                        now,
                        now,
                    ),
                )

    def mark_tool_results(self, conversation_id: str, tool_results: list[dict]):
        now = time.time()
        with self._lock, self._connect() as conn:
            for item in tool_results:
                tool_use_id = item["tool_use_id"]
                row = conn.execute(
                    "SELECT dedupe_key, name, input_json FROM tool_calls WHERE tool_use_id = ?",
                    (tool_use_id,),
                ).fetchone()
                if row:
                    conn.execute(
                        """
                        UPDATE tool_calls
                        SET conversation_id = ?, result_json = ?, status = 'completed', updated_at = ?
                        WHERE tool_use_id = ?
                        """,
                        (conversation_id, json.dumps(item.get("content"), ensure_ascii=False), now, tool_use_id),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO tool_calls (
                            tool_use_id, conversation_id, thread_id, key_id, dedupe_key, name,
                            input_json, result_json, status, created_at, updated_at
                        ) VALUES (?, ?, NULL, NULL, NULL, 'unknown', '{}', ?, 'completed', ?, ?)
                        """,
                        (tool_use_id, conversation_id, json.dumps(item.get("content"), ensure_ascii=False), now, now),
                    )

    def get_tool_call(self, tool_use_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tool_calls WHERE tool_use_id = ?",
                (tool_use_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_completed_tool_result(self, conversation_id: str, dedupe_key: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM tool_calls
                WHERE conversation_id = ? AND dedupe_key = ? AND status = 'completed'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (conversation_id, dedupe_key),
            ).fetchone()
        return dict(row) if row else None

    def get_recent_tool_activity(self, conversation_id: str, limit: int = 10) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT tool_use_id, name, input_json, result_json, status, dedupe_key, updated_at
                FROM tool_calls
                WHERE conversation_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]
