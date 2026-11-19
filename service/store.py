"""SQLite persistence for service runs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunStore:
    def __init__(self, path: Path | str = Path("artifacts/agent_runs/runs.sqlite")) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS runs ("
            "run_id TEXT PRIMARY KEY, instruction TEXT NOT NULL, status TEXT NOT NULL, "
            "attempts INTEGER NOT NULL, result TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        self._connection.commit()

    def save(self, run_id: str, instruction: str, result: dict[str, Any]) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO runs (run_id, instruction, status, attempts, result, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                instruction,
                str(result.get("status", "unknown")),
                int(result.get("attempts", 0)),
                json.dumps(result, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._connection.commit()

    def get(self, run_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT run_id, instruction, status, attempts, result, created_at FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row[0],
            "instruction": row[1],
            "status": row[2],
            "attempts": row[3],
            "result": json.loads(row[4]),
            "created_at": row[5],
        }

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT run_id, instruction, status, attempts, created_at FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {"run_id": row[0], "instruction": row[1], "status": row[2], "attempts": row[3], "created_at": row[4]}
            for row in rows
        ]

    def close(self) -> None:
        self._connection.close()
