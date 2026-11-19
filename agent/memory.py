"""Episodic memory + retrieval (sparse RAG) for the drone agent.

The memory stores past missions (instruction, label/plan, outcome) and retrieves
the most similar ones with BM25 over Chinese character uni/bi-grams.  It is
intentionally dependency-light: ``rank_bm25`` plus SQLite, so the agent has a
reproducible retrieval baseline before any dense-embedding upgrade.

Records are the same dicts used by the v4 dataset (``instruction``, ``plan``,
``expected_response``), so retrieved items can be passed directly to
``TransformersPlanner.generate(..., examples=...)``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Plus


def tokenize(text: str) -> list[str]:
    """Chinese-friendly tokenization for BM25: ASCII words + char uni/bi-grams."""
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_.-]+", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
    tokens.extend(chinese)
    tokens.extend(left + right for left, right in zip(chinese, chinese[1:]))
    return tokens or ["<empty>"]


@dataclass
class MemoryRecord:
    id: str
    instruction: str
    category: str
    target: dict[str, Any] | None
    outcome: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "instruction": self.instruction,
            "category": self.category,
            "target": self.target,
            "outcome": self.outcome,
        }

    def as_example(self) -> dict[str, Any]:
        """Shape accepted by ``build_messages`` / ``training_messages``."""
        example: dict[str, Any] = {"id": self.id, "instruction": self.instruction, "category": self.category}
        if isinstance(self.target, dict) and self.target.get("status") == "needs_clarification":
            example["expected_response"] = self.target
        else:
            example["plan"] = self.target
        return example


class MissionMemory:
    """In-memory BM25 index with optional SQLite persistence."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._records: list[MemoryRecord] = []
        self._dirty = True
        self._bm25: BM25Plus | None = None
        self._corpus_tokens: list[list[str]] = []
        self._db_path = db_path
        self._connection: sqlite3.Connection | None = None
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(str(db_path))
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS missions ("
                "id TEXT PRIMARY KEY, instruction TEXT NOT NULL, category TEXT NOT NULL, "
                "target TEXT, outcome TEXT NOT NULL)"
            )
            self._connection.commit()
            self._load_from_sqlite()

    # ------------------------------------------------------------- persistence
    def _load_from_sqlite(self) -> None:
        assert self._connection is not None
        for row in self._connection.execute("SELECT id, instruction, category, target, outcome FROM missions"):
            identifier, instruction, category, target_text, outcome = row
            target = json.loads(target_text) if target_text else None
            self._records.append(MemoryRecord(id=identifier, instruction=instruction, category=category, target=target, outcome=outcome))
        self._dirty = True

    def add(self, record: dict[str, Any], outcome: str = "seed") -> None:
        """Add a dataset-style record; ``target`` is the plan or rejection payload."""
        target = record.get("expected_response") or record.get("plan")
        item = MemoryRecord(
            id=str(record.get("id") or f"mem-{len(self._records)}"),
            instruction=str(record["instruction"]),
            category=str(record.get("category", "unknown")),
            target=target,
            outcome=outcome,
        )
        if any(existing.id == item.id for existing in self._records):
            return
        self._records.append(item)
        self._dirty = True
        if self._connection is not None:
            self._connection.execute(
                "INSERT OR REPLACE INTO missions (id, instruction, category, target, outcome) VALUES (?, ?, ?, ?, ?)",
                (item.id, item.instruction, item.category, json.dumps(item.target, ensure_ascii=False) if item.target else None, item.outcome),
            )
            self._connection.commit()

    def seed_from_jsonl(self, path: Path, outcome: str = "seed") -> int:
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            self.add(json.loads(line), outcome=outcome)
            count += 1
        return count

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    # --------------------------------------------------------------- retrieval
    def _ensure_index(self) -> None:
        if not self._dirty:
            return
        self._corpus_tokens = [tokenize(record.instruction) for record in self._records]
        self._bm25 = BM25Plus(self._corpus_tokens) if self._corpus_tokens else None
        self._dirty = False

    def retrieve(self, query: str, k: int = 2, exclude_ids: set[str] | None = None, category: str | None = None) -> list[dict[str, Any]]:
        """Return up to ``k`` examples with positive BM25 similarity.

        ``category`` optionally filters the memory to one category, which is
        what the targeted retrieval policy uses.
        """
        self._ensure_index()
        if self._bm25 is None:
            return []
        exclude = exclude_ids or set()
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(self._records)), key=lambda index: scores[index], reverse=True)
        results: list[dict[str, Any]] = []
        seen_instructions: set[str] = set()
        for index in order:
            if len(results) >= k:
                break
            if scores[index] <= 0:
                break
            record = self._records[index]
            if record.id in exclude or record.instruction in seen_instructions:
                continue
            if category is not None and record.category != category:
                continue
            seen_instructions.add(record.instruction)
            results.append({**record.as_example(), "outcome": record.outcome, "retrieval_score": round(float(scores[index]), 4)})
        return results

    @property
    def size(self) -> int:
        return len(self._records)

    def stats(self) -> dict[str, Any]:
        categories: dict[str, int] = {}
        for record in self._records:
            categories[record.category] = categories.get(record.category, 0) + 1
        return {"size": self.size, "categories": categories}
