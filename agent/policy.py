"""Retrieval policy: decide whether, and for which category, memory may help.

Naive RAG over normal successful plans biased the 0.5B planner away from
``needs_clarification``: unsafe/underspecified test tasks retrieved normal
plans and the model then produced invalid plans instead of rejecting.  The
policy below does two things:

1. **Safety gate** - requests with risk keywords or too few concrete numeric
   constraints skip retrieval entirely, so refusal behaviour is not diluted.
2. **Targeted retrieval** - only the categories where the v4 LoRA planner is
   known to be weak (circles, rectangles, multi-drone plans) get examples;
   point/multi-waypoint tasks already score ~100% and do not need memory.

This is deliberately explainable; a learned risk/category classifier is the
natural next step.
"""

from __future__ import annotations

import re

RISK_KEYWORDS = (
    "禁飞区",
    "不提供",
    "没有",
    "未说明",
    "不说明",
    "任意",
    "不限",
    "最快",
    "不要 json",
    "不要json",
    "代码",
    "那边",
    "那里",
    "瞬间",
    "同时",
    "不得超过",
    "矛盾",
    "越界",
)

# v4 task categories; only weak categories are allowed to retrieve examples.
WEAK_CATEGORIES = ("circle", "rectangle", "multi_drone")


def infer_category(instruction: str) -> str:
    """Cheap keyword-based category inference used only to filter retrieval."""
    text = instruction.lower()
    if "半边长" in text or "方形" in text or "矩形" in text:
        return "rectangle"
    if "圆" in text or "半径" in text or "巡航一圈" in text:
        return "circle"
    if len(re.findall(r"drone-\d", text)) >= 2 or any(word in text for word in ("两架", "三架", "2架", "3架")):
        return "multi_drone"
    if "依次经过" in text:
        return "multi"
    if "坐标" in text:
        return "point"
    return "unknown"


def retrieval_allowed(instruction: str) -> tuple[bool, str]:
    """Return (allowed, reason). Only specified, non-risky requests use memory."""
    text = instruction.lower()
    for keyword in RISK_KEYWORDS:
        if keyword in text:
            return False, f"risk_keyword:{keyword}"
    if len(re.findall(r"-?\d+(?:\.\d+)?", instruction)) < 2:
        return False, "too_few_numeric_constraints"
    return True, "ok"


def route_memory(instruction: str) -> tuple[bool, str, str | None]:
    """Return (use_memory, reason, category_filter)."""
    allowed, reason = retrieval_allowed(instruction)
    if not allowed:
        return False, reason, None
    category = infer_category(instruction)
    if category not in WEAK_CATEGORIES:
        return False, f"category_not_weak:{category}", category
    return True, "weak_category", category
