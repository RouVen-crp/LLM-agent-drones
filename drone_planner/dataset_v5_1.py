"""v5.1 dataset revision: train-only unsupported-capability rejection family.

The v5 validation and test splits stay byte-identical so router evaluations on
v5 remain comparable; only the training rejection set changes.  This lets the
router learn "reject unsupported capabilities" without leaking the test
``unsupported_ellipse`` family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset_v5 import (
    TRAIN_FAMILIES,
    RejectionFamily,
    _select_rejections,
    _stratified_replace,
    generate_records_v5,
    split_records_v5,
)

UNSUPPORTED_SHAPE_FAMILY = RejectionFamily(
    "unsupported_shape",
    "reject",
    "绘制一个{shape}轨迹{detail}",
    [
        {"shape": shape, "detail": detail}
        for shape in ("八字形", "莫比乌斯环", "球面螺旋", "文字", "logo", "立方体", "三维心形", "双环", "花瓣螺旋", "波浪面")
        for detail in ("", "并保持对称", "并返回起点", "并保持等距")
    ],
)

UNSUPPORTED_CONIC_FAMILY = RejectionFamily(
    "unsupported_conic",
    "reject",
    "以 (0.0, 0.0) 为中心、长半轴 {a} 米、短半轴 {b} 米画{shape}巡航后返航",
    [
        {"shape": shape, "a": a, "b": b}
        for shape in ("双曲线", "抛物线", "悬链线", "心形线", "摆线", "玫瑰线", "渐开线", "螺旋线")
        for a, b in ((0.1, 0.05), (0.08, 0.03), (0.06, 0.04), (0.05, 0.02), (0.07, 0.05))
    ],
)

MISSING_ZONE_FAMILY = RejectionFamily(
    "missing_zone_constraint",
    "clarify",
    "绕开{zone}后巡航，但{detail}",
    [
        {"zone": zone, "detail": detail}
        for zone in ("限飞区", "障碍区", "危险区", "管制区", "临时管制区", "管控区")
        for detail in ("不提供位置", "没有边界", "未说明范围", "不说明安全走廊", "没有给出坐标")
    ],
)

TRAIN_EXTRA_FAMILIES = [UNSUPPORTED_SHAPE_FAMILY, UNSUPPORTED_CONIC_FAMILY, MISSING_ZONE_FAMILY]


def write_dataset_v5_1(output_dir: Path, base_dir: Path, scale: int = 20, seed: int = 42) -> dict[str, Any]:
    base_manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    if base_manifest.get("dataset_version") != "v5":
        raise ValueError(f"base_dir 不是 v5 数据集：{base_manifest.get('dataset_version')}")

    output_dir.mkdir(parents=True, exist_ok=True)
    normal_train = split_records_v5(generate_records_v5(scale=scale, seed=seed), seed=seed)["train"]
    rng = random.Random(seed + 1001)
    prompts = _select_rejections(TRAIN_FAMILIES + TRAIN_EXTRA_FAMILIES, 16 * scale, rng)
    _stratified_replace(normal_train, prompts, rng)

    for name in ("validation", "test"):
        shutil.copyfile(base_dir / f"{name}.jsonl", output_dir / f"{name}.jsonl")
    with (output_dir / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in normal_train:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    families = TRAIN_FAMILIES + TRAIN_EXTRA_FAMILIES
    train_counts = Counter(row["risk_family"] for row in normal_train if row.get("expect_rejection"))
    manifest: dict[str, Any] = {
        "dataset_version": "v5.1",
        "base_version": "v5",
        "generator": "drone_planner.dataset_v5_1",
        "scale": scale,
        "seed": seed,
        "splits": {name: sum(1 for _ in (output_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()) for name in ("train", "validation", "test")},
        "evaluation": base_manifest["evaluation"],
        "rejections": {
            "train": sum(1 for row in normal_train if row.get("expect_rejection")),
            "validation": base_manifest["rejections"]["validation"],
            "test": base_manifest["rejections"]["test"],
        },
        "rejection_families": {
            "train": {family.name: {"action": family.action, "count": train_counts.get(family.name, 0)} for family in families},
            "validation": base_manifest["rejection_families"]["validation"],
            "test": base_manifest["rejection_families"]["test"],
        },
        "protocol": {
            "normal_instruction_label_contract": base_manifest["protocol"]["normal_instruction_label_contract"],
            "unsafe_split": base_manifest["protocol"]["unsafe_split"],
            "action_contract": base_manifest["protocol"]["action_contract"],
            "train_only_revision": "v5.1 adds an unsupported-capability rejection family only to train; validation/test are byte-identical to v5.",
            "historical_comparison": "Router results should be compared on the same v5 test split; Planner retraining must use v5.1 train.",
        },
        "base_sha256": base_manifest["sha256"],
    }
    manifest["sha256"] = {name: hashlib.sha256((output_dir / f"{name}.jsonl").read_bytes()).hexdigest() for name in ("train", "validation", "test")}
    if manifest["sha256"]["test"] != base_manifest["sha256"]["test"]:
        raise RuntimeError("v5.1 test split changed; refusing to write a non-comparable revision.")
    if manifest["sha256"]["validation"] != base_manifest["sha256"]["validation"]:
        raise RuntimeError("v5.1 validation split changed; refusing to write a non-comparable revision.")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="在 v5 基础上生成 v5.1（仅训练集新增不支持能力拒绝族）。")
    parser.add_argument("--base-dir", type=Path, default=Path("artifacts/dataset_contract_v5"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dataset_contract_v5_1"))
    parser.add_argument("--scale", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = write_dataset_v5_1(args.output_dir, args.base_dir, scale=args.scale, seed=args.seed)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "version": manifest["dataset_version"],
                "splits": manifest["splits"],
                "rejections": manifest["rejections"],
                "test_sha": manifest["sha256"]["test"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
