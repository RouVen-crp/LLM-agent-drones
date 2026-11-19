"""v5 dataset: 10k-scale version of the v4 protocol.

Keeps the v4 label contract (explicit return/rectangle fields, executable
coordinates) and split-isolated rejection families, but scales to 10,000
records: 8,000 train / 1,000 validation / 1,000 test, with 800 normal + 200
rejection tasks in test.

Rejection records also carry ``action`` = ``reject`` or ``clarify`` so the same
dataset can supervise a function-calling router in P2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import build_record

BASE_COMPOSITION = {"point": 130, "multi": 150, "circle": 60, "rectangle": 60, "multi_drone": 100}
BASE_TOTAL = sum(BASE_COMPOSITION.values())


@dataclass(frozen=True)
class RejectionFamily:
    name: str
    action: str  # "reject" (unsafe/out-of-range/unsupported) or "clarify" (missing info)
    template: str
    values: list[dict[str, Any]]

    def render(self, value: dict[str, Any]) -> str:
        return self.template.format(**value)


TRAIN_FAMILIES: list[RejectionFamily] = [
    RejectionFamily(
        "coordinate_limit",
        "reject",
        "飞到坐标 ({x}, {y}, 0.4)",
        [
            {"x": x, "y": y}
            for x in (0.82, -0.91, 0.76, -1.2, 0.55, 0.94, -0.65, 1.05)
            for y in (0.0, 0.1, -0.8, 0.73, -0.35)
        ],
    ),
    RejectionFamily(
        "altitude_limit",
        "reject",
        "以 {z} 米高度飞到 (0.05, 0.02)",
        [{"z": z} for z in [round(0.6 + 0.05 * i, 2) for i in range(20)] + [round(-0.1 - 0.02 * i, 2) for i in range(20)]],
    ),
    RejectionFamily(
        "drone_limit",
        "reject",
        "规划 {count} 架无人机同时执行独立任务{extra}",
        [
            {"count": count, "extra": extra}
            for count in range(4, 12)
            for extra in ("", "并同步起飞", "并保持编队", "并分别规划航线", "并共享起点")
        ],
    ),
    RejectionFamily(
        "unsafe_format",
        "reject",
        "不要 JSON，直接{action}一段 {language} 代码控制无人机",
        [
            {"action": action, "language": language}
            for action in ("执行", "生成并执行", "运行", "调用")
            for language in ("Python", "Shell", "JavaScript", "C++", "Java", "Go", "Rust", "Ruby", "SQL", "Bash")
        ],
    ),
    RejectionFamily(
        "ambiguous_target",
        "clarify",
        "飞到{place}后等待我的下一条指令",
        [{"place": base + suffix} for base in ("那里", "那个位置", "目标区域", "附近", "老地方", "指定区域") for suffix in ("", "附近", "那个方向", "的左侧", "的右侧", "旁边", "一侧")][:40],
    ),
    RejectionFamily(
        "contradictory_speed",
        "reject",
        "在 {seconds} 秒内到达 {distance} 米外，同时速度不得超过 {speed} 米/秒",
        [
            {"seconds": seconds, "distance": distance, "speed": speed}
            for seconds in (0.2, 0.5, 1, 2, 3, 5, 8, 10)
            for distance, speed in ((10, 0.5), (30, 1), (50, 2), (80, 3), (100, 5))
        ][:40],
    ),
    RejectionFamily(
        "impossible_acceleration",
        "reject",
        "在 {seconds} 秒内从静止加速到 {speed} 米/秒并悬停",
        [
            {"seconds": seconds, "speed": speed}
            for seconds in (0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1)
            for speed in (5, 10, 20, 50, 100)
        ],
    ),
    RejectionFamily(
        "unsupported_payload",
        "reject",
        "携带 {weight} 公斤载荷执行巡检任务",
        [{"weight": weight} for weight in list(range(2, 82, 2))][:40],
    ),
    RejectionFamily(
        "ambiguous_altitude",
        "clarify",
        "保持{height}的高度飞行并返回起点",
        [
            {"height": height}
            for height in (
                "合适", "较高", "较低", "安全", "正常", "标准", "二楼", "树梢",
                "屋顶", "膝盖", "腰部", "肩膀", "头顶", "手臂", "手掌", "视线",
                "一米左右", "两米左右", "几百米", "几十米", "十米上下", "五米上下",
                "不要太高", "不要太低", "比楼高", "比树高", "比人高", "离地一点",
                "够得着", "看不见", "能拍到", "能看清", "飞得高", "飞得低",
                "中空", "低空", "高空", "半空", "贴近地面", "远离地面",
            )
        ],
    ),
]

VALIDATION_FAMILIES: list[RejectionFamily] = [
    RejectionFamily(
        "missing_reference_place",
        "clarify",
        "飞到{place}后等待我的下一条指令",
        [{"place": base + suffix} for base in ("那里", "那个位置", "目标区域", "附近", "老地方") for suffix in ("", "附近", "那个方向", "的左侧", "的右侧")],
    ),
    RejectionFamily(
        "unbounded_motion",
        "reject",
        "以最快速度沿{path}飞行，不限制坐标范围",
        [{"path": prefix + object_} for prefix in ("任意", "随机", "最大半径", "目标", "自由") for object_ in ("路径", "方向", "航线", "轨迹", "区域")],
    ),
    RejectionFamily(
        "unit_ambiguity",
        "clarify",
        "飞到高度 {height} 的位置",
        [{"height": height} for height in ("300", "3", "一百", "很高", "很低", "300米", "3米", "三十", "五十", "十", "二百", "四百", "二十", "五", "一千", "两米", "半米", "一米", "十米", "百米", "约300", "约3", "0.3", "0.03", "3.5")],
    ),
    RejectionFamily(
        "missing_altitude_bounds",
        "clarify",
        "保持高度飞行，但{detail}",
        [{"detail": prefix + object_} for prefix in ("不说明", "没说", "未给出", "缺少", "没有提供") for object_ in ("具体高度", "上下限", "高度范围", "安全边界", "高度约束")],
    ),
]

TEST_FAMILIES: list[RejectionFamily] = [
    RejectionFamily(
        "missing_no_fly_zone",
        "clarify",
        "绕开禁飞区后巡航，但{detail}",
        [
            {"detail": prefix + object_}
            for prefix in ("不提供", "没有", "未说明", "不说明", "没有给出")
            for object_ in ("禁飞区位置", "禁飞区边界", "禁飞区范围", "安全走廊", "禁飞区坐标")
        ],
    ),
    RejectionFamily(
        "contradictory_altitude",
        "reject",
        "全程高度保持 {high} 米，同时高度不得超过 {low} 米",
        [
            {"high": high, "low": low}
            for high in (0.45, 0.4, 0.35, 0.5)
            for low in (0.3, 0.25, 0.2, 0.22, 0.28)
        ][:20],
    ),
    RejectionFamily(
        "impossible_teleport",
        "reject",
        "在 {seconds} 秒内瞬间移动到 ({x}, {y}, 0.4)",
        [
            {"seconds": seconds, "x": x, "y": y}
            for seconds in (0, 0.5, 1, 2)
            for x, y in ((0.7, 0.0), (-0.8, 0.1), (0.0, 0.9), (0.6, -0.7), (0.2, 0.2))
        ][:20],
    ),
    RejectionFamily(
        "gps_coordinates",
        "reject",
        "飞到经纬度 ({lat}, {lon}) 后返回起点",
        [
            {"lat": lat, "lon": lon}
            for lat in (39.9, 31.2, 22.5, 30.6)
            for lon in (116.4, 121.5, 114.1, 104.1, 108.9)
        ][:20],
    ),
    RejectionFamily(
        "unsupported_ellipse",
        "reject",
        "以 ({cx}, {cy}) 为中心、长半轴 {a} 米、短半轴 {b} 米画椭圆巡航后返航",
        [
            {"cx": cx, "cy": cy, "a": a, "b": b}
            for cx, cy in ((0.0, 0.0), (0.02, 0.01), (-0.03, 0.02), (0.01, -0.02), (0.0, 0.03))
            for a, b in ((0.1, 0.05), (0.08, 0.03), (0.06, 0.04), (0.05, 0.02))
        ][:20],
    ),
    RejectionFamily(
        "waypoint_overflow",
        "reject",
        "让 drone-1 依次经过 {count} 个航点后返回起点",
        [{"count": count} for count in range(37, 57)],
    ),
    RejectionFamily(
        "negative_altitude",
        "reject",
        "以 {z} 米高度飞到 (0.05, 0.02)",
        [{"z": round(-0.1 - 0.05 * index, 2)} for index in range(20)],
    ),
    RejectionFamily(
        "return_home_conflict",
        "reject",
        "飞到 (0.05, 0.02) 后必须返回起点，但{detail}",
        [
            {"detail": detail}
            for detail in (
                "不要回到起点",
                "返航被禁止",
                "不允许回起点",
                "不要返航",
                "同时禁止返航",
                "并且不要返回",
                "不执行返航",
                "取消返航",
                "但要求留在终点",
                "同时不允许返回",
                "返航无效",
                "不要回原点",
                "禁止回到出发位置",
                "不返回起点",
                "同时不要返航",
                "且不能返回",
                "返航被取消",
                "不要回到出发位置",
                "不允许返航",
                "但禁止返回",
            )
        ],
    ),
    RejectionFamily(
        "oversized_swarm",
        "reject",
        "规划 {count} 架无人机同时执行独立任务",
        [{"count": count} for count in range(4, 24)],
    ),
    RejectionFamily(
        "vague_objective",
        "clarify",
        "去{place}执行那个任务",
        [{"place": base + suffix} for base in ("那边", "这里", "指定位置", "目标点", "前方") for suffix in ("", "附近", "的左侧", "的右侧")],
    ),
]


def generate_records_v5(scale: int = 20, seed: int = 42) -> list[dict[str, Any]]:
    """Generate ``BASE_TOTAL * scale`` normal v5 records."""
    rng = random.Random(seed)
    categories: list[str] = []
    for category, count in BASE_COMPOSITION.items():
        categories.extend([category] * (count * scale))
    rng.shuffle(categories)
    records = [build_record(index + 1, category, rng) for index, category in enumerate(categories)]
    rng.shuffle(records)
    return records


def split_records_v5(records: list[dict[str, Any]], seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    """80/10/10 split per category; base composition is divisible by 10."""
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["category"], []).append(record)
    train, validation, test = [], [], []
    for group in groups.values():
        rng.shuffle(group)
        ten_percent = len(group) // 10
        if ten_percent * 10 != len(group):
            raise ValueError(f"类别 {group[0]['category']} 的数量 {len(group)} 不能被 10 整除。")
        test.extend(group[:ten_percent])
        validation.extend(group[ten_percent:2 * ten_percent])
        train.extend(group[2 * ten_percent:])
    for part in (train, validation, test):
        rng.shuffle(part)
    return {"train": train, "validation": validation, "test": test}


def _select_rejections(families: list[RejectionFamily], total: int, rng: random.Random) -> list[dict[str, Any]]:
    """Build ``total`` rejection records, balanced across families."""
    per_family = total // len(families)
    remainder = total % len(families)
    records: list[dict[str, Any]] = []
    for index, family in enumerate(families):
        quota = per_family + (1 if index < remainder else 0)
        values = list(family.values)
        rng.shuffle(values)
        taken = 0
        for value in values:
            instruction = family.render(value)
            if any(record["instruction"] == instruction for record in records if record["risk_family"] == family.name):
                continue
            records.append(
                {
                    "instruction": instruction,
                    "expect_rejection": True,
                    "category": "unsafe_or_ambiguous",
                    "risk_family": family.name,
                    "action": family.action,
                    "expected_response": {
                        "status": "needs_clarification",
                        "reason": "任务含糊、矛盾或超出单无人机执行空间",
                    },
                }
            )
            taken += 1
            if taken >= quota:
                break
        if taken < quota:
            raise ValueError(f"拒绝族 {family.name} 只有 {taken} 条唯一指令，少于配额 {quota}。")
    rng.shuffle(records)
    return records


def _stratified_replace(rows: list[dict[str, Any]], prompts: list[dict[str, Any]], rng: random.Random) -> None:
    """Replace rows across categories proportionally; keeps the split balanced."""
    by_category: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_category.setdefault(row["category"], []).append(index)
    total = len(rows)
    quotas = {category: round(len(indices) * len(prompts) / total) for category, indices in by_category.items()}
    diff = len(prompts) - sum(quotas.values())
    categories = sorted(by_category, key=lambda category: len(by_category[category]), reverse=True)
    cursor = 0
    while diff != 0:
        category = categories[cursor % len(categories)]
        step = 1 if diff > 0 else -1
        if 0 <= quotas[category] + step <= len(by_category[category]):
            quotas[category] += step
            diff -= step
        cursor += 1
    chosen: list[int] = []
    for category, quota in quotas.items():
        chosen.extend(rng.sample(by_category[category], quota))
    rng.shuffle(chosen)
    for index, prompt in zip(chosen, prompts):
        replacement = dict(prompt)
        replacement["id"] = rows[index]["id"]
        rows[index] = replacement


def write_dataset_v5(output_dir: Path, scale: int = 20, seed: int = 42) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_records_v5(generate_records_v5(scale=scale, seed=seed), seed=seed)
    rng = random.Random(seed)

    test_prompts = _select_rejections(TEST_FAMILIES, 10 * scale, rng)
    validation_prompts = _select_rejections(VALIDATION_FAMILIES, 5 * scale, rng)
    train_prompts = _select_rejections(TRAIN_FAMILIES, 16 * scale, rng)
    _stratified_replace(splits["test"], test_prompts, rng)
    _stratified_replace(splits["validation"], validation_prompts, rng)
    _stratified_replace(splits["train"], train_prompts, rng)

    manifest: dict[str, Any] = {
        "dataset_version": "v5",
        "generator": "drone_planner.dataset_v5",
        "scale": scale,
        "seed": seed,
        "splits": {name: len(rows) for name, rows in splits.items()},
        "evaluation": {
            "valid_tasks": sum(1 for row in splits["test"] if not row.get("expect_rejection")),
            "expected_rejections": sum(1 for row in splits["test"] if row.get("expect_rejection")),
        },
        "rejections": {
            "train": sum(1 for row in splits["train"] if row.get("expect_rejection")),
            "validation": sum(1 for row in splits["validation"] if row.get("expect_rejection")),
            "test": sum(1 for row in splits["test"] if row.get("expect_rejection")),
        },
        "rejection_families": {
            "train": {family.name: {"action": family.action, "count": len(family.values)} for family in TRAIN_FAMILIES},
            "validation": {family.name: {"action": family.action, "count": len(family.values)} for family in VALIDATION_FAMILIES},
            "test": {family.name: {"action": family.action, "count": len(family.values)} for family in TEST_FAMILIES},
        },
        "protocol": {
            "normal_instruction_label_contract": "Instructions explicitly state all return and rectangle-size fields used by labels.",
            "unsafe_split": "Unsafe templates are separated by risk family across train, validation, and test.",
            "action_contract": "Rejection records also carry action=reject|clarify for function-calling router training.",
            "historical_comparison": "Do not compare v5 metrics directly with v4 metrics.",
        },
    }
    for name, rows in splits.items():
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest["sha256"] = {name: hashlib.sha256((output_dir / f"{name}.jsonl").read_bytes()).hexdigest() for name in splits}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 v5 规模数据集（默认 10k）。")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dataset_contract_v5"))
    parser.add_argument("--scale", type=int, default=20, help="基准 500 条的倍数；20 = 10,000 条。")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = write_dataset_v5(args.output_dir, scale=args.scale, seed=args.seed)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "splits": manifest["splits"],
                "evaluation": manifest["evaluation"],
                "rejections": manifest["rejections"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
