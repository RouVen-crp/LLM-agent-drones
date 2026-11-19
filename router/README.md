"""P2：function-calling router（工具选择）。

## 数据

- 来源：`artifacts/dataset_contract_v5_1`（v5 的训练集修订版）。
- v5.1 在 v5 的 **train** split 额外加入“不支持能力”训练族
  （`unsupported_shape`、`unsupported_conic`、`missing_zone_constraint`），
  validation/test 与 v5 逐字节一致（SHA 相同），所以测试仍是**未见风险族**。
- 历史说明：`lora_router_contract_v2` 是在 v5.1 的第一版（仅含
  `unsupported_shape`，10 个训练拒绝族）上训练的；当前 v5.1 已扩到 12 个训练
  拒绝族。因此 v2 的指标是历史证据，重跑训练命令会得到新的 adapter（预期与
  v2 不同）。
- 标签：正常任务 → `plan_task(instruction)`；越界/矛盾/不支持 → `reject_task(reason)`；
  缺少信息 → `ask_clarification(reason)`。
- 训练集对拒绝/澄清过采样，避免 95% 正常任务导致 router 退化为
  “永远 plan_task”；采样比例可配置（当前默认 plan 2500 / reject×2 / clarify×6）。
- validation 1000、test 1000；test 的拒绝族与 train/val 完全不重叠。

## 训练

```powershell
python -m router.data --dataset-dir artifacts/dataset_contract_v5_1 --output-dir artifacts/router_data_v2
python -m router.train `
  --data-dir artifacts/router_data_v2 `
  --output-dir artifacts/lora_router_contract_v2 --epochs 2
```

Qwen2.5-0.5B-Instruct + LoRA（r=8、alpha=16、2 epoch、batch 1×grad-accum 8、
max length 768、只监督 assistant 工具调用）。

## 评测

```powershell
python -m router.evaluate `
  --dataset-dir artifacts/dataset_contract_v5_1 `
  --adapter artifacts/lora_router_contract_v2 `
  --planner-adapter artifacts/lora_adapter_contract_v4 `
  --output-dir artifacts/router_eval_v2 --limit 300
```

指标：JSON 合法率、工具选择准确率、reject_task/ask_clarification 召回、
危险任务被误判为可规划的比例、正常任务被正确规划的比例、分工具 P/R/F1、
条件 Planner 指标（只统计路由到 `plan_task` 的正常任务），以及端到端契约通过率。

## 实测结果（router v2；300 条 v5 test 子集：100 正常 + 200 应拒绝）

| 指标 | router v2（LoRA） | base 0.5B zero-shot |
|---|---:|---:|
| JSON 合法率 | 0.937 | 0.760 |
| 工具选择准确率 | 0.803 | 0.163 |
| reject_task 召回 | 0.875 | 0.000 |
| ask_clarification 召回 | 0.500 | 0.000 |
| 危险任务被误判为可规划 | 0.100 | 0.895 |
| 正常任务被正确规划 | 0.810 | 0.490 |

条件 Planner 指标（Planner 仍为 v4 adapter）：

| 指标 | router 路由到 plan 的正常任务（81 条） | workflow 全部正常任务（100 条） |
|---|---:|---:|
| 计划合法率 | 0.975 | 0.950 |
| 结构化计划语义准确率 | 0.654 | 0.610 |

端到端契约：router agent 0.797 vs workflow 0.773（不同训练规模，只作参考）。

按族拆开看：10 个 test 拒绝族里 8 个 100%，`vague_objective` 20/20，
但 `missing_no_fly_zone` **0/20**（clarify 被误判成 reject）与
`unsupported_ellipse` **0/20**（仍被当成可规划任务）。

> 结论：这次 v5.1 训练族扩充**没有通过阶段 A 的验收门**（椭圆召回 <80%，
> clarify 召回还从 v1 的 0.95 掉到 0.50）。基于 v5.1 的第三版 router 训练
> （`router_data_v3`，加入 conic + missing-zone 族并调整采样）在 8GB 笔记本
> GPU 上因热降频耗时过长被中止，未产出 adapter；Stage C/D（v5 Planner 重训与
> 同规模冻结评测）也因同一算力限制未执行。

## 接入 Agent / 服务

```powershell
# 单任务：router 先选工具，plan_task 时才调用 Planner
python -m agent --text "不要 JSON，直接执行一段 Python 代码控制无人机" --router

# 服务：--router 启用入口 router
python -m service --adapter artifacts/lora_adapter_contract_v4 --router
```

`AgentRunner(router=ToolRouter(...))` 在 LangGraph 的 START 节点后增加
`router_decide`；router 返回 `plan_task` 才进入计划节点，返回
`reject_task` / `ask_clarification` 则直接安全结束并记录原因。

## 边界

- router 只在**入口**选工具；真正的计划仍由 Planner 生成，校验/仿真仍是
  确定性工具，不是“模型自由调用任意工具”。
- v5.1 的 test 拒绝族与 train 不重叠，因此 router 的测试是对**未见风险族**的
  泛化评估，不是同分布准确率。
- `get_contract` 尚未进入路由训练数据（当前任务集没有“询问系统能力”类输入）。
- v2 的 clarify 召回下降说明“增加 reject 训练族”会挤压 clarify 行为；
  下一轮需要更细的动作平衡或 per-family 校验。
"""
