# 自然语言无人机任务规划与单无人机仿真验证

面向 LLM 应用算法岗位的可复现实验项目：将中文任务指令转换为受约束的**结构化飞行计划**，在离线评测中比较 Planner，并用 PyBullet 验证选定无人机的单机执行。

## 仓库范围

本仓库只提交代码、协议与文档：

- `artifacts/`（数据集、LoRA adapter、评测产物）由命令生成，不提交；README 中的产物路径需要先按命令重新生成。
- `data/`、`.venv/`、`.bench/`、`reference_docs/` 以及个人面试/论文材料不提交。
- 仅保留 `sources/__init__.py` 与 `sources/TD3_resources/`（含 TD3 示例权重）；旧 Leviosa 的 LLM 规划、语音与实验日志不提交。
- PID 单机执行基线不依赖 `sources/`；TD3 对照需要该目录下的权重。

## 真实能力边界

- Planner 只生成 JSON，不执行模型生成的 Python 代码。
- 支持至多 3 架无人机的**独立航迹规划**；不声称多机协同控制、同步或碰撞规避。
- 稳定演示入口只接收文本。音频是预留接口，当前会明确拒绝执行。
- 运行时校验覆盖 JSON、契约与执行空间边界；逐航点语义匹配只用于带标注的离线评测。
- PID 是当前单机 PyBullet 执行基线；TD3 保留为对照，不宣称其已成功跟踪计划。
- `sources/` 保存旧原型和底层资源，不是新版 Planner 的稳定入口。

新版流程：

```text
中文任务指令
  -> Qwen Planner（Zero-shot / Few-shot / LoRA-SFT）
  -> 结构化飞行计划 JSON
  -> JSON、契约与空间边界校验
  -> 离线字段级语义评测 / 安全拒绝评测
  -> selected_drone_id 的 PID 或 TD3 单机 PyBullet 对照执行
```

执行空间为 `x/y ∈ [-0.2, 0.2]m`、`z ∈ [0.2, 0.5]m`。任务执行成功要求实际轨迹未越界、未提前结束，并在规定时限内以不超过 `0.05m` 的误差达到每个航点及必要的返航点。

## v4：当前可信实验协议

v4 保留原有逐航点 JSON Schema，但修复了历史协议问题：

- 指令明确写出返航与矩形半边长，标签可由输入唯一推出；
- 应拒绝任务按风险模板族切分，训练、验证、测试之间不复用模板族；
- Few-shot 仅选择正确配对的有效任务示例，且覆盖不同任务类别；
- 每次实验保存数据集 SHA-256、完整运行参数、逐例 JSONL、按任务类型统计和字段级失败分布；
- 同时报告非法任务拒绝率与合法任务误拒率。

v2/v3 产物均作为历史证据保留，**不得与 v4 指标直接比较或混合引用**。

## 运行

```powershell
# 核心契约测试
.\.venv\Scripts\python.exe -m unittest -v drone_planner.tests.test_contracts

# 生成 v4 数据集（不覆盖历史数据）
.\.venv\Scripts\python.exe -m drone_planner.dataset --output-dir artifacts/dataset_contract_v4 --seed 42 --include-training-rejections

# 训练本地 Qwen2.5-0.5B LoRA adapter
.\.venv\Scripts\python.exe -m drone_planner.train --dataset-dir artifacts/dataset_contract_v4 --output-dir artifacts/lora_adapter_contract_v4 --epochs 3 --seed 42

# Zero-shot、Few-shot、LoRA-SFT 对照；PID 仅在语义正确计划上执行
.\.venv\Scripts\python.exe -m drone_planner.experiments --dataset-dir artifacts/dataset_contract_v4 --adapter artifacts/lora_adapter_contract_v4 --output-dir artifacts/evaluation_contract_v4 --variants zero_shot few_shot lora_sft --simulate --simulator pid

# 直接执行标注计划，校准 PID 或 TD3 的执行上限
.\.venv\Scripts\python.exe -m drone_planner.oracle_simulation --dataset-dir artifacts/dataset_contract_v4 --output-dir artifacts/oracle_simulation_v4_pid --simulator pid
.\.venv\Scripts\python.exe -m drone_planner.oracle_simulation --dataset-dir artifacts/dataset_contract_v4 --output-dir artifacts/oracle_simulation_v4_td3_probe --simulator td3 --end 5
```

稳定文本演示默认使用 PID：

```powershell
.\.venv\Scripts\python.exe -m drone_planner --text "让 drone-1 以 0.35 米高度依次经过 (0.05, 0.00)、(0.10, 0.05) 后返回起点" --adapter artifacts/lora_adapter_contract_v4 --simulate --simulator pid --output artifacts/runs/v4_demo.json
```

## Agent 编排与评测（P1/P3）

`agent/` 把 v4 的 Planner、校验器与 PID 执行器封装成工具，用 LangGraph 串成
`retrieve_memory → plan → validate → repair（最多 3 次）→ simulate` 的有状态流程；
每一步写入可审计的 tool trace，支持 `seed_plan_text` 复用首轮规划做对照评测。

```powershell
# 工具 schema（不加载模型）
.\.venv\Scripts\python.exe -m agent --print-tools

# 单任务：规划 -> 校验 -> PID 仿真
.\.venv\Scripts\python.exe -m agent --text "让 drone-1 以 0.35 米高度依次经过 (0.05, 0.00)、(0.10, 0.05) 后返回起点" --simulate

# direct / agent / agent+memory 对照（结果在 artifacts/agent_eval_v2/）
.\.venv\Scripts\python.exe -m agent.evaluate --dataset-dir artifacts/dataset_contract_v4 --adapter artifacts/lora_adapter_contract_v4 --output-dir artifacts/agent_eval_v2 --limit 30 --memory
```

30 条子集上的结论：agent 修复把契约通过率从 96.7% 提到 100%；朴素 RAG 会
把拒绝率从 100% 拉到 60%，加风险门控后拒绝恢复到 100%，但语义准确率没有
提升（0.55 vs 0.60）、时延升到 9s，因此 **memory 默认关闭**，只保留模块与
负结果分析。边界：这是 workflow agent（Planner 只负责生成计划/拒绝，工具由
状态机编排），不是“模型自主选择任意工具”；数字不能与 v4 的 40+10 全量结果混引。

## Agent 服务化（P5）

```powershell
.\.venv\Scripts\python.exe -m service --adapter artifacts/lora_adapter_contract_v4
# 可选：--router 启用 P2 函数调用 router
# GET /healthz, GET /v1/tools, POST /v1/missions, POST /v1/missions/stream (SSE),
# GET /v1/missions/{run_id}, POST /v1/tools/{name}
```

运行记录持久化到 `artifacts/agent_runs/runs.sqlite`；单任务同步执行、SSE 逐节点
推送，属于演示级服务，不含鉴权、限流、队列与 Docker 部署。

## 数据集 v5 / v5.1（10k）与 P2 函数调用 Router

v5 把 v4 协议扩到 **10,000 条**（8000 train / 1000 validation / 1000 test，
测试集 800 正常 + 200 应拒绝），拒绝样本扩到 23 个风险族、按 split 隔离，
并带 `action=reject|clarify` 供 router 监督。生成器与 manifest：

```powershell
.\.venv\Scripts\python.exe -m drone_planner.dataset_v5 --output-dir artifacts/dataset_contract_v5 --scale 20
```

v5.1 是训练集修订版：train 额外加入“不支持能力”族
（`unsupported_shape` / `unsupported_conic` / `missing_zone_constraint`），
validation/test 与 v5 逐字节一致，保证 router 评测仍用同一测试集：

```powershell
.\.venv\Scripts\python.exe -m drone_planner.dataset_v5_1 --base-dir artifacts/dataset_contract_v5 --output-dir artifacts/dataset_contract_v5_1
```

P2 用 Qwen2.5-0.5B + LoRA 训练入口 router（plan_task / reject_task /
ask_clarification），数据 `artifacts/router_data_v2`、adapter
`artifacts/lora_router_contract_v2`、评测 `artifacts/router_eval_v2/`：

```powershell
.\.venv\Scripts\python.exe -m router.data --dataset-dir artifacts/dataset_contract_v5_1 --output-dir artifacts/router_data_v2
.\.venv\Scripts\python.exe -m router.train --data-dir artifacts/router_data_v2 --output-dir artifacts/lora_router_contract_v2 --epochs 2
.\.venv\Scripts\python.exe -m router.evaluate --dataset-dir artifacts/dataset_contract_v5_1 --adapter artifacts/lora_router_contract_v2 --output-dir artifacts/router_eval_v2 --limit 300
```

300 条 v5 test 子集（100 正常 + 200 应拒绝）上的结果：

| 指标 | router v2（LoRA） | base 0.5B zero-shot |
|---|---:|---:|
| 工具选择准确率 | 0.803 | 0.163 |
| reject_task 召回 | 0.875 | 0.000 |
| ask_clarification 召回 | 0.500 | 0.000 |
| 危险任务被误判为可规划 | 0.100 | 0.895 |
| 正常任务被正确规划 | 0.810 | 0.490 |

条件 Planner 指标（Planner 仍为 v4 adapter）：router 路由到 plan 的 81 条正常
任务上计划合法率 0.975、语义准确率 0.654；workflow 全部 100 条正常任务为
0.950 / 0.610。端到端契约 0.797 vs 0.773（不同训练规模，仅作参考）。

说明：`lora_router_contract_v2` 训练于 v5.1 的早期版本（10 个训练拒绝族）；
当前 v5.1 已扩到 12 族，重跑训练命令会得到新 adapter，指标会不同。

**阶段 A 未通过验收门**：`unsupported_ellipse` 仍 0/20，clarify 召回从上一版
0.95 掉到 0.50（`missing_no_fly_zone` 0/20）。第三版训练数据（conic +
missing-zone 族）已生成，但在 8GB 笔记本 GPU 上因热降频被中止，未产出 adapter；
Stage C（v5 Planner 重训）与 Stage D（同规模冻结评测）同样因算力限制未执行。
详见 `.scratch/router-v5-rollout/spec.md` 与 `router/README.md`。


## 面试表述

> 构建中文无人机任务指令到结构化飞行计划的 LLM Planner，基于合成但可审计的 v4 数据协议完成 Zero-shot、Few-shot 与 LoRA-SFT 对照；通过字段级失败归因、安全拒绝与合法任务误拒指标分析模型边界，并以 PID 与 TD3 的单机 PyBullet 对照验证 Planner 与控制器的独立影响。

只用实际产物中的 v4 指标替换简历数字。不要写“多机协同控制”“真实飞行数据”“TD3 已成功执行”或“语义错误自动反馈重规划”。
