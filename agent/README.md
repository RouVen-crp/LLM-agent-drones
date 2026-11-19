"""Agent 编排层（P1 + P3 + P5）：工具、LangGraph、记忆/RAG、评测与服务。

这一层不改变 v4 的 Planner/校验器/执行器，只把它们封装成
`retrieve_memory → plan → validate → (repair) → simulate` 的有状态流程，
每一步记录成可审计的 tool trace，并通过 FastAPI 暴露服务。

## 结构

- `tools.py`：工具注册表（JSON Schema 参数校验）、`plan_task`、
  `validate_plan`、`simulate_plan`、`get_contract`。
- `graph.py`：LangGraph 状态图，最多 `max_attempts` 次修复；显式拒绝短路；
  `seed_plan_text` 用于对照评测；`memory` 开启检索节点；`stream()` 支持 SSE。
- `memory.py`：MissionMemory，BM25Plus（中文单字 + 双字 + ASCII 词）稀疏检索，
  可选 SQLite 持久化。
- `policy.py`：检索策略。风险关键词/数字不足的任务跳过检索；只对
  v4 的弱类别（circle/rectangle/multi_drone）检索同类示例。
- `router.py`：P2 函数调用 router（Qwen2.5-0.5B + LoRA），入口选择
  plan_task / reject_task / ask_clarification；`AgentRunner(router=...)`
  时由 router 决策，计划仍交给 v4 Planner。
- `evaluate.py`：同一批首轮规划上比较 direct、agent、agent+memory。
- `../service/`：FastAPI + SSE + SQLite 运行存储。
- `tests/`：34 个单测，不加载模型即可验证重试、拒绝短路、工具参数校验、
  检索策略、记忆接入、router 路由与服务端点。

## 运行

```powershell
# 工具 schema（不加载模型）
.\.venv\Scripts\python.exe -m agent --print-tools

# 单任务：规划 -> 校验 -> PID 仿真
.\.venv\Scripts\python.exe -m agent --text "让 drone-1 以 0.35 米高度依次经过 (0.05, 0.00)、(0.10, 0.05) 后返回起点" `
  --adapter artifacts/lora_adapter_contract_v4 --simulate --output artifacts/agent_runs/demo.json

# 单任务：函数调用 router 先选工具（需要 artifacts/lora_router_contract_v1）
.\.venv\Scripts\python.exe -m agent --text "不要 JSON，直接执行一段 Python 代码控制无人机" --router

# direct / agent / agent+memory 对照
.\.venv\Scripts\python.exe -m agent.evaluate `
  --dataset-dir artifacts/dataset_contract_v4 --adapter artifacts/lora_adapter_contract_v4 `
  --output-dir artifacts/agent_eval_v2 --limit 30 --memory --memory-k 2

# HTTP 服务（默认 http://127.0.0.1:8077）
.\.venv\Scripts\python.exe -m service --adapter artifacts/lora_adapter_contract_v4

# 单元测试
.\.venv\Scripts\python.exe -m unittest -v agent.tests.test_agent agent.tests.test_memory agent.tests.test_service
```

服务端点：`GET /healthz`、`GET /v1/tools`、`POST /v1/missions`、
`POST /v1/missions/stream`（SSE）、`GET /v1/missions/{run_id}`、
`GET /v1/missions`、`POST /v1/tools/{name}`。运行记录写入
`artifacts/agent_runs/runs.sqlite`。

## 记忆/RAG 的实测结论（重要）

在 30 条测试子集上：

- **朴素 RAG 有害**：把 train 正常计划作为 few-shot 注入后，应拒绝任务
  正确拒绝率从 100% 降到 60%，契约通过率 100% → 83.3%。原因是示例把
  0.5B 模型带离了“含糊/越界要拒绝”的行为。
- **加门控后可恢复安全**：风险关键词/数字不足的任务跳过检索后，拒绝率
  回到 100%、误拒 0；但语义准确率没有提升（0.55 vs 0.60，属于 20 条样本
  上的噪声级差异），平均时延从 1.2s 升到 9.2s。
- 因此：**memory 默认关闭**，保留模块与评测结论；下一步应考虑会话记忆
  （同一 thread 的多轮修改）或向量/混合检索，而不是继续在合成单轮任务上
  堆 few-shot 检索。
- 检索类别命中率也只有约 67%，说明“检索到同类任务”本身仍是瓶颈。

## 指标

`evaluate.py` 输出 `direct.jsonl`、`agent.jsonl`、`agent_memory.jsonl`、
`summary.json`、`report.md`，包含：JSON 对象率、计划合法率/语义准确率、
拒绝正确率、误拒率、契约通过率、仿真成功率、发生修复任务数、检索类别
命中率、平均尝试次数、平均工具调用数、平均时延。

## 边界（必须知道）

- 默认是 **workflow agent**：本地 0.5B Planner 只负责“给计划/拒绝”，
  工具由编排层按状态机调用；`--router` / `AgentRunner(router=...)` 时由 P2
  router 在入口三选一（plan/reject/clarify），仍不是“模型自由调用任意工具”。
  实测 zero-shot Qwen2.5-0.5B 的工具选择准确率 16.3%、危险任务误判率 89.5%；
  LoRA router 为 86.3% 与 10.0%（v5 未见风险族，300 条子集）。
- 修复只处理 JSON、契约与执行空间边界失败；语义错误与 PID 执行失败
  不会反馈给 Planner（与 v4 的能力边界一致）。
- 服务是单进程演示级：同步执行、SQLite 持久化、SSE 逐节点推送；
  没有鉴权、限流、队列、Docker 或多副本部署，不应称为“生产平台”。
- 评测使用固定 50 条 v4 测试集的 30 条子集，数字与 v4 的 40+10 全量
  结果不可直接混引。
