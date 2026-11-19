Type: spec
Status: ready-for-agent

# Router 改进与 v5 Planner 重训的分阶段发布

## Problem Statement

当前项目已经有三块可运行的能力：v5 的 10k 合成指令任务集、一个在 v5 训练集上微调的 function-calling router，以及一个能执行工具调用、校验、PID 单无人机执行与 Agent 级评测的编排层。但三者的训练规模不一致，导致结论不可比：

- 数据集已扩到 10,000 条，但主 Planner 仍是在 v4 的 400 条训练样本上得到的 LoRA-SFT Planner；v5 的规模只被 router 使用。
- Router 在 v5 的未见风险族上工具选择准确率为 86.3%，但 `unsupported_ellipse` 一族 0/20，全部被误判为可规划任务；也就是说“按能力边界拒绝”仍不成立。
- 目前“router agent 85.7% vs workflow planner 77.3%”的端到端对比中，router 使用 v5 的 8k 训练数据、workflow 使用 v4 的 400 条 adapter，不能作为“router 比 Planner 强”的证据。
- 缺少一个冻结的、同数据规模的评测口径：既没有区分“Planner 实际处理的任务子集”与“端到端任务”的条件指标，也没有明确何时必须重训 Planner、何时只需重评。

从项目负责人视角，问题是：**我需要一个可执行的发布顺序，把“router 改进”和“Planner 重训”拆成两个独立工作流，并明确每一阶段的验收门槛、评测口径和产物版本，避免用混杂数字写简历或做面试结论。**

## Solution

按四个阶段发布，每个阶段有明确的决策门与产物；前一阶段不通过就不进入下一阶段。

### 阶段 A：Router-only 改进（不动 Planner）

- 在 v5 训练集中增加一个“不支持能力/不支持形状”的拒绝族，专用于训练集，测试集的 `unsupported_ellipse` 保持不可见；目标是让 router 学到“能力边界”而不是记住“椭圆”这个词。
- 用同一套 balanced 采样与 LoRA 超参重新训练 router，产出新的 router adapter 版本。
- 重跑 router 评测：整体工具选择准确率、`reject_task` / `ask_clarification` 召回、危险任务误判率、按拒绝族拆分的准确率，基线仍保留 zero-shot base。
- 通过门：`unsupported_ellipse` 族准确率显著提升（目标 ≥80%），整体工具选择准确率不低于上一版（86.3%），其他 8 个已达 100% 的族不退化。

### 阶段 B：用现有 Planner 重评（不重训）

- Router 改变后，进入 `plan_task` 的任务子集与比例会变，必须重新评测，而不是重训 Planner。
- 同时报告两类口径：
  - 条件口径：只统计被 router 路由到 `plan_task` 的任务，Planner 的计划合法率、结构化计划语义准确率、执行评测结果；
  - 端到端口径：包含 router 正确拒绝/澄清的任务，统计契约通过率、安全拒绝率、误拒率、平均工具调用数与时延。
- 明确记录当前 Planner 仍是 v4 规模训练；此阶段结论只能用于验证 router 改进，不作为最终“同规模”结论。

### 阶段 C：v5 Planner 重训（数据集规模化工作流）

- 用 v5 的 8,000 条训练样本重训 LoRA-SFT Planner，新建 adapter 版本，不覆盖 v4 adapter；超参与 v4 保持一致，记录数据哈希、训练日志、验证损失。
- 在 v5 的 1,000 条测试集（800 正常 + 200 应拒绝）上做执行评测：结构化计划语义准确率、计划约束合法率、安全拒绝率、误拒率、PID 单无人机执行成功率。
- 通过门：Planner 在 v5 测试集上的结构化计划语义准确率不低于 v4 Planner 在 v4 上的水平（62.5%），且安全拒绝率与误拒率不退化；若未达到，保留 v4 adapter 作为主 Planner，并记录原因。

### 阶段 D：同规模冻结评测与对外数字

- 使用阶段 C 的 v5 Planner + 阶段 A 的 router，冻结一次评测：固定测试子集（至少 300 条；最终报告可扩展到 1,000 条），固定 seed、提示、解码参数和 adapter 版本。
- 同时跑 workflow 基线（同一 v5 Planner 直接规划/拒绝），得到同规模的 router agent vs workflow 对比；两者使用同一数据规模与同一测试集。
- 产物：summary、逐条预测 JSONL、按族拆分报告、失败案例分析、面试/简历可用的数字表；所有旧版本产物保留。

### 验收清单

- [ ] v5.1 数据集包含新的训练集拒绝族，测试集 SHA 与 v5 保持一致，训练/验证/测试的拒绝族仍完全不重叠。
- [ ] 新 router 在固定测试子集上 `unsupported_ellipse` 召回 ≥ 80%，整体工具选择准确率 ≥ 86.3%。
- [ ] 阶段 B 的条件指标与端到端指标都已重算，并明确标注 Planner 未重训。
- [ ] v5 Planner 训练完成，v5 测试集指标达标；v4 adapter 与旧评测产物未被覆盖。
- [ ] 阶段 D 的冻结报告中，router 与 workflow 使用同一 Planner、同一测试集、同一数据规模。
- [ ] 简历/面试只引用阶段 D 的数字，并注明测试子集大小与数据版本。

## User Stories

1. As a project owner, I want the 10k dataset to have a versioned revision when the training rejection families change, so that older router evaluations remain reproducible and traceable.
2. As a data engineer, I want the new unsupported-shape family to live only in the training split, so that the test set still measures generalization to unseen risk families.
3. As a data engineer, I want the test split byte-identical to the current v5 test split, so that router improvements can be compared on the same evaluation tasks.
4. As a router trainer, I want the new family to use shapes that differ from the test ellipse family, so that the router learns the concept of “unsupported capability” rather than memorizing the word “ellipse”.
5. As a router trainer, I want balanced training data with a documented sampling ratio, so that the router does not collapse into always calling `plan_task`.
6. As a router trainer, I want a new adapter revision rather than overwriting the previous adapter, so that before/after comparisons are possible.
7. As an evaluator, I want per-family tool-selection accuracy, so that I can see whether the improvement came from the intended family and whether other families regressed.
8. As an evaluator, I want correct metric denominators for `reject_task` recall, `ask_clarification` recall and unsafe-plan rate, so that aggregate numbers are not diluted by the class distribution.
9. As an evaluator, I want a zero-shot base baseline on the exact same prompts, so that the value of LoRA fine-tuning is quantified rather than assumed.
10. As an engineer, I want a documented decision rule that router-only changes do not require Planner retraining, so that time is not wasted on a 2–3 hour training run when only the gate changed.
11. As an engineer, I want the decision rule to also state when Planner retraining is required (input contract change, dataset-scale switch, base model change), so that future changes are handled consistently.
12. As an evaluator, I want conditional Planner metrics computed only on tasks routed to `plan_task`, so that router filtering does not silently change the meaning of Planner accuracy.
13. As an evaluator, I want end-to-end contract metrics that include router rejection and clarification, so that the full agent behavior is measured.
14. As a project owner, I want the v5 Planner trained on the full 8k train split with a new adapter version, so that the expanded dataset actually affects the main Planner.
15. As a project owner, I want v4 Planner metrics and artifacts to remain untouched, so that historical claims stay valid and comparable.
16. As an evaluator, I want the frozen comparison to use the same v5 Planner in both router-agent and workflow arms, so that the comparison isolates the routing layer rather than training scale.
17. As a report author, I want a fixed test subset, fixed seed and frozen adapter versions, so that the final numbers can be reproduced and defended in interviews.
18. As a report author, I want per-family failure analysis and representative traces, so that the remaining weakness (unsupported capabilities) is documented honestly.
19. As a service owner, I want the router-enabled agent path to be covered by tests, so that enabling the router in the service does not break plan/reject/clarify semantics.
20. As a maintainer, I want dataset, router, Planner and service changes to be separable, so that each stage can be reviewed and rolled back independently.
21. As a candidate, I want resume bullets to be updated only after the frozen evaluation, so that no number mixes training scales or test subsets.
22. As an engineer, I want wall-clock and compute cost recorded for router retraining, Planner retraining and evaluation, so that the next iteration can be planned realistically.

## Implementation Decisions

### 数据集与协议

- 在 v5 基础上创建一个版本化的数据修订（v5.1），只改变 **train** split 的拒绝族构成：新增一个“不支持能力/不支持形状”训练族；validation 与 test 的拒绝族、样本数与 SHA 保持不变。
- 新训练族使用与测试椭圆不同的不支持形状（例如八字形、莫比乌斯环、球面、文字/图案等），避免 test 族泄漏；每个族的模板与参数值必须唯一，生成器需在族内去重并在配额不足时报错。
- v5.1 的 manifest 继续记录 split 大小、拒绝族-动作-数量、SHA-256 与协议说明；旧 v5 manifest 与产物保留。
- 正常任务的标签契约不变：任务指令显式写出返航与矩形半边长等标签字段，结构化飞行计划仍由确定性校验器验证。

### Router 数据与训练

- 训练标签保持三工具动作空间：`plan_task`、`reject_task`、`ask_clarification`；映射规则为正常任务 → `plan_task`，越界/矛盾/不支持能力 → `reject_task`，缺少必要信息 → `ask_clarification`。
- 训练集采用 balanced 采样（正常任务抽样 + 拒绝/澄清过采样），并在数据生成脚本中输出每类数量；采样 seed 固定。
- Router 训练沿用 LoRA-SFT 配置（r=8、alpha=16、2 epoch、batch 1 + 梯度累积 8、最大长度 768，仅监督 assistant 工具调用），产出新 adapter 版本；训练日志、run config、验证损失必须落盘。
- Router 输出契约不变：单个 JSON 对象 `{"name": ..., "arguments": {...}}`；`plan_task` 只携带原始任务指令，不向 Planner 传递 router 私有字段。

### Planner 训练

- v5 Planner 复用现有 LoRA-SFT 训练流程与超参，在 v5.1 的 8,000 条 train 上训练 3 epoch；新建 adapter 版本，不覆盖 v4 adapter。
- 训练前记录数据集 SHA-256、训练参数、基座模型版本；训练后记录 train/validation loss、训练耗时与峰值显存。
- v5 Planner 的评测使用现有执行评测流程：先做 JSON/契约/执行空间校验，再做字段级语义匹配，最后对语义正确计划做 PID 单无人机执行。

### 评测与指标

- Router 评测必须输出：JSON 合法率、整体工具选择准确率、分工具 precision/recall/F1、`reject_task` 召回、`ask_clarification` 召回、危险任务被误判为可规划的比例、正常任务被正确规划的比例、按拒绝族拆分的准确率。
- 所有比例的分母必须是对应子集（例如 reject 召回的分母是应 `reject_task` 的样本），不得用全量样本做分母。
- 端到端评测必须区分：条件口径（仅被 `plan_task` 处理的任务）与全量口径（包含拒绝/澄清）。
- 冻结评测必须固定：测试子集、seed、提示模板、解码参数、router adapter 版本、Planner adapter 版本；预测逐条落盘，报告可重算。
- 对比基线至少包含：同提示的 zero-shot base router、同 v5 Planner 的 workflow 基线；不得用 v4 规模 Planner 的 workflow 数字冒充同规模基线。

### 编排与服务

- Agent 运行时保持现有状态机：START → router 决策 → `plan_task`/`reject_task`/`ask_clarification` → 校验/修复/执行/结束；router 只做入口工具选择，不改变 Planner 的输入契约。
- 语义状态保持：`plan_task` 成功 → accepted；`reject_task` → rejected；`ask_clarification` → clarification；无法解析的 router 输出 → invalid（fail closed）。
- 服务层保持现有端点与运行持久化；启用 router 时行为与 CLI 一致，不新增对外 API 破坏性变更。

### 版本与产物

- 所有新阶段使用新的版本目录或版本标识（数据集修订、router adapter、Planner adapter、评测报告），旧产物不覆盖。
- 最终报告标注数据版本、adapter 版本、测试子集大小、训练规模；v4 与 v5 指标不得混引。

## Testing Decisions

好的测试只验证外部可观察行为，不验证实现细节；优先使用现有最高层接缝，不新增低层接缝。

### 选择的接缝

1. **数据集 manifest 接缝**：以生成后的 manifest 与 split 文件为外部行为。验证 v5.1 的 split 大小、拒绝族隔离、动作分布、训练集新增族存在、validation/test 的 SHA 与 v5 一致、正常计划仍通过契约校验。现有 `test_dataset_v5` 是同类先例。
2. **Router 数据接缝**：以“任务指令 → 期望工具调用”的转换与 balanced 采样结果为外部行为。验证正常/越界/缺信息分别映射到三个工具，采样后每类数量符合配置，训练/验证/测试的拒绝族不重叠。用纯函数测试，不加载模型。
3. **Router 指标接缝**：以评测汇总函数为外部行为。用合成预测验证分母正确性（分组召回不被全量稀释）、分工具 P/R/F1、危险任务误判率、端到端契约语义（`plan_task` 看计划合法性，拒绝/澄清看动作完全匹配）。不加载模型。
4. **Agent 路由接缝**：复用现有 `AgentRunner` + fake router 测试。验证 `plan_task` 进入计划节点、`reject_task` 跳过 Planner、`ask_clarification` 进入 clarification 状态、非法 router 输出 fail closed、工具轨迹顺序正确。现有 router graph 测试是同类先例。
5. **服务接线接缝**：复用现有服务测试，通过 fake router 验证 router 启用时请求仍走相同端点、状态与轨迹正确。现有 service 测试是同类先例。
6. **Planner 训练/评测接缝**：以 CLI 产物为外部行为。验证 v5.1 数据集可被训练流程消费、adapter 可被重新加载、v5 测试集可被执行评测并产出 summary；训练质量作为验收运行记录，不写成单元测试断言（避免模型随机性导致脆弱）。
7. **冻结评测接缝（最高层）**：以评测 CLI 的 summary/report 为外部行为。验证报告 schema、任务数、族覆盖、adapter 版本与 seed 记录完整；指标阈值作为发布门槛记录在验收清单中，不在自动测试里硬断言模型精度。

### 测试原则

- 不在单元测试中加载 Transformer 模型；模型相关验证交给固定的验收运行。
- 新增测试优先落在现有测试文件中已有的同类接缝，不新建平行测试框架。
- 对失败路径（非法 router 输出、工具异常、越界计划）必须有断言，确保 fail closed。
- 数据集与评测报告必须可复算：测试读取产物文件，不重跑生成逻辑之外的随机过程。

### Prior art

- 数据集：`drone_planner.tests.test_dataset_v5`
- 契约与评测语义：`drone_planner.tests.test_contracts`
- Agent 路由：`agent.tests.test_agent.RouterGraphTests`
- 服务接线：`agent.tests.test_service`
- 指标分母/工具语义：新增纯函数测试，参照 router 评测代码的现有行为。

## Out of Scope

- 记忆/RAG 的继续改进；上一轮已得到负结果，本轮不调整记忆策略。
- 服务生产化：Docker、鉴权、限流、请求队列、多副本、可观测性平台。
- MCP server、多 Agent 协作、Agent 间通信。
- 更换基座模型（例如改用 1.5B/3B router 或 API 模型）；本轮仍用 Qwen2.5-0.5B。
- TD3 训练或控制器改进；执行评测仍使用现有 PID 基线。
- 真实飞行数据、VLM 评审、多无人机协同执行。
- 修改 v4 数据集、v4 Planner adapter 或 v4 评测产物；v4 作为历史证据冻结。
- 追求 1,000 条全量 router 评测的统计显著性优化；300 条固定子集即可作为本轮发布口径，最终报告可自愿扩展。

## Further Notes

### 当前基线（供验收对照）

- Router：300 条 v5 test 子集（100 正常 + 200 应拒绝），工具选择准确率 0.863，`reject_task` 召回 0.875，`ask_clarification` 召回 0.950，危险任务误判率 0.100；按族看 8/10 族 100%，`vague_objective` 18/20，`unsupported_ellipse` 0/20。
- Zero-shot base：工具选择准确率 0.163，危险任务误判率 0.895。
- 端到端：router agent 0.857 vs v4 规模 workflow 0.773（不同训练规模，仅作历史参考，不作为发布结论）。
- v5 Planner 尚未训练；v5 的 10k 规模目前只服务 router 与数据协议。

### 决策规则（写入团队约定）

- **Router-only 变更**（新增拒绝族、调整采样、换 router adapter）：不重训 Planner；必须重跑 router 指标、条件 Planner 指标与端到端指标。
- **Planner 输入契约变更**（`plan_task` 参数变化、提示模板变化、约束字段变化）：重训或至少重评 Planner。
- **数据集规模/版本切换**（v4 → v5）：重训 Planner，并在同规模下重跑端到端。
- **基座模型或 tokenizer 变更**：router 与 Planner 都需要重新训练与重评。

### 成本估算

- Router 重训：约 25 分钟 GPU（当前 3,673 条、2 epoch 的实测为 24 分 38 秒）。
- Router 评测：300 条约 10–15 分钟（含 base 基线、router、Planner 端到端）。
- v5 Planner 重训：约 2–3 小时 GPU（8,000 条、3 epoch、batch 1 + 梯度累积 8）。
- v5 Planner 评测：1,000 条测试集约 30 分钟（含 PID 仿真，可按子集缩短）。

### 领域词汇缺口

- `CONTEXT.md` 目前没有“router / 工具调用 / 条件指标 / 端到端契约”等术语；本轮涉及的概念建议在后续 domain-modeling 中补充，以免不同报告使用不同口径。

### 风险

- 训练集新增“不支持形状”族后，router 可能只学会列举的训练形状名；验收必须看 test 族的实际召回，而不是训练损失。
- v5 Planner 重训后语义准确率可能低于 v4（数据更难、类别更多），此时需要保留 v4 作为主 Planner 并如实报告，不得为了“10k”而牺牲指标口径。
- 同规模端到端对比仍受 router 与 Planner 的交互影响：router 误放的危险任务会被 Planner 处理，Planner 自身的拒绝能力也会影响端到端契约率；报告中必须把 router 漏放率与 Planner 条件指标分开列。
- 数据集版本与 adapter 版本增多后，容易混引数字；所有报告必须固定版本号与测试子集大小。

## Comments

- 由上一轮对话中的建议执行顺序整理而来：先做 router-only 改进并重评，再重训 v5 Planner，最后做同规模冻结评测。
- 接缝选择按现有测试先例（数据集、router 指标、Agent 路由、服务接线）优先，未引入新的模型内单元测试。
- 2026-11-19 实现进展：
  - 已完成：v5.1 数据集修订（train-only 新增 `unsupported_shape` / `unsupported_conic` / `missing_zone_constraint`，validation/test 与 v5 逐字节一致，含测试）；router 数据映射/采样与指标分母的纯函数测试；router 评测增加“条件 Planner 指标”；v2 router 评测产物 `artifacts/router_eval_v2/`。
  - 阶段 A 未通过验收门：router v2 的 `unsupported_ellipse` 仍 0/20，clarify 召回 0.50，`missing_no_fly_zone` 0/20。第三版训练数据（`artifacts/router_data_v3/`）已生成，但训练在 8GB 笔记本 GPU 上因热降频被中止，未产出 v3 adapter。
  - 阶段 C/D 未执行：v5 Planner 未重训，同规模冻结评测缺失；原因是同一算力限制（预计 2–3 小时 GPU，实际降频后显著更长）。
  - 下一步：在冷却/外接散热或云 GPU 上先完成 `lora_router_contract_v3` 训练与阶段 A 门禁复测，再按本 spec 进入阶段 C/D。
