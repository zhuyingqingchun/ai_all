# Day 9 工作内容报告：断言规范化与数据集防护

## 1. 当前阶段结论

当前 Day 8.2 验证套件已经通过门禁：主数据集通过率达到门槛、主数据集 `plan_error` 为 0、回归数据集 100% 通过，并且已经形成了“主集 + 回归集 + 统一验收”的结构。fileciteturn12file0

这说明：
- 多步 planner 的主要已知缺陷已经修复；
- 当前多步任务模块已经从“修补阶段”进入“可持续迭代阶段”；
- 下一版不应再优先改 planner 主逻辑，而应优先补齐**评测层的稳定性和防误报能力**。

## 2. 为什么下一版要做这件事

本轮最后一个 `answer_error` 的根因不是模型退化，而是**数据集断言对动态字段写得过死**：
- 工具链执行是正确的；
- 多步整合是正确的；
- 失败来自对具体时间字面值的硬匹配；
- 最终通过修改断言而不是修改 agent 主逻辑解决。

这说明当前项目需要一层新的“工程护栏”：

1. **动态字段不要用静态字面值硬匹配**
   - 当前时间
   - 天气实时值
   - 温度/风速
   - 日期/星期

2. **数据集在被人工或小模型修改时，需要先 lint 再运行**
   - 避免再次出现“断言过死”或“数据集字段写错”的问题。

## 3. Day 9 的目标

Day 9 不改 planner，不改工具，不改 memory 主逻辑，只做下面三件事：

### 3.1 断言规范化
给多步数据集增加一套统一规则：
- 静态语义：继续使用 `expected_contains_all` / `expected_contains_any`
- 动态字段：优先改用
  - `expected_regex_any`
  - `expected_regex_all`
  - 或更宽松的语义匹配

### 3.2 数据集 guardrails
新增数据集 lint，专门检查：
- `expected_contains_all` / `expected_contains_any` 里是否写死了 `HH:MM`
- 是否写死了类似 `2026-04-15` 这样的日期
- 是否写死了 `24.8°C` 这类实时数值
- 是否出现结构错误（缺 `turns`、字段类型不对等）

### 3.3 运行门禁前置
把 lint 放到正式评测之前：
- 先过 dataset guardrails
- 再跑 Day 8.2 suite
- 若 lint 失败，则直接阻断评测

## 4. 本次补丁范围

本次补丁是**低风险、可叠加**的，不碰你当前已经跑通的 planner / evaluator 主逻辑，只新增：

- `src/vllm_agent_eval/assertion_guardrails.py`
- `scripts/lint_multistep_dataset.py`
- `scripts/run_day9_dataset_guardrails.sh`

## 5. 预期收益

做完 Day 9 之后，你会得到：

1. **断言更稳**
   - 不会再因为动态时间值写死而误报 `answer_error`

2. **数据集更安全**
   - 在改数据集时，能提前发现脆弱断言

3. **评测更可信**
   - 失败更可能是真能力问题，而不是样本设计问题

## 6. Day 9 验收标准

### 必须满足
- `python scripts/lint_multistep_dataset.py datasets/day8_multistep_dataset.json` 返回 0
- `python scripts/lint_multistep_dataset.py datasets/day8_multistep_regression_suite.json` 返回 0
- `bash scripts/run_multistep_82_suite.sh next80b_fp8` 仍然通过

### 额外建议
- 把所有带固定时间字面值的断言改成 regex 或更宽松的语义断言
- 把 lint 加进你后续所有 suite 运行脚本

## 7. 这版之后的下一步

Day 9 完成后，再做 Day 10 会更合理：
- 统一 answer evaluator 的断言表达能力
- 支持 regex / forbidden / semantic match
- 把动态字段规范正式接入 evaluator

也就是说：
- **Day 9 = guardrails**
- **Day 10 = evaluator 断言能力升级**

