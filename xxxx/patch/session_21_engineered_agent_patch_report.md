# 第21轮：工程化成熟代理补丁说明

## 目标
基于当前单主模型 GUI Agent 基线，补充更工程化、低 token、低侵入的控制逻辑：

1. 轻量状态摘要，替代长历史堆叠。
2. 前 3 步默认禁止 `COMPLETE`。
3. 重复点击时启用“重瞄”策略，而不是重复完全相同坐标。
4. 早期阶段谨慎 `SCROLL`，优先可见目标点击。
5. 保持只通过 `BaseAgent._call_api()` 调单个视觉模型，不与官方替换版 `agent_base.py` 冲突。

## 变更文件
- `agent.py`
- `utils/state_manager.py`

## 关键设计

### 1. `agent.py`
- 新增 `_build_system_prompt()`：
  - 强约束只输出一个动作。
  - 固定为显式语法：`CLICK:[[x,y]] / TYPE:['内容'] / ...`
  - 强调“不要因为不确定就 COMPLETE”。
  - 强调“点偏时应重瞄附近小区域”。
- 新增 `_build_user_text()`：
  - 只传压缩状态摘要，不传冗长历史。
- `act()`：
  - 首步仍走 `OPEN` 启发式。
  - 解析后统一走 `state.postprocess()`。
  - parser 失败走 `safe_fallback()`。

### 2. `utils/state_manager.py`
- Agent 状态增加：
  - `current_subgoal`
  - `page_guess`
  - `last_typed_text`
  - `repeated_action_count`
  - `retry_cursor`
- 新增轻状态摘要：
  - 用户目标
  - 目标应用
  - 当前子目标
  - 页面猜测
  - 最近动作
  - 最近输入文本
  - 是否处于重瞄阶段
- `postprocess()`：
  - `COMPLETE` 在前 3 步直接拦截并转为 fallback。
  - `SCROLL` 在早期阶段若未明显卡住，则转为保守点击。
  - `CLICK` 若与上一步相同，则自动施加偏移进行重瞄。
- `safe_fallback()`：
  - 早期优先 `OPEN`
  - 有上次点击点则对该点做 retry offsets
  - 有最近输入则尝试点击底部确认区域
  - 明显卡住后才执行 `SCROLL`

## 为什么这版更工程化

这版没有引入重依赖，也没有引入会与评测替换机制冲突的复杂架构，而是把成熟 GUI Agent 的几种常用工程思想压缩到了低风险实现里：

- **状态压缩**：降低 token 占用。
- **controller 硬约束**：把高风险决策从模型收回到代码层。
- **重复点击重瞄**：用小范围扰动替代纯重复盲点点击。
- **早期禁 COMPLETE**：降低“任务未完成就结束”的大失分风险。

## 校验结果
已完成：
1. `git apply --check` 通过。
2. `git apply` 通过。
3. `py_compile` 通过。
4. `Agent.act()` 最小 smoke test 通过。
5. 额外验证：模型若输出 `COMPLETE:[]`，在第 2 步会被代码层拦截，不会直接提前结束。

## 使用建议
先在你已经打通的主模型上继续测试：
- `doubao-1-5-vision-pro-32k-250115`

先看三个指标：
1. 过早 `COMPLETE` 是否下降。
2. 连续相同 `CLICK` 是否减少。
3. `SCROLL` 是否更谨慎。

若这三项改善，再考虑进一步做局部裁剪重瞄或弱页面解析。
