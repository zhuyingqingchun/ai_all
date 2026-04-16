# Day 8.2 执行清单

目标：
1. 重新运行 Day 8 主数据集
2. 固定运行 3 条回归样本
3. 输出统一验收结果，不再手工分别检查

## 一、准备

1. 确认 vLLM 服务已启动
2. 确认当前仓库已经合入 Day 8.1 planner 修复
3. 进入项目根目录 `day8_vllm_agent_project`

## 二、应用补丁

```bash
git apply /path/to/day8_2_execution_and_suite.patch
```

如果需要先试运行：

```bash
git apply --check /path/to/day8_2_execution_and_suite.patch
```

## 三、执行命令

### 1. 跑 Day 8.2 验证套件

```bash
bash scripts/run_multistep_82_suite.sh next80b_fp8
```

### 2. 指定自定义主数据集或回归数据集

```bash
PYTHONPATH=src python -m vllm_agent_eval.cli_multistep_suite \
  --model next80b_fp8 \
  --main-dataset datasets/day8_multistep_dataset.json \
  --regression-dataset datasets/day8_multistep_regression_suite.json
```

## 四、需要重点查看的输出

- `results/day8_2/day8_2_main_eval_results.json`
- `results/day8_2/day8_2_regression_eval_results.json`
- `results/day8_2/day8_2_suite_summary.json`

## 五、验收门槛

主数据集：
- pass_rate >= 0.875
- 不允许出现 `plan_error`

回归数据集：
- pass_rate == 1.0
- 不允许出现 `plan_error`

总体验收：
- `overall_passed == true`

## 六、失败后先看哪里

1. `results/day8_2/day8_2_regression_eval_traces.jsonl`
2. `results/day8_2/day8_2_main_eval_traces.jsonl`
3. 重点看：
   - `plan_raw`
   - `plan.route_source`
   - `failure_type`
   - `failure_detail`

## 七、这份 8.2 补丁新增的内容

- `src/vllm_agent_eval/cli_multistep_suite.py`
- `datasets/day8_multistep_regression_suite.json`
- `scripts/run_multistep_82_suite.sh`
- `DAY8_2_EXECUTION_CHECKLIST.md`
- `multistep_dataset.py` 中的回归集 loader / saver

## 八、完成标准

满足以下条件才算 Day 8.2 完成：
- 主集通过率提升并稳定
- 回归集 3/3 保持通过
- `fallback_single_direct_answer` 不再吞掉可执行 step
- `calculator` 在多步任务里不再丢步
