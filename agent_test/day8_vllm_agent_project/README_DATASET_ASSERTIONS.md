# Dataset Assertion Guidelines

## Supported assertion fields

### 1. `expected_contains_all`
回答中必须全部包含这些固定字符串。

适合：
- 城市名
- 工具名
- 静态任务标签

不适合：
- 当前时间
- 当前温度
- 风速
- 当前日期

### 2. `expected_contains_any`
回答中至少包含这些字符串中的一个。

### 3. `expected_regex_all`
回答中必须匹配所有这些正则表达式。

适合：
- `\d{1,2}:\d{2}` 形式的时间
- `星期[一二三四五六日天]`
- 日期格式

### 4. `expected_regex_any`
回答中至少匹配一个正则表达式。

### 5. `forbidden_contains_any`
回答中不允许出现这些字符串。

适合：
- fallback 模板文案
- 未触发工具调用的错误提示
- 不希望暴露给用户的内部提示

## Examples

### Bad
```json
{
  "expected_contains_all": ["北京", "上海", "11:30"]
}
```

### Good
```json
{
  "expected_contains_all": ["北京", "上海"],
  "expected_regex_any": ["\\b\\d{1,2}:\\d{2}\\b"]
}
```

### Good with forbidden assertions
```json
{
  "expected_contains_all": ["南京"],
  "forbidden_contains_any": ["未触发任何工具调用"]
}
```

## Guardrails

`run_day9_dataset_guardrails.sh` 应作为 suite 前置门禁运行：

```bash
bash scripts/run_day9_dataset_guardrails.sh
```

## Recommendation

动态字段优先使用：
- `expected_regex_any`
- `expected_regex_all`

避免把动态值直接写死进 `expected_contains_all` / `expected_contains_any`
