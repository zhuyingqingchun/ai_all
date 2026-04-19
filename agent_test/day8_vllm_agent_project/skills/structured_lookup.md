# structured_lookup

## 作用
查询本地结构化配置表中的确定性条目。

## 何时使用
- 用户要查询固定配置值、模型 profile、伺服器参数
- 用户给出了明确 key 或可映射 key

## 何时不要使用
- 用户在问开放式知识
- 用户问题更适合天气、时间、计算或单位换算

## 参数格式
```json
{
  "key": "servo_max_torque"
}
```

## 成功返回示例
```json
{
  "ok": true,
  "data": {
    "key": "servo_max_torque",
    "value": "2.5 N·m",
    "category": "servo_spec"
  },
  "error": null
}
```

## 失败返回示例
```json
{
  "ok": false,
  "data": null,
  "error": "未找到配置项：unknown_config"
}
```

## 常见组合
- `structured_lookup -> calculator`
- `structured_lookup -> unit_convert`

## 标准例句
1. 先查一下 qwen2.5-7b_context_length 的配置值，再算 sqrt(144)。
2. 先查一下 servo_max_torque 的配置值，再把 2.5 kg 转成 g。
3. 先查 unknown_config 的配置值，再说明哪一步失败了。
