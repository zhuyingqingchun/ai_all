# unit_convert

## 作用
把确定性数值从一个单位换算成另一个单位。

## 何时使用
- 用户明确要求单位换算
- 后续还要基于换算结果继续计算

## 何时不要使用
- 用户只是在问天气、时间或配置值
- 用户没有提供可解析的数值与单位

## 参数格式
```json
{
  "value": 72,
  "from_unit": "km/h",
  "to_unit": "m/s"
}
```

## 当前支持单位
- 速度：`km/h`, `m/s`
- 质量：`kg`, `g`
- 长度：`cm`, `m`
- 温度：`°C`, `°F`

## 成功返回示例
```json
{
  "ok": true,
  "data": {
    "value": 72.0,
    "from_unit": "km/h",
    "to_unit": "m/s",
    "result": 20.0
  },
  "error": null
}
```

## 失败返回示例
```json
{
  "ok": false,
  "data": null,
  "error": "单位换算失败：暂不支持单位：mph"
}
```

## 常见组合
- `unit_convert -> calculator`
- `structured_lookup -> unit_convert`

## 标准例句
1. 先把 72 km/h 转成 m/s，再算 20 秒能走多远。
2. 把 2.5 kg 换算成 g。
3. 先查配置值，再把 500 cm 转成 m。
