# note_lookup

## 作用
按 note id 精确读取单条笔记。

## 何时使用
- 用户已经给出明确的 note id
- 用户需要查看某条固定笔记的标题、内容、标签

## 何时不要使用
- 用户只有主题关键词，没有明确 note id
- 用户实际上是要查配置值，不是查笔记

## 参数格式
```json
{
  "note_id": "servo_thermal_limit_note"
}
```

## 成功返回示例
```json
{
  "ok": true,
  "data": {
    "id": "servo_thermal_limit_note",
    "title": "舵机热降额说明",
    "content": "舵机在 95 °C 开始热降额，110 °C 触发停机保护。",
    "tags": ["热降额", "温度", "舵机"],
    "source": "mock_note_lookup"
  },
  "error": null
}
```

## 常见组合
- `note_lookup -> unit_convert`
- `note_lookup -> calculator`
- `text_search -> note_lookup`

## 标准例句
1. 先查一下 servo_thermal_limit_note 这条笔记，再把 95 °C 转成 °F。
2. 先搜索 offboard failsafe，再查一下 px4_offboard_requirements_note 这条笔记。
3. 查一下 battery_warmup_note 这条笔记。
