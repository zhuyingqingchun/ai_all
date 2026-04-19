# text_search

## 作用
在本地 note 语料中做关键字检索，返回按命中得分排序的结果。

## 何时使用
- 用户要在本地笔记、记录、说明中搜索某个主题
- 用户给的是关键词，而不是精确 note id

## 何时不要使用
- 用户已经给出了精确 note id，更适合 `note_lookup`
- 用户想查询结构化配置值，更适合 `structured_lookup`

## 参数格式
```json
{
  "query": "热降额",
  "top_k": 3
}
```

## 成功返回示例
```json
{
  "ok": true,
  "data": {
    "query": "热降额",
    "hits": [
      {"id": "servo_thermal_limit_note", "score": 2}
    ]
  },
  "error": null
}
```

## 常见组合
- `text_search -> calculator`
- `text_search -> note_lookup`
- `text_search -> structured_lookup`

## 标准例句
1. 先搜索 热降额，再算 95-20，并总结结论。
2. 先搜索 offboard failsafe，再查一下 px4_offboard_requirements_note 这条笔记。
3. 先搜索 上下文长度，再查一下 next80b_fp8_context_length 的配置值。
