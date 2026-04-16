from __future__ import annotations

from typing import Tuple

from .config import KEEP_RECENT_MESSAGES, MAX_RECENT_MESSAGES, SUMMARY_MAX_TOKENS
from .prompts import SUMMARY_SYSTEM_PROMPT
from .schemas import ConversationMemory


def maybe_summarize(memory: ConversationMemory, chat_client) -> Tuple[bool, float]:
    if len(memory.recent_messages) <= MAX_RECENT_MESSAGES:
        return False, 0.0
    old_chunk = memory.recent_messages[:-KEEP_RECENT_MESSAGES]
    memory.recent_messages = memory.recent_messages[-KEEP_RECENT_MESSAGES:]
    transcript = "\n".join([("用户" if m["role"] == "user" else "助手") + f"：{m['content']}" for m in old_chunk])
    existing_summary = memory.summary.strip() if memory.summary.strip() else "无"
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": f"已有摘要：\n{existing_summary}\n\n需要压缩的新对话：\n{transcript}\n\n请输出更新后的摘要。"},
    ]
    try:
        new_summary, latency_ms = chat_client.chat(messages, max_tokens=SUMMARY_MAX_TOKENS)
        if new_summary.strip():
            memory.summary = new_summary.strip()
        return True, latency_ms
    except Exception:
        fallback = []
        if memory.summary.strip():
            fallback.append(memory.summary.strip())
        fallback.append("更早对话涉及：" + "；".join([m["content"][:40] for m in old_chunk if m["role"] == "user"][:6]))
        memory.summary = " ".join(fallback).strip()
        return True, 0.0
