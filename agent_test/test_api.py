from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy"
)

resp = client.chat.completions.create(
    model="next80b_fp8",
    messages=[
        {"role": "system", "content": "你是一个简洁的中文助手。"},
        {"role": "user", "content": "请用一句话解释什么是 agent。"}
    ],
    temperature=0,
    max_tokens=128,
)

print(resp.choices[0].message.content)
