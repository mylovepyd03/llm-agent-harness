"""
2단계: 도구를 하나 정의하고, LLM이 "이 도구를 이 인자로 불러줘"라는
요청(tool_calls)을 만들어내는지 확인.
아직 실제 위키피디아 호출은 안 함 - 모델이 도구 호출 형식을 뱉는지만 확인.
"""
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

# 도구 스펙: "이런 함수가 있다"를 모델에게 설명해주는 명세서
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "위키피디아에서 질병명이나 의학 용어를 검색해서 요약 설명을 가져온다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 질병명 또는 의학 용어 (예: '당뇨병', '편두통')",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def chat(messages, tools=None):
    payload = {"model": MODEL, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["message"]


if __name__ == "__main__":
    # 도구가 필요할 법한 질문
    messages = [{"role": "user", "content": "편두통이 뭔지 설명해줘."}]
    message = chat(messages, tools=TOOLS)

    print("--- 모델이 돌려준 message ---")
    print(message)

    if message.get("tool_calls"):
        print("\n--- 모델이 요청한 도구 호출 ---")
        for call in message["tool_calls"]:
            print(call["function"]["name"], call["function"]["arguments"])
    else:
        print("\n(도구 호출 없이 바로 답변함)")
