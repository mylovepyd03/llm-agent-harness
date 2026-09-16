"""
3단계: 실제 도구 실행 + 결과 반영 + 최종 답변까지 - ReAct 루프 한 바퀴 완성.

흐름: 질문 -> 모델이 도구 호출 요청 -> 진짜 위키피디아 호출 ->
      결과를 대화에 추가 -> 모델 재호출 -> 최종 답변
"""
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

# 위키피디아는 User-Agent 없는 요청을 403으로 차단함 - 누가 호출하는지 식별 가능하게 표시
WIKI_HEADERS = {
    "User-Agent": "llm-agent-harness-learning-project/0.1 (personal study, contact: paranvit@gmail.com)"
}

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
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        # temperature 0 = 매번 가장 확률 높은 답만 고르게 해서 일관성을 높임
        "options": {"temperature": 0},
    }
    if tools:
        payload["tools"] = tools
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["message"]


def search_wikipedia(query: str) -> str:
    """진짜 위키피디아 API를 호출해서 요약 텍스트를 가져오는 함수."""
    # 1) 검색어와 가장 가까운 문서 제목 찾기
    search_resp = requests.get(
        "https://ko.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
        },
        headers=WIKI_HEADERS,
    )
    results = search_resp.json()["query"]["search"]
    if not results:
        return f"'{query}'에 대한 위키피디아 검색 결과가 없습니다."
    title = results[0]["title"]

    # 2) 그 문서의 도입부 요약(extract) 가져오기
    extract_resp = requests.get(
        "https://ko.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": title,
            "format": "json",
        },
        headers=WIKI_HEADERS,
    )
    pages = extract_resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    extract = page.get("extract", "").strip()
    return f"[위키피디아 - {title}]\n{extract[:1000]}"


AVAILABLE_TOOLS = {
    "search_wikipedia": search_wikipedia,
}


def run_agent(user_question: str) -> str:
    messages = [{"role": "user", "content": user_question}]

    # 1차 호출: 모델이 도구가 필요한지 판단
    message = chat(messages, tools=TOOLS)
    messages.append(message)

    if not message.get("tool_calls"):
        # 도구 없이 바로 답할 수 있는 질문이었던 경우
        return message["content"]

    # 2) 모델이 요청한 도구를 실제로 실행
    for call in message["tool_calls"]:
        name = call["function"]["name"]
        args = call["function"]["arguments"]
        print(f"[도구 호출] {name}({args})")

        func = AVAILABLE_TOOLS.get(name)
        result = func(**args) if func else f"알 수 없는 도구: {name}"
        print(f"[도구 결과]\n{result}\n")

        # 3) 실행 결과를 대화 기록에 추가
        messages.append({"role": "tool", "content": result})

    # 4) 결과를 반영해서 최종 답변 생성
    final_message = chat(messages)
    return final_message["content"]


if __name__ == "__main__":
    answer = run_agent("편두통이 뭔지 설명해줘.")
    print("=== 최종 답변 ===")
    print(answer)
