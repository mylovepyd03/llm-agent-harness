"""
4단계: 두 번째 도구 추가 - 건강보험심사평가원 질병정보서비스(공공데이터포털)로
정확한 병명/코드를 조회.

이제 도구가 2개(search_wikipedia, search_disease_code)라서, 모델이 질문 종류에
따라 어떤 도구를 쓸지(또는 둘 다 쓸지) 스스로 판단하는지 확인하는 단계.
"""
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

WIKI_HEADERS = {
    "User-Agent": "llm-agent-harness-learning-project/0.1 (personal study, contact: paranvit@gmail.com)"
}

DISEASE_API_URL = "http://apis.data.go.kr/B551182/diseaseInfoService1/getDissNameCodeList1"
DISEASE_API_KEY = os.environ.get("DISEASE_INFO_SERVICE_KEY")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "위키피디아에서 질병명이나 의학 용어의 일반적인 설명/개요를 가져온다. '~가 뭐야', '~에 대해 설명해줘' 같은 질문에 사용.",
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
    },
    {
        "type": "function",
        "function": {
            "name": "search_disease_code",
            "description": (
                "건강보험심사평가원 공식 데이터에서 정확한 질병명, 표준 질병 코드(KCD), "
                "영문 병명을 조회한다. '정확한 병명이 뭐야', '이 증상이랑 비슷한 병명 찾아줘', "
                "'질병 코드가 뭐야' 같은 질문에 사용."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "검색할 질병명 키워드 (예: '편두통', '위염')",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
]


def chat(messages, tools=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        # temperature 0 = 일관성, num_predict = 생성 토큰 상한(무한 생성 방지)
        "options": {"temperature": 0, "num_predict": 512},
    }
    if tools:
        payload["tools"] = tools
    # 타임아웃 없으면 모델이 이상 동작할 때 코드가 영원히 멈출 수 있음
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]


# ---------------------------------------------------------------------------
# 여기서부터는 "LLM이 뭘 했나"가 아니라 "하네스가 그 결과를 검증하는 부분"이다.
# 즉 chat()은 그대로 두고, chat()이 돌려준 결과가 믿을 만한지를 우리 코드가
# 따로 검사한다 - 이게 하네스가 LLM을 신뢰하지 않고 방어하는 지점.
# ---------------------------------------------------------------------------

def is_response_ok(message: dict) -> bool:
    """
    2) 하네스가 출력을 검사하는 부분.
    모델 응답이 "믿을 만한지"를 판단한다. 두 경우만 정상으로 인정:
      - 도구를 부르고 싶으면 진짜 tool_calls 필드로 요청했을 것
      - 도구가 필요 없으면 content에 멀쩡한 텍스트가 들어있을 것
    """
    if message.get("tool_calls"):
        return True  # 정상적인 구조화된 도구 호출

    content = message.get("content", "")
    if not content:
        return False  # 도구 호출도 없고 텍스트도 없으면 이상함

    # 텍스트로 도구 호출을 흉내낸 경우 (예: content가 JSON처럼 생김)
    fake_tool_call = '"name"' in content and (
        '"parameters"' in content or '"arguments"' in content
    )
    # 같은 조각(2~20글자)이 5번 이상 연속 반복되는 경우 (반복 루프)
    repetition = re.search(r"(.{2,20}?)\1{4,}", content) is not None

    return not (fake_tool_call or repetition)


def call_with_retry(messages, tools=None, max_retries: int = 1):
    """
    3) 실패를 감지하는 부분 + 4) 재시도하는 부분을 합친 함수.
    chat()을 부르고 -> is_response_ok()로 검사 -> 이상하면 다시 부르고
    -> 그래도 이상하면(재시도 횟수 소진) None을 돌려줘서 포기.
    """
    for attempt in range(max_retries + 1):
        message = chat(messages, tools=tools)
        if is_response_ok(message):
            return message
        print(f"[경고] 비정상 응답 감지 (시도 {attempt + 1}/{max_retries + 1}) - 재시도")
    return None  # 재시도까지 다 실패


def search_wikipedia(query: str) -> str:
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
        timeout=10,
    )
    results = search_resp.json()["query"]["search"]
    if not results:
        return f"'{query}'에 대한 위키피디아 검색 결과가 없습니다."
    title = results[0]["title"]

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
        timeout=10,
    )
    pages = extract_resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    extract = page.get("extract", "").strip()
    return f"[위키피디아 - {title}]\n{extract[:1000]}"


def search_disease_code(keyword: str) -> str:
    """건강보험심사평가원 질병정보서비스로 정확한 병명/코드를 조회."""
    if not DISEASE_API_KEY:
        return "질병정보서비스 API 키가 설정되지 않았습니다 (.env의 DISEASE_INFO_SERVICE_KEY 확인)."

    response = requests.get(
        DISEASE_API_URL,
        params={
            "ServiceKey": DISEASE_API_KEY,
            "pageNo": 1,
            "numOfRows": 5,
            "sickType": 1,       # 3단상병 코드 기준
            "medTp": 1,          # 의과(양방)
            "diseaseType": "SICK_NM",
            "searchText": keyword,
        },
        timeout=10,
    )
    response.raise_for_status()

    # XML 응답 파싱 (표준 라이브러리 xml.etree 사용 - 별도 설치 불필요)
    import xml.etree.ElementTree as ET

    root = ET.fromstring(response.text)
    items = root.findall(".//item")
    if not items:
        return f"'{keyword}'와(과) 일치하는 공식 질병 정보를 찾지 못했습니다."

    lines = ["[건강보험심사평가원 - 질병정보서비스]"]
    for item in items:
        sick_nm = item.findtext("sickNm", "")
        sick_cd = item.findtext("sickCd", "")
        sick_eng_nm = item.findtext("sickEngNm", "")
        lines.append(f"- {sick_nm} (코드: {sick_cd}, 영문명: {sick_eng_nm})")
    return "\n".join(lines)


AVAILABLE_TOOLS = {
    "search_wikipedia": search_wikipedia,
    "search_disease_code": search_disease_code,
}


def run_agent(user_question: str) -> str:
    messages = [{"role": "user", "content": user_question}]

    # 1차 호출: 도구가 필요한지 LLM이 판단 (실패하면 1회 재시도까지 자동으로 함)
    message = call_with_retry(messages, tools=TOOLS)
    if message is None:
        return "[오류] 모델이 계속 비정상적인 응답을 내서 포기했습니다. 질문을 바꿔서 다시 시도해보세요."
    messages.append(message)

    if not message.get("tool_calls"):
        return message["content"]

    for call in message["tool_calls"]:
        name = call["function"]["name"]
        args = call["function"]["arguments"]
        print(f"[도구 호출] {name}({args})")

        func = AVAILABLE_TOOLS.get(name)
        result = func(**args) if func else f"알 수 없는 도구: {name}"
        print(f"[도구 결과]\n{result}\n")

        messages.append({"role": "tool", "content": result})

    # 2차 호출: 도구 결과를 반영한 최종 답변 (여기도 동일하게 검증+재시도 적용)
    final_message = call_with_retry(messages)
    if final_message is None:
        return "[오류] 최종 답변 생성 중 모델이 계속 비정상적인 응답을 내서 포기했습니다."
    return final_message["content"]


if __name__ == "__main__":
    for question in [
        "편두통이 뭔지 설명해줘.",
        "편두통의 정확한 질병 코드가 뭐야?",
    ]:
        print(f"\n########## 질문: {question} ##########")
        answer = run_agent(question)
        print("=== 최종 답변 ===")
        print(answer)
