"""
5단계: 세 번째 도구 추가 - 질병관리청 국가건강정보포털로 증상/원인/치료 정보 조회.

이제 도구가 3개(위키피디아 개요 / 공식 병명·코드 / 증상·원인·치료 상세)라서,
질문 종류에 따라 모델이 셋 중 알맞은 걸 골라 쓰는지 확인하는 단계.
"""
import json
import os
import re
import ssl
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

WIKI_HEADERS = {
    "User-Agent": "llm-agent-harness-learning-project/0.1 (personal study, contact: paranvit@gmail.com)"
}

DISEASE_API_URL = "http://apis.data.go.kr/B551182/diseaseInfoService1/getDissNameCodeList1"
DISEASE_API_KEY = os.environ.get("DISEASE_INFO_SERVICE_KEY")

KDCA_HEALTH_INFO_URL = "https://api.kdca.go.kr/api/provide/healthInfo"
KDCA_TOKEN = os.environ.get("KDCA_HEALTH_INFO_TOKEN")
KDCA_INDEX_PATH = os.path.join(os.path.dirname(__file__), "data", "kdca_disease_index.json")


class _LegacySSLAdapter(HTTPAdapter):
    """api.kdca.go.kr가 오래된 TLS 재협상 방식을 써서, 기본 SSL 설정으로는 접속이 막힌다.
    이 어댑터가 그 legacy 옵션을 허용하도록 SSL 컨텍스트를 살짝 풀어준다."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.options |= 0x4  # SSL_OP_LEGACY_SERVER_CONNECT
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


_kdca_session = requests.Session()
_kdca_session.mount("https://", _LegacySSLAdapter())

with open(KDCA_INDEX_PATH, encoding="utf-8") as f:
    KDCA_DISEASE_INDEX: dict = json.load(f)


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
                "영문 병명을 조회한다. '정확한 병명이 뭐야', '질병 코드가 뭐야' 같은 질문에 사용."
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
    {
        "type": "function",
        "function": {
            "name": "search_health_info",
            "description": (
                "질병관리청 국가건강정보포털에서 특정 질환의 증상, 원인, 치료법 등 "
                "상세한 의료 정보를 조회한다. '증상이 뭐야', '원인이 뭐야', '어떻게 치료해', "
                "'관리 방법 알려줘' 같은 질문에 사용."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "검색할 질병명 키워드 (예: '감기', '편두통')",
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
        "options": {"temperature": 0, "num_predict": 512},
    }
    if tools:
        payload["tools"] = tools
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]


def is_response_ok(message: dict) -> bool:
    if message.get("tool_calls"):
        return True

    content = message.get("content", "")
    if not content:
        return False

    fake_tool_call = '"name"' in content and (
        '"parameters"' in content or '"arguments"' in content
    )
    repetition = re.search(r"(.{2,20}?)\1{4,}", content) is not None

    return not (fake_tool_call or repetition)


def call_with_retry(messages, tools=None, max_retries: int = 1):
    for attempt in range(max_retries + 1):
        message = chat(messages, tools=tools)
        if is_response_ok(message):
            return message
        print(f"[경고] 비정상 응답 감지 (시도 {attempt + 1}/{max_retries + 1}) - 재시도")
    return None


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
    if not DISEASE_API_KEY:
        return "질병정보서비스 API 키가 설정되지 않았습니다 (.env의 DISEASE_INFO_SERVICE_KEY 확인)."

    response = requests.get(
        DISEASE_API_URL,
        params={
            "ServiceKey": DISEASE_API_KEY,
            "pageNo": 1,
            "numOfRows": 5,
            "sickType": 1,
            "medTp": 1,
            "diseaseType": "SICK_NM",
            "searchText": keyword,
        },
        timeout=10,
    )
    response.raise_for_status()

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


def _find_cntnts_sn(keyword: str):
    """병명 키워드로 로컬 인덱스에서 cntntsSn 번호를 찾는다.
    정확히 일치하는 이름이 없으면, 키워드를 포함하는 이름 중 가장 짧은 것을 고른다."""
    if keyword in KDCA_DISEASE_INDEX:
        return keyword, KDCA_DISEASE_INDEX[keyword]

    candidates = [name for name in KDCA_DISEASE_INDEX if keyword in name]
    if not candidates:
        return None, None

    best = min(candidates, key=len)
    return best, KDCA_DISEASE_INDEX[best]


def search_health_info(keyword: str) -> str:
    """질병관리청 국가건강정보포털에서 증상/원인/치료 정보를 조회."""
    if not KDCA_TOKEN:
        return "국가건강정보포털 API 토큰이 설정되지 않았습니다 (.env의 KDCA_HEALTH_INFO_TOKEN 확인)."

    matched_name, sn = _find_cntnts_sn(keyword)
    if sn is None:
        return f"'{keyword}'에 대한 국가건강정보포털 데이터가 없습니다 (인덱스에 없는 질환)."

    response = _kdca_session.get(
        KDCA_HEALTH_INFO_URL,
        params={"TOKEN": KDCA_TOKEN, "cntntsSn": sn},
        timeout=10,
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)
    sections = root.findall(".//cntntsCl")
    if not sections:
        return f"'{matched_name}'에 대한 상세 내용이 없습니다."

    lines = [f"[국가건강정보포털 - {matched_name}]"]
    total_len = 0
    for sec in sections:
        name = sec.findtext("CNTNTS_CL_NM", "")
        content = (sec.findtext("CNTNTS_CL_CN", "") or "").strip()
        if not content or content.startswith("http"):
            continue  # 첨부파일 다운로드 링크 등은 건너뜀
        entry = f"## {name}\n{content}"
        lines.append(entry)
        total_len += len(entry)
        if total_len > 1500:  # 너무 길어지지 않게 상한
            break

    return "\n\n".join(lines)


AVAILABLE_TOOLS = {
    "search_wikipedia": search_wikipedia,
    "search_disease_code": search_disease_code,
    "search_health_info": search_health_info,
}


def run_agent(user_question: str) -> str:
    messages = [{"role": "user", "content": user_question}]

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

    final_message = call_with_retry(messages)
    if final_message is None:
        return "[오류] 최종 답변 생성 중 모델이 계속 비정상적인 응답을 내서 포기했습니다."
    return final_message["content"]


if __name__ == "__main__":
    for question in [
        "감기 증상이랑 치료법 알려줘.",
        "편두통의 정확한 질병 코드가 뭐야?",
    ]:
        print(f"\n########## 질문: {question} ##########")
        answer = run_agent(question)
        print("=== 최종 답변 ===")
        print(answer)
