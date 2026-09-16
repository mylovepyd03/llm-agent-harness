"""
7단계: 대화형 인터페이스 - "내 손안의 의사" 에이전트를 실제로 써볼 수 있게.

지금까지(01~06)는 코드에 미리 박아둔 질문으로만 테스트했다.
여기서는 터미널에서 사용자가 직접 질문을 입력하고, 에이전트가 답하는 반복 루프(REPL)를 추가한다.
run_agent()를 비롯한 하네스 로직 자체는 06단계와 동일 - 바뀐 건 "누가 질문을 넣어주느냐"뿐이다.
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

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_CONTACT = {"tool": "llm-agent-harness-learning-project", "email": "paranvit@gmail.com"}


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
    {
        "type": "function",
        "function": {
            "name": "search_pubmed",
            "description": (
                "PubMed(미국 국립의학도서관)에서 특정 질환/의학 주제에 관한 최신 연구 논문을 "
                "검색한다. '관련 논문 찾아줘', '연구 결과가 뭐야', '최신 연구는' 같은 질문에 사용."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "검색할 질병명 키워드, 한글도 가능 (예: '편두통', 'migraine')",
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


# 도구 인자로 허용할 문자: 한글/영문/숫자/공백 + 의학 용어에 흔한 기호(하이픈, 괄호, 쉼표, 점,
# 슬래시, %, +, 가운뎃점). 한자, 낱개 자음/모음, 그 외 이상한 기호가 섞이면 깨진 인자로 본다.
ALLOWED_QUERY_CHARS = re.compile(r"^[가-힣a-zA-Z0-9\s\-.,()/%+·]*$")


def is_valid_tool_query(text: str) -> bool:
    """도구에 넘기려는 문자열 인자가 정상적인지 검사. 비어있거나 허용 문자 밖의
    글자(한자, 낱개 자모 등)가 섞여 있으면 깨진 것으로 판단한다."""
    text = text.strip()
    if not text:
        return False
    return ALLOWED_QUERY_CHARS.match(text) is not None


def is_response_ok(message: dict) -> bool:
    tool_calls = message.get("tool_calls")
    if tool_calls:
        for call in tool_calls:
            args = call.get("function", {}).get("arguments", {})
            for value in args.values():
                if isinstance(value, str) and not is_valid_tool_query(value):
                    return False  # 도구 호출 형식은 정상이지만 인자 내용이 깨짐
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


def _get_disease_items(keyword: str):
    if not DISEASE_API_KEY:
        return []

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
    return [
        {
            "sickNm": item.findtext("sickNm", ""),
            "sickCd": item.findtext("sickCd", ""),
            "sickEngNm": item.findtext("sickEngNm", ""),
        }
        for item in root.findall(".//item")
    ]


def search_disease_code(keyword: str) -> str:
    items = _get_disease_items(keyword)
    if not DISEASE_API_KEY:
        return "질병정보서비스 API 키가 설정되지 않았습니다 (.env의 DISEASE_INFO_SERVICE_KEY 확인)."
    if not items:
        return f"'{keyword}'와(과) 일치하는 공식 질병 정보를 찾지 못했습니다."

    lines = ["[건강보험심사평가원 - 질병정보서비스]"]
    for item in items:
        lines.append(f"- {item['sickNm']} (코드: {item['sickCd']}, 영문명: {item['sickEngNm']})")
    return "\n".join(lines)


def _find_cntnts_sn(keyword: str):
    if keyword in KDCA_DISEASE_INDEX:
        return keyword, KDCA_DISEASE_INDEX[keyword]

    candidates = [name for name in KDCA_DISEASE_INDEX if keyword in name]
    if not candidates:
        return None, None

    best = min(candidates, key=len)
    return best, KDCA_DISEASE_INDEX[best]


def search_health_info(keyword: str) -> str:
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
            continue
        entry = f"## {name}\n{content}"
        lines.append(entry)
        total_len += len(entry)
        if total_len > 1500:
            break

    return "\n\n".join(lines)


def _to_english_query(keyword: str) -> str:
    has_korean = re.search(r"[가-힣]", keyword) is not None
    if not has_korean:
        return keyword

    items = _get_disease_items(keyword)
    if items and items[0]["sickEngNm"]:
        return items[0]["sickEngNm"]
    return keyword


def search_pubmed(keyword: str) -> str:
    english_query = _to_english_query(keyword)

    search_resp = requests.get(
        PUBMED_ESEARCH_URL,
        params={
            **PUBMED_CONTACT,
            "db": "pubmed",
            "term": english_query,
            "retmode": "json",
            "retmax": 5,
            "sort": "relevance",
        },
        timeout=10,
    )
    search_resp.raise_for_status()
    id_list = search_resp.json().get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return f"'{keyword}'({english_query})에 대한 PubMed 논문을 찾지 못했습니다."

    summary_resp = requests.get(
        PUBMED_ESUMMARY_URL,
        params={**PUBMED_CONTACT, "db": "pubmed", "id": ",".join(id_list), "retmode": "json"},
        timeout=10,
    )
    summary_resp.raise_for_status()
    result = summary_resp.json().get("result", {})

    lines = [f"[PubMed 검색어: {english_query}]"]
    for pmid in id_list:
        doc = result.get(pmid)
        if not doc:
            continue
        title = doc.get("title", "제목 없음")
        journal = doc.get("fulljournalname", "")
        pubdate = doc.get("pubdate", "")
        lines.append(
            f"- {title} ({journal}, {pubdate})\n  https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        )

    return "\n".join(lines)


AVAILABLE_TOOLS = {
    "search_wikipedia": search_wikipedia,
    "search_disease_code": search_disease_code,
    "search_health_info": search_health_info,
    "search_pubmed": search_pubmed,
}

# 우리 도구 함수들이 "검색 결과 없음"일 때 돌려주는 문구들 - 이 중 하나라도 들어있으면
# 그 도구 호출은 실패(=근거 없음)로 취급한다.
FAILURE_MARKERS = [
    "찾지 못했습니다",
    "결과가 없습니다",
    "데이터가 없습니다",
    "내용이 없습니다",
    "설정되지 않았습니다",
]


def is_tool_result_failure(result: str) -> bool:
    """도구 실행 결과가 '진짜 정보'인지 '실패 메시지'인지 판별."""
    return any(marker in result for marker in FAILURE_MARKERS)


def run_agent(user_question: str) -> str:
    messages = [{"role": "user", "content": user_question}]

    message = call_with_retry(messages, tools=TOOLS)
    if message is None:
        return "[오류] 모델이 계속 비정상적인 응답을 내서 포기했습니다. 질문을 바꿔서 다시 시도해보세요."
    messages.append(message)

    if not message.get("tool_calls"):
        return message["content"]

    any_success = False
    for call in message["tool_calls"]:
        name = call["function"]["name"]
        args = call["function"]["arguments"]
        print(f"  [도구 호출] {name}({args})")

        func = AVAILABLE_TOOLS.get(name)
        result = func(**args) if func else f"알 수 없는 도구: {name}"

        if not is_tool_result_failure(result):
            any_success = True

        messages.append({"role": "tool", "content": result})

    if not any_success:
        # 근거가 하나도 없는데 LLM한테 "그래도 답해봐"라고 넘기면 자기 지식으로
        # 지어낼 위험이 있으므로, 아예 LLM을 다시 부르지 않고 여기서 끝낸다.
        return "검색 결과를 얻지 못해 답변할 수 없습니다. 다른 검색어로 다시 시도해보세요."

    final_message = call_with_retry(messages)
    if final_message is None:
        return "[오류] 최종 답변 생성 중 모델이 계속 비정상적인 응답을 내서 포기했습니다."
    return final_message["content"]


def main():
    print("=" * 50)
    print("내 손안의 의사 (학습용 에이전트)")
    print("증상/병명을 물어보세요. 종료하려면 'q' 또는 '종료' 입력.")
    print("(로컬 3B 모델이라 질문마다 수십 초~1분 정도 걸릴 수 있어요)")
    print("=" * 50)

    while True:
        try:
            question = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not question:
            continue
        if question in ("q", "quit", "종료", "exit"):
            print("종료합니다.")
            break

        answer = run_agent(question)
        print(f"\n답변> {answer}")


if __name__ == "__main__":
    main()
