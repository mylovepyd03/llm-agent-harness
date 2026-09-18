"""
B단계 마무리: A단계 도구 호출 에이전트 + B단계 RAG를 하나로 통합.

기존 07_interactive_agent.py의 search_health_info(정확한 병명만 가능)와
search_pubmed(제목만 보고 짐작)를 RAG 버전으로 교체한다:
  - search_symptom_info: 증상 문장을 임베딩 검색으로 관련 질환에 매칭 (11단계 RAG)
  - search_pubmed_deep: PubMed 초록을 번역/종합해서 근거 기반으로 답변 (13단계 RAG)

중요한 설계 포인트: RAG 도구는 내부에서 이미 "검증된 근거로 다듬어진 답변"을
완성해서 돌려준다. 그런데 이걸 바깥쪽 에이전트(1차 LLM 호출)에게 다시 넘겨서
"이 결과 보고 답변 써줘"라고 하면, 바깥쪽 LLM이 또 한 번 재해석하면서
할루시네이션이 재발할 위험이 있다 (실제로 13단계에서 이 문제를 겪었음).
그래서 RAG 도구가 호출되면, 그 결과를 바깥쪽 LLM에게 넘기지 않고
**그대로 최종 답변으로 반환**한다 (DIRECT_ANSWER_TOOLS).
"""
import json
import math
import os
import re
import ssl
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter

load_dotenv()

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

AGENT_MODEL = "llama3.2"       # 메인 에이전트(도구 선택, 일반 도구 결과 요약)
RAG_MODEL = "llama3.1"         # RAG 내부 번역/종합 전용 (llama3.2는 이 작업에서 문자 오염 문제 있었음)
EMBED_MODEL = "bge-m3"

WIKI_HEADERS = {
    "User-Agent": "llm-agent-harness-learning-project/0.1 (personal study, contact: paranvit@gmail.com)"
}
DISEASE_API_URL = "http://apis.data.go.kr/B551182/diseaseInfoService1/getDissNameCodeList1"
DISEASE_API_KEY = os.environ.get("DISEASE_INFO_SERVICE_KEY")

KDCA_EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "data", "kdca_embeddings_clean.json")
MIN_SIMILARITY_SYMPTOM = 0.45

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_CONTACT = {"tool": "llm-agent-harness-learning-project", "email": "paranvit@gmail.com"}
MIN_SIMILARITY_PUBMED = 0.4


class _LegacySSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.options |= 0x4
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


with open(KDCA_EMBEDDINGS_PATH, encoding="utf-8") as f:
    KDCA_CORPUS: dict = json.load(f)


# ---------------------------------------------------------------------------
# 공통: LLM 호출 + 두 겹의 검증(하네스)
# ---------------------------------------------------------------------------

def chat(messages, tools=None, model: str = AGENT_MODEL, temperature: float = 0) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 512},
    }
    if tools:
        payload["tools"] = tools
    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["message"]


# 도구 인자로 허용할 문자 (07단계와 동일)
ALLOWED_QUERY_CHARS = re.compile(r"^[가-힣a-zA-Z0-9\s\-.,()/%+·]*$")


def is_valid_tool_query(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    return ALLOWED_QUERY_CHARS.match(text) is not None


def has_short_chunk_repetition(content: str) -> bool:
    return re.search(r"(.{2,20}?)\1{4,}", content) is not None


def has_sentence_repetition(content: str, min_len: int = 15, min_repeats: int = 3) -> bool:
    sentences = re.split(r"(?<=[.!?])\s+", content)
    counts: dict[str, int] = {}
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < min_len:
            continue
        counts[sentence] = counts.get(sentence, 0) + 1
    return any(count >= min_repeats for count in counts.values())


_ALLOWED_SCRIPT_RANGES = (
    (0xAC00, 0xD7A3), (0x3131, 0x318E), (0x0000, 0x024F), (0x2000, 0x206F),
)


def has_unexpected_script(content: str) -> bool:
    for ch in content:
        if ch.isspace():
            continue
        if any(lo <= ord(ch) <= hi for lo, hi in _ALLOWED_SCRIPT_RANGES):
            continue
        return True
    return False


# 가짜 도구 호출(텍스트로 JSON 흉내) 감지 - 모델이 만드는 JSON이 항상 깔끔하진
# 않아서(예: "parameters\":  처럼 필드명 뒤에 엉뚱한 백슬래시가 낌), 정확한
# 문자열 일치 대신 "백슬래시가 몇 개 껴도 통과하는" 정규식으로 느슨하게 잡는다.
_FAKE_TOOL_NAME_RE = re.compile(r'"name"\\*\s*:')
_FAKE_TOOL_PARAMS_RE = re.compile(r'"(parameters|arguments)\\*"')


def has_fake_tool_call_text(content: str) -> bool:
    stripped = content.strip()
    name_like = _FAKE_TOOL_NAME_RE.search(content) is not None
    if not name_like:
        return False
    # 중괄호로 시작하면서 "name" 필드가 보이면, parameters/arguments 여부와
    # 상관없이 이미 자연어 답변이 아니라 JSON을 흉내내려던 것으로 본다.
    if stripped.startswith("{"):
        return True
    return _FAKE_TOOL_PARAMS_RE.search(content) is not None


# 의미 검증: "문법은 맞는데 뜻이 없는 한글"(예: "국구괭가하")은 문자 화이트리스트로는
# 못 잡는다. KDCA 611개 질환 임베딩과 비교해서 "의료 도메인과 조금이라도 관련 있는
# 내용인가"를 추가로 검사한다. 임계값 0.5는 실측으로 정함:
#   실제 병명/증상 문장 유사도: 0.569~0.731
#   헛소리/무관한 문장 유사도: 0.296~0.483
# 그 사이(약 0.48~0.57)에 뚜렷한 간격이 있어서 0.5로 잡으면 둘을 안전하게 가른다.
def is_query_semantically_valid(query: str, threshold: float = 0.5) -> bool:
    query_vec = embed(query)
    best_score = max(cosine_similarity(query_vec, e["embedding"]) for e in KDCA_CORPUS.values())
    return best_score >= threshold


def is_outer_response_ok(message: dict) -> bool:
    """1차(에이전트) 응답 검증: 도구 호출이면 인자까지(문법+의미), 아니면 반복/가짜호출/문자오염 체크."""
    tool_calls = message.get("tool_calls")
    if tool_calls:
        for call in tool_calls:
            args = call.get("function", {}).get("arguments", {})
            for value in args.values():
                if not isinstance(value, str):
                    continue
                if not is_valid_tool_query(value):
                    return False
                if not is_query_semantically_valid(value):
                    return False
        return True

    content = message.get("content", "")
    if not content:
        return False
    return not (
        has_fake_tool_call_text(content)
        or has_short_chunk_repetition(content)
        or has_sentence_repetition(content)
        or has_unexpected_script(content)
    )


def is_rag_generation_ok(content: str) -> bool:
    """RAG 내부 생성(번역/종합) 검증: 반복 + 낯선 문자 섞임 체크."""
    if not content:
        return False
    if has_short_chunk_repetition(content):
        return False
    if has_sentence_repetition(content):
        return False
    if has_unexpected_script(content):
        return False
    return True


def call_with_retry(messages, tools=None, model: str = AGENT_MODEL, max_retries: int = 1, inner: bool = False):
    """1차 시도는 temperature=0(일관성), 실패하면 재시도부터는 temperature를
    올려서(0.6) 다른 출력이 나올 여지를 준다. temperature=0은 같은 입력에 거의
    항상 같은(고장난) 출력을 내서, 아무것도 안 바꾸고 재시도해봐야 소용없다는
    걸 실제로 확인했음 - 단, 온도를 올리면 새로운 형태로 깨질 수도 있어서
    재시도 결과도 반드시 같은 검증기를 다시 통과해야만 받아들인다."""
    checker = is_rag_generation_ok if inner else is_outer_response_ok
    for attempt in range(max_retries + 1):
        temperature = 0 if attempt == 0 else 0.6
        message = chat(messages, tools=tools, model=model, temperature=temperature)
        ok = checker(message.get("content", "")) if inner else checker(message)
        if ok:
            return message
        print(f"    [경고] 비정상 응답 감지 (시도 {attempt + 1}/{max_retries + 1}, temp={temperature}) - 재시도")
    return None


# ---------------------------------------------------------------------------
# 도구 1, 2: A단계 그대로 (위키피디아, 공식 병명/코드)
# ---------------------------------------------------------------------------

def search_wikipedia(query: str) -> str:
    query = clean_disease_query(query)
    search_resp = requests.get(
        "https://ko.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1},
        headers=WIKI_HEADERS, timeout=10,
    )
    results = search_resp.json()["query"]["search"]
    if not results:
        return f"'{query}'에 대한 위키피디아 검색 결과가 없습니다."
    title = results[0]["title"]

    extract_resp = requests.get(
        "https://ko.wikipedia.org/w/api.php",
        params={"action": "query", "prop": "extracts", "exintro": True, "explaintext": True, "titles": title, "format": "json"},
        headers=WIKI_HEADERS, timeout=10,
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
        params={"ServiceKey": DISEASE_API_KEY, "pageNo": 1, "numOfRows": 5, "sickType": 1,
                "medTp": 1, "diseaseType": "SICK_NM", "searchText": keyword},
        timeout=10,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    return [
        {"sickNm": item.findtext("sickNm", ""), "sickCd": item.findtext("sickCd", ""),
         "sickEngNm": item.findtext("sickEngNm", "")}
        for item in root.findall(".//item")
    ]


def search_disease_code(keyword: str) -> str:
    keyword = clean_disease_query(keyword)
    if not DISEASE_API_KEY:
        return "질병정보서비스 API 키가 설정되지 않았습니다."
    items = _get_disease_items(keyword)
    if not items:
        return f"'{keyword}'와(과) 일치하는 공식 질병 정보를 찾지 못했습니다."
    lines = ["[건강보험심사평가원 - 질병정보서비스]"]
    for item in items:
        lines.append(f"- {item['sickNm']} (코드: {item['sickCd']}, 영문명: {item['sickEngNm']})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 도구 3: 증상 기반 RAG (11단계) - "직접 답변" 도구
# ---------------------------------------------------------------------------

def embed(text: str, model: str = EMBED_MODEL) -> list[float]:
    response = requests.post(OLLAMA_EMBED_URL, json={"model": model, "prompt": text}, timeout=30)
    response.raise_for_status()
    return response.json()["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def search_kdca(query: str, top_k: int = 5):
    query_vec = embed(query)
    scored = [(name, cosine_similarity(query_vec, e["embedding"]), e) for name, e in KDCA_CORPUS.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


SYMPTOM_RAG_SYSTEM_PROMPT = (
    "당신은 국가건강정보포털 자료를 바탕으로 답하는 친절한 건강정보 도우미입니다.\n"
    "반드시 아래 [참고자료]에 있는 내용만 근거로 답하세요.\n"
    "참고자료에 없는 내용은 절대 지어내지 말고, 모른다고 답하세요.\n"
    "이것은 진단이 아니라 참고 정보이므로, 확정적으로 단언하지 말고 "
    "'~일 가능성이 있습니다' 같은 표현을 쓰고, 증상이 지속되면 병원 진료를 권하세요.\n"
    "질문 문장을 그대로 되풀이하지 말고, 바로 본론(답)부터 말하세요.\n"
    "따뜻하고 공감하는 어투로, 참고자료에 있는 원인/증상/관리방법을 충분히 풀어서 "
    "친절하게 설명하세요. 한두 문장으로 짧게 끝내지 마세요.\n"
    "답변 끝에는 진단을 좁히는 데 도움될 후속 질문을 하나 자연스럽게 덧붙이세요 "
    "(예: '~한 증상도 있으신가요?'). 참고자료에 실제로 나오는 증상/요인에 대해서만 "
    "물어보세요."
)


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """max_chars 근처에서 자르되, 문장 중간이 아니라 그 문장이 끝나는 지점까지 포함한다.
    글자 수 제한보다 "문장을 안 끊는 것"이 우선이라, max_chars를 조금 넘어가도 된다
    (그냥 글자 수로 자르면 문장이 중간에 뚝 끊기고, 그 잘린 문장을 LLM이 그대로
    베껴 써서 답변도 중간에 끊기는 문제가 실제로 발생했음)."""
    if len(text) <= max_chars:
        return text
    for i in range(max_chars, len(text)):
        if text[i] in ".!?":
            return text[: i + 1]
    return text  # 그 뒤로 문장부호가 아예 없으면 끝까지 다 포함


def search_symptom_info(symptom_or_keyword: str) -> str:
    """증상 문장이나 병명을 자유롭게 받아서, 관련 질환을 의미 기반으로 찾아 답한다."""
    results = search_kdca(symptom_or_keyword, top_k=5)
    if not results or results[0][1] < MIN_SIMILARITY_SYMPTOM:
        return "관련된 건강정보를 찾지 못해 답변할 수 없습니다. 증상을 좀 더 구체적으로 말씀해주세요."

    print("    [검색됨]", ", ".join(f"{n}({s:.2f})" for n, s, _ in results[:3]))
    context = "\n\n".join(
        f"[{name}]\n{truncate_at_sentence(entry['text'], 800)}" for name, score, entry in results[:3]
    )
    messages = [
        {"role": "system", "content": SYMPTOM_RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"[참고자료]\n{context}\n\n[질문]\n{symptom_or_keyword}"},
    ]
    # 종합(합성) 단계는 llama3.2가 약함 - PubMed 때와 같은 이유로 RAG_MODEL(llama3.1) 사용
    message = call_with_retry(messages, model=RAG_MODEL, inner=True)
    if message is None:
        return "[오류] 모델이 계속 비정상적인 응답을 내서 포기했습니다."
    return message["content"]


# ---------------------------------------------------------------------------
# 도구 4: PubMed 심층 RAG (13단계) - "직접 답변" 도구
# ---------------------------------------------------------------------------

def _to_english_query(keyword: str) -> str:
    if not re.search(r"[가-힣]", keyword):
        return keyword
    items = _get_disease_items(keyword)
    if items and items[0]["sickEngNm"]:
        return items[0]["sickEngNm"]
    return keyword


def _pubmed_esearch(query: str, retmax: int = 10) -> list[str]:
    response = requests.get(
        PUBMED_ESEARCH_URL,
        params={**PUBMED_CONTACT, "db": "pubmed", "term": query, "retmode": "json", "retmax": retmax, "sort": "relevance"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("esearchresult", {}).get("idlist", [])


def _pubmed_efetch_abstracts(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    response = requests.get(
        PUBMED_EFETCH_URL,
        params={**PUBMED_CONTACT, "db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"},
        timeout=15,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    articles = []
    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle", "") or ""
        year = article.findtext(".//JournalIssue/PubDate/Year", "") or ""
        journal = article.findtext(".//Journal/Title", "") or ""
        parts = [(n.text or "") for n in article.findall(".//Abstract/AbstractText")]
        abstract = " ".join(p.strip() for p in parts if p.strip())
        if not abstract:
            continue
        articles.append({"title": title, "year": year, "journal": journal, "abstract": abstract})
    return articles


PUBMED_TRANSLATE_SYSTEM_PROMPT = (
    "당신은 의학 논문 초록을 한국어로 옮기는 번역 도우미입니다.\n"
    "아래 영어 초록에 있는 사실만 한국어로 정리하세요.\n"
    "원문에 없는 내용을 추가하거나, 추측하거나, 다른 지식을 끌어오지 마세요.\n"
    "3~5문장으로 간결하게 쓰세요."
)
PUBMED_SYNTHESIZE_SYSTEM_PROMPT = (
    "당신은 여러 논문 요약을 종합해서 답하는 의학 연구 요약 도우미입니다.\n"
    "반드시 아래 [참고 요약]에 있는 내용만 근거로 답하세요.\n"
    "참고 요약에 없는 내용은 절대 지어내지 말고 언급하지 마세요.\n"
    "[참고 요약]에 실제로 주어진 논문 제목 외의 다른 논문/저자/출판연도를 "
    "새로 만들어서 인용하지 마세요 - 오직 주어진 제목만 언급하세요.\n"
    "이것은 진단이 아니라 연구 요약이므로 단정적으로 말하지 마세요.\n"
    "질문 문장을 그대로 되풀이하지 말고, 바로 본론(답)부터 말하세요."
)


def _translate_abstract(article: dict) -> str | None:
    messages = [
        {"role": "system", "content": PUBMED_TRANSLATE_SYSTEM_PROMPT},
        {"role": "user", "content": f"제목: {article['title']}\n초록: {article['abstract'][:1500]}"},
    ]
    message = call_with_retry(messages, model=RAG_MODEL, inner=True)
    return message["content"] if message else None


def search_pubmed_deep(keyword: str) -> str:
    """PubMed 논문을 실제로 찾아 번역/종합해서 근거 기반으로 답한다 (시간이 좀 걸림)."""
    keyword = clean_disease_query(keyword)
    english_query = _to_english_query(keyword)
    pmids = _pubmed_esearch(english_query, retmax=10)
    articles = _pubmed_efetch_abstracts(pmids)
    if not articles:
        return f"'{keyword}'({english_query})에 대한 관련 논문을 찾지 못해 답변할 수 없습니다."

    query_vec = embed(english_query)
    for art in articles:
        art["score"] = cosine_similarity(query_vec, embed(f"{art['title']} {art['abstract']}"[:2000]))
    articles.sort(key=lambda a: a["score"], reverse=True)
    top_articles = articles[:3]

    if top_articles[0]["score"] < MIN_SIMILARITY_PUBMED:
        return f"'{keyword}'({english_query})에 대한 관련 논문을 찾지 못해 답변할 수 없습니다."

    print("    [검색됨]", ", ".join(f"{a['title'][:30]}...({a['score']:.2f})" for a in top_articles))

    blocks = []
    for art in top_articles:
        ko = _translate_abstract(art)
        status = "성공" if ko else "실패(건너뜀)"
        print(f"    [번역 {status}] {art['title'][:30]}...")
        if ko:
            blocks.append(f"[{art['title']}] ({art['journal']}, {art['year']})\n{ko}")

    if not blocks:
        return "논문 요약을 만들지 못해 답변할 수 없습니다."

    context = "\n\n".join(blocks)
    messages = [
        {"role": "system", "content": PUBMED_SYNTHESIZE_SYSTEM_PROMPT},
        {"role": "user", "content": f"[참고 요약]\n{context}\n\n[질문]\n{keyword}에 대한 최신 연구 결과를 알려줘."},
    ]
    message = call_with_retry(messages, model=RAG_MODEL, inner=True)
    if message is None:
        return "[오류] 모델이 계속 비정상적인 응답을 내서 포기했습니다."
    return message["content"]


# ---------------------------------------------------------------------------
# 도구 등록 + 에이전트 루프
# ---------------------------------------------------------------------------

TOOLS = [
    {"type": "function", "function": {
        "name": "search_wikipedia",
        "description": "위키피디아에서 질병명이나 의학 용어의 일반적인 설명/개요를 가져온다. '~가 뭐야' 같은 질문에 사용.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "검색할 질병명 또는 의학 용어"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "search_disease_code",
        "description": "정확한 질병명, 표준 질병 코드(KCD), 영문 병명을 조회한다. '질병 코드가 뭐야' 같은 질문에 사용.",
        "parameters": {"type": "object", "properties": {
            "keyword": {"type": "string", "description": "정확히 아는 병명 키워드"}}, "required": ["keyword"]},
    }},
    {"type": "function", "function": {
        "name": "search_symptom_info",
        "description": (
            "증상을 설명하는 자유로운 문장이나 병명을 받아서, 의미 기반 검색으로 관련 질환의 "
            "증상/원인/치료 정보를 찾아 답한다. 정확한 병명을 몰라도 사용 가능 - "
            "'머리 아프고 속이 메스꺼운데 뭘까' 같은 증상 설명 질문에 사용."
        ),
        "parameters": {"type": "object", "properties": {
            "symptom_or_keyword": {"type": "string", "description": "증상 설명 문장 또는 병명"}},
            "required": ["symptom_or_keyword"]},
    }},
    {"type": "function", "function": {
        "name": "search_pubmed_deep",
        "description": (
            "PubMed 논문 초록을 실제로 찾아 번역하고 종합해서 근거 기반으로 답한다. "
            "시간이 좀 걸리지만 신뢰도 높은 연구 요약이 필요할 때 사용. "
            "'관련 논문 찾아줘', '연구 결과가 뭐야' 같은 질문에 사용."
        ),
        "parameters": {"type": "object", "properties": {
            "keyword": {"type": "string", "description": "검색할 질병명/의학 주제"}}, "required": ["keyword"]},
    }},
]

AVAILABLE_TOOLS = {
    "search_wikipedia": search_wikipedia,
    "search_disease_code": search_disease_code,
    "search_symptom_info": search_symptom_info,
    "search_pubmed_deep": search_pubmed_deep,
}

# 이 도구들은 내부에서 이미 근거 기반 최종 답변을 완성해서 돌려준다 -
# 바깥쪽 에이전트가 또 재해석하면 할루시네이션이 재발할 수 있어 그대로 반환한다.
DIRECT_ANSWER_TOOLS = {"search_symptom_info", "search_pubmed_deep"}


class MedicalConversationState:
    """messages(대화 원문)와 별도로 유지하는 구조화된 상태.
    LLM을 전혀 안 쓰고, 도구가 호출될 때마다 규칙 기반으로만 기록한다 -
    이 세션 내내 겪은 LLM 불안정성을 이 부분에는 아예 끌어들이지 않기 위해서."""

    def __init__(self):
        self.diseases: dict[str, dict] = {}  # name -> {"sources": [...], "count": n, "first_turn", "last_turn"}
        self.symptoms: list[dict] = []       # [{"text": ..., "turn": n}]
        self.turn = 0
        self.last_topic: str | None = None   # 가장 최근에 다룬 주제 - "그럼", "그거" 같은 지칭어 해결용
        self.awaiting_followup_reply = False  # 직전 답변이 후속 질문으로 끝났으면 True -
                                               # 다음 입력이 "네", "심해요"처럼 지칭어 없는
                                               # 짧은 대답이어도 이전 주제를 이어받게 함

    def record_disease(self, name: str, source: str):
        """같은 질환이 또 언급되면 새 항목을 만들지 않고, 기존 항목의 횟수/최근턴/출처만 갱신."""
        if name in self.diseases:
            entry = self.diseases[name]
            entry["count"] += 1
            entry["last_turn"] = self.turn
            if source not in entry["sources"]:
                entry["sources"].append(source)
        else:
            self.diseases[name] = {
                "sources": [source], "count": 1, "first_turn": self.turn, "last_turn": self.turn,
            }

    def record_symptom(self, text: str, source: str):
        """완전히 똑같은 문장이 또 나오면 중복 저장하지 않음 (자유 문장이라 횟수 집계는 안 함)."""
        if any(s["text"] == text for s in self.symptoms):
            return
        self.symptoms.append({"text": text, "turn": self.turn, "source": source})

    def summary(self) -> str:
        if not self.diseases and not self.symptoms:
            return "아직 기록된 내용이 없습니다."
        lines = []
        if self.diseases:
            lines.append("[언급된 질환]")
            for name, info in self.diseases.items():
                times = f"{info['count']}번" if info["count"] > 1 else "1번"
                lines.append(f"- {name} ({times} 언급, 출처: {', '.join(info['sources'])})")
        if self.symptoms:
            lines.append("\n[언급된 증상]")
            for s in self.symptoms:
                lines.append(f"- {s['text']} (출처: {s['source']})")
        return "\n".join(lines)


def find_known_term_in_text(text: str) -> str | None:
    """사용자 원문에 KDCA 사전(611개)에 있는 병명이 그대로 들어있으면 찾아서 반환.
    LLM이 그 단어를 '다시 타이핑'하다가 깨뜨리는 걸(예: 편두통 -> 국구괭가하)
    피하기 위해, LLM한테 시키지 않고 원문에서 그대로 오려쓴다.
    긴 이름부터 검사해야 "편두통"이 있을 때 "두통"으로 먼저 매칭되는 걸 방지함."""
    for name in sorted(KDCA_CORPUS.keys(), key=len, reverse=True):
        if name in text:
            return name
    return None


def clean_disease_query(query: str) -> str:
    """LLM이 도구 인자에 조사/수식어를 붙여서 만드는 문제(예: '당뇨병' 대신
    '당뇨병의 설명', '당뇨병의 원인')를 정리한다. 위키피디아/공식코드/PubMed는
    "정확한 병명"이어야 제대로 매칭되는데, 수식어가 붙으면 완전히 다른 문서가
    검색될 수 있음 (실제로 "당뇨병의 설명"으로 검색해서 "당뇨병성 케톤산증"이
    나온 사례 발생). KDCA 사전에 있는 병명이 쿼리 안에 포함돼 있으면 그 병명만
    추출해서 쓰고, 못 찾으면 원래 쿼리를 그대로 둔다."""
    known = find_known_term_in_text(query)
    return known if known else query


def _record_tool_call(state: MedicalConversationState, name: str, args: dict):
    """도구 호출 하나를 보고 state에 규칙 기반으로 기록. LLM 사용 안 함.
    도구 실행 때와 똑같이 clean_disease_query를 거쳐서 기록해야, "당뇨병"과
    "당뇨병의 원인"이 state에서 서로 다른 질환으로 중복 기록되는 걸 막을 수 있음."""
    if name == "search_wikipedia":
        state.record_disease(clean_disease_query(args.get("query", "")), "search_wikipedia")
    elif name == "search_disease_code":
        state.record_disease(clean_disease_query(args.get("keyword", "")), "search_disease_code")
    elif name == "search_pubmed_deep":
        state.record_disease(clean_disease_query(args.get("keyword", "")), "search_pubmed_deep")
    elif name == "search_symptom_info":
        value = args.get("symptom_or_keyword", "")
        # KDCA 사전에 있는 정확한 병명이면 "질환"으로, 아니면 자유 서술이니 "증상"으로 기록
        if value in KDCA_CORPUS:
            state.record_disease(value, "search_symptom_info")
        else:
            state.record_symptom(value, "search_symptom_info")


# ---------------------------------------------------------------------------
# 도구 선택 방식 전환: 지금까지는 LLM(TOOLS + call_with_retry)이 4개 도구 중
# 뭘 쓸지 매번 자유롭게 판단했는데, 하루 종일 겪은 불안정성의 상당 부분이
# "작은 모델이 도구 선택 자체를 못 미더워한다"는 거였음. 그래서 순서를
# 코드로 고정하는 파이프라인으로 바꾼다 - LLM은 이제 도구 선택에 관여하지
# 않고, PubMed 내부(번역/종합)에서만 쓰인다.
#   1) KDCA 건강포털 RAG 먼저 시도
#   2) 실패하면(근거 부족) 위키피디아로 대체
#   3) 정확한 병명을 알아냈으면 공식 코드도 같이
#   4) 질문에 "논문"/"연구" 같은 표현이 있으면 PubMed 심층 검색 추가
# (TOOLS/AVAILABLE_TOOLS/is_outer_response_ok 등 기존 LLM 기반 선택 로직은
# 아래 함수들이 재사용하는 검색 함수 자체는 그대로 쓰되, "선택" 부분만 대체됨)
# ---------------------------------------------------------------------------

RESEARCH_KEYWORDS = ("논문", "연구", "study", "리서치", "research")


def wants_research(user_question: str) -> bool:
    return any(kw in user_question for kw in RESEARCH_KEYWORDS)


# 이 마커가 있을 때만 "이전 주제 이어받기"를 적용한다. 마커 없이 무조건
# last_topic을 붙이면 "오늘 날씨 어때?"처럼 완전히 무관한 새 질문까지
# 직전 질환(예: 당뇨병) 얘기로 착각해서 "당뇨병 날씨" 같은 헛소리가 나오는
# 버그가 실제로 발생함 - 지칭어가 있을 때만 좁혀서 적용.
FOLLOWUP_MARKERS = ("그럼", "그거", "그건", "그게", "그것", "이거", "저거")


def resolve_query_with_context(user_question: str, state: MedicalConversationState):
    """이번 질문에서 병명을 직접 찾고, 지칭어(그럼/그거 등)가 있을 때만
    state.last_topic을 힌트로 앞에 붙인다. LLM 없이 순수 문자열 처리로
    '그럼 치료법은?' 같은 질문이 이전 주제를 잃지 않게 한다."""
    known_term = find_known_term_in_text(user_question)
    if known_term:
        return user_question, known_term
    # 지칭어("그럼"/"그거")가 있거나, 직전 답변이 후속 질문으로 끝나서 지금이
    # 그 답인 경우(예: "네", "심해요")에는 지칭어가 없어도 이전 주제를 이어받는다.
    is_followup = (
        any(marker in user_question for marker in FOLLOWUP_MARKERS)
        or state.awaiting_followup_reply
    )
    if state.last_topic and is_followup:
        implied_term = state.last_topic if state.last_topic in KDCA_CORPUS else None
        return f"{state.last_topic} {user_question}", implied_term
    return user_question, None


def _is_failure_message(text: str) -> bool:
    return ("찾지 못" in text) or ("답변할 수 없습니다" in text) or ("설정되지 않았습니다" in text)


def run_agent_turn(messages: list, user_question: str, state: MedicalConversationState) -> str:
    """C단계: messages(대화 원문)와 state(구조화된 요약)를 둘 다 세션 내내 이어받는다.
    도구 선택은 LLM이 아니라 고정된 파이프라인 순서로 결정한다."""
    state.turn += 1
    messages.append({"role": "user", "content": user_question})

    rag_query, known_term = resolve_query_with_context(user_question, state)
    parts: list[str] = []

    # 1) 건강포털 RAG 먼저 시도
    rag_result = search_symptom_info(rag_query)
    if _is_failure_message(rag_result):
        # 2) 실패하면 위키피디아로 대체 (아까 "위염"처럼 KDCA에 내용이 비어있는 경우 등)
        print("  [파이프라인] 건강포털 RAG 실패 -> 위키피디아로 대체")
        wiki_result = search_wikipedia(known_term or user_question)
        if _is_failure_message(wiki_result):
            return (
                "이 질문은 제가 다루는 건강/의료 정보 범위를 벗어났거나, "
                "제가 아는 정보로는 답변하기 어려운 내용인 것 같아요. "
                "증상이나 병명을 조금 더 구체적으로 말씀해주시겠어요?"
            )
        parts.append(wiki_result)
        if known_term:
            state.record_disease(known_term, "search_wikipedia(fallback)")
        else:
            state.record_symptom(user_question, "search_wikipedia(fallback)")
    else:
        print("  [파이프라인] 건강포털 RAG 성공")
        parts.append(rag_result)
        if known_term:
            state.record_disease(known_term, "search_symptom_info")
        else:
            state.record_symptom(user_question, "search_symptom_info")

    # 3) 정확한 병명을 알면 공식 코드도 같이
    if known_term:
        code_result = search_disease_code(known_term)
        if not _is_failure_message(code_result):
            parts.append(code_result)
            state.record_disease(known_term, "search_disease_code")

    # 4) "논문"/"연구" 같은 표현이 있으면 PubMed 추가 (정확한 병명이 있을 때만 - 검색어가 필요해서)
    if known_term and wants_research(user_question):
        print("  [파이프라인] '논문/연구' 표현 감지 -> PubMed 추가")
        pubmed_result = search_pubmed_deep(known_term)
        parts.append(pubmed_result)
        state.record_disease(known_term, "search_pubmed_deep")

    state.last_topic = known_term or user_question
    combined = "\n\n".join(parts)
    # 이번 답변이 후속 질문으로 끝났으면, 다음 입력이 "네"/"심해요"처럼 지칭어가
    # 없어도 이전 주제를 이어받게 표시해둔다 (마지막 200자만 검사 - 너무 앞부분의
    # 물음표까지 "질문으로 끝났다"고 오판하지 않도록).
    state.awaiting_followup_reply = "?" in combined[-200:]
    messages.append({"role": "assistant", "content": combined})
    return combined


def main():
    print("=" * 50)
    print("내 손안의 의사 (C단계: 멀티턴 대화 + 구조화된 상태 기억)")
    print("도구 4개: 위키피디아 / 공식병명코드 / 증상RAG / PubMed심층RAG")
    print("이전 대화 맥락을 기억합니다. '요약해줘'라고 하면 지금까지 기록을 보여줍니다.")
    print("종료하려면 'q' 또는 '종료' 입력.")
    print("=" * 50)

    messages: list = []                          # 대화 원문(자유 텍스트)
    state = MedicalConversationState()           # 구조화된 요약(질환/증상 목록)

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
        if "요약" in question or "기록" in question:
            print(f"\n[지금까지의 기록]\n{state.summary()}")
            continue
        print(f"\n답변> {run_agent_turn(messages, question, state)}")


if __name__ == "__main__":
    main()
