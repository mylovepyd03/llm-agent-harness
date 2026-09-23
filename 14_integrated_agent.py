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
그래서 도구 결과는 바깥쪽 LLM에게 넘기지 않고 **그대로 최종 답변으로 반환**한다.

(C단계에서 더 나아가서, 이제는 "어떤 도구를 쓸지"도 LLM이 아니라 코드가 고정
순서로 정한다 - run_agent_turn() 참고. LLM은 PubMed 번역/종합/재설명처럼
"검색된 근거를 문장으로 만드는" 좁은 역할만 맡는다.)
"""
import json
import math
import os
import re
import ssl
import xml.etree.ElementTree as ET
from datetime import datetime

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter

load_dotenv()

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

RAG_MODEL = "llama3.1"         # 유일한 채팅 모델 - 번역/종합/재설명 전용
                                # (llama3.2는 도구 선택 불안정 + 문자 오염 문제가 있어 완전히 은퇴시킴)
EMBED_MODEL = "bge-m3"

WIKI_HEADERS = {
    "User-Agent": "llm-agent-harness-learning-project/0.1 (personal study, contact: paranvit@gmail.com)"
}
DISEASE_API_URL = "http://apis.data.go.kr/B551182/diseaseInfoService1/getDissNameCodeList1"
DISEASE_API_KEY = os.environ.get("DISEASE_INFO_SERVICE_KEY")

KDCA_EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "data", "kdca_embeddings_clean.json")
MIN_SIMILARITY_SYMPTOM = 0.6
# 0.45였을 때 "주사피부염인데 어떻게 조심해야돼"가 코퍼스에 없는데도 "열상"(0.58,
# 칼에 베인 상처 - 완전히 무관)을 근거인 것처럼 받아들여서 엉뚱한 내용을 확신에
# 차서 답한 사례를 발견함(2026-09-23). 실측해보니 정답 문서 자체도 0.52~0.69
# 범위라 "틀린 매칭"과 "맞는 매칭"이 겹쳐서, 단순 문턱값으로 완전히 가를 순
# 없었음 - 그래도 0.6으로 올리면 정확한 병명/강한 매칭(0.61~0.74)은 그대로
# 통과하고, 애매한 매칭(주사피부염 사례 포함, 0.5대)은 걸러져서 위키피디아로
# 넘어감 - "약한 근거로 확신에 찬 오답" 대신 "모르면 바로 다음 수단으로 넘어간다"는
# 원칙(사용자 지시)에 맞춘 것. 대가: "속이 쓰리고 아파요"→위식도역류질환(0.58)처럼
# 실제로 괜찮았을 애매한 매칭도 이제 위키피디아로 넘어감(정확함 우선, 보수적 선택).

PUBMED_DEBUG_LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "pubmed_debug_log.jsonl")

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
# 공통: LLM 호출 + 응답 검증(하네스)
# ---------------------------------------------------------------------------

def chat(messages, model: str = RAG_MODEL, temperature: float = 0, num_predict: int = 512) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["message"]


def has_short_chunk_repetition(content: str) -> bool:
    return re.search(r"(.{2,20}?)\1{4,}", content) is not None


def has_sentence_repetition(content: str, min_len: int = 15, min_repeats: int = 2) -> bool:
    # min_repeats=3이었을 때, "치료" 항목과 "예방" 항목에 완전히 똑같은 문장이
    # 정확히 2번(예: 금연 설명)만 반복되는 걸 놓친 사례를 실제로 발견함 -
    # 기존 정상 답변 6개로 오탐 검사한 뒤 2로 낮춰도 안전한 것을 확인함.
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


def call_with_retry(messages, model: str = RAG_MODEL, max_retries: int = 1, num_predict: int = 512):
    """1차 시도는 temperature=0(일관성), 실패하면 재시도부터는 temperature를
    올려서(0.6) 다른 출력이 나올 여지를 준다. temperature=0은 같은 입력에 거의
    항상 같은(고장난) 출력을 내서, 아무것도 안 바꾸고 재시도해봐야 소용없다는
    걸 실제로 확인했음 - 단, 온도를 올리면 새로운 형태로 깨질 수도 있어서
    재시도 결과도 반드시 같은 검증기를 다시 통과해야만 받아들인다.

    (예전엔 여기서 "도구 선택" 응답과 "RAG 내부 생성" 응답을 다른 검증기로 나눠
    검사했는데, 도구 선택 자체가 결정론적 파이프라인으로 바뀌면서 이 함수가
    RAG 내부 생성 검증에만 쓰이게 됨 - is_rag_generation_ok 하나로 통일함.)"""
    for attempt in range(max_retries + 1):
        temperature = 0 if attempt == 0 else 0.6
        message = chat(messages, model=model, temperature=temperature, num_predict=num_predict)
        if is_rag_generation_ok(message.get("content", "")):
            return message
        print(f"    [경고] 비정상 응답 감지 (시도 {attempt + 1}/{max_retries + 1}, temp={temperature}) - 재시도")
    return None


# ---------------------------------------------------------------------------
# 도구 1, 2: A단계 그대로 (위키피디아, 공식 병명/코드)
# ---------------------------------------------------------------------------

MIN_SIMILARITY_WIKIPEDIA = 0.4
# 위키피디아 자체 검색(srsearch)은 KDCA 임베딩 검색과 달리 "이 결과가 실제로
# 관련 있는가"를 알려주는 점수가 없어서, 오타나 낯선 표기가 섞이면 완전히
# 무관한 문서를 fuzzy하게 찾아오는 걸 확인함(예: "주사피부부염"(오타) -> "나혜석"
# [무관한 인물], "로사시아" -> "사시"[사팔뜨기], 2026-09-23). 그래서 문서를
# 찾아온 뒤 우리 자신의 임베딩으로 "질문과 본문이 실제로 관련 있는지" 한 번 더
# 검증한다. 실측: 무관한 오검색 0.17~0.28, 정말 관련 있는 검색 0.53~0.71로
# 뚜렷한 간격이 있어서 0.4로 잡음(KDCA 쪽과 달리 여긴 깔끔하게 갈림).
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
    if not extract:
        return f"'{query}'에 대한 위키피디아 검색 결과가 없습니다."

    relevance = cosine_similarity(embed(query), embed(extract[:1000]))
    if relevance < MIN_SIMILARITY_WIKIPEDIA:
        print(f"    [경고] 위키피디아 결과 '{title}'가 질문과 무관해 보임(유사도 {relevance:.2f}) - 버림")
        return f"'{query}'에 대한 위키피디아 검색 결과가 없습니다."

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
    "구체적인 약물 이름(예: 특정 성분명, 제품명)을 절대 나열하거나 지어내지 마세요. "
    "약물치료가 필요하면 '약물치료' 정도로만 일반적으로 언급하세요.\n"
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


def search_symptom_info(symptom_or_keyword: str, question_text: str | None = None) -> str:
    """증상 문장이나 병명을 자유롭게 받아서, 관련 질환을 의미 기반으로 찾아 답한다.
    search_text(=symptom_or_keyword)는 검색(임베딩 매칭) 전용이고, question_text는
    LLM에게 보여줄 [질문] 부분 전용이다 - 분리하는 이유는 아래 참고."""
    question_text = question_text or symptom_or_keyword

    results = search_kdca(symptom_or_keyword, top_k=5)
    if not results or results[0][1] < MIN_SIMILARITY_SYMPTOM:
        return "관련된 건강정보를 찾지 못해 답변할 수 없습니다. 증상을 좀 더 구체적으로 말씀해주세요."

    print("    [검색됨]", ", ".join(f"{n}({s:.2f})" for n, s, _ in results[:3]))
    context = "\n\n".join(
        f"[{name}]\n{truncate_at_sentence(entry['text'], 800)}" for name, score, entry in results[:3]
    )
    messages = [
        {"role": "system", "content": SYMPTOM_RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"[참고자료]\n{context}\n\n[질문]\n{question_text}"},
    ]
    # 종합(합성) 단계는 llama3.2가 약함 - PubMed 때와 같은 이유로 RAG_MODEL(llama3.1) 사용.
    # num_predict: 친절하고 상세하게 답하도록 프롬프트를 늘렸더니 512토큰 한도에
    # 걸려 문장이 뚝 끊기는 경우가 실제로 발생해서 넉넉하게 늘림.
    message = call_with_retry(messages, model=RAG_MODEL, num_predict=1024)
    if message is None:
        return "[오류] 모델이 계속 비정상적인 응답을 내서 포기했습니다."
    return message["content"]


# ---------------------------------------------------------------------------
# 도구 4: PubMed 심층 RAG (13단계) - "직접 답변" 도구
# ---------------------------------------------------------------------------

TERM_TRANSLATE_SYSTEM_PROMPT = (
    "당신은 한국어 의학 용어를 표준 영어 의학 용어로 번역하는 도우미입니다.\n"
    "주어진 한국어 병명/의학 용어에 해당하는 영어 이름만 한 줄로 출력하세요.\n"
    "설명, 따옴표, 다른 말은 절대 덧붙이지 마세요. 확실하지 않아도 가장 가능성\n"
    "높은 영어 이름을 추정해서 답하세요."
)


def _translate_term_to_english(keyword: str) -> str:
    """HIRA 공식 코드 DB에 없는 병명(예: '주사피부염')은 한글 그대로 PubMed에
    넘기면 검색 결과가 0건이 된다(실측 확인) - PubMed는 영어 문헌 DB라서 당연함.
    HIRA 조회가 실패했을 때 llama3.1에게 영어 이름을 추정 번역시켜서 이 문제를
    우회한다."""
    messages = [
        {"role": "system", "content": TERM_TRANSLATE_SYSTEM_PROMPT},
        {"role": "user", "content": keyword},
    ]
    message = call_with_retry(messages, model=RAG_MODEL)
    if message is None:
        return keyword
    translated = message["content"].strip().strip("\"'").splitlines()[0].strip()
    return translated if translated else keyword


def _to_english_query(keyword: str) -> str:
    if not re.search(r"[가-힣]", keyword):
        return keyword
    items = _get_disease_items(keyword)
    if items and items[0]["sickEngNm"]:
        return items[0]["sickEngNm"]
    return _translate_term_to_english(keyword)


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
PUBMED_SIMPLIFY_SYSTEM_PROMPT = (
    "당신은 의학 연구 요약을 일반인이 이해하기 쉽게 다시 설명하는 도우미입니다.\n"
    "아래 [원문]의 사실 내용은 절대 바꾸거나 빼거나 새로 추가하지 마세요 - 새로운 "
    "정보, 수치, 연도, 논문을 지어내지 마세요. 오직 표현만 쉽게 바꾸는 것이 목표입니다.\n"
    "어려운 의학 용어나 줄임말이 나오면 쉬운 말로 풀어 쓰거나, 용어 뒤에 괄호로 "
    "짧은 설명을 덧붙이세요 (예: '기관지 과민성(기관지가 자극에 예민하게 반응하는 상태)').\n"
    "원문에 있는 문장/정보량을 크게 늘리거나 줄이지 말고, 있는 내용을 더 쉬운 말로 "
    "바꾸는 데만 집중하세요."
)


def _log_pubmed_stage(stage: str, keyword: str, content: str) -> None:
    """PubMed 답변 생성 단계별 원문을 파일에 상시 남긴다. "동맥경화"처럼 무관한
    내용이 섞이는 현상이 재발했을 때, 그 순간 터미널을 보고 있지 않았어도 이
    로그를 대조해서 어느 단계(synthesize/simplify)에서 처음 생겼는지 바로
    확인하기 위한 것 - 원인 확정에 실패했던 이전 시도 때문에 추가함."""
    try:
        with open(PUBMED_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "keyword": keyword,
                "stage": stage,
                "content": content,
            }, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _has_fabricated_citation(content: str, valid_years: set[str]) -> bool:
    """답변에 등장하는 4자리 연도가 실제로 검색된 논문 연도 목록에 하나도 없으면,
    형식만 그럴듯한(예: "대한당뇨병학회 (2018)") 지어낸 인용일 가능성이 높다고 본다."""
    years_mentioned = set(re.findall(r"\b(?:19|20)\d{2}\b", content))
    return bool(years_mentioned - valid_years)


def _simplify_for_layperson(content: str, valid_years: set[str], keyword: str = "") -> str:
    """전문용어 위주 답변을 일반인이 이해하기 쉽게 다시 설명한다. 새 사실을
    지어낼 위험이 있으므로, 실패/의심스러우면 원문(전문용어 버전)을 그대로 반환해서
    안전을 우선한다 - 이해하기 쉬운 것보다 정확한 게 더 중요."""
    messages = [
        {"role": "system", "content": PUBMED_SIMPLIFY_SYSTEM_PROMPT},
        {"role": "user", "content": f"[원문]\n{content}"},
    ]
    for attempt in range(2):
        message = call_with_retry(messages, model=RAG_MODEL, num_predict=1024)
        if message is None:
            print("    [경고] 쉬운 설명 생성 실패 - 원문 그대로 사용")
            return content
        simplified = message["content"]
        _log_pubmed_stage(f"simplify_attempt_{attempt + 1}", keyword, simplified)
        if _has_fabricated_citation(simplified, valid_years):
            print(f"    [경고] 쉬운 설명 중 없는 연도 인용 감지 - 재시도 ({attempt + 1}/2)")
            continue
        if len(simplified) > len(content) * 1.8:
            print(f"    [경고] 쉬운 설명이 원문보다 훨씬 길어짐(내용 추가 의심) - 재시도 ({attempt + 1}/2)")
            continue
        return simplified
    print("    [경고] 쉬운 설명이 계속 의심스러워 원문 그대로 사용")
    return content


def _translate_abstract(article: dict) -> str | None:
    messages = [
        {"role": "system", "content": PUBMED_TRANSLATE_SYSTEM_PROMPT},
        {"role": "user", "content": f"제목: {article['title']}\n초록: {article['abstract'][:1500]}"},
    ]
    message = call_with_retry(messages, model=RAG_MODEL)
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
    _log_pubmed_stage("search_articles", keyword, "\n\n".join(
        f"[{a['title']}] ({a['journal']}, {a['year']})\n{a['abstract']}" for a in top_articles
    ))

    blocks = []
    for art in top_articles:
        ko = _translate_abstract(art)
        status = "성공" if ko else "실패(건너뜀)"
        print(f"    [번역 {status}] {art['title'][:30]}...")
        if ko:
            blocks.append(f"[{art['title']}] ({art['journal']}, {art['year']})\n{ko}")
            _log_pubmed_stage(f"translate:{art['title'][:50]}", keyword, ko)

    if not blocks:
        return "논문 요약을 만들지 못해 답변할 수 없습니다."

    context = "\n\n".join(blocks)
    valid_years = {art["year"] for art in top_articles if art.get("year")}
    messages = [
        {"role": "system", "content": PUBMED_SYNTHESIZE_SYSTEM_PROMPT},
        {"role": "user", "content": f"[참고 요약]\n{context}\n\n[질문]\n{keyword}에 대한 최신 연구 결과를 알려줘."},
    ]
    for attempt in range(2):
        message = call_with_retry(messages, model=RAG_MODEL, num_predict=1024)
        if message is None:
            return "[오류] 모델이 계속 비정상적인 응답을 내서 포기했습니다."
        _log_pubmed_stage(f"synthesize_attempt_{attempt + 1}", keyword, message["content"])
        if not _has_fabricated_citation(message["content"], valid_years):
            return _simplify_for_layperson(message["content"], valid_years, keyword)
        print(f"    [경고] 실제 논문에 없는 연도 인용 감지 - 재시도 ({attempt + 1}/2)")
    return "논문 내용을 실제 자료 그대로 요약하지 못해 답변할 수 없습니다. 다시 시도해주세요."


# ---------------------------------------------------------------------------
# 상태 관리 + 결정론적 파이프라인
# (예전엔 여기에 LLM에게 줄 도구 스펙(TOOLS)과 이름→함수 매핑(AVAILABLE_TOOLS)이
#  있었지만, 도구 "선택"을 코드가 고정 순서로 정하는 구조로 바뀌면서 필요 없어짐 -
#  각 도구 함수는 run_agent_turn()이 직접 호출한다.)
# ---------------------------------------------------------------------------

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


_TRAILING_PARTICLES_RE = re.compile(
    r"(이었을때|였을때|일때|을때|할때|이라면|라면|이었으면|이나|라서|이라서|인데|인지|"
    r"이면|이고|이지만|이라도|일까요|일까|인가요|인가|랑|이랑|과|와|은|는|이|가|을|를|"
    r"의|도|만|면)+$"
)


def clean_disease_query(query: str) -> str:
    """LLM이 도구 인자에 조사/수식어를 붙여서 만드는 문제(예: '당뇨병' 대신
    '당뇨병의 설명', '당뇨병의 원인')를 정리한다. 위키피디아/공식코드/PubMed는
    "정확한 병명"이어야 제대로 매칭되는데, 수식어가 붙으면 완전히 다른 문서가
    검색될 수 있음 (실제로 "당뇨병의 설명"으로 검색해서 "당뇨병성 케톤산증"이
    나온 사례 발생). KDCA 사전에 있는 병명이 쿼리 안에 포함돼 있으면 그 병명만
    추출해서 쓴다.

    KDCA 사전에도 없는 완전히 새 단어면(예: "주사피부염") 예전엔 원본 문장을
    그대로 반환했는데, 그 문장을 통째로 위키피디아 검색에 넘기면("주사피부염인데
    어떻게 조심해야돼?") 물음표·어미까지 다 들어가서 완전히 무관한 문서(실제로
    "나혜석"이라는 인물 문서)가 나오는 걸 확인함(2026-09-23). 그래서 known_term을
    못 찾으면, 문장 맨 앞 어절에서 흔한 조사/연결어미를 떼어내 병명 후보를
    만들어본다 - 완벽하진 않지만("머리가 지끈거리고 아파요" -> "머리" 정도로만
    깎임) 문장 전체를 그대로 넘기는 것보다는 훨씬 안전하다."""
    known = find_known_term_in_text(query)
    if known:
        return known
    stripped = query.strip()
    if not stripped:
        return query
    first_chunk = stripped.split()[0]
    candidate = _TRAILING_PARTICLES_RE.sub("", first_chunk)
    return candidate if candidate else query


# ---------------------------------------------------------------------------
# 도구 선택 방식: LLM이 4개 도구 중 뭘 쓸지 매번 자유롭게 판단하게 했더니
# 하루 종일 겪은 불안정성의 상당 부분이 "작은 모델이 도구 선택 자체를 못
# 미더워한다"는 거였음. 그래서 순서를 코드로 고정하는 파이프라인으로 바꿈 -
# LLM은 도구 선택에 전혀 관여하지 않고, PubMed 내부(번역/종합/재설명)에서만
# 쓰인다.
#   1) KDCA 건강포털 RAG 먼저 시도
#   2) 실패하면(근거 부족) 위키피디아로 대체
#   3) 정확한 병명을 알아냈으면 공식 코드도 같이
#   4) 질문에 "논문"/"연구" 같은 표현이 있으면 PubMed 심층 검색 추가
# ---------------------------------------------------------------------------

RESEARCH_KEYWORDS = ("논문", "연구", "study", "리서치", "research")


def wants_research(user_question: str) -> bool:
    return any(kw in user_question for kw in RESEARCH_KEYWORDS)


# 이 마커가 있을 때만 "이전 주제 이어받기"를 적용한다. 마커 없이 무조건
# last_topic을 붙이면 "오늘 날씨 어때?"처럼 완전히 무관한 새 질문까지
# 직전 질환(예: 당뇨병) 얘기로 착각해서 "당뇨병 날씨" 같은 헛소리가 나오는
# 버그가 실제로 발생함 - 지칭어가 있을 때만 좁혀서 적용.
FOLLOWUP_MARKERS = ("그럼", "그거", "그건", "그게", "그것", "이거", "저거")

# "네 심해요"처럼 짧은 대답은 마커도 없고, awaiting_followup_reply도(모델이 항상
# "?"로 안 끝내서) 못 잡는 경우가 실제로 발생함. 순수 임베딩 점수로 구분해보려 했으나
# "네 심해요"(0.46)와 "그럼 치료법은?"(0.59)이 둘 다 그 자체로도 그럴듯하게 매칭돼서
# 점수만으론 구분 불가 - 대신 "짧은 문장인가"를 추가 신호로 씀 (실제 증상 질문은
# 이보다 길게 나오는 경향이 있어서 안전한 경계로 확인함).
SHORT_REPLY_MAX_CHARS = 8


def resolve_query_with_context(user_question: str, state: MedicalConversationState):
    """이번 질문에서 병명을 직접 찾고, 지칭어(그럼/그거 등)가 있을 때만
    state.last_topic을 힌트로 활용한다. LLM 없이 순수 문자열 처리로
    '그럼 치료법은?' 같은 질문이 이전 주제를 잃지 않게 한다.

    검색용 텍스트(search_text)와 LLM에게 보여줄 질문(question_text)을 분리해서
    반환한다 - 후속 대답("네 심해요")을 검색어에까지 섞어버리면, "심해요" 같은
    말이 임베딩을 엉뚱한 문서(예: 응급 기도폐쇄 처치)로 끌고 가는 걸 실제로
    확인했음. 검색은 이전 주제 하나로만 안전하게 하고, 그 대답 내용은 LLM한테
    질문으로만 보여줘서 답변에 반영되게 한다.

    반환값: (search_text, question_text, known_term)"""
    known_term = find_known_term_in_text(user_question)
    if known_term:
        return user_question, user_question, known_term
    is_followup = (
        any(marker in user_question for marker in FOLLOWUP_MARKERS)
        or state.awaiting_followup_reply
        or len(user_question.strip()) <= SHORT_REPLY_MAX_CHARS
    )
    if state.last_topic and is_followup:
        implied_term = state.last_topic if state.last_topic in KDCA_CORPUS else None
        combined_question = f"{state.last_topic} {user_question}"
        return state.last_topic, combined_question, implied_term
    return user_question, user_question, None


def _is_failure_message(text: str) -> bool:
    # 도구들이 반환하는 실패 메시지는 전부 "[오류]"로 시작하거나 아래 문구 중
    # 하나를 포함하도록 맞춰 왔는데, 새 실패 메시지를 추가할 때마다 여기 목록에
    # 반영하는 걸 깜빡해서 실제로 실패 메시지가 성공으로 오인된 사례가 두 번
    # 있었음(위키피디아 "검색 결과가 없습니다", search_symptom_info의 "[오류]
    # 모델이... 포기했습니다") - "[오류]" 접두사 체크를 추가해서 앞으로 새
    # 오류 메시지를 추가해도 이 접두사만 지키면 자동으로 잡히게 함.
    return (
        text.startswith("[오류]")
        or ("찾지 못" in text)
        or ("답변할 수 없습니다" in text)
        or ("설정되지 않았습니다" in text)
        or ("검색 결과가 없습니다" in text)
    )


def run_agent_turn(messages: list, user_question: str, state: MedicalConversationState) -> str:
    """C단계: messages(대화 원문)와 state(구조화된 요약)를 둘 다 세션 내내 이어받는다.
    도구 선택은 LLM이 아니라 고정된 파이프라인 순서로 결정한다.

    네트워크/Ollama 호출은 전부 이 함수 안에서(직접 또는 하위 도구 함수를 통해)
    일어나는데, 그중 하나라도 실패하면(Ollama 재시작 중, 외부 API 순단 등)
    requests 예외가 여기까지 그대로 올라온다. 그걸 여기서 잡아서 messages/state를
    이번 턴 이전 상태로 되돌리고 친절한 오류 메시지로 답해야, 그 예외가 main()의
    대화 루프 전체를 죽여서 지금까지의 멀티턴 기록을 날리는 걸 막을 수 있다."""
    state.turn += 1
    messages.append({"role": "user", "content": user_question})

    try:
        search_text, question_text, known_term = resolve_query_with_context(user_question, state)
        parts: list[str] = []

        # 1) 건강포털 RAG 먼저 시도 (검색은 search_text로, LLM 질문은 question_text로 - 분리 이유는 resolve_query_with_context 참고)
        rag_result = search_symptom_info(search_text, question_text)
        if _is_failure_message(rag_result):
            # 2) 실패하면 위키피디아로 대체 (아까 "위염"처럼 KDCA에 내용이 비어있는 경우 등)
            print("  [파이프라인] 건강포털 RAG 실패 -> 위키피디아로 대체")
            wiki_result = search_wikipedia(known_term or user_question)
            if not _is_failure_message(wiki_result):
                parts.append(wiki_result)
                if known_term:
                    state.record_disease(known_term, "search_wikipedia(fallback)")
                else:
                    # 위키피디아 결과 형식 "[위키피디아 - 제목]\n..."에서 실제 문서 제목을
                    # 뽑아 기록. 정확한 병명이 아니라 사용자의 증상 서술로 여기까지 왔으므로,
                    # 서술 자체는 "증상"으로, 매칭된 병명은 "추정 질환"으로 같이 남긴다 -
                    # 여러 턴에 걸쳐 증상이 쌓이고 같은 질환이 후보로 반복되면 문진처럼
                    # 좁혀지는 걸 state.summary()에서 볼 수 있게 하기 위함.
                    title_match = re.match(r"\[위키피디아 - (.+?)\]", wiki_result)
                    if title_match:
                        matched_name = title_match.group(1)
                        state.record_symptom(user_question, f"search_wikipedia(fallback, 추정: {matched_name})")
                        state.record_disease(matched_name, "search_wikipedia(fallback, 증상 매칭 추정)")
            else:
                # 3) 위키피디아까지 실패하면 PubMed로 마지막 시도 (건강정보포털/
                # 위키피디아 둘 다 없는 병명이라도, PubMed는 영어 의학 문헌
                # 전체를 대상으로 하므로 찾을 가능성이 있음 - "주사피부염"처럼
                # 국내 자료엔 드물지만 해외 연구는 많은 경우가 실제로 있었음).
                print("  [파이프라인] 위키피디아도 실패 -> PubMed로 마지막 시도")
                fallback_keyword = known_term or clean_disease_query(user_question)
                pubmed_result = search_pubmed_deep(fallback_keyword)
                if _is_failure_message(pubmed_result):
                    messages.pop()
                    state.turn -= 1
                    return (
                        "이 질문은 제가 다루는 건강/의료 정보 범위를 벗어났거나, "
                        "제가 아는 정보로는 답변하기 어려운 내용인 것 같아요. "
                        "증상이나 병명을 조금 더 구체적으로 말씀해주시겠어요?"
                    )
                parts.append(pubmed_result)
                state.record_symptom(user_question, f"search_pubmed_deep(fallback, 추정: {fallback_keyword})")
                state.record_disease(fallback_keyword, "search_pubmed_deep(fallback)")
        else:
            print("  [파이프라인] 건강포털 RAG 성공")
            parts.append(rag_result)
            if known_term:
                state.record_disease(known_term, "search_symptom_info")
            else:
                # 정확한 병명을 직접 말한 게 아니라 증상을 서술해서 의미 검색으로
                # 매칭된 경우다. 사용자가 실제로 한 말(증상 서술)은 "증상"으로,
                # 검색이 찾아낸 가장 비슷한 병명은 "추정 질환"으로 따로 기록한다 -
                # 매칭된 병명만 확정 진단처럼 남기면 오해의 소지가 있고, 서술 자체를
                # 버리면 "증상을 묻고 꼬리를 물어 문진한다"는 기능 자체가 안 남는다.
                top_match = search_kdca(search_text, top_k=1)
                if top_match:
                    matched_name = top_match[0][0]
                    state.record_symptom(user_question, f"search_symptom_info(추정: {matched_name})")
                    state.record_disease(matched_name, "search_symptom_info(증상 매칭 추정)")

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
    except requests.exceptions.RequestException as e:
        # 이번 턴에서 건드린 것(방금 넣은 user 메시지, turn 증가)을 되돌려서,
        # 다음 질문이 "실패한 턴"의 영향을 받지 않고 깨끗하게 이어지도록 한다.
        messages.pop()
        state.turn -= 1
        print(f"  [오류] 외부 서비스 호출 실패: {type(e).__name__}: {e}")
        return (
            "지금 Ollama나 외부 정보 서비스(위키피디아/KDCA/PubMed) 중 하나에 연결하지 "
            "못했습니다. 인터넷 연결과 Ollama 실행 상태(`ollama serve`)를 확인하고 "
            "다시 질문해주세요."
        )

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
        try:
            print(f"\n답변> {run_agent_turn(messages, question, state)}")
        except Exception as e:
            # run_agent_turn 안에서 이미 requests 예외는 잡아서 롤백까지 하지만,
            # 예상 못 한 다른 버그(예: 코드 오류)까지 여기서 한 번 더 막아서
            # 세션 전체(지금까지의 멀티턴 기록)가 죽는 것만은 방지한다.
            print(f"  [오류] 예상치 못한 문제: {type(e).__name__}: {e}")
            print("답변> 죄송해요, 방금 질문 처리 중 문제가 생겼어요. 다시 한번 질문해주시겠어요?")


if __name__ == "__main__":
    main()
