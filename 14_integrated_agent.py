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
CONTEXT_SCORE_MARGIN = 0.06
# 참고자료로 넣을 문서를 1등 점수에서 이만큼 안쪽으로만 제한한다(위 search_symptom_info
# 참고). 실측 기준: 소화불량 0.68 / 대사증후군 0.56 - 0.12 차이면 다른 주제로 보는 게 맞고,
# 문진 경로의 소화불량 0.578 / 복통 0.565처럼 0.013 차이면 둘 다 관련 있는 자료로 본다.

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

# MedlinePlus(미국 NIH 소비자 건강정보). 키 없이 쓸 수 있고, 애초에 건강 주제만
# 담긴 DB라서 위키피디아처럼 "만화/인물 문서가 걸리는" 문제가 구조적으로 없다.
MEDLINEPLUS_SEARCH_URL = "https://wsearch.nlm.nih.gov/ws/query"
MIN_SIMILARITY_MEDLINEPLUS = 0.5


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
# 검증한다. 완전히 무관한 오검색은 0.17~0.28이라 0.4 밑이면 그냥 버린다.

CONFIDENT_SIMILARITY_WIKIPEDIA = 0.6
# MIN_SIMILARITY_WIKIPEDIA만으론 부족한 사례를 발견함: "배가아파"처럼 짧고
# 막연한 증상 질문은 KDCA가 실패한 뒤 위키피디아 폴백에서 "배"라는 공통 어근
# 때문에 "배가로근"(복부 근육 해부학) 같은 일반 신체부위 문서가 0.49~0.59
# 정도로 그럴듯하게 매칭됨(2026-10-01) - 완전 무관하진 않지만("배"는 맞으니)
# "왜 아픈지"에는 답이 안 됨. 이걸 문턱값만 더 올려서 거르려 하면 "주사
# (질병)"(0.66) 같은 진짜 좋은 매칭까지 같이 걸러질 위험이 있음. 그래서
# "검색을 포기하진 않되, 확신에 찬 척도 안 하기"로 절충: 0.4~0.6 사이(애매한
# 매칭)는 _hedge_wikipedia_answer()로 LLM이 찾은 내용을 반영해서 자연스럽게
# 되묻게 하고, 0.6 이상(확실한 매칭)만 원문 그대로 보여준다.


def _wikipedia_candidate_titles(query: str) -> list[str]:
    """srsearch 1위 + prefixsearch(제목이 쿼리로 시작하는 문서들)로 후보를
    모은다. "주사피부염" 같은 합성어는 srsearch만으론 관련 문서를 못 찾는 걸
    확인함 - 흔한 뜻("주사"=주사기)이 검색 순위에서 이겨버려서, 특이한 뜻의
    문서(실제로 로사시아는 "주사 (질병)"라는 제목으로 존재함)는 안 뜸.
    제목이 쿼리의 앞부분으로 시작하는 문서까지 후보로 넓혀서, 아래에서
    임베딩으로 재선별한다(2026-09-24에 "주사 (질병)" 사례로 발견)."""
    titles: list[str] = []

    def _add(new_titles):
        for t in new_titles:
            if t not in titles:
                titles.append(t)

    search_resp = requests.get(
        "https://ko.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1},
        headers=WIKI_HEADERS, timeout=10,
    )
    _add(r["title"] for r in search_resp.json()["query"]["search"])

    prefixes = {query}
    if len(query) > 2:
        prefixes.add(query[:2])
    if len(query) > 4:
        prefixes.add(query[: len(query) // 2])

    for prefix in prefixes:
        prefix_resp = requests.get(
            "https://ko.wikipedia.org/w/api.php",
            params={"action": "query", "list": "prefixsearch", "pssearch": prefix, "format": "json", "pslimit": 8},
            headers=WIKI_HEADERS, timeout=10,
        )
        _add(r["title"] for r in prefix_resp.json()["query"]["prefixsearch"])

    return titles[:10]


def _wikipedia_page_info(title: str) -> dict | None:
    """본문 요약과 영어 langlink(있으면)를 한 번에 가져온다. langlink는 사람이
    큐레이션한 언어 간 매핑이라, PubMed용 영문 번역에 LLM 추측보다 더 믿을 만함
    (예: "주사 (질병)" -> 영어 langlink가 정확히 "Rosacea")."""
    resp = requests.get(
        "https://ko.wikipedia.org/w/api.php",
        params={"action": "query", "prop": "extracts|langlinks", "exintro": True, "explaintext": True,
                "lllang": "en", "titles": title, "format": "json"},
        headers=WIKI_HEADERS, timeout=10,
    )
    page = next(iter(resp.json()["query"]["pages"].values()))
    if "missing" in page:
        return None
    extract = page.get("extract", "").strip()
    if not extract:
        return None
    langlinks = page.get("langlinks")
    en_title = langlinks[0]["*"] if langlinks else None
    return {"title": page.get("title", title), "extract": extract, "en_title": en_title}


def _best_wikipedia_match(query: str) -> dict | None:
    """query에 대한 후보 문서들(_wikipedia_candidate_titles)을 모아서, 우리
    자신의 임베딩으로 실제로 가장 관련 있어 보이는 문서 하나를 고른다. 1위
    후보도 관련성이 임계값 미만이면 None(= 못 찾음으로 처리)."""
    candidates = _wikipedia_candidate_titles(query)
    if not candidates:
        return None
    query_vec = embed(query)
    best, best_score = None, -1.0
    for title in candidates:
        info = _wikipedia_page_info(title)
        if not info:
            continue
        score = cosine_similarity(query_vec, embed(info["extract"][:1000]))
        if score > best_score:
            best, best_score = info, score
    if best is None or best_score < MIN_SIMILARITY_WIKIPEDIA:
        if best is not None:
            print(f"    [경고] 위키피디아 후보 중 가장 가까운 '{best['title']}'도 무관해 보임"
                  f"(유사도 {best_score:.2f}) - 버림")
        return None
    best["score"] = best_score
    return best


WIKI_HEDGE_SYSTEM_PROMPT = (
    "당신은 건강 정보를 챙겨주는 친절한 도우미입니다.\n"
    "아래 [위키피디아 내용]은 검색은 됐지만, 사용자 질문과 정확히 같은 주제가 "
    "아닐 가능성이 높습니다(단어 일부만 겹쳐서 찾아진 것일 수 있음).\n"
    "**위키피디아 내용을 설명하거나 요약하지 마세요.** 구체적인 답을 드리기엔 "
    "지금 정보가 부족하다는 점을 한 문장 정도로 짧게만 알리고, 바로 증상을 더"
    "구체적으로 설명해달라고 자연스럽게 되물으세요(언제부터, 어떤 느낌으로, "
    "다른 증상은 없는지 등). 전체 답변을 2~3문장 이내로 짧게 유지하세요.\n"
    "**절대로 위키피디아 내용을 사용자 증상의 원인이라고 추측하거나 연결짓지 "
    "마세요** (예: '이 근육 때문에 아프신 것 같다' 같은 표현 금지). 참고자료에 "
    "없는 인과관계는 지어내면 안 됩니다.\n"
    "질문 문장을 그대로 반복하지 말고, 간결하게 답하세요."
)


def _hedge_wikipedia_answer(query: str, title: str, extract: str) -> str:
    """매칭이 애매한(관련은 있어 보이지만 확신할 정도는 아닌) 위키피디아 결과를
    그대로 보여주지 않고, LLM이 "이게 정확히 맞는지는 모르겠다"는 걸 알리면서
    찾은 내용에 나온 용어로 자연스럽게 되묻게 한다. 고정된 문구("더 구체적으로
    말씀해주세요")로 틀에 박히게 되묻는 대신, 실제로 찾은 내용과 연결된 질문을
    하도록 하기 위함."""
    messages = [
        {"role": "system", "content": WIKI_HEDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"[사용자 질문]\n{query}\n\n[위키피디아 - {title}]\n{extract[:1000]}"},
    ]
    message = call_with_retry(messages, model=RAG_MODEL, num_predict=512)
    if message is None:
        # LLM 생성도 실패하면 안전하게 원문 + 정적 안내로 대체
        return (
            f"[위키피디아 - {title}]\n{extract[:1000]}\n\n"
            "(참고: 정확히 어떤 증상을 말씀하시는지와는 다를 수 있어요. "
            "조금 더 구체적으로 말씀해주시겠어요?)"
        )
    return message["content"]


MIN_MEDICAL_RELEVANCE = 0.5
# 위키피디아는 의료 백과사전이 아니라서, 질문과 단어가 겹치기만 하면 만화 제목
# ("배가본드")이나 인물("나혜석")처럼 의료와 전혀 상관없는 문서도 올라온다.
# "질문과 관련 있는가"(MIN_SIMILARITY_WIKIPEDIA)와 "애초에 의료/건강 내용인가"는
# 다른 질문이라, 후자를 따로 검사한다 - 이미 갖고 있는 KDCA 질환 코퍼스(624개)와
# 비교해서 그 중 하나라도 비슷하면 의료 영역 내용으로 본다.
# 실측(2026-10-01): 의료 아님 - 배가본드(만화) 0.371, 나혜석(인물) 0.387,
# 배가사리(어류) 0.465 / 의료 - 주사(질병) 0.612, 쇼그렌증후군 0.662,
# 피부염 0.816. 그 사이가 뚜렷하게 비어서 0.5로 잡음(해부학 문서는 배가로근
# 0.541, 머리 0.594로 통과하지만, 이건 질문 관련도가 낮아서 어차피 되묻기
# 경로로 빠지므로 원문이 그대로 노출되진 않음).


def _is_medical_content(text: str) -> bool:
    """이 텍스트가 의료/건강 영역 내용인지 KDCA 질환 코퍼스와 비교해서 판단."""
    vec = embed(text[:1000])
    best = max(cosine_similarity(vec, entry["embedding"]) for entry in KDCA_CORPUS.values())
    return best >= MIN_MEDICAL_RELEVANCE


WIKI_SUMMARY_SYSTEM_PROMPT = (
    "당신은 위키피디아 자료를 바탕으로 답하는 친절한 건강정보 도우미입니다.\n"
    "반드시 아래 [참고자료]에 있는 내용만 근거로, 사용자 질문에 맞게 요약해서 "
    "설명하세요. 참고자료에 없는 내용은 절대 지어내지 마세요.\n"
    "백과사전 문장을 그대로 옮기지 말고, 사용자가 읽기 쉬운 말로 풀어서 "
    "정리하세요(어려운 용어는 괄호로 짧게 풀어주면 좋습니다).\n"
    "구체적인 약물 이름은 나열하지 말고, 필요하면 '약물치료' 정도로만 언급하세요.\n"
    "이건 진단이 아니라 참고 정보이므로 단정하지 말고, 증상이 지속되면 병원 진료를 "
    "권하세요.\n"
    "답변 끝에는 증상을 좁히는 데 도움될 후속 질문을 하나 자연스럽게 덧붙이세요.\n"
    "질문 문장을 그대로 되풀이하지 말고, 바로 본론부터 4~6문장 정도로 답하세요."
)


def _summarize_wikipedia_answer(query: str, title: str, extract: str) -> str:
    """위키피디아 원문을 그대로 쏟아내지 않고, 질문에 맞게 요약해서 답한다.
    KDCA RAG(search_symptom_info)는 이미 이렇게 동작하는데 위키피디아 폴백만
    원문을 그대로 반환하고 있었어서, 같은 방식으로 맞춘 것. 요약 실패 시에는
    안전하게 원문을 그대로 돌려준다(정보를 아예 잃지 않도록)."""
    messages = [
        {"role": "system", "content": WIKI_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": f"[참고자료 - {title}]\n{extract[:1500]}\n\n[질문]\n{query}"},
    ]
    message = call_with_retry(messages, model=RAG_MODEL, num_predict=1024)
    if message is None:
        print("    [경고] 위키피디아 요약 실패 - 원문 그대로 사용")
        return f"[위키피디아 - {title}]\n{extract[:1000]}"
    return f"{message['content']}\n\n(출처: 위키피디아 '{title}')"


def search_wikipedia(query: str) -> str:
    query = clean_disease_query(query)
    match = _best_wikipedia_match(query)
    if match is None:
        return f"'{query}'에 대한 위키피디아 검색 결과가 없습니다."
    if not _is_medical_content(match["extract"]):
        print(f"    [경고] 위키피디아 결과 '{match['title']}'가 의료/건강 내용이 아님 - 버림")
        return f"'{query}'에 대한 위키피디아 검색 결과가 없습니다."
    if match["score"] < CONFIDENT_SIMILARITY_WIKIPEDIA:
        print(f"    [경고] 위키피디아 매칭이 애매함('{match['title']}', 유사도 {match['score']:.2f}) - 되묻는 답으로 전환")
        return _hedge_wikipedia_answer(query, match["title"], match["extract"])
    return _summarize_wikipedia_answer(query, match["title"], match["extract"])


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


def search_symptom_info(symptom_or_keyword: str, question_text: str | None = None,
                        min_similarity: float | None = None) -> str:
    """증상 문장이나 병명을 자유롭게 받아서, 관련 질환을 의미 기반으로 찾아 답한다.
    search_text(=symptom_or_keyword)는 검색(임베딩 매칭) 전용이고, question_text는
    LLM에게 보여줄 [질문] 부분 전용이다 - 분리하는 이유는 아래 참고.

    min_similarity를 따로 넘길 수 있게 둔 이유: 문진으로 정보를 모으고 검색어를
    정규화한 뒤의 검색은 근거가 더 탄탄해서(실측: 구어체 원문은 '파라티푸스'
    0.52로 엉뚱하게 매칭되지만, 정규화한 '상복부 통증 식후 악화'는 소화불량
    0.578/복통 0.565로 적절하게 매칭됨) 기본 기준(0.6)보다 조금 낮춰도 안전하다."""
    question_text = question_text or symptom_or_keyword
    threshold = MIN_SIMILARITY_SYMPTOM if min_similarity is None else min_similarity

    results = search_kdca(symptom_or_keyword, top_k=5)
    if not results or results[0][1] < threshold:
        return "관련된 건강정보를 찾지 못해 답변할 수 없습니다. 증상을 좀 더 구체적으로 말씀해주세요."

    print("    [검색됨]", ", ".join(f"{n}({s:.2f})" for n, s, _ in results[:3]))
    # 1등과 점수가 많이 떨어지는 항목은 참고자료에서 뺀다. 예전엔 무조건 상위 3개를
    # 넣었는데, "소화불량"(0.68)과 함께 "대사증후군"(0.56)·"보툴리눔독소증"(0.54)까지
    # 들어가서 LLM이 "소화불량의 원인이 대사증후군, 보툴리눔독소증과 관련 있을 수
    # 있다"는 식으로 없는 인과관계를 만들어내는 일이 실제로 발생함(2026-10-01).
    top_score = results[0][1]
    selected = [r for r in results[:3] if r[1] >= max(threshold, top_score - CONTEXT_SCORE_MARGIN)]
    if len(selected) < len(results[:3]):
        dropped = [f"{n}({s:.2f})" for n, s, _ in results[:3] if (n, s) not in [(sn, ss) for sn, ss, _ in selected]]
        print(f"    [참고자료 제외] 1등과 차이가 커서 제외: {', '.join(dropped)}")
    context = "\n\n".join(
        f"[{name}]\n{truncate_at_sentence(entry['text'], 800)}" for name, score, entry in selected
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
# 도구 5: MedlinePlus(미국 NIH) RAG - "직접 답변" 도구
#
# 설계 원칙(사용자 지시): **번역은 '병명 키워드 매핑'부터.** 자유 번역에 맡기면
# "주사"를 injection으로 오역하는 식의 사고가 나므로, _to_english_query()의
# 매핑 순서(HIRA 공식 영문명 -> 위키피디아 언어간 링크 -> 그래도 없으면 LLM 추정)를
# 그대로 쓴다. 그리고 MedlinePlus가 문서마다 주는 title/altTitle(= 주제 태그)에
# 그 병명이 직접 걸리는지를 1순위 근거로 보고, 임베딩 유사도는 보조로만 쓴다.
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """MedlinePlus 응답 본문에는 <p>, <ul> 같은 태그와 검색어 하이라이트
    <span class="qt0">가 그대로 들어있어서 걷어낸다."""
    return re.sub(r"\s+", " ", _HTML_TAG_RE.sub(" ", text or "")).strip()


def _medlineplus_search(english_term: str, retmax: int = 5) -> list[dict]:
    response = requests.get(
        MEDLINEPLUS_SEARCH_URL,
        params={"db": "healthTopics", "term": english_term, "retmax": retmax},
        headers=WIKI_HEADERS, timeout=10,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    docs = []
    for doc in root.findall(".//document"):
        title, summary, alt_titles = "", "", []
        for content in doc.findall("content"):
            value = _strip_html("".join(content.itertext()))
            name = content.get("name")
            if name == "title":
                title = value
            elif name == "altTitle":
                alt_titles.append(value)
            elif name == "FullSummary":
                summary = value
        if title and summary:
            docs.append({"title": title, "alt_titles": alt_titles,
                         "summary": summary, "url": doc.get("url", "")})
    return docs


def _tag_match_kind(english_term: str, doc: dict) -> str | None:
    """병명이 문서의 주제 태그에 어떻게 걸렸는지 구분해서 돌려준다.

    - "title": 문서 제목 자체가 그 병명 (예: 'rosacea' -> "Rosacea") = 가장 확실
    - "alt": 제목은 더 넓은 주제인데 altTitle로만 걸림
      (예: 'gastritis' -> "Stomach Disorders"의 altTitle) = 그 병을 '포함하는'
      일반 문서라는 뜻이므로, 이걸로 그 병을 구체적으로 설명하게 하면 LLM이
      없는 내용을 지어낸다(실제 발생: 위염의 증상/원인/수술 여부를 날조,
      2026-10-02). 그래서 이 경우 프롬프트에 "넓은 주제의 일반 설명"이라고
      명시해서 날조를 막는다.
    - None: 태그에 안 걸림(임베딩 유사도로만 판단)
    """
    term = english_term.strip().lower()
    if len(term) < 4:
        return None
    title = doc["title"].lower()
    if title and (term == title or term in title or title in term):
        return "title"
    for alt in doc["alt_titles"]:
        a = alt.lower()
        if a and (term == a or term in a or a in term):
            return "alt"
    return None


MEDLINEPLUS_SUMMARY_SYSTEM_PROMPT = (
    "당신은 미국 NIH MedlinePlus의 영문 건강정보를 한국어로 정리해주는 "
    "친절한 건강정보 도우미입니다.\n"
    "반드시 아래 [참고자료]에 있는 내용만 근거로, 한국어로 번역하면서 사용자 "
    "질문에 맞게 요약하세요. 참고자료에 없는 내용은 절대 지어내지 마세요.\n"
    "영어 단어를 그대로 두지 말고 자연스러운 한국어로 옮기세요(필요하면 의학 "
    "용어 뒤에 괄호로 짧은 설명을 덧붙이세요).\n"
    "구체적인 약물 이름은 나열하지 말고, 필요하면 '약물치료' 정도로만 언급하세요.\n"
    "이건 진단이 아니라 참고 정보이므로 단정하지 말고, 증상이 지속되면 병원 진료를 "
    "권하세요.\n"
    "질문 문장을 그대로 되풀이하지 말고, 바로 본론부터 4~6문장 정도로 답하세요."
)


def _summarize_medlineplus(keyword: str, doc: dict, match_kind: str | None) -> str:
    caveat = ""
    if match_kind != "title":
        # 제목이 질문한 병명과 다르면(더 넓은 주제 문서면) 그 사실을 LLM에게
        # 분명히 알려서, 자료에 없는 내용을 그 병명의 증상/원인/치료라고
        # 지어내는 것을 막는다.
        caveat = (
            f"\n\n[주의] 이 자료는 '{doc['title']}'라는 더 넓은 주제의 일반 설명이며, "
            f"'{keyword}'만을 다룬 자료가 아닙니다. 자료에 적혀 있지 않은 내용을 "
            f"'{keyword}'의 증상·원인·치료라고 쓰지 마세요. 자료에 담긴 일반적인 "
            f"내용만 전달하고, '{keyword}'에 대한 구체적인 설명이 자료에 없으면 "
            "그 사실을 솔직히 밝히세요."
        )
    messages = [
        {"role": "system", "content": MEDLINEPLUS_SUMMARY_SYSTEM_PROMPT},
        {"role": "user",
         "content": f"[참고자료 - {doc['title']}]\n{doc['summary'][:1500]}\n\n[질문]\n{keyword}{caveat}"},
    ]
    message = call_with_retry(messages, model=RAG_MODEL, num_predict=1024)
    if message is None:
        return "[오류] 모델이 계속 비정상적인 응답을 내서 포기했습니다."
    source = f"(출처: MedlinePlus '{doc['title']}'"
    source += f" - {doc['url']})" if doc["url"] else ")"
    return f"{message['content']}\n\n{source}"


def search_medlineplus(keyword: str) -> str:
    """MedlinePlus(NIH)에서 병명으로 건강정보를 찾아 한국어로 요약해 답한다."""
    keyword = clean_disease_query(keyword)
    english_term = _to_english_query(keyword)
    if re.search(r"[가-힣]", english_term):
        # 영문 매핑에 실패하면(한글이 그대로 남으면) 영문 DB 검색이 무의미하다
        return f"'{keyword}'의 영문 병명을 확인하지 못해 MedlinePlus 검색 결과가 없습니다."

    docs = _medlineplus_search(english_term)
    if not docs and " " in english_term:
        # HIRA 공식 영문명은 "Gastritis and duodenitis"처럼 여러 단어인 경우가 있고,
        # 그 구문 전체로는 MedlinePlus에서 0건이 나온다(실측). 핵심 단어 하나로
        # 다시 찾아본다("Gastritis" -> Stomach Disorders 문서가 정상 매칭됨).
        head_term = english_term.split()[0]
        print(f"    [MedlinePlus] '{english_term}' 0건 -> 핵심 단어 '{head_term}'로 재검색")
        docs = _medlineplus_search(head_term)
        if docs:
            english_term = head_term
    if not docs:
        return f"'{keyword}'({english_term})에 대한 MedlinePlus 검색 결과가 없습니다."

    # 1순위: 제목이 곧 그 병명인 문서, 2순위: altTitle로 걸린(더 넓은 주제) 문서,
    # 3순위: 태그에 안 걸려서 임베딩 유사도로만 보는 문서
    kinds = {id(d): _tag_match_kind(english_term, d) for d in docs}
    by_title = [d for d in docs if kinds[id(d)] == "title"]
    by_alt = [d for d in docs if kinds[id(d)] == "alt"]
    pool = by_title or by_alt or docs
    query_vec = embed(english_term)
    scored = [(d, cosine_similarity(query_vec, embed(d["summary"][:1000]))) for d in pool]
    scored.sort(key=lambda x: x[1], reverse=True)
    best, best_score = scored[0]
    match_kind = kinds[id(best)]

    print(f"    [MedlinePlus] '{english_term}' -> '{best['title']}' "
          f"(태그={match_kind or '없음'}, 유사도={best_score:.2f})")

    # 제목이 곧 그 병명이면 그 자체로 강한 근거이므로 유사도 기준을 적용하지 않는다.
    # 그 외(더 넓은 주제이거나 태그에 안 걸린 경우)는 유사도 기준을 지켜야 한다.
    if match_kind != "title" and best_score < MIN_SIMILARITY_MEDLINEPLUS:
        return f"'{keyword}'({english_term})에 대한 MedlinePlus 검색 결과가 없습니다."

    return _summarize_medlineplus(keyword, best, match_kind)


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
    # HIRA 공식 DB에 없으면, 다음으로 위키피디아의 사람이 큐레이션한 언어 간
    # 링크를 시도한다 - LLM 번역은 "주사"(주사기 vs 로사시아 옛 한자어) 같은
    # 중의적 단어를 흔한 뜻으로 오역하는 걸 실제로 확인했음(2026-09-24,
    # "주사피부염" -> "Intramuscular panniculitis"로 완전히 틀리게 번역됨).
    # 위키피디아 langlink는 "주사 (질병)" -> "Rosacea"로 정확함.
    match = _best_wikipedia_match(keyword)
    if match and match.get("en_title"):
        return match["en_title"]
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
        # 문진(問診) 상태: 증상이 막연해서 되물었을 때, 원래 호소와 이후 받은
        # 답변들을 모아둔다. 이게 없으면 "배가 살살 아파" -> (되묻기) -> "저녁부터"
        # 라고 답했을 때 "저녁부터"를 새로운 증상 검색어로 취급해서 "못 찾았다"고
        # 답하는 버그가 생김(2026-10-01 실제 발생). 사용자의 답변은 증상이 아니라
        # 시간/빈도/정도일 수 있으므로, 독립 질의가 아니라 원래 호소의 '추가 정보'로
        # 누적해서 다뤄야 한다.
        self.pending_complaint: str | None = None
        self.followup_answers: list[str] = []

    def start_interview(self, complaint: str):
        self.pending_complaint = complaint
        self.followup_answers = []

    def add_interview_answer(self, answer: str):
        self.followup_answers.append(answer)

    def interview_description(self) -> str:
        """원래 증상 호소 + 지금까지 받은 답변들을 합친 설명."""
        parts = [self.pending_complaint or ""] + self.followup_answers
        return " ".join(p for p in parts if p).strip()

    def end_interview(self):
        self.pending_complaint = None
        self.followup_answers = []

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
            if lines:
                lines.append("")
            lines.append("[언급된 증상]")
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


def resolve_query_with_context(user_question: str, state: MedicalConversationState,
                               intent: dict | None = None):
    """이번 질문에서 병명을 직접 찾고, 지칭어(그럼/그거 등)가 있을 때만
    state.last_topic을 힌트로 활용한다. LLM 없이 순수 문자열 처리로
    '그럼 치료법은?' 같은 질문이 이전 주제를 잃지 않게 한다.

    검색용 텍스트(search_text)와 LLM에게 보여줄 질문(question_text)을 분리해서
    반환한다 - 후속 대답("네 심해요")을 검색어에까지 섞어버리면, "심해요" 같은
    말이 임베딩을 엉뚱한 문서(예: 응급 기도폐쇄 처치)로 끌고 가는 걸 실제로
    확인했음. 검색은 이전 주제 하나로만 안전하게 하고, 그 대답 내용은 LLM한테
    질문으로만 보여줘서 답변에 반영되게 한다.

    intent(판단 레이어 결과)가 있으면 추가 신호로 쓴다:
      - intent가 "followup"이면 지칭어가 없어도 이전 주제를 이어받는다
      - 원문에 사전 병명이 그대로 없어도(예: "당뇨" -> 사전엔 "당뇨병"), 판단
        레이어가 정리한 병명이 사전에 있으면 known_term으로 인정한다.
        사전에 있는지는 여기서 코드로 다시 확인하므로, 모델이 병명을 지어내도
        사전에 없는 이름이면 known_term이 되지 않는다.

    반환값: (search_text, question_text, known_term)"""
    known_term = find_known_term_in_text(user_question)
    if known_term:
        return user_question, user_question, known_term
    is_followup = (
        any(marker in user_question for marker in FOLLOWUP_MARKERS)
        or state.awaiting_followup_reply
        or len(user_question.strip()) <= SHORT_REPLY_MAX_CHARS
        or (intent is not None and intent["intent"] == "followup")
    )
    if state.last_topic and is_followup:
        implied_term = state.last_topic if state.last_topic in KDCA_CORPUS else None
        combined_question = f"{state.last_topic} {user_question}"
        return state.last_topic, combined_question, implied_term
    if (intent is not None and intent["intent"] == "disease_info"
            and intent["disease_name"] in KDCA_CORPUS):
        return user_question, user_question, intent["disease_name"]
    return user_question, user_question, None


# 증상을 호소하는 표현들. 이 표현이 들어있으면 "병명"이 아니라 "증상 서술"로 본다.
# 왜 필요한가: clean_disease_query는 조사를 떼어내다 보니 "속이 매스꺼워" -> "속",
# "배가 살살 아파" -> "배"처럼 1글자 조각을 만들어내는데, 그 조각으로 위키피디아를
# 검색하면 "속옷", "배(과일/선박)" 같은 전혀 다른 문서가 나오고, 심지어 그걸
# 길게 요약해서 답하는 일이 실제로 발생함(2026-10-01 실측). 위키피디아/PubMed는
# "정확한 병명"으로 찾을 때만 쓸모가 있는 도구라서, 병명이 아닌 질의로는 아예
# 호출하지 않고 대신 증상을 더 물어본다.
_SYMPTOM_COMPLAINT_RE = re.compile(
    "아프|아파|쑤시|쑤셔|결리|결려|뻐근|저리|저려|메스꺼|매스꺼|메슥|울렁|구역|토할|"
    "띵|어지럽|어지러|답답|거북|쓰리|쓰려|쓰림|간지럽|가렵|부었|부어|따갑|화끈|"
    "열나|열이 나|불편해|이상해|힘들어|피곤해"
)
MIN_NAME_CANDIDATE_CHARS = 3


def is_name_like_query(user_question: str, candidate: str) -> bool:
    """위키피디아/PubMed(정확한 병명으로 찾는 도구)에 넘길 만한 질의인지 판단.
    1) 병명 후보가 너무 짧으면(2글자 이하) 조사만 떼어낸 조각일 가능성이 높고,
    2) 증상 호소 표현이 들어있으면 병명이 아니라 증상 서술이므로 둘 다 거른다."""
    if len(candidate.strip()) < MIN_NAME_CANDIDATE_CHARS:
        return False
    return _SYMPTOM_COMPLAINT_RE.search(user_question) is None


CLARIFY_SYSTEM_PROMPT = (
    "당신은 환자의 증상을 차근차근 물어보는 친절한 건강 상담 도우미입니다.\n"
    "사용자가 증상을 아주 짧게만 말해서, 아직 어떤 질환인지 추측할 정보가 "
    "부족한 상황입니다.\n"
    "**절대 질환명을 추측하거나 원인을 단정하지 마세요.** 아는 척하지 말고, "
    "자료에 없는 설명을 덧붙이지도 마세요.\n"
    "[지금까지 들은 내용]에 이미 나온 것은 다시 묻지 말고, 아직 모르는 것만 "
    "2가지 이내로 구체적으로 물어보세요(그 증상에 실제로 맞는 질문이어야 합니다 - "
    "예: 언제부터인지, 어떤 느낌/정도인지, 같이 나타나는 다른 증상, 식사·수면·"
    "활동과의 관계 등).\n"
    "전체 3문장 이내로 짧고 따뜻하게 답하세요."
)


def ask_for_more_detail(complaint: str, answers: list[str] | None = None) -> str:
    """병명이 아니라 짧은 증상 호소일 때, 엉뚱한 검색 결과를 들이대는 대신
    증상을 더 물어본다. 고정 문구가 아니라 LLM이 그 증상에 맞는 질문을 하도록 하고,
    이미 들은 내용(answers)은 또 묻지 않도록 함께 넘긴다."""
    heard = "\n".join(f"- {a}" for a in (answers or [])) or "- (아직 없음)"
    messages = [
        {"role": "system", "content": CLARIFY_SYSTEM_PROMPT},
        {"role": "user", "content": f"[증상 호소]\n{complaint}\n\n[지금까지 들은 내용]\n{heard}"},
    ]
    message = call_with_retry(messages, model=RAG_MODEL, num_predict=400)
    if message is None:
        return (
            "증상을 조금 더 자세히 알려주시면 관련 정보를 찾아드릴 수 있어요. "
            "언제부터 그러셨는지, 어떤 느낌인지, 같이 나타나는 다른 증상은 없는지 "
            "말씀해주시겠어요?"
        )
    return message["content"]


SYMPTOM_NORMALIZE_SYSTEM_PROMPT = (
    "당신은 환자가 구어체로 말한 증상 설명을, 의학 자료 검색에 쓸 검색어로 "
    "바꿔주는 도우미입니다.\n"
    "주어진 설명에 **실제로 적혀 있는** 증상/부위/양상만 사용해서, 의학 자료에 "
    "쓰이는 표현의 명사 위주 검색어로 다시 쓰세요.\n"
    "**설명에 없는 말은 절대 추가하지 마세요.** 특히 시간·상황·악화요인(식후, "
    "야간, 운동 시 등)은 설명에 그 말이 실제로 있을 때만 쓰세요. 질환명을 "
    "추측해서 넣는 것도 금지입니다.\n"
    "아래 예시는 '바꾸는 방식'만 참고하고, 예시에 나온 단어를 가져다 쓰지 마세요.\n"
    "예: '목이 칼칼하고 기침이 나' -> '인후 통증 기침'\n"
    "설명이나 따옴표 없이, 검색어만 한 줄로 출력하세요."
)


def normalize_symptom_query(description: str) -> str:
    """구어체 증상 설명을 KDCA 자료에 가까운 검색어로 바꾼다. KDCA 코퍼스는
    의학 문서 문체라서 "배가 살살 아파 저녁부터" 같은 구어체와는 임베딩 거리가
    멀다 - 문진으로 정보를 모아도 그대로는 검색이 안 맞아서, 검색 직전에 한 번
    표현만 정규화한다(질환명 추측은 금지해서 할루시네이션 여지를 줄임).
    실패하면 원문을 그대로 쓴다."""
    messages = [
        {"role": "system", "content": SYMPTOM_NORMALIZE_SYSTEM_PROMPT},
        {"role": "user", "content": description},
    ]
    message = call_with_retry(messages, model=RAG_MODEL, num_predict=120)
    if message is None:
        return description
    normalized = message["content"].strip().strip("\"'").splitlines()[0].strip()
    return normalized or description


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


MAX_INTERVIEW_ROUNDS = 2
# 문진으로 몇 번까지 더 물어볼지. 계속 되묻기만 하면 사용자가 지치므로, 2번까지
# 물어본 뒤에는 모은 내용으로 검색해보고 안 되면 솔직하게 마무리한다.

MIN_SIMILARITY_INTERVIEW = 0.55
# 문진으로 모은 정보 + 검색어 정규화를 거친 뒤의 검색에 쓰는 기준(기본 0.6보다 낮음).
# 근거: 구어체 원문으로 검색하면 엉뚱한 게 1등이 되지만(파라티푸스 0.52),
# 정규화를 거치면 의학적으로 적절한 항목이 1등이 됨(소화불량 0.578, 복통 0.565).
# 즉 이 경로는 입력 품질이 더 좋아서 조금 낮은 점수도 신뢰할 만하다. 대신 답변은
# SYMPTOM_RAG_SYSTEM_PROMPT 규칙대로 "~일 가능성이 있습니다"로만 말하고 진료를 권한다.


_DISEASE_SUFFIX_RE = re.compile(r"(염|병|증|암|증후군|장애|궤양|결핵|중독|종양)$")


def _looks_like_interview_answer(user_question: str) -> bool:
    """문진 중에 들어온 입력이 "우리 질문에 대한 답변"인지 판단.

    사용자가 병명을 말하면(사전에 있는 병명이거나, 병명 접미사로 끝나는 단어)
    문진을 접고 평소 파이프라인으로 보낸다. 그 외에는 - 시간("저녁부터"),
    빈도, 정도, 상황 설명 등 무엇이든 - 우리 질문에 대한 답변으로 본다.

    처음엔 is_name_like_query로 판단했는데, "저녁부터"가 4글자이고 증상 표현도
    없어서 "병명 같다"로 오판 -> 위키피디아에서 '저녁'을 검색하는 버그가 있었음
    (2026-10-01). 답변은 병명이 아닌 게 정상이므로, "병명인지"만 좁게 보고
    나머지는 전부 답변으로 처리하는 쪽이 맞다."""
    if find_known_term_in_text(user_question):
        return False
    return _DISEASE_SUFFIX_RE.search(clean_disease_query(user_question)) is None


def _record_fallback_result(state: MedicalConversationState, user_question: str,
                            known_term: str | None, asked_unknown_disease: bool,
                            source_topic: str | None, tool_label: str,
                            asked_name: str | None = None):
    """폴백 도구(MedlinePlus/위키피디아/PubMed)로 답했을 때 state에 기록한다.

    예전엔 "known_term이 없으면 사용자가 증상을 서술한 것"이라고 가정해서 질문
    문장을 그대로 '증상'에 넣었는데, 그러면 병명을 물어본 경우에도
    "주사피부염일때 어떻게 해야해?"가 증상 목록에 쌓이고 질환은 영문 제목
    ("Rosacea")으로 남는 오염이 생긴다(2026-10-02 사용자 지적). 지금은
    "코퍼스에 없는 병명을 물어본 경우"를 구분할 수 있으므로 그에 맞게 기록한다."""
    if known_term:
        state.record_disease(known_term, tool_label)
        state.last_topic = known_term
        return
    if asked_unknown_disease:
        # 사용자가 물어본 병명 자체를 질환으로 기록(한국어 그대로), 출처에 어떤
        # 자료에서 찾았는지 남긴다. 증상 목록은 건드리지 않는다.
        asked_name = asked_name or clean_disease_query(user_question)
        label = f"{tool_label}({source_topic})" if source_topic else tool_label
        state.record_disease(asked_name, label)
        state.last_topic = asked_name
        return
    if source_topic:
        # 증상을 서술해서 여기까지 온 경우에만 서술을 '증상'으로 남기고,
        # 매칭된 주제는 '추정 질환'으로 기록한다.
        state.record_symptom(user_question, f"{tool_label}(추정: {source_topic})")
        state.record_disease(source_topic, f"{tool_label}(증상 매칭 추정)")


def _handle_interview_answer(messages: list, user_question: str,
                             state: MedicalConversationState) -> str | None:
    """문진 답변을 원래 호소에 누적하고, 모인 설명으로 다시 검색해본다.
    - 검색 성공 -> 답변하고 문진 종료
    - 실패 & 아직 더 물어볼 여유 있음 -> 아직 안 물어본 걸 추가 질문
    - 실패 & 여유 없음 -> 솔직하게 마무리(병원 권유)
    반환값이 None이면 호출자가 평소 파이프라인을 계속 진행한다."""
    state.add_interview_answer(user_question)
    description = state.interview_description()
    rounds = len(state.followup_answers)
    print(f"  [문진] 답변 누적({rounds}회): {description}")

    normalized = normalize_symptom_query(description)
    print(f"  [문진] 검색어 정규화: {normalized}")
    rag_result = search_symptom_info(normalized, description,
                                     min_similarity=MIN_SIMILARITY_INTERVIEW)

    if not _is_failure_message(rag_result):
        print("  [문진] 모인 정보로 건강포털 RAG 성공 -> 문진 종료")
        answer = (
            f"말씀해주신 내용({description})을 종합해보면 이런 가능성을 참고해보실 "
            f"수 있어요.\n\n{rag_result}"
        )
        state.record_symptom(description, "문진 완료(증상 종합)")
        top_match = search_kdca(normalized, top_k=1)
        if top_match:
            state.record_disease(top_match[0][0], "search_symptom_info(문진 종합 추정)")
            state.last_topic = top_match[0][0]
        state.end_interview()
        state.awaiting_followup_reply = "?" in answer[-200:]
        messages.append({"role": "assistant", "content": answer})
        return answer

    if rounds < MAX_INTERVIEW_ROUNDS:
        print("  [문진] 아직 정보가 부족 -> 추가 질문")
        combined = ask_for_more_detail(state.pending_complaint or description,
                                       state.followup_answers)
        state.awaiting_followup_reply = True
        messages.append({"role": "assistant", "content": combined})
        return combined

    print("  [문진] 충분히 물어봤지만 매칭 실패 -> 솔직하게 마무리")
    combined = (
        f"지금까지 말씀해주신 내용({description})만으로는 제가 가진 자료에서 "
        "어떤 질환인지 좁히기 어려웠어요. 증상이 계속되거나 심해지면 "
        "가까운 병원에서 진료를 받아보시는 게 좋겠습니다. "
        "혹시 짐작되는 병명이 있으시면 그 이름으로 다시 물어봐주셔도 돼요."
    )
    state.record_symptom(description, "문진 종합(질환 특정 실패)")
    state.end_interview()
    state.awaiting_followup_reply = False
    messages.append({"role": "assistant", "content": combined})
    return combined



# ---------------------------------------------------------------------------
# 판단 레이어 (2026-10-02 추가): 질문 이해만 Claude Haiku에게 맡긴다
#
# 지금까지 "이 질문이 뭘 묻는 건지"는 전부 규칙으로 판단했다:
#   병명인지 증상인지 -> _SYMPTOM_COMPLAINT_RE + 3글자 기준
#   문진 답변인지     -> 병명 접미사(염/병/증...) 검사
#   논문을 원하는지   -> "논문/연구" 키워드
#   이전 주제 잇기    -> "그럼/그거" 마커
# 버그를 하나 고칠 때마다 규칙이 하나씩 늘었고("속" -> 속옷, "저녁부터" -> 병명
# 오판), 처음 보는 표현에는 계속 약했다. 그런데 이 규칙들은 결국 "이 사람이
# 뭘 묻는 거지?"라는 한 가지 질문을 쪼개서 답하고 있었다.
#
# 그래서 그 판단 하나만 큰 모델(Haiku)에게 객관식으로 맡긴다. 프로젝트 전체
# 결론("판단은 코드, LLM은 좁은 역할")은 그대로다 - Haiku는 정해진 선택지 중
# 하나를 고르는 좁은 역할만 하고, 검색·답변 생성은 여전히 로컬(bge-m3,
# llama3.1)이 한다. 질문 1번에 Haiku 호출 1번(약 0.4원).
#
# 안전장치:
#   - API 키가 없거나, 패키지가 없거나, 호출이 실패하면 classify_turn()이 None을
#     돌려주고, 그러면 파이프라인은 예전 규칙 그대로 동작한다(앱이 죽지 않음)
#   - 병명은 Haiku가 말했다고 믿지 않고, KDCA 사전에 실제로 있는지 코드로 확인
#   - "논문" 키워드 규칙은 Haiku 판단과 OR로 유지(놓치는 쪽만 보완)
#   - 끄고 싶으면 .env에 USE_INTENT_LAYER=0
#   - 매 판단을 data/intent_log.jsonl에 규칙 판단과 나란히 남긴다 -> 나중에
#     "규칙 vs Haiku" 판단이 어디서 갈리는지 비교 분석할 수 있다
# ---------------------------------------------------------------------------
try:
    import anthropic
except ImportError:
    anthropic = None

INTENT_MODEL = os.environ.get("INTENT_MODEL", "claude-haiku-4-5-20251001")
USE_INTENT_LAYER = os.environ.get("USE_INTENT_LAYER", "1") != "0"
INTENT_LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "intent_log.jsonl")
INTENT_TIMEOUT_SECONDS = 15

INTENT_LABELS = ("symptom", "disease_info", "interview_answer", "followup", "off_topic")

INTENT_SYSTEM_PROMPT = (
    "당신은 한국어 건강 상담 챗봇의 '접수 담당'입니다. 답변은 하지 않고, 사용자의 "
    "이번 입력이 어떤 종류인지만 분류해서 report_intent 도구로 보고합니다.\n\n"
    "[intent 선택지]\n"
    "- symptom: 자기 몸의 증상을 호소하거나 서술함 (예: '속이 매스꺼워', "
    "'머리가 띵하고 어지러워요')\n"
    "- disease_info: 특정 병명/질환에 대해 물어봄 (예: '위염이 뭐야?', "
    "'주사피부염일 때 어떻게 해야 해?', '당뇨 있으면 뭘 조심해?')\n"
    "- interview_answer: [문진 중]인 상황에서, 우리가 던진 질문에 답하거나 같은 "
    "증상에 대한 정보를 덧붙임 (시간 '저녁부터', 빈도, 정도, 악화 요인, 동반 "
    "증상 등). 문진 중이 아니면 절대 고르지 마세요.\n"
    "- followup: 직전 대화 주제에 이어서 묻는 말 (예: '그럼 치료법은?', "
    "'그건 전염돼?', '네 심해요')\n"
    "- off_topic: 건강·의료와 명백히 무관함 (예: '오늘 날씨 어때?'). 조금이라도 "
    "건강과 관련 있으면 고르지 마세요.\n\n"
    "[disease_name]\n"
    "사용자가 병명을 **직접 말했을 때만** 표준 한국어 병명으로 적으세요 "
    "(예: '당뇨' -> '당뇨병'). 증상만 말했다면 병명을 추측하지 말고 빈 문자열로 "
    "두세요. followup이면 이어받는 주제의 병명을 적어도 됩니다.\n\n"
    "[wants_research]\n"
    "논문, 연구 결과, 최신 근거, 임상시험 등 학술 근거를 원하면 true.\n\n"
    "[red_flag]\n"
    "즉시 응급 진료가 필요할 수 있는 신호(가슴 통증·압박감, 숨이 참, 갑작스러운 "
    "마비·말 어눌함·얼굴 처짐, 의식 저하, 심한 출혈, 토혈·혈변, 갑자기 시작된 "
    "생애 최악의 두통, 자해 위험 등)가 입력에 있으면 true. 애매하면 false."
)

INTENT_TOOL = {
    "name": "report_intent",
    "description": "사용자 입력의 분류 결과를 보고한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": list(INTENT_LABELS)},
            "disease_name": {"type": "string", "description": "직접 언급된 병명(표준 한국어). 없으면 빈 문자열."},
            "wants_research": {"type": "boolean"},
            "red_flag": {"type": "boolean"},
            "reason": {"type": "string", "description": "판단 근거 한 줄"},
        },
        "required": ["intent", "disease_name", "wants_research", "red_flag", "reason"],
    },
}

_intent_client = None
_intent_disabled_reason: str | None = None


def _get_intent_client():
    """클라이언트는 처음 한 번만 만든다. 쓸 수 없는 이유가 있으면 한 번만 알리고 None."""
    global _intent_client, _intent_disabled_reason
    if _intent_client is not None or _intent_disabled_reason is not None:
        return _intent_client
    if not USE_INTENT_LAYER:
        _intent_disabled_reason = "USE_INTENT_LAYER=0"
    elif anthropic is None:
        _intent_disabled_reason = "anthropic 패키지 없음 (pip install anthropic)"
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        _intent_disabled_reason = ".env에 ANTHROPIC_API_KEY 없음"
    if _intent_disabled_reason:
        print(f"  [판단 레이어] 꺼짐: {_intent_disabled_reason} -> 규칙 기반으로 동작")
        return None
    # accept-encoding을 gzip으로 고정하는 이유: 이 환경의 anthropic SDK가 내부적으로
    # 쓰는 httpx2가 brotli 응답을 풀 때 설치된 brotli 바인딩과 호출 규약이 맞지 않아
    # "TypeError: process() takes no keyword arguments"로 모든 요청이 실패했음
    # (APIConnectionError로 보여서 네트워크 문제처럼 보였지만 실제론 응답 압축 해제
    # 단계 문제였음). brotli를 안 받으면 그 경로를 아예 타지 않는다.
    _intent_client = anthropic.Anthropic(
        timeout=INTENT_TIMEOUT_SECONDS, max_retries=1,
        default_headers={"accept-encoding": "gzip"},
    )
    return _intent_client


def _last_assistant_text(messages: list, max_chars: int = 300) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m["content"][-max_chars:]
    return ""


def _validate_intent(raw: dict) -> dict | None:
    """모델 출력이 형식에 맞는지 코드로 다시 확인한다(틀리면 None -> 규칙으로)."""
    if not isinstance(raw, dict) or raw.get("intent") not in INTENT_LABELS:
        return None
    return {
        "intent": raw["intent"],
        "disease_name": str(raw.get("disease_name") or "").strip(),
        "wants_research": bool(raw.get("wants_research")),
        "red_flag": bool(raw.get("red_flag")),
        "reason": str(raw.get("reason") or "")[:200],
    }


def _log_intent(user_question: str, state: MedicalConversationState, intent: dict | None,
                error: str | None = None) -> None:
    """Haiku 판단과 기존 규칙 판단을 나란히 기록한다(비교 분석용). 실패해도 무시."""
    try:
        candidate = clean_disease_query(user_question)
        rule = {
            "known_term": find_known_term_in_text(user_question),
            "name_like": is_name_like_query(user_question, candidate),
            "interview_answer": (bool(state.pending_complaint)
                                 and _looks_like_interview_answer(user_question)),
            "wants_research": wants_research(user_question),
        }
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "question": user_question,
            "in_interview": bool(state.pending_complaint),
            "model": INTENT_MODEL,
            "intent": intent,
            "rule": rule,
            "error": error,
        }
        os.makedirs(os.path.dirname(INTENT_LOG_PATH), exist_ok=True)
        with open(INTENT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def classify_turn(user_question: str, state: MedicalConversationState,
                  messages: list) -> dict | None:
    """이번 입력의 의도를 Haiku로 분류한다. 실패하면 None(= 예전 규칙으로 동작)."""
    client = _get_intent_client()
    if client is None:
        return None

    context_lines = []
    if state.pending_complaint:
        answers = ", ".join(state.followup_answers) or "(아직 없음)"
        context_lines.append(f"[문진 중] 원래 호소: {state.pending_complaint} / 받은 답변: {answers}")
    else:
        context_lines.append("[문진 중 아님]")
    if state.last_topic:
        context_lines.append(f"[직전 주제] {state.last_topic}")
    last_reply = _last_assistant_text(messages)
    if last_reply:
        context_lines.append(f"[직전 챗봇 답변 끝부분] {last_reply}")
    context_lines.append(f"[이번 사용자 입력] {user_question}")

    try:
        resp = client.messages.create(
            model=INTENT_MODEL,
            max_tokens=300,
            # temperature는 설치된 anthropic SDK(1.11.0)에서 지원하지 않아 뺐다
            # (넘기면 TypeError). 분류 출력은 report_intent 도구 스키마(enum)로
            # 이미 제약돼 있어서 흔들릴 여지가 적다.
            system=INTENT_SYSTEM_PROMPT,
            tools=[INTENT_TOOL],
            tool_choice={"type": "tool", "name": "report_intent"},
            messages=[{"role": "user", "content": "\n".join(context_lines)}],
        )
        raw = next((b.input for b in resp.content if b.type == "tool_use"), None)
        intent = _validate_intent(raw)
        if intent is None:
            print(f"  [판단 레이어] 형식이 맞지 않는 응답 -> 규칙 기반으로 대체: {raw}")
            _log_intent(user_question, state, None, error=f"invalid: {raw}")
            return None
    except Exception as e:  # 네트워크, 인증, 잔액 부족 등 무엇이든 -> 규칙으로
        print(f"  [판단 레이어] 호출 실패 -> 규칙 기반으로 대체: {type(e).__name__}: {e}")
        _log_intent(user_question, state, None, error=f"{type(e).__name__}: {e}")
        return None

    print(f"  [판단 레이어] {intent['intent']}"
          f"{' / 병명=' + intent['disease_name'] if intent['disease_name'] else ''}"
          f"{' / 논문' if intent['wants_research'] else ''}"
          f"{' / 응급신호' if intent['red_flag'] else ''}"
          f" ({intent['reason']})")
    _log_intent(user_question, state, intent)
    return intent


RED_FLAG_NOTICE = (
    "⚠️ 말씀하신 내용에는 바로 진료가 필요할 수 있는 신호가 있어요. 가슴 통증, "
    "숨이 참, 갑작스러운 마비나 말 어눌함, 의식이 흐려짐 같은 증상이 있다면 지금 "
    "바로 119에 전화하거나 가까운 응급실로 가세요."
)

OFF_TOPIC_MESSAGE = (
    "이 질문은 제가 다루는 건강/의료 정보 범위를 벗어났거나, "
    "제가 아는 정보로는 답변하기 어려운 내용인 것 같아요. "
    "증상이나 병명을 조금 더 구체적으로 말씀해주시겠어요?"
)


def _asked_disease_name(user_question: str, intent: dict | None) -> str | None:
    """사용자가 '특정 병명'을 물었다면 그 병명, 아니면 None.
    판단 레이어가 있으면 그 판단을, 없으면 예전 규칙(is_name_like_query)을 쓴다."""
    if intent is not None:
        if intent["intent"] == "disease_info" and intent["disease_name"]:
            return intent["disease_name"]
        return None
    candidate = clean_disease_query(user_question)
    return candidate if is_name_like_query(user_question, candidate) else None


def run_agent_turn(messages: list, user_question: str, state: MedicalConversationState) -> str:
    """한 턴의 입구: 먼저 판단 레이어로 질문을 분류하고, 그 결과를 들고 기존
    파이프라인(_run_pipeline)을 돈다. 응급 신호가 있으면 답변 맨 앞에 안내를 붙인다."""
    intent = classify_turn(user_question, state, messages)
    answer = _run_pipeline(messages, user_question, state, intent)
    if intent is not None and intent["red_flag"]:
        answer = f"{RED_FLAG_NOTICE}\n\n{answer}"
    return answer


def _run_pipeline(messages: list, user_question: str, state: MedicalConversationState,
                  intent: dict | None = None) -> str:
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
        # 0) 문진 중이면(직전에 우리가 증상을 더 물어봤고, 이번 입력이 그 답변이면)
        # 이 입력을 독립된 질의로 검색하지 않는다. 사용자의 답변은 증상이 아니라
        # 시간("저녁부터")·빈도·정도일 수 있어서, 그걸 그대로 검색하면 "저녁부터는
        # 찾지 못했다"는 엉뚱한 답이 나옴(실제 발생). 원래 호소에 누적해서 다룬다.
        # (판단 레이어가 있으면 "문진 답변인지"를 그 판단으로, 없으면 예전 규칙으로)
        if state.pending_complaint:
            is_interview_answer = (
                intent["intent"] == "interview_answer" if intent is not None
                else _looks_like_interview_answer(user_question)
            )
            if is_interview_answer:
                answer = _handle_interview_answer(messages, user_question, state)
                if answer is not None:
                    return answer

        # 건강과 무관한 질문이면 검색을 아예 돌리지 않는다(예전엔 KDCA -> MedlinePlus
        # -> 위키 -> PubMed를 다 돌고 나서야 같은 안내 문구로 끝났음)
        if intent is not None and intent["intent"] == "off_topic":
            print("  [파이프라인] 판단 레이어: 건강과 무관한 질문 -> 검색 생략")
            messages.pop()
            state.turn -= 1
            return OFF_TOPIC_MESSAGE

        search_text, question_text, known_term = resolve_query_with_context(user_question, state, intent)
        asked_name = _asked_disease_name(user_question, intent)
        parts: list[str] = []

        # 1) 건강포털 RAG 먼저 시도 (검색은 search_text로, LLM 질문은 question_text로 - 분리 이유는 resolve_query_with_context 참고)
        rag_result = search_symptom_info(search_text, question_text)

        # 사용자가 "특정 병명"을 물었는데 그 병명이 우리 코퍼스에 없으면(known_term이
        # None인데 질의가 병명 꼴이면), KDCA에서 나온 매칭은 정의상 '다른 질환'이다.
        # 그걸 그 병명의 답으로 내놓으면 안 된다 - 실제로 "주사피부염일때 어떻게
        # 해야해?"가 "열상"(0.62, 칼에 베인 상처)으로 매칭돼서 "주사피부염이 생기면
        # 깨끗한 수건으로 지혈하라"는 날조 답변이 나왔음(2026-10-02). 임계값을 더
        # 올려서 막으려 하면 당뇨병(0.630)·고혈압(0.627) 같은 정상 조회가 깨지므로,
        # "물어본 병명과 매칭된 병명이 다르다"는 사실 자체로 판단한다.
        asked_unknown_disease = not known_term and asked_name is not None
        if asked_unknown_disease and not _is_failure_message(rag_result):
            print("  [파이프라인] 코퍼스에 없는 병명 질문 -> KDCA 매칭(다른 질환) 보류, 외부 소스 우선")

        if _is_failure_message(rag_result) or asked_unknown_disease:
            # 2) 건강포털에서 못 찾았을 때: 위키피디아/PubMed는 "정확한 병명"으로
            # 찾는 도구라서, 병명이 아니라 짧은 증상 호소("속이 매스꺼워")면
            # 외부 검색을 아예 하지 않고 증상을 더 물어본다. 이 판단 없이 검색을
            # 돌렸다가 "속" -> "속옷" 문서, "배" -> PubMed 정신건강 논문처럼
            # 엉뚱한 자료를 답변으로 내놓는 사례가 실제로 있었음(2026-10-01).
            if intent is not None:
                go_interview = not known_term and asked_name is None
            else:
                name_candidate = known_term or clean_disease_query(user_question)
                go_interview = not is_name_like_query(user_question, name_candidate)
            if go_interview:
                print("  [파이프라인] 병명이 아닌 증상 호소로 판단 -> 외부 검색 생략, 문진 시작")
                combined = ask_for_more_detail(user_question)
                state.record_symptom(user_question, "증상 호소(문진 시작)")
                state.start_interview(user_question)
                state.last_topic = None
                state.awaiting_followup_reply = True
                messages.append({"role": "assistant", "content": combined})
                return combined

            # 2-1) 먼저 MedlinePlus(미국 NIH). 위키피디아보다 앞에 두는 이유:
            # 애초에 건강 주제만 모아둔 DB여서 비의료 문서가 걸릴 일이 없고,
            # 주제 태그(altTitle)로 병명이 직접 매칭되는지 확인할 수 있어 근거가
            # 더 분명하다. 대신 영문 DB라 병명 영문 매핑이 안 되면 건너뛴다.
            lookup = known_term or (asked_name if intent is not None else None) or user_question
            print("  [파이프라인] 건강포털 RAG 실패 -> MedlinePlus(NIH) 시도")
            medline_result = search_medlineplus(lookup)
            if not _is_failure_message(medline_result):
                parts.append(medline_result)
                topic_match = re.search(r"\(출처: MedlinePlus '(.+?)'", medline_result)
                _record_fallback_result(
                    state, user_question, known_term, asked_unknown_disease,
                    topic_match.group(1) if topic_match else None, "search_medlineplus",
                    asked_name=asked_name,
                )
                combined = "\n\n".join(parts)
                state.awaiting_followup_reply = "?" in combined[-200:]
                messages.append({"role": "assistant", "content": combined})
                return combined

            print("  [파이프라인] MedlinePlus 실패 -> 위키피디아로 대체")
            wiki_result = search_wikipedia(lookup)
            if not _is_failure_message(wiki_result):
                parts.append(wiki_result)
                # 위키 답변 형식 두 가지를 모두 지원: 요약 실패 시 원문 그대로
                # 반환하는 "[위키피디아 - 제목]\n..." 형식과, 요약 성공 시
                # 끝에 붙는 "(출처: 위키피디아 '제목')" 형식.
                title_match = (
                    re.match(r"\[위키피디아 - (.+?)\]", wiki_result)
                    or re.search(r"\(출처: 위키피디아 '(.+?)'\)", wiki_result)
                )
                _record_fallback_result(
                    state, user_question, known_term, asked_unknown_disease,
                    title_match.group(1) if title_match else None, "search_wikipedia(fallback)",
                    asked_name=asked_name,
                )
            else:
                # 3) 위키피디아까지 실패하면 PubMed로 마지막 시도 (건강정보포털/
                # 위키피디아 둘 다 없는 병명이라도, PubMed는 영어 의학 문헌
                # 전체를 대상으로 하므로 찾을 가능성이 있음 - "주사피부염"처럼
                # 국내 자료엔 드물지만 해외 연구는 많은 경우가 실제로 있었음).
                print("  [파이프라인] 위키피디아도 실패 -> PubMed로 마지막 시도")
                fallback_keyword = known_term or asked_name or clean_disease_query(user_question)
                pubmed_result = search_pubmed_deep(fallback_keyword)
                if _is_failure_message(pubmed_result):
                    # 외부 소스가 전부 실패했지만, 코퍼스에 없는 병명을 물어서
                    # 보류해둔 KDCA 매칭이 있으면 버리지 말고 "직접 자료는 아니다"라고
                    # 분명히 밝히면서 참고용으로 보여준다(정보를 아예 잃지 않도록).
                    if asked_unknown_disease and not _is_failure_message(rag_result):
                        print("  [파이프라인] 외부 소스 전부 실패 -> 보류했던 KDCA 매칭을 참고용으로 안내")
                        combined = (
                            f"'{fallback_keyword}'에 대한 직접적인 자료는 제가 가진 "
                            "건강정보포털·MedlinePlus·위키피디아·PubMed에서 찾지 못했어요. "
                            "아래는 증상이 비슷해 보이는 다른 질환 자료라서, 여쭤보신 "
                            "질환과 다를 수 있다는 점을 꼭 감안해주세요.\n\n"
                            f"{rag_result}"
                        )
                        # 물어본 병명 자체는 질환으로 남기되, 자료를 못 찾았다는
                        # 사실을 출처에 분명히 적어둔다(증상 목록은 건드리지 않음).
                        state.record_disease(fallback_keyword, "자료 미확인(유사 질환만 참고 안내)")
                        state.awaiting_followup_reply = "?" in combined[-200:]
                        messages.append({"role": "assistant", "content": combined})
                        return combined
                    messages.pop()
                    state.turn -= 1
                    return OFF_TOPIC_MESSAGE
                parts.append(pubmed_result)
                _record_fallback_result(
                    state, user_question, known_term, asked_unknown_disease,
                    fallback_keyword, "search_pubmed_deep(fallback)",
                    asked_name=asked_name,
                )
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
        research = wants_research(user_question) or (intent is not None and intent["wants_research"])
        if known_term and research:
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
