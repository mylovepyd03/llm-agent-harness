"""
B단계 5단계: PubMed에도 같은 RAG 패턴 적용 - 제목만 보고 짐작하지 않고
실제 초록(abstract)을 가져와서, 그 내용을 근거로만 답하게 함.

KDCA 버전과 다른 점: 611개를 미리 다 모아둘 수 없으니(논문은 수천만 건),
질문이 들어올 때마다 PubMed에서 후보 논문 초록을 그때그때 가져와서
임베딩으로 재순위화(rerank)한다.

흐름: 키워드 영문 번역 -> PubMed에서 후보 10개 검색 -> 초록 가져오기
      -> 임베딩으로 질문과 가장 관련있는 3~5개 추리기 -> 그 초록을 근거로 LLM 답변
"""
import math
import os
import re
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
CHAT_MODEL = "llama3.2"  # 메인 에이전트 모델(도구 선택 등)은 그대로 유지
TRANSLATE_MODEL = "llama3.1"  # 번역 전용 - llama3.2:3b는 영→한 번역 시 문자가 섞이는 문제가 있어서 8B로 교체
SYNTHESIZE_MODEL = "llama3.1"  # 종합(최종답변) 단계도 llama3.2:3b가 근거 없는 내용을 지어내서 테스트 삼아 8B로 교체
EMBED_MODEL = "bge-m3"

DISEASE_API_URL = "http://apis.data.go.kr/B551182/diseaseInfoService1/getDissNameCodeList1"
DISEASE_API_KEY = os.environ.get("DISEASE_INFO_SERVICE_KEY")

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_CONTACT = {"tool": "llm-agent-harness-learning-project", "email": "paranvit@gmail.com"}

MIN_SIMILARITY = 0.4  # 이 미만이면 "관련 논문 없음"으로 판단

RAG_SYSTEM_PROMPT = (
    "당신은 PubMed 논문 초록을 바탕으로 답하는 의학 연구 요약 도우미입니다.\n"
    "반드시 아래 [참고 논문]에 있는 내용만 근거로 답하세요.\n"
    "논문에 없는 내용은 절대 지어내지 말고, 모른다고 답하세요.\n"
    "어떤 논문(제목)에서 나온 내용인지 언급하면서 답하세요.\n"
    "이것은 진단이 아니라 연구 요약이므로 단정적으로 말하지 마세요."
)

# 1단계: 초록 하나를 한국어로 "번역/정리"만 시키는 프롬프트.
# 여러 초록을 한꺼번에 종합+번역하는 건 3B 모델에게 너무 어려운 작업이라,
# "번역/정리"와 "종합 답변 작성"을 완전히 분리된 두 단계로 쪼갠다.
TRANSLATE_SYSTEM_PROMPT = (
    "당신은 의학 논문 초록을 한국어로 옮기는 번역 도우미입니다.\n"
    "아래 영어 초록에 있는 사실만 한국어로 정리하세요.\n"
    "원문에 없는 내용을 추가하거나, 추측하거나, 다른 지식을 끌어오지 마세요.\n"
    "3~5문장으로 간결하게 쓰세요."
)

# 2단계: 번역된 요약들만 근거로 종합 답변을 쓰게 하는 프롬프트.
SYNTHESIZE_SYSTEM_PROMPT = (
    "당신은 여러 논문 요약을 종합해서 답하는 의학 연구 요약 도우미입니다.\n"
    "반드시 아래 [참고 요약]에 있는 내용만 근거로 답하세요.\n"
    "참고 요약에 없는 내용은 절대 지어내지 말고 언급하지 마세요.\n"
    "어떤 논문(제목)에서 나온 내용인지 언급하면서 답하세요.\n"
    "이것은 진단이 아니라 연구 요약이므로 단정적으로 말하지 마세요."
)


def _to_english_query(keyword: str) -> str:
    """한글 병명을 PubMed 검색용 영문으로 변환 (HIRA 영문명 재활용)."""
    has_korean = re.search(r"[가-힣]", keyword) is not None
    if not has_korean:
        return keyword
    if not DISEASE_API_KEY:
        return keyword

    response = requests.get(
        DISEASE_API_URL,
        params={
            "ServiceKey": DISEASE_API_KEY,
            "pageNo": 1,
            "numOfRows": 1,
            "sickType": 1,
            "medTp": 1,
            "diseaseType": "SICK_NM",
            "searchText": keyword,
        },
        timeout=10,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    item = root.find(".//item")
    if item is not None:
        eng = item.findtext("sickEngNm", "")
        if eng:
            return eng
    return keyword


def _pubmed_esearch(query: str, retmax: int = 10) -> list[str]:
    response = requests.get(
        PUBMED_ESEARCH_URL,
        params={
            **PUBMED_CONTACT,
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": retmax,
            "sort": "relevance",
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("esearchresult", {}).get("idlist", [])


def _pubmed_efetch_abstracts(pmids: list[str]) -> list[dict]:
    """PMID 목록으로 제목+초록 텍스트를 가져온다."""
    if not pmids:
        return []

    response = requests.get(
        PUBMED_EFETCH_URL,
        params={
            **PUBMED_CONTACT,
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
        },
        timeout=15,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)

    articles = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", "")
        title = article.findtext(".//ArticleTitle", "") or ""
        year = article.findtext(".//JournalIssue/PubDate/Year", "") or ""
        journal = article.findtext(".//Journal/Title", "") or ""

        abstract_parts = [
            (node.text or "") for node in article.findall(".//Abstract/AbstractText")
        ]
        abstract = " ".join(part.strip() for part in abstract_parts if part.strip())

        if not abstract:
            continue  # 초록이 없는 논문은 근거로 못 쓰니 건너뜀

        articles.append(
            {"pmid": pmid, "title": title, "year": year, "journal": journal, "abstract": abstract}
        )
    return articles


def embed(text: str) -> list[float]:
    response = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    response.raise_for_status()
    return response.json()["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def pubmed_retrieve(keyword: str, top_k: int = 3):
    """PubMed 검색(1차, 키워드 기반) + 임베딩 재순위화(2차, 의미 기반)."""
    english_query = _to_english_query(keyword)
    pmids = _pubmed_esearch(english_query, retmax=10)
    articles = _pubmed_efetch_abstracts(pmids)
    if not articles:
        return english_query, []

    query_vec = embed(english_query)
    for art in articles:
        art_vec = embed(f"{art['title']} {art['abstract']}"[:2000])
        art["score"] = cosine_similarity(query_vec, art_vec)

    articles.sort(key=lambda a: a["score"], reverse=True)
    return english_query, articles[:top_k]


def build_context(articles: list[dict], chars_per_abstract: int = 700) -> str:
    blocks = []
    for art in articles:
        blocks.append(
            f"[{art['title']}] ({art['journal']}, {art['year']})\n"
            f"{art['abstract'][:chars_per_abstract]}"
        )
    return "\n\n".join(blocks)


def chat(messages, model: str = CHAT_MODEL) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 512},
    }
    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["message"]


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


# 영→한 번역 중에 낯선 문자 체계가 섞여 나오는 고장을 반복 관찰함
# (한자 "头痛", 키릴 문자 "міг라인" 등 - 매번 다른 문자셋으로 샘).
# 특정 문자셋 하나씩 막는 대신, "한국어 답변에 있을 법한 문자만 허용"하는
# 화이트리스트 방식으로 일반화 - 07_interactive_agent.py의 도구 인자 검증과 같은 패턴.
# 허용: 한글 음절/자모, 기본 라틴+라틴 확장(영문/숫자/약물기호), 일반 구두점, 공백.
_ALLOWED_SCRIPT_RANGES = (
    (0xAC00, 0xD7A3),  # 한글 음절
    (0x3131, 0x318E),  # 한글 자모
    (0x0000, 0x024F),  # 기본 라틴 + 라틴 확장 (영문자/숫자/기본 기호)
    (0x2000, 0x206F),  # 일반 구두점 (따옴표, 줄임표, 대시 등)
)


def has_unexpected_script(content: str) -> bool:
    """허용 범위 밖의 문자(한자, 키릴 문자 등)가 하나라도 섞이면 True."""
    for ch in content:
        if ch.isspace():
            continue
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in _ALLOWED_SCRIPT_RANGES):
            continue
        return True
    return False


def is_generation_ok(content: str) -> bool:
    if not content:
        return False
    if has_short_chunk_repetition(content):
        return False
    if has_sentence_repetition(content):
        return False
    if has_unexpected_script(content):
        return False
    return True


def call_with_retry(messages, max_retries: int = 1, model: str = CHAT_MODEL):
    for attempt in range(max_retries + 1):
        message = chat(messages, model=model)
        if is_generation_ok(message.get("content", "")):
            return message
        print(f"[경고] 비정상 응답 감지 (시도 {attempt + 1}/{max_retries + 1}) - 재시도")
    return None


def translate_abstract(article: dict) -> str | None:
    """1단계: 초록 하나만 한국어로 충실하게 번역/정리 (종합·해석은 아직 안 함).
    번역은 llama3.2:3b가 약한 작업이라 더 큰 TRANSLATE_MODEL을 씀."""
    messages = [
        {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"제목: {article['title']}\n초록: {article['abstract'][:1500]}",
        },
    ]
    message = call_with_retry(messages, model=TRANSLATE_MODEL)
    if message is None:
        return None
    return message["content"]


def build_translated_context(articles: list[dict]) -> str:
    blocks = []
    for art in articles:
        if not art.get("ko_summary"):
            continue
        blocks.append(
            f"[{art['title']}] (PMID: {art['pmid']}, {art['journal']}, {art['year']})\n"
            f"{art['ko_summary']}"
        )
    return "\n\n".join(blocks)


def answer_with_pubmed_rag(keyword: str) -> str:
    # 1) 검색(retrieval): PubMed 키워드 검색 + 임베딩 재순위화
    english_query, articles = pubmed_retrieve(keyword, top_k=3)

    if not articles or articles[0]["score"] < MIN_SIMILARITY:
        return f"'{keyword}'({english_query})에 대한 관련 논문을 찾지 못해 답변할 수 없습니다."

    print(
        "  [검색됨]",
        ", ".join(f"{a['title'][:40]}...({a['score']:.2f})" for a in articles),
    )

    # 2) 1단계 번역: 초록마다 따로따로 한국어로 정리 (종합 X, 번역/정리만)
    for art in articles:
        art["ko_summary"] = translate_abstract(art)
        status = "성공" if art["ko_summary"] else "실패(건너뜀)"
        print(f"  [번역 {status}] {art['title'][:40]}...")

    context = build_translated_context(articles)
    if not context:
        return "논문 요약을 만들지 못해 답변할 수 없습니다."

    # 3) 2단계 종합: 번역된 요약들만 근거로 최종 답변 작성
    messages = [
        {"role": "system", "content": SYNTHESIZE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"[참고 요약]\n{context}\n\n[질문]\n{keyword}에 대한 최신 연구 결과를 알려줘.",
        },
    ]

    message = call_with_retry(messages, model=SYNTHESIZE_MODEL)
    if message is None:
        return "[오류] 모델이 계속 비정상적인 응답을 내서 포기했습니다."
    return message["content"]


if __name__ == "__main__":
    for keyword in ["편두통", "코딩테스트"]:  # 두 번째는 일부러 관련 논문 없을 만한 키워드
        print(f"\n########## 키워드: {keyword} ##########")
        answer = answer_with_pubmed_rag(keyword)
        print("=== 최종 답변 ===")
        print(answer)
