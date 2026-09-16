"""
B단계 4단계: RAG 전체 흐름 완성 - 검색(retrieval) + LLM 답변 생성(generation).

흐름: 증상 문장 -> 임베딩 검색으로 관련 질환 top-k 찾기 -> 그 내용을 LLM에게
"참고자료"로 주면서 반드시 그 안에서만 답하라고 지시 -> 최종 답변.

핵심 규칙: 근거(검색 결과)가 부실하면 LLM을 아예 부르지 않는다.
LLM이 부르더라도 "참고자료에 없으면 모른다고 답하라"고 강하게 지시한다.
"""
import json
import math
import os
import re

import requests

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
CHAT_MODEL = "llama3.2"
EMBED_MODEL = "bge-m3"

EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "data", "kdca_embeddings_clean.json")

# 검색 1등 유사도가 이 값보다 낮으면 "관련 자료 없음"으로 보고 LLM을 부르지 않음.
# (감기 vs 당뇨병처럼 무관한 쌍이 ~0.41, 관련 있는 쌍은 0.53~0.61로 나온 실측 기준)
MIN_SIMILARITY = 0.45

RAG_SYSTEM_PROMPT = (
    "당신은 국가건강정보포털 자료를 바탕으로 답하는 건강정보 도우미입니다.\n"
    "반드시 아래 [참고자료]에 있는 내용만 근거로 답하세요.\n"
    "참고자료에 없는 내용은 절대 지어내지 말고, 모른다고 답하세요.\n"
    "이것은 진단이 아니라 참고 정보이므로, 확정적으로 단언하지 말고 "
    "'~일 가능성이 있습니다' 같은 표현을 쓰고, 증상이 지속되면 병원 진료를 권하세요."
)


def embed_query(text: str) -> list[float]:
    response = requests.post(
        OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=30
    )
    response.raise_for_status()
    return response.json()["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def load_corpus() -> dict:
    with open(EMBEDDINGS_PATH, encoding="utf-8") as f:
        return json.load(f)


def search(query: str, corpus: dict, top_k: int = 5):
    query_vec = embed_query(query)
    scored = [
        (name, cosine_similarity(query_vec, entry["embedding"]), entry)
        for name, entry in corpus.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def build_context(results, n: int = 3, chars_per_doc: int = 500) -> str:
    """검색 결과 상위 n개를 LLM에게 줄 참고자료 텍스트로 조립."""
    blocks = []
    for name, score, entry in results[:n]:
        blocks.append(f"[{name}]\n{entry['text'][:chars_per_doc]}")
    return "\n\n".join(blocks)


def chat(messages) -> dict:
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 512},
    }
    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]


def has_short_chunk_repetition(content: str) -> bool:
    """짧은 조각(2~20글자)이 연달아 5번 이상 반복되는 경우 (예: '세요세요세요...')."""
    return re.search(r"(.{2,20}?)\1{4,}", content) is not None


def has_sentence_repetition(content: str, min_len: int = 15, min_repeats: int = 3) -> bool:
    """문장/문단 단위로 통째로 여러 번 반복되는 경우 (짧은 조각 반복 감지로는 못 잡음).
    문장을 나눠서, 같은 문장(min_len자 이상)이 min_repeats번 이상 나오면 반복으로 판단."""
    sentences = re.split(r"(?<=[.!?])\s+", content)
    counts: dict[str, int] = {}
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < min_len:
            continue  # 너무 짧은 문장은 우연히 반복될 수 있어 제외
        counts[sentence] = counts.get(sentence, 0) + 1
    return any(count >= min_repeats for count in counts.values())


def is_generation_ok(content: str) -> bool:
    """도구 호출이 없는 순수 생성 응답용 검증 - 두 종류의 반복 루프를 모두 체크."""
    if not content:
        return False
    if has_short_chunk_repetition(content):
        return False
    if has_sentence_repetition(content):
        return False
    return True


def call_with_retry(messages, max_retries: int = 1):
    for attempt in range(max_retries + 1):
        message = chat(messages)
        if is_generation_ok(message.get("content", "")):
            return message
        print(f"[경고] 비정상 응답 감지 (시도 {attempt + 1}/{max_retries + 1}) - 재시도")
    return None


def answer_with_rag(question: str, corpus: dict) -> str:
    # 1) 검색(retrieval)
    results = search(question, corpus, top_k=5)

    if not results or results[0][1] < MIN_SIMILARITY:
        return "관련된 건강정보를 찾지 못해 답변할 수 없습니다. 증상을 좀 더 구체적으로 말씀해주세요."

    print("  [검색됨]", ", ".join(f"{name}({score:.2f})" for name, score, _ in results[:3]))

    # 2) 답변 생성(generation) - 검색 결과를 근거로만 답하게 강제
    context = build_context(results, n=3)
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"[참고자료]\n{context}\n\n[질문]\n{question}"},
    ]

    message = call_with_retry(messages)
    if message is None:
        return "[오류] 모델이 계속 비정상적인 응답을 내서 포기했습니다."
    return message["content"]


if __name__ == "__main__":
    corpus = load_corpus()

    for question in [
        "머리가 아프고 속이 메스꺼워요",
        "배가 아프고 설사를 해요",
        "코딩을 잘하는 방법이 뭐야",  # 관련 자료가 없는 질문 - 거부해야 정상
    ]:
        print(f"\n########## 질문: {question} ##########")
        answer = answer_with_rag(question, corpus)
        print("=== 최종 답변 ===")
        print(answer)
