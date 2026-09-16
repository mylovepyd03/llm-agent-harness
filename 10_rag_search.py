"""
B단계 3단계: 벡터 유사도 검색만 따로 테스트 (아직 LLM 없이).

사용자 질문(증상 문장)을 임베딩하고, 611개 질환 벡터와 코사인 유사도를 계산해서
가장 가까운 것 몇 개를 찾는다. "정확한 키워드"가 아니라 "의미가 비슷한 것"을
찾는 게 핵심이라, 여기서는 LLM 없이 이 검색 자체가 말이 되는지만 확인한다.
"""
import json
import math
import os

import requests

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "bge-m3"
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "data", "kdca_embeddings.json")


def embed_query(text: str) -> list[float]:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
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


def search(query: str, corpus: dict, top_k: int = 5) -> list[tuple[str, float, dict]]:
    """질문과 가장 비슷한 질환 top_k개를 (이름, 유사도, 항목) 형태로 반환."""
    query_vec = embed_query(query)

    scored = []
    for name, entry in corpus.items():
        score = cosine_similarity(query_vec, entry["embedding"])
        scored.append((name, score, entry))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    corpus = load_corpus()
    print(f"말뭉치 {len(corpus)}개 로드 완료\n")

    test_questions = [
        "머리가 아프고 속이 메스꺼워요",
        "기침이 심하고 콧물이 나요",
        "배가 아프고 설사를 해요",
        "가슴이 답답하고 숨쉬기 힘들어요",
    ]

    for question in test_questions:
        print(f"########## 질문: {question} ##########")
        results = search(question, corpus, top_k=5)
        for name, score, entry in results:
            print(f"  {score:.3f}  {name}")
        print()
