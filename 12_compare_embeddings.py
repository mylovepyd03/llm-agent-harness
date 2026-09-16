"""
B단계 검색 품질 개선 실험: 상투문구("'이것만은 꼭 기억하세요'") 제거 후 재임베딩,
기존(원본 텍스트) 임베딩과 나란히 검색 결과를 비교.

기존 kdca_embeddings.json은 건드리지 않고, 정제된 버전은
kdca_embeddings_clean.json으로 따로 저장해서 비교한다.
"""
import json
import math
import os
import re

import requests

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "bge-m3"

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "kdca_corpus.json")
OLD_EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "data", "kdca_embeddings.json")
CLEAN_EMBEDDINGS_PATH = os.path.join(
    os.path.dirname(__file__), "data", "kdca_embeddings_clean.json"
)

# 문서 시작부에 반복되는 상투문구. 작은따옴표로 감싸인 형태 그대로 제거.
BOILERPLATE = re.compile(r"'이것만은 꼭 기억하세요'\n?")

MAX_EMBED_CHARS = 2000


def clean_text(text: str) -> str:
    return BOILERPLATE.sub("", text).strip()


def embed(text: str) -> list[float]:
    response = requests.post(
        OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text[:MAX_EMBED_CHARS]}, timeout=30
    )
    response.raise_for_status()
    return response.json()["embedding"]


def build_clean_embeddings():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus: dict = json.load(f)

    clean = {}
    print(f"정제 후 재임베딩 시작 ({len(corpus)}개)")
    for i, (name, entry) in enumerate(corpus.items(), 1):
        cleaned_text = clean_text(entry["text"])
        clean[name] = {"sn": entry["sn"], "text": cleaned_text, "embedding": embed(cleaned_text)}
        if i % 100 == 0:
            print(f"  진행: {i}/{len(corpus)}")

    with open(CLEAN_EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False)
    print(f"저장 완료: {CLEAN_EMBEDDINGS_PATH}")
    return clean


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def embed_query(text: str) -> list[float]:
    response = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    response.raise_for_status()
    return response.json()["embedding"]


def search(query_vec, corpus, top_k=5):
    scored = [(name, cosine_similarity(query_vec, e["embedding"])) for name, e in corpus.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    if os.path.exists(CLEAN_EMBEDDINGS_PATH):
        print("기존 정제 임베딩 파일 재사용")
        with open(CLEAN_EMBEDDINGS_PATH, encoding="utf-8") as f:
            clean_corpus = json.load(f)
    else:
        clean_corpus = build_clean_embeddings()

    with open(OLD_EMBEDDINGS_PATH, encoding="utf-8") as f:
        old_corpus = json.load(f)

    test_questions = [
        "머리가 아프고 속이 메스꺼워요",
        "기침이 심하고 콧물이 나요",
        "배가 아프고 설사를 해요",
        "가슴이 답답하고 숨쉬기 힘들어요",
    ]

    for question in test_questions:
        qvec = embed_query(question)
        old_results = search(qvec, old_corpus)
        clean_results = search(qvec, clean_corpus)

        print(f"\n########## 질문: {question} ##########")
        print("[기존(상투문구 포함)]")
        for name, score in old_results:
            print(f"  {score:.3f}  {name}")
        print("[정제 후]")
        for name, score in clean_results:
            print(f"  {score:.3f}  {name}")
