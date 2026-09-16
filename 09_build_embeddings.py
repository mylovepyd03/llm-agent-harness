"""
B단계 2단계: 말뭉치를 임베딩 벡터로 변환.

kdca_corpus.json(611개 질환 텍스트)을 nomic-embed-text 모델로 벡터화해서
저장한다. nomic-embed-text는 "search_document: "(문서용) / "search_query: "(질문용)
접두어를 붙이면 검색 품질이 좋아진다고 공식 안내하고 있어서 그대로 따른다.
"""
import json
import os
import time

import requests

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "bge-m3"  # nomic-embed-text는 한국어 구분력이 약해서 다국어 특화 모델로 교체

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "kdca_corpus.json")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "data", "kdca_embeddings.json")

MAX_EMBED_CHARS = 2000


def embed(text: str) -> list[float]:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text[:MAX_EMBED_CHARS]},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embedding"]


if __name__ == "__main__":
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus: dict = json.load(f)

    items = list(corpus.items())
    print(f"전체 {len(items)}개 임베딩 시작")

    start = time.time()
    for i, (name, entry) in enumerate(items, 1):
        entry["embedding"] = embed(entry["text"])
        if i % 100 == 0:
            print(f"  진행: {i}/{len(items)} ({time.time()-start:.0f}초 경과)")

    elapsed = time.time() - start
    print(f"완료: {elapsed:.1f}초")

    with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    print(f"저장 완료: {EMBEDDINGS_PATH}")
