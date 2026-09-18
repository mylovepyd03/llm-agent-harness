"""
알려진 한계 3번 해결: KDCA API가 내용을 안 채워둔 13개 질환(위염, 천식 등)을
위키피디아로 보충해서 kdca_embeddings_clean.json에 추가한다.

이렇게 하면 "위염이 뭐야?" 같은 질문이 더 이상 "위십이지장 궤양" 같은 유사
항목으로 대체 매칭되지 않고, 실제 위염 내용으로 답할 수 있게 된다.
"""
import json
import os

import requests

WIKI_HEADERS = {
    "User-Agent": "llm-agent-harness-learning-project/0.1 (personal study, contact: paranvit@gmail.com)"
}
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "bge-m3"

INDEX_PATH = os.path.join(os.path.dirname(__file__), "data", "kdca_disease_index.json")
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "kdca_corpus.json")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "data", "kdca_embeddings_clean.json")


def fetch_wikipedia_extract(query: str) -> str | None:
    search_resp = requests.get(
        "https://ko.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1},
        headers=WIKI_HEADERS, timeout=10,
    )
    results = search_resp.json()["query"]["search"]
    if not results:
        return None
    title = results[0]["title"]

    extract_resp = requests.get(
        "https://ko.wikipedia.org/w/api.php",
        params={"action": "query", "prop": "extracts", "exintro": True, "explaintext": True, "titles": title, "format": "json"},
        headers=WIKI_HEADERS, timeout=10,
    )
    pages = extract_resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    extract = page.get("extract", "").strip()
    return extract if extract else None


def embed(text: str) -> list[float]:
    response = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text[:2000]}, timeout=30)
    response.raise_for_status()
    return response.json()["embedding"]


if __name__ == "__main__":
    with open(INDEX_PATH, encoding="utf-8") as f:
        full_index: dict = json.load(f)
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus: dict = json.load(f)
    with open(EMBEDDINGS_PATH, encoding="utf-8") as f:
        embeddings: dict = json.load(f)

    missing = [name for name in full_index if name not in corpus]
    print(f"보충 대상 {len(missing)}개: {missing}")

    added = 0
    for name in missing:
        # 괄호 안 부가설명 제거해서 검색 (예: "교통사고(일반)" -> "교통사고")
        search_name = name.split("(")[0].strip()
        extract = fetch_wikipedia_extract(search_name)
        if not extract:
            print(f"  [실패] {name}: 위키피디아에서도 못 찾음")
            continue

        entry = {"sn": full_index[name], "text": extract, "source": "wikipedia_fallback", "embedding": embed(extract)}
        corpus[name] = {"sn": entry["sn"], "text": entry["text"]}
        embeddings[name] = entry
        added += 1
        print(f"  [성공] {name}: {len(extract)}자")

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False)

    print(f"\n완료: {added}/{len(missing)}개 보충됨")
