"""
B단계 1단계: RAG용 말뭉치(corpus) 만들기.

kdca_disease_index.json에 있는 병명 624개 전부에 대해 실제 정의/증상/원인/치료
텍스트를 KDCA API로 가져와서, 검색 가능한 형태로 로컬 파일에 저장한다.
(일부 항목은 "위염"처럼 실제 내용이 비어있는 경우가 있어 그런 건 건너뛴다.)
"""
import json
import os
import ssl
import time
import xml.etree.ElementTree as ET

import requests
from requests.adapters import HTTPAdapter

KDCA_HEALTH_INFO_URL = "https://api.kdca.go.kr/api/provide/healthInfo"
KDCA_TOKEN = "1a0a4b2b8e71"
INDEX_PATH = os.path.join(os.path.dirname(__file__), "data", "kdca_disease_index.json")
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "kdca_corpus.json")


class _LegacySSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.options |= 0x4
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


session = requests.Session()
session.mount("https://", _LegacySSLAdapter())


def fetch_one(name: str, sn: int) -> dict | None:
    """질환 하나의 본문 텍스트를 가져온다. 내용이 없으면 None."""
    response = session.get(
        KDCA_HEALTH_INFO_URL, params={"TOKEN": KDCA_TOKEN, "cntntsSn": sn}, timeout=10
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    sections = root.findall(".//cntntsCl")

    texts = []
    for sec in sections:
        content = (sec.findtext("CNTNTS_CL_CN", "") or "").strip()
        if not content or content.startswith("http") or content.startswith("<table"):
            continue  # 첨부파일 링크, HTML 표는 건너뜀 (검색에 안 쓸 노이즈)
        texts.append(content)

    if not texts:
        return None

    full_text = "\n".join(texts)
    return {"sn": sn, "text": full_text[:3000]}  # 너무 길면 임베딩 비용도 커지니 상한


if __name__ == "__main__":
    with open(INDEX_PATH, encoding="utf-8") as f:
        disease_index: dict = json.load(f)

    items = list(disease_index.items())
    print(f"전체 {len(items)}개 질환 본문 수집 시작")

    start = time.time()
    corpus = {}
    empty_count = 0
    for i, (name, sn) in enumerate(items, 1):
        result = fetch_one(name, sn)
        if result is None:
            empty_count += 1
        else:
            corpus[name] = result

        if i % 100 == 0:
            print(f"  진행: {i}/{len(items)} ({time.time()-start:.0f}초 경과)")

    elapsed = time.time() - start
    print(f"완료: {elapsed:.1f}초")
    print(f"내용 있음: {len(corpus)}개, 내용 없음(스킵): {empty_count}개")

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {CORPUS_PATH}")
