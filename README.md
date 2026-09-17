# llm-agent-harness

LLM 기반 에이전트/하네스 시스템을 구현해보는 개인 학습 프로젝트.
[toy-llm-project](../toy-llm-project)(트랜스포머를 밑바닥부터 구현)와는 목적이 다름 —
여기서는 **이미 있는 LLM을 "두뇌"로 가져다 쓰고, 그 주변에 도구 호출/검색/상태 관리 같은
"시스템"을 어떻게 쌓는지**를 배우는 게 목표. 파이썬 초보자 학습용, 매 단계 개념 설명 +
작은 단위로 확인하며 진행.

**주제**: "내 손안의 의사" — 증상/병명을 물어보면 위키피디아, 공식 질병정보(HIRA),
증상/원인/치료 정보(질병관리청), 관련 연구 논문(PubMed)까지 찾아서 답해주는 에이전트.
도구 자체보다 "LLM 혼자서는 못 하는 걸 도구/검증/재시도로 어떻게 보완하는가"를 배우는
용도라 주제(의료)는 소재일 뿐, 핵심은 하네스 구조.

## 두 프로젝트 비교

| | toy-llm-project | llm-agent-harness (이 프로젝트) |
|---|---|---|
| 배우는 것 | 트랜스포머 내부 구조 (attention, 학습 루프 등) | 에이전트 구조 (도구 호출, 검색, 상태 관리) |
| LLM | 직접 만듦 (파라미터 9만개 미니 GPT) | 기존 오픈소스 모델을 가져다 씀 (Ollama) |
| 학습(훈련) 여부 | O (직접 학습시킴) | X (이미 학습된 모델을 호출만 함) |

## 실행 환경

- LLM: **Ollama**로 로컬에서 실행 (API 키/비용 불필요)
  - 채팅 모델: `llama3.2` (3B, 약 2GB) — 도구 호출(tool calling) 지원
  - 임베딩 모델: `bge-m3` (1.2GB) — 다국어(한국어 포함) 임베딩. `nomic-embed-text`는
    한국어 구분력이 낮아서 교체함
  - 확인: `curl http://localhost:11434/api/version`
- 파이썬: conda base 환경 사용 (`/opt/anaconda3/bin/python`) — toy-llm-project와 동일
- 프로젝트 위치: `/Users/princess-yedam/llm-agent-harness`
- API 키: `.env` 파일에 보관 (git에 안 올라감, `.gitignore` 처리됨)
  - `DISEASE_INFO_SERVICE_KEY`: 공공데이터포털 건강보험심사평가원 질병정보서비스
  - `KDCA_HEALTH_INFO_TOKEN`: 질병관리청 국가건강정보포털
- git: 로컬 저장소 + GitHub 비공개 저장소(`mylovepyd03/llm-agent-harness`) 연결됨

## 로드맵 (난이도순, 아직 확정 아님 — 진행하며 조정)

1. [x] 프로젝트 폴더 만들기
2. [x] 로컬 LLM 환경 구성 (Ollama 설치 + llama3.2 다운로드)
3. [x] **A. 기초 도구 호출 에이전트** — LLM이 도구를 스스로 판단해서 호출하는 최소
   에이전트 루프 (ReAct 패턴). 도구 4개(위키피디아/공식병명코드/증상정보/PubMed) +
   응답 검증·재시도 하네스까지 완성.
4. [ ] **B. RAG 질의응답 에이전트** — 진행 중. 증상 문장을 임베딩 검색으로 관련 질환에
   매칭하는 RAG는 완성. PubMed 논문 기반 RAG는 llama3.2:3b의 영→한 번역 한계로 막힘
   (다음 단계 후보: 번역 전용 모델 추가).
5. [ ] **C. 미니 하네스** — 여러 단계를 스스로 계획하고, 도구를 순서대로 호출하며 상태를
   기억하는 좀 더 범용적인 구조 (지금 이 대화 — Claude Code — 가 하는 일의 축소판)

## 파일 구성

### A단계 — 기초 도구 호출 에이전트

| 파일 | 내용 |
|---|---|
| `01_hello_llm.py` | Ollama 기본 호출 확인 (도구 없음) |
| `02_tool_schema.py` | 도구 스펙만 주고 LLM이 `tool_calls` 구조로 요청하는지 확인 |
| `03_full_loop.py` | 실제 도구 실행 + 결과 반영 + 최종 답변까지 (ReAct 루프 완성) |
| `04_disease_tool.py` | 도구 2개로 확장 + 반복 루프 감지/재시도 로직 추가 |
| `05_health_info_tool.py` | 도구 3개(질병관리청 증상/원인/치료 추가) |
| `06_pubmed_tool.py` | 도구 4개(PubMed 논문 검색 추가, 영문 자동 번역) |
| `07_interactive_agent.py` | 터미널 대화형 CLI + 도구 인자 검증 + 실행 결과 검증 |

### B단계 — RAG (진행 중)

| 파일 | 내용 |
|---|---|
| `08_build_corpus.py` | KDCA 611개 질환 본문(정의/증상/원인/치료) 수집 |
| `09_build_embeddings.py` | bge-m3로 611개 텍스트 임베딩 |
| `10_rag_search.py` | 순수 벡터 유사도 검색만 테스트 (LLM 없이) |
| `11_rag_agent.py` | 검색+생성 RAG 완성. 근거 부족하면 LLM 호출 자체를 생략 |
| `12_compare_embeddings.py` | 상투문구 제거 전후 검색 품질 비교 실험 |
| `13_pubmed_rag.py` | PubMed 초록 기반 RAG. 번역 단계 한자 혼입 이슈로 현재 막힘 |

### 데이터

- `data/kdca_disease_index.json` — 병명 624개 → 국가건강정보포털 콘텐츠번호 매핑
- `data/kdca_corpus.json` — 611개 질환 실제 본문 텍스트
- `data/kdca_embeddings.json` / `kdca_embeddings_clean.json` — bge-m3 임베딩 (clean 버전이 기본)

## 하네스가 방어하는 것들

LLM 혼자는 못 믿어서 코드로 감싼 부분들:
- 응답 시간/생성 길이 무제한 방지 (`timeout`, `num_predict`)
- 반복 루프(짧은 조각 반복 + 문장/문단 반복) 감지 → 재시도
- 가짜 도구 호출(텍스트로 JSON 흉내) 감지
- 도구 인자 검증 (한자·깨진 문자 섞이면 실패로 판단)
- 도구 실행/검색 결과가 부실하면 **LLM을 아예 다시 안 부르고** 실패로 종료
  (근거 없이 LLM이 자체 지식으로 지어내는 것 방지)

## 알려진 한계

- 대화형 CLI(`07_interactive_agent.py`)는 Claude Code 채팅 환경 안에서 실시간 입력이
  안 됨 — 별도 macOS 터미널 앱에서 직접 실행해야 진짜 대화형으로 써짐
- llama3.2:3b는 영어→한국어 의학 번역 시 한자(중국어)가 섞여 나오는 경우가 있음
  (PubMed RAG가 막힌 이유)

## 진행 방식

- 한 번에 다 만들지 않고, 작은 단위로 쪼개서 구현 → 실행 결과 확인 → 다음 단계로
- 코드 쓰기 전에 개념 설명 먼저
- 버그/한계를 발견하면 숨기지 않고 실제 로그와 함께 보고 → 최소 범위로만 수정
