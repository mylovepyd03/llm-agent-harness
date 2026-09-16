# llm-agent-harness

LLM 기반 에이전트/하네스 시스템을 구현해보는 개인 학습 프로젝트.
[toy-llm-project](../toy-llm-project)(트랜스포머를 밑바닥부터 구현)와는 목적이 다름 —
여기서는 **이미 있는 LLM을 "두뇌"로 가져다 쓰고, 그 주변에 도구 호출/검색/상태 관리 같은
"시스템"을 어떻게 쌓는지**를 배우는 게 목표. 파이썬 초보자 학습용, 매 단계 개념 설명 +
작은 단위로 확인하며 진행.

## 두 프로젝트 비교

| | toy-llm-project | llm-agent-harness (이 프로젝트) |
|---|---|---|
| 배우는 것 | 트랜스포머 내부 구조 (attention, 학습 루프 등) | 에이전트 구조 (도구 호출, 검색, 상태 관리) |
| LLM | 직접 만듦 (파라미터 9만개 미니 GPT) | 기존 오픈소스 모델을 가져다 씀 (Ollama) |
| 학습(훈련) 여부 | O (직접 학습시킴) | X (이미 학습된 모델을 호출만 함) |

## 실행 환경

- LLM: **Ollama**로 로컬에서 실행 (API 키/비용 불필요)
  - 설치: `brew install ollama` (완료)
  - 서비스: `brew services start ollama` (완료, 백그라운드 상시 실행)
  - 모델: `llama3.2` (3B, 약 2GB) — 도구 호출(tool calling) 지원
  - 확인: `curl http://localhost:11434/api/version`
- 파이썬: conda base 환경 사용 (`/opt/anaconda3/bin/python`) — toy-llm-project와 동일
- 프로젝트 위치: `/Users/princess-yedam/llm-agent-harness`
- git 저장소 아님 (로컬 전용)

## 로드맵 (난이도순, 아직 확정 아님 — 진행하며 조정)

1. [x] 프로젝트 폴더 만들기
2. [x] 로컬 LLM 환경 구성 (Ollama 설치 + llama3.2 다운로드)
3. [ ] **A. 기초 도구 호출 에이전트** — LLM이 계산기/파일 읽기 같은 간단한 도구를 스스로
   판단해서 호출하는 최소 에이전트 루프 (ReAct 패턴)
4. [ ] **B. RAG 질의응답 에이전트** — toy-llm-project에서 겪은 "환각(hallucination)" 문제를
   실제로 해결. 문서 검색 + LLM 답변 결합
5. [ ] **C. 미니 하네스** — 여러 단계를 스스로 계획하고, 도구를 순서대로 호출하며 상태를
   기억하는 좀 더 범용적인 구조 (지금 이 대화 — Claude Code — 가 하는 일의 축소판)

## 진행 방식

- 한 번에 다 만들지 않고, 작은 단위로 쪼개서 구현 → 실행 결과 확인 → 다음 단계로
- 코드 쓰기 전에 개념 설명 먼저
