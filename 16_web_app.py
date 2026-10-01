"""
C단계 이후: 지금까지 만든 결정론적 파이프라인(14_integrated_agent.py)을 그대로
가져다 쓰되, 터미널 input() 대신 브라우저로 접속하는 웹 UI를 씌운다.

중요한 설계 포인트: 에이전트 로직(run_agent_turn, MedicalConversationState 등)은
단 한 줄도 다시 안 만든다 - 14_integrated_agent.py를 모듈로 그대로 불러와서
재사용한다. 이 파일이 새로 하는 일은 딱 하나, "HTTP 요청 <-> 함수 호출"을
연결하는 것뿐이다. 파일 이름이 숫자로 시작해서(`14_integrated_agent`) 일반
import 문법으로는 못 불러오므로 importlib로 불러온다(테스트할 때 썼던 방식과 동일).

실행: /opt/anaconda3/bin/python -m uvicorn 16_web_app:app --reload
그다음 브라우저에서 http://localhost:8000 접속.

주의: 세션/사용자 구분이 없는 가장 단순한 버전이다 - 서버 하나에 대화 상태
(messages, state)가 전역으로 하나만 있다. 개인이 혼자 로컬에서 쓰는 걸
전제로 한 설계이고, 여러 명이 동시에 쓰면 대화가 서로 섞인다(다음에 다룰 한계).
"""
import importlib.util
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

_AGENT_PATH = os.path.join(os.path.dirname(__file__), "14_integrated_agent.py")
_spec = importlib.util.spec_from_file_location("integrated_agent", _AGENT_PATH)
agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent)

app = FastAPI(title="내 손안의 의사")

# 서버 전역 대화 상태 - 위 모듈 docstring 참고
messages: list = []
state = agent.MedicalConversationState()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    answer = agent.run_agent_turn(messages, req.question, state)
    return ChatResponse(answer=answer)


@app.get("/api/summary")
def summary() -> dict:
    return {"summary": state.summary()}


@app.post("/api/reset")
def reset() -> dict:
    messages.clear()
    global state
    state = agent.MedicalConversationState()
    return {"ok": True}


_PAGE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>내 손안의 의사</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 720px;
         margin: 0 auto; padding: 16px; background: #f7f7f8; }
  h1 { font-size: 1.2rem; }
  #log { display: flex; flex-direction: column; gap: 10px; margin-bottom: 90px; }
  .msg { padding: 10px 14px; border-radius: 12px; max-width: 85%; white-space: pre-wrap;
         line-height: 1.5; }
  .user { align-self: flex-end; background: #4f7cff; color: white; }
  .bot  { align-self: flex-start; background: white; border: 1px solid #ddd; }
  .pending { opacity: 0.6; }
  form { position: fixed; bottom: 0; left: 0; right: 0; display: flex; gap: 8px;
         padding: 12px; background: #f7f7f8; border-top: 1px solid #ddd; }
  input { flex: 1; padding: 10px; border-radius: 8px; border: 1px solid #ccc; font-size: 1rem; }
  button { padding: 10px 16px; border-radius: 8px; border: none; background: #4f7cff;
           color: white; font-size: 1rem; cursor: pointer; }
  button:disabled { opacity: 0.5; }
  #summaryBtn { background: #888; margin-left: 6px; }
</style>
</head>
<body>
<h1>내 손안의 의사</h1>
<div id="log"></div>
<form id="form">
  <input id="input" autocomplete="off" placeholder="증상이나 궁금한 병명을 물어보세요">
  <button id="sendBtn" type="submit">보내기</button>
  <button id="summaryBtn" type="button">요약</button>
  <button id="resetBtn" type="button">새 대화</button>
</form>
<script>
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');

function addMsg(text, who) {
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  div.textContent = text;
  log.appendChild(div);
  window.scrollTo(0, document.body.scrollHeight);
  return div;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  addMsg(question, 'user');
  input.value = '';
  sendBtn.disabled = true;
  const pending = addMsg('생각하는 중... (검색 내용에 따라 수십 초 걸릴 수 있어요)', 'bot pending');
  try {
    const resp = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question}),
    });
    const data = await resp.json();
    pending.remove();
    addMsg(data.answer, 'bot');
  } catch (err) {
    pending.remove();
    addMsg('오류가 발생했어요: ' + err, 'bot');
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});

document.getElementById('summaryBtn').addEventListener('click', async () => {
  const resp = await fetch('/api/summary');
  const data = await resp.json();
  addMsg(data.summary, 'bot');
});

document.getElementById('resetBtn').addEventListener('click', async () => {
  await fetch('/api/reset', {method: 'POST'});
  log.innerHTML = '';
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE
