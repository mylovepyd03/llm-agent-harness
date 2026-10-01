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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>내 손안의 의사</title>
<style>
  :root {
    --accent: #3b6fe0; --accent-dark: #2f59b8; --bg: #eef1f6; --card: #ffffff;
    --text: #1f2430; --muted: #8a93a6; --border: #e4e7ee;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo",
                 "Pretendard", system-ui, sans-serif;
    margin: 0; background: var(--bg); color: var(--text);
    display: flex; justify-content: center;
  }
  .app {
    width: 100%; max-width: 720px; min-height: 100vh; background: var(--card);
    display: flex; flex-direction: column;
    box-shadow: 0 0 40px rgba(20, 30, 60, 0.06);
  }
  header {
    padding: 18px 20px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent-dark));
    color: white; position: sticky; top: 0; z-index: 5;
  }
  header .icon { font-size: 1.4rem; }
  header h1 { font-size: 1.05rem; margin: 0; font-weight: 600; }
  header p { margin: 2px 0 0; font-size: 0.75rem; opacity: 0.85; }
  header .spacer { flex: 1; }
  .iconbtn {
    background: rgba(255,255,255,0.18); border: none; color: white;
    border-radius: 8px; padding: 7px 11px; font-size: 0.78rem; cursor: pointer;
  }
  .iconbtn:hover { background: rgba(255,255,255,0.3); }
  #log { flex: 1; display: flex; flex-direction: column; gap: 12px;
         padding: 18px 16px 100px; overflow-y: auto; }
  .empty-hint { margin: auto; text-align: center; color: var(--muted); font-size: 0.88rem; }
  .row { display: flex; gap: 8px; max-width: 88%; }
  .row.user { align-self: flex-end; flex-direction: row-reverse; }
  .row.bot { align-self: flex-start; }
  .avatar { flex-shrink: 0; width: 28px; height: 28px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.95rem; background: var(--bg); }
  .msg { padding: 11px 15px; border-radius: 16px; white-space: pre-wrap;
         line-height: 1.55; font-size: 0.93rem; }
  .row.user .msg { background: var(--accent); color: white; border-bottom-right-radius: 4px; }
  .row.bot .msg { background: #f3f5f9; color: var(--text); border-bottom-left-radius: 4px; }
  .row.pending .msg { color: var(--muted); font-style: italic; }
  form {
    position: fixed; bottom: 0; left: 50%; transform: translateX(-50%);
    width: 100%; max-width: 720px; display: flex; gap: 8px;
    padding: 12px 14px calc(12px + env(safe-area-inset-bottom));
    background: var(--card); border-top: 1px solid var(--border);
  }
  input {
    flex: 1; padding: 11px 14px; border-radius: 22px; border: 1px solid var(--border);
    font-size: 0.93rem; background: var(--bg); color: var(--text); outline: none;
  }
  input:focus { border-color: var(--accent); }
  button[type=submit] {
    padding: 0 18px; border-radius: 22px; border: none; background: var(--accent);
    color: white; font-size: 0.9rem; font-weight: 600; cursor: pointer;
  }
  button[type=submit]:hover { background: var(--accent-dark); }
  button:disabled { opacity: 0.5; cursor: default; }
</style>
</head>
<body>
<div class="app">
  <header>
    <span class="icon">🩺</span>
    <div>
      <h1>내 손안의 의사</h1>
      <p>증상/병명을 물어보면 찾아서 알려드려요</p>
    </div>
    <div class="spacer"></div>
    <button class="iconbtn" id="summaryBtn" type="button">요약</button>
    <button class="iconbtn" id="resetBtn" type="button">새 대화</button>
  </header>
  <div id="log"><div class="empty-hint">예: "위염이 뭐야?", "머리가 아프고 속이 메스꺼워요"</div></div>
  <form id="form">
    <input id="input" autocomplete="off" placeholder="증상이나 궁금한 병명을 물어보세요">
    <button id="sendBtn" type="submit">보내기</button>
  </form>
</div>
<script>
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');

const emptyHint = log.querySelector('.empty-hint');

function addMsg(text, who, pending) {
  if (emptyHint) emptyHint.remove();
  const row = document.createElement('div');
  row.className = 'row ' + who + (pending ? ' pending' : '');
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = who === 'user' ? '🙂' : '🩺';
  const bubble = document.createElement('div');
  bubble.className = 'msg';
  bubble.textContent = text;
  row.appendChild(avatar);
  row.appendChild(bubble);
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
  return row;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  addMsg(question, 'user');
  input.value = '';
  sendBtn.disabled = true;
  const pending = addMsg('생각하는 중... (검색 내용에 따라 수십 초 걸릴 수 있어요)', 'bot', true);
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
  log.innerHTML = '<div class="empty-hint">예: "위염이 뭐야?", "머리가 아프고 속이 메스꺼워요"</div>';
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE
