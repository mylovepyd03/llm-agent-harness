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
    --teal: #0f9b8e;
    --teal-dark: #0a7d72;
    --teal-soft: #e6f6f4;
    --ink: #17212b;
    --muted: #7b8794;
    --line: #e8edf2;
    --card: #ffffff;
  }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }

  body {
    margin: 0; min-height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo",
                 Pretendard, "Noto Sans KR", system-ui, sans-serif;
    color: var(--ink);
    background:
      radial-gradient(900px 500px at 12% -5%, #d9f1ee 0%, transparent 60%),
      radial-gradient(800px 500px at 95% 8%, #e4ecfb 0%, transparent 55%),
      linear-gradient(180deg, #f6f9fb 0%, #eef3f7 100%);
    background-attachment: fixed;
    display: flex; justify-content: center; align-items: stretch;
  }

  .app {
    width: 100%; max-width: 760px; display: flex; flex-direction: column;
    min-height: 100vh; background: rgba(255,255,255,0.72);
    backdrop-filter: blur(12px);
    border-left: 1px solid rgba(255,255,255,0.7);
    border-right: 1px solid rgba(255,255,255,0.7);
    box-shadow: 0 10px 60px rgba(23, 45, 70, 0.10);
  }

  /* ── 헤더 ───────────────────────────────── */
  header {
    position: sticky; top: 0; z-index: 10;
    display: flex; align-items: center; gap: 12px;
    padding: 16px 20px;
    background: linear-gradient(135deg, var(--teal) 0%, #11867f 55%, #0e6f82 100%);
    color: #fff;
    box-shadow: 0 6px 24px rgba(13, 120, 110, 0.22);
  }
  .logo {
    width: 40px; height: 40px; flex-shrink: 0; border-radius: 13px;
    display: grid; place-items: center; font-size: 1.25rem;
    background: rgba(255,255,255,0.2);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.28);
  }
  .titles h1 { margin: 0; font-size: 1.02rem; font-weight: 700; letter-spacing: -0.2px; }
  .titles .sub {
    margin: 3px 0 0; font-size: 0.72rem; opacity: 0.9;
    display: flex; align-items: center; gap: 6px;
  }
  .dot {
    width: 6px; height: 6px; border-radius: 50%; background: #7dffb2;
    box-shadow: 0 0 0 0 rgba(125,255,178,0.8); animation: pulse 2.2s infinite;
  }
  @keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(125,255,178,0.7); }
    70%  { box-shadow: 0 0 0 7px rgba(125,255,178,0); }
    100% { box-shadow: 0 0 0 0 rgba(125,255,178,0); }
  }
  .spacer { flex: 1; }
  .ghost {
    background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.22);
    color: #fff; border-radius: 10px; padding: 7px 12px;
    font-size: 0.76rem; font-weight: 600; cursor: pointer;
    transition: background 0.18s ease, transform 0.12s ease;
  }
  .ghost:hover { background: rgba(255,255,255,0.3); }
  .ghost:active { transform: scale(0.96); }

  /* ── 대화 영역 ──────────────────────────── */
  #log {
    flex: 1; display: flex; flex-direction: column; gap: 14px;
    padding: 24px 18px 132px; overflow-y: auto; scroll-behavior: smooth;
  }
  #log::-webkit-scrollbar { width: 8px; }
  #log::-webkit-scrollbar-thumb { background: #d4dde6; border-radius: 99px; }

  .welcome { margin: auto 0; text-align: center; padding: 10px; }
  .welcome .big { font-size: 2.6rem; }
  .welcome h2 { margin: 10px 0 6px; font-size: 1.1rem; font-weight: 700; }
  .welcome p { margin: 0 0 18px; font-size: 0.86rem; color: var(--muted); line-height: 1.6; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
  .chip {
    background: var(--card); border: 1px solid var(--line); color: var(--teal-dark);
    border-radius: 999px; padding: 9px 15px; font-size: 0.82rem; font-weight: 600;
    cursor: pointer; transition: all 0.18s ease;
    box-shadow: 0 2px 8px rgba(23, 45, 70, 0.05);
  }
  .chip:hover { background: var(--teal-soft); border-color: #bfe6e1; transform: translateY(-1px); }

  .row { display: flex; gap: 9px; max-width: 86%; animation: rise 0.3s ease both; }
  .row.user { align-self: flex-end; flex-direction: row-reverse; }
  .row.bot { align-self: flex-start; }
  @keyframes rise {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: none; }
  }
  .avatar {
    flex-shrink: 0; width: 30px; height: 30px; border-radius: 50%;
    display: grid; place-items: center; font-size: 0.92rem;
    background: var(--card); box-shadow: 0 2px 7px rgba(23,45,70,0.09);
  }
  .row.bot .avatar { background: var(--teal-soft); }
  .msg {
    padding: 12px 16px; border-radius: 18px; white-space: pre-wrap;
    line-height: 1.62; font-size: 0.93rem; word-break: break-word;
    box-shadow: 0 2px 10px rgba(23, 45, 70, 0.06);
  }
  .row.user .msg {
    background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dark) 100%);
    color: #fff; border-bottom-right-radius: 5px;
  }
  .row.bot .msg {
    background: var(--card); color: var(--ink);
    border: 1px solid var(--line); border-bottom-left-radius: 5px;
  }

  /* 타이핑 인디케이터 */
  .typing { display: flex; gap: 4px; padding: 4px 2px; }
  .typing i {
    width: 7px; height: 7px; border-radius: 50%; background: #b6c2cd;
    animation: blink 1.3s infinite ease-in-out;
  }
  .typing i:nth-child(2) { animation-delay: 0.18s; }
  .typing i:nth-child(3) { animation-delay: 0.36s; }
  @keyframes blink {
    0%, 80%, 100% { opacity: 0.3; transform: translateY(0); }
    40%           { opacity: 1;   transform: translateY(-3px); }
  }
  .hint-text { font-size: 0.76rem; color: var(--muted); margin-top: 5px; }

  /* ── 입력창 ─────────────────────────────── */
  form {
    position: fixed; bottom: 0; left: 50%; transform: translateX(-50%);
    width: 100%; max-width: 760px; display: flex; gap: 9px; align-items: center;
    padding: 14px 16px calc(16px + env(safe-area-inset-bottom));
    background: linear-gradient(180deg, rgba(246,249,251,0) 0%, rgba(246,249,251,0.95) 38%, #f6f9fb 100%);
  }
  input {
    flex: 1; padding: 14px 18px; border-radius: 999px;
    border: 1px solid var(--line); background: var(--card);
    font-size: 0.94rem; color: var(--ink); outline: none;
    box-shadow: 0 4px 18px rgba(23, 45, 70, 0.07);
    transition: border-color 0.18s ease, box-shadow 0.18s ease;
  }
  input::placeholder { color: #a9b4bf; }
  input:focus {
    border-color: var(--teal);
    box-shadow: 0 0 0 3px rgba(15,155,142,0.14), 0 4px 18px rgba(23,45,70,0.07);
  }
  .send {
    width: 48px; height: 48px; flex-shrink: 0; border: none; border-radius: 50%;
    background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dark) 100%);
    color: #fff; font-size: 1.1rem; cursor: pointer;
    display: grid; place-items: center;
    box-shadow: 0 6px 18px rgba(15, 155, 142, 0.34);
    transition: transform 0.14s ease, box-shadow 0.18s ease;
  }
  .send:hover { transform: translateY(-1px); box-shadow: 0 9px 22px rgba(15,155,142,0.4); }
  .send:active { transform: scale(0.94); }
  .send:disabled { opacity: 0.45; cursor: default; transform: none; box-shadow: none; }

  @media (max-width: 480px) {
    #log { padding: 18px 13px 126px; }
    .row { max-width: 92%; }
    .titles .sub { display: none; }
  }
</style>
</head>
<body>
<div class="app">
  <header>
    <div class="logo">🩺</div>
    <div class="titles">
      <h1>내 손안의 의사</h1>
      <div class="sub"><span class="dot"></span> 국가건강정보포털 · MedlinePlus(NIH) · 위키피디아 · PubMed 기반</div>
    </div>
    <div class="spacer"></div>
    <button class="ghost" id="summaryBtn" type="button">요약</button>
    <button class="ghost" id="resetBtn" type="button">새 대화</button>
  </header>

  <div id="log"></div>

  <form id="form">
    <input id="input" autocomplete="off" placeholder="어떤 증상이 있으신가요?">
    <button class="send" id="sendBtn" type="submit" aria-label="보내기">↑</button>
  </form>
</div>
<script>
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');

const EXAMPLES = ['위염이 뭐야?', '머리가 아프고 속이 메스꺼워요', '당뇨병 관련 논문 찾아줘'];

function showWelcome() {
  log.innerHTML = '';
  const box = document.createElement('div');
  box.className = 'welcome';
  box.innerHTML =
    '<div class="big">🩺</div>' +
    '<h2>어디가 불편하신가요?</h2>' +
    '<p>증상을 자유롭게 설명해주시면 관련 질환 정보를 찾아드려요.<br>' +
    '정확한 병명을 모르셔도 괜찮아요.</p>';
  const chips = document.createElement('div');
  chips.className = 'chips';
  EXAMPLES.forEach((text) => {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.type = 'button';
    chip.textContent = text;
    chip.addEventListener('click', () => { input.value = text; send(); });
    chips.appendChild(chip);
  });
  box.appendChild(chips);
  log.appendChild(box);
}

function clearWelcome() {
  const w = log.querySelector('.welcome');
  if (w) w.remove();
}

function addMsg(text, who) {
  clearWelcome();
  const row = document.createElement('div');
  row.className = 'row ' + who;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = who === 'user' ? '🙂' : '🩺';
  const bubble = document.createElement('div');
  bubble.className = 'msg';
  bubble.textContent = text;
  row.append(avatar, bubble);
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
  return row;
}

function addTyping() {
  clearWelcome();
  const row = document.createElement('div');
  row.className = 'row bot';
  row.innerHTML =
    '<div class="avatar">🩺</div>' +
    '<div class="msg"><div class="typing"><i></i><i></i><i></i></div>' +
    '<div class="hint-text">자료를 찾고 있어요 · 수십 초 걸릴 수 있어요</div></div>';
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
  return row;
}

async function send() {
  const question = input.value.trim();
  if (!question || sendBtn.disabled) return;
  addMsg(question, 'user');
  input.value = '';
  sendBtn.disabled = true;
  const typing = addTyping();
  try {
    const resp = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question}),
    });
    const data = await resp.json();
    typing.remove();
    addMsg(data.answer, 'bot');
  } catch (err) {
    typing.remove();
    addMsg('오류가 발생했어요: ' + err, 'bot');
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener('submit', (e) => { e.preventDefault(); send(); });

document.getElementById('summaryBtn').addEventListener('click', async () => {
  const resp = await fetch('/api/summary');
  const data = await resp.json();
  addMsg(data.summary, 'bot');
});

document.getElementById('resetBtn').addEventListener('click', async () => {
  await fetch('/api/reset', {method: 'POST'});
  showWelcome();
});

showWelcome();
input.focus();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE
