"""
FastAPI 法律问答服务
================================================================
单端口：用户输入 → 系统预生成背景（首次）或直接多轮追问
点"下一个问题"清除记忆，重新开始

流程：
  首次提问 → Stage 1-4 预生成（不回显）→ 返回结果 + 记忆种子
  后续追问 → 自动拼接 history → Stage 1-4 → 返回
  点"下一个问题" → 清除 memory，重新首次流程
"""

import sys
import os
import uuid
import time
import re
from pathlib import Path
from typing import Optional

# 注入 demo 路径
DEMO_DIR = Path(__file__).parent
sys.path.insert(0, str(DEMO_DIR))

from pipeline import LegalRAGPipeline
from session_memory import SessionMemory
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# =====================================================================
#  配置
# =====================================================================

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "ollama")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
PORT = int(os.environ.get("PORT", "8000"))

# =====================================================================
#  FastAPI App
# =====================================================================

app = FastAPI(title="法律问答助手", version="1.0")


# =====================================================================
#  会话存储（内存，每个会话一个 SessionMemory）
# =====================================================================

class ChatSession:
    def __init__(self):
        self.memory = SessionMemory()
        self.pipeline: Optional[LegalRAGPipeline] = None
        self.first_question: str = ""      # 首次问题
        self.first_result_sent: bool = False  # 首次结果是否已发送

    def reset(self):
        self.memory = SessionMemory()
        self.first_question = ""
        self.first_result_sent = False


sessions: dict[str, ChatSession] = {}


def get_session(session_id: str) -> ChatSession:
    if session_id not in sessions:
        sessions[session_id] = ChatSession()
    return sessions[session_id]


# =====================================================================
#  请求 / 响应模型
# =====================================================================

class AskRequest(BaseModel):
    session_id: str
    question: str


class AskResponse(BaseModel):
    session_id: str
    is_first: bool          # 是否是首次提问
    answer: str              # 最终回答
    law_entities: list[str]  # 涉及的法律名
    articles: list[dict]     # 精排法条
    timings: dict[str, float]  # 各阶段耗时


class ResetRequest(BaseModel):
    session_id: str


class HealthResponse(BaseModel):
    status: str
    session_count: int


# =====================================================================
#  API 路由
# =====================================================================

@app.post("/api/ask-stream")
def ask_stream(req: AskRequest):
    """
    SSE 流式接口：
      Stage 1-4 同步跑完 → 先发送阶段结果 + 法律名 + 法条
      Stage 5 → 每个 token 实时写进 SSE
    前端用 fetch + ReadableStream 接收。
    """
    import asyncio
    import json
    from fastapi.responses import StreamingResponse

    async def event_stream():
        session = get_session(req.session_id)
        is_first = (session.first_question == "")

        if is_first:
            session.first_question = req.question
            session.pipeline = LegalRAGPipeline(
                embed_backend=EMBED_BACKEND,
                llm_backend=LLM_BACKEND,
                embed_model=EMBED_MODEL,
                llm_model=LLM_MODEL,
                enable_hyde=False,
            )
            session.first_result_sent = False

        history_str = session.memory.get_history_str()

        # Stage 1-4（毫秒级，全部同步跑完）
        t0 = time.time()
        rewritten = session.pipeline.rewriter(req.question, history_str)
        entity_hint = session.pipeline.fetcher.keyword_lookup(rewritten)
        matched = entity_hint or session.pipeline.law_matcher(rewritten, top_k=3)
        candidates = session.pipeline.fetcher(matched)
        ranked = session.pipeline.ranker(rewritten, candidates, top_k=10, law_filter=matched)
        stage_ms = round((time.time() - t0) * 1000)

        articles = [
            {
                "law": doc.metadata.get("law_title", ""),
                "article_no": doc.metadata.get("article_no", ""),
                "score": round(score, 3),
            }
            for doc, score in ranked
        ]

        # 先发送元数据（Stage 1-4 结果）
        meta = {
            "type": "meta",
            "is_first": is_first,
            "rewritten": rewritten,
            "law_entities": matched,
            "articles": articles,
            "stage_ms": stage_ms,
            "question": req.question,
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        # Stage 5 流式：累积 token 列表 + SSE 发 token
        tokens: list[str] = []
        ttft_recorded = [False]

        def chunk_callback(delta: str):
            tokens.append(delta)
            payload = json.dumps({"type": "chunk", "delta": delta}, ensure_ascii=False)
            yield f"data: {payload}\n\n"

        t0 = time.time()
        llm_meta = session.pipeline.qa.stream(req.question, ranked, callback=chunk_callback)
        total_ms = round((time.time() - t0) * 1000)
        answer_text = "".join(tokens)

        # 发 done
        done = {
            "type": "done",
            "total_ms": total_ms,
            "answer_ms": llm_meta.get("total_ms", 0),
            "ttft_ms": llm_meta.get("ttft_ms", 0),
            "tokens": llm_meta.get("tokens", 0),
        }
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

        # 写完后再写入记忆
        session.memory.record(req.question, answer_text, matched, ranked)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    核心接口：
    - 首次提问：初始化 pipeline + 预生成背景 + 返回结果
    - 追问：直接用 memory.get_history_str() 拼接历史
    """
    session = get_session(req.session_id)
    is_first = (session.first_question == "")

    # 首次：初始化 pipeline
    if is_first:
        session.first_question = req.question
        session.pipeline = LegalRAGPipeline(
            embed_backend=EMBED_BACKEND,
            llm_backend=LLM_BACKEND,
            embed_model=EMBED_MODEL,
            llm_model=LLM_MODEL,
            enable_hyde=False,
        )
        session.first_result_sent = False

    # 获取历史
    history_str = session.memory.get_history_str()

    # ---------- Stage 1: 改写 ----------
    t0 = time.time()
    rewritten = session.pipeline.rewriter(req.question, history_str)
    t_rewrite = time.time() - t0

    # ---------- Stage 1.75: 实体抽取 ----------
    t0 = time.time()
    entity_hint = session.pipeline.fetcher.keyword_lookup(rewritten)
    t_entity = time.time() - t0

    # ---------- Stage 2: 法律名 ----------
    t0 = time.time()
    if entity_hint:
        matched = entity_hint
    else:
        matched = session.pipeline.law_matcher(rewritten, top_k=3)
    t_match = time.time() - t0

    # ---------- Stage 3: 取法条 ----------
    t0 = time.time()
    candidates = session.pipeline.fetcher(matched)
    t_fetch = time.time() - t0

    # ---------- Stage 4: 精排 ----------
    t0 = time.time()
    ranked = session.pipeline.ranker(rewritten, candidates, top_k=10, law_filter=matched)
    t_rank = time.time() - t0

    # ---------- Stage 5: 回答 ----------
    t0 = time.time()
    answer, llm_meta = session.pipeline.qa(req.question, ranked)
    t_answer = time.time() - t0

    # 写入记忆
    session.memory.record(
        question=req.question,
        answer=answer,
        matched_laws=matched,
        ranked_articles=ranked,
    )

    timings = {
        "rewrite_ms": round(t_rewrite * 1000),
        "entity_ms": round(t_entity * 1000),
        "match_ms": round(t_match * 1000),
        "fetch_ms": round(t_fetch * 1000),
        "rank_ms": round(t_rank * 1000),
        "answer_ms": round(t_answer * 1000),
        "total_ms": round((t_rewrite + t_entity + t_match + t_fetch + t_rank + t_answer) * 1000),
    }

    articles = [
        {
            "law": doc.metadata.get("law_title", ""),
            "article_no": doc.metadata.get("article_no", ""),
            "score": round(score, 3),
        }
        for doc, score in ranked
    ]

    return AskResponse(
        session_id=req.session_id,
        is_first=is_first,
        answer=answer,
        law_entities=matched,
        articles=articles,
        timings=timings,
    )


@app.post("/api/reset")
def reset(req: ResetRequest):
    """清除当前会话记忆，重新开始"""
    if req.session_id in sessions:
        sessions[req.session_id].reset()
    return {"status": "ok", "session_id": req.session_id}


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", session_count=len(sessions))


@app.get("/api/new-session")
def new_session():
    """创建新会话，返回 session_id"""
    sid = str(uuid.uuid4())
    sessions[sid] = ChatSession()
    return {"session_id": sid}


# =====================================================================
#  前端页面
# =====================================================================

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>法律问答助手</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --green:       #22c55e;
    --green-dark:  #16a34a;
    --green-light: #dcfce7;
    --cyan:        #06b6d4;
    --cyan-light:  #cffafe;
    --bg:          #f8fafc;
    --surface:     #ffffff;
    --border:      #e2e8f0;
    --text:         #1e293b;
    --text-light:   #64748b;
    --accent:       #0d9488;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ---- 顶栏 ---- */
  header {
    background: linear-gradient(135deg, var(--green-dark), var(--accent));
    color: #fff;
    padding: 0 24px;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 12px rgba(34,197,94,0.25);
    position: sticky; top: 0; z-index: 100;
  }
  header .logo { font-size: 18px; font-weight: 700; letter-spacing: 1px; }
  header .logo span { opacity: 0.8; font-weight: 400; }
  .header-btn {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    color: #fff; padding: 6px 16px; border-radius: 20px;
    font-size: 13px; cursor: pointer; transition: background 0.2s;
  }
  .header-btn:hover { background: rgba(255,255,255,0.3); }

  /* ---- 主容器 ---- */
  main {
    flex: 1; max-width: 760px; width: 100%;
    margin: 0 auto; padding: 24px 16px;
    display: flex; flex-direction: column;
  }

  /* ---- 提示卡 ---- */
  .tip-card {
    background: linear-gradient(135deg, var(--green-light), var(--cyan-light));
    border: 1px solid #bbf7d0;
    border-radius: 12px; padding: 16px 20px;
    font-size: 13px; color: var(--text);
    margin-bottom: 16px; line-height: 1.6;
  }
  .tip-card strong { color: var(--green-dark); }

  /* ---- 消息列表 ---- */
  .messages { flex: 1; overflow-y: auto; margin-bottom: 16px; }

  .msg { margin-bottom: 20px; animation: fadeIn 0.3s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

  .msg-q {
    display: flex; align-items: flex-start; gap: 10px;
  }
  .msg-q .avatar {
    width: 34px; height: 34px; border-radius: 50%;
    background: var(--green); color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700; flex-shrink: 0;
  }
  .msg-q .bubble {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px; font-size: 14px; line-height: 1.7;
    max-width: 88%;
  }

  .msg-a .avatar {
    width: 34px; height: 34px; border-radius: 50%;
    background: var(--cyan); color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; flex-shrink: 0;
  }
  .msg-a .bubble {
    background: linear-gradient(135deg, #f0fdf4, #ecfeff);
    border: 1px solid #bbf7d0;
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px; font-size: 14px; line-height: 1.8;
    max-width: 88%;
  }

  .law-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
  .law-tag {
    background: var(--green-light); color: var(--green-dark);
    border: 1px solid #86efac; border-radius: 20px;
    padding: 2px 10px; font-size: 12px; font-weight: 500;
  }

  .articles { margin-top: 10px; }
  .article-item {
    display: flex; align-items: baseline; gap: 8px;
    font-size: 12px; color: var(--text-light);
    padding: 4px 0; border-bottom: 1px dashed var(--border);
  }
  .article-item .idx { color: var(--cyan); font-weight: 600; min-width: 20px; }
  .article-item .law { color: var(--green-dark); font-weight: 500; }
  .article-item .no { color: var(--text); font-weight: 600; }
  .article-item .score { margin-left: auto; color: #94a3b8; font-size: 11px; }

  .timing-bar {
    margin-top: 10px; font-size: 11px; color: #94a3b8;
    display: flex; gap: 8px; flex-wrap: wrap;
  }
  .timing-bar span { background: #f1f5f9; padding: 2px 8px; border-radius: 10px; }

  /* ---- 输入区 ---- */
  .input-area {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    display: flex; align-items: center;
    padding: 8px 8px 8px 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    gap: 10px;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .input-area:focus-within {
    border-color: var(--green);
    box-shadow: 0 4px 20px rgba(34,197,94,0.15);
  }
  .input-area input {
    flex: 1; border: none; outline: none;
    font-size: 15px; color: var(--text);
    background: transparent; line-height: 1.5;
  }
  .input-area input::placeholder { color: #94a3b8; }
  .send-btn {
    background: linear-gradient(135deg, var(--green), var(--accent));
    color: #fff; border: none;
    border-radius: 50%; width: 40px; height: 40px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; transition: transform 0.15s, opacity 0.15s; flex-shrink: 0;
  }
  .send-btn:hover { transform: scale(1.08); }
  .send-btn:active { transform: scale(0.95); }
  .send-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  /* ---- 加载 ---- */
  .loading {
    display: flex; align-items: center; gap: 8px;
    color: var(--text-light); font-size: 13px;
    padding: 8px 0;
  }
  .spinner {
    width: 18px; height: 18px;
    border: 2px solid var(--green-light);
    border-top-color: var(--green);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ---- 下一个问题按钮 ---- */
  .next-btn-wrap { text-align: center; margin-bottom: 12px; }
  .next-btn {
    background: transparent; border: 1px dashed var(--border);
    color: var(--text-light); font-size: 13px;
    padding: 8px 24px; border-radius: 20px;
    cursor: pointer; transition: all 0.2s;
  }
  .next-btn:hover {
    border-color: var(--green); color: var(--green-dark);
    background: var(--green-light);
  }

  /* ---- 空状态 ---- */
  .empty {
    text-align: center; padding: 60px 20px; color: #94a3b8;
  }
  .empty-icon { font-size: 48px; margin-bottom: 12px; }
  .empty p { font-size: 14px; line-height: 1.6; }
</style>
</head>
<body>

<header>
  <div class="logo">⚖️ 法律问答助手 <span>· 本地 RAG</span></div>
  <button class="header-btn" onclick="startNew()">+ 新会话</button>
</header>

<main>
  <div class="tip-card" id="tipCard">
    👋 您好！请输入您的法律问题，例如：
    <strong>"草原保护有什么方针？"</strong>、
    <strong>"醉驾怎么处理？"</strong>、
    <strong>"试用期最长多久？"</strong>
  </div>

  <div class="messages" id="messages">
    <div class="empty" id="emptyState">
      <div class="empty-icon">📖</div>
      <p>输入问题开始咨询<br>系统将基于法律法规提供专业解答</p>
    </div>
  </div>

  <div class="next-btn-wrap" id="nextWrap" style="display:none">
    <button class="next-btn" onclick="startNew()">➕ 下一个问题，清除当前记忆</button>
  </div>

  <div class="input-area">
    <input id="questionInput" type="text" placeholder="输入法律问题，回车发送 ..."
       autocomplete="off" onkeydown="if(event.key==='Enter') sendQuestion()" />
    <button class="send-btn" id="sendBtn" onclick="sendQuestion()">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
        <line x1="22" y1="2" x2="11" y2="13"/>
        <polygon points="22 2 15 22 22 11 2 13 2 22"/>
      </svg>
    </button>
  </div>
</main>

<script>
  let sessionId = "";
  let busy = false;

  // 初始化会话
  async function init() {
    const r = await fetch("/api/new-session");
    const d = await r.json();
    sessionId = d.session_id;
  }

  // 发送问题（SSE 流式）
  async function sendQuestion() {
    if (busy) return;
    const input = document.getElementById("questionInput");
    const q = input.value.trim();
    if (!q) return;

    busy = true;
    document.getElementById("sendBtn").disabled = true;
    input.value = "";

    // 清空空状态
    const empty = document.getElementById("emptyState");
    if (empty) empty.remove();

    // 显示用户消息
    appendMsg("q", q);
    document.getElementById("tipCard").style.display = "none";

    // 创建回答气泡（先显示，法律标签和法条列表后续动态填入）
    const bubbleId = "bubble-" + Date.now();
    const messages = document.getElementById("messages");
    messages.insertAdjacentHTML("beforeend",
      `<div class="msg msg-a" id="${bubbleId}">
        <div class="avatar">⚖️</div>
        <div class="bubble">
          <div class="law-tags" id="${bubbleId}-tags"></div>
          <div id="${bubbleId}-answer" class="answer-text" style="white-space:pre-wrap;line-height:1.8"></div>
          <div class="articles" id="${bubbleId}-arts"></div>
          <div class="timing-bar" id="${bubbleId}-timing"></div>
        </div>
      </div>`);
    messages.scrollTop = messages.scrollHeight;

    let rawAnswer = "";
    let stageMs = 0;
    let totalMs = 0;

    try {
      const resp = await fetch("/api/ask-stream", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ session_id: sessionId, question: q }),
      });

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value, { stream: true });
        const lines = text.split("\n");
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          let data;
          try { data = JSON.parse(line.slice(6)); } catch { continue; }
          if (data.type === "meta") {
            // Stage 1-4 结果到达，显示法律标签和法条列表
            stageMs = data.stage_ms;
            if (data.law_entities && data.law_entities.length) {
              const tagsEl = document.getElementById(bubbleId + "-tags");
              if (tagsEl) tagsEl.innerHTML = data.law_entities
                .map(l => `<span class="law-tag">${l}</span>`).join("");
            }
            if (data.articles && data.articles.length) {
              const artsEl = document.getElementById(bubbleId + "-arts");
              if (artsEl) artsEl.innerHTML = data.articles.map((a, i) =>
                `<div class="article-item">
                  <span class="idx">${i+1}.</span>
                  <span class="law">${a.law}</span>
                  <span class="no">${a.article_no}</span>
                  <span class="score">${a.score}</span>
                </div>`
              ).join("");
            }
          } else if (data.type === "chunk") {
            // 逐 token 显示答案
            rawAnswer += data.delta;
            const ansEl = document.getElementById(bubbleId + "-answer");
            if (ansEl) ansEl.textContent = rawAnswer;
            messages.scrollTop = messages.scrollHeight;
          } else if (data.type === "done") {
            // 结束时显示耗时
            totalMs = data.total_ms;
            const ttftMs = data.ttft_ms || 0;
            const timingEl = document.getElementById(bubbleId + "-timing");
            if (timingEl) {
              timingEl.innerHTML =
                `<span>检索 ${stageMs}ms</span>` +
                `<span>TTFT ${ttftMs}ms</span>` +
                `<span>回答 ${data.answer_ms}ms</span>` +
                `<span>总计 ${totalMs}ms</span>`;
            }
            document.getElementById("nextWrap").style.display = "block";
          }
        }
      }
    } catch(e) {
      const ansEl = document.getElementById(bubbleId + "-answer");
      if (ansEl) ansEl.textContent = "⚠️ 请求失败：" + e.message;
    }

    busy = false;
    document.getElementById("sendBtn").disabled = false;
    input.focus();
  }

  function appendMsg(role, text) {
    const messages = document.getElementById("messages");
    const avatar = role === "q" ? "👤" : "⚖️";
    messages.insertAdjacentHTML("beforeend",
      `<div class="msg msg-${role}">
        <div class="avatar">${avatar}</div>
        <div class="bubble">${escapeHtml(text)}</div>
      </div>`);
    messages.scrollTop = messages.scrollHeight;
  }

  async function startNew() {
    await fetch("/api/reset", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ session_id: sessionId }),
    });
    document.getElementById("messages").innerHTML =
      '<div class="empty" id="emptyState"><div class="empty-icon">📖</div><p>输入问题开始咨询<br>系统将基于法律法规提供专业解答</p></div>';
    document.getElementById("tipCard").style.display = "block";
    document.getElementById("nextWrap").style.display = "none";
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/\\n/g, "<br>");
  }

  init();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT


# =====================================================================
#  启动
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    print(f"启动法律问答服务 http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
