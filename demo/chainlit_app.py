"""
法律问答助手 — Chainlit 前端
启动：chainlit run demo/chainlit_app.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import chainlit as cl
from pipeline import LegalRAGPipeline
from session_memory import SessionMemory

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        import os
        _pipeline = LegalRAGPipeline(
            embed_backend=os.environ.get("EMBED_BACKEND", "ollama"),
            llm_backend=os.environ.get("LLM_BACKEND", "ollama"),
            embed_model=os.environ.get("EMBED_MODEL", "bge-m3"),
            llm_model=os.environ.get("LLM_MODEL", "qwen2.5:7b"),
            enable_hyde=False,
        )
    return _pipeline


@cl.on_chat_start
async def start():
    get_pipeline()
    cl.user_session.set("memory", SessionMemory())
    cl.user_session.set("first_q", "")

    await cl.Message(
        content="## 法律问答助手\n\n"
                "基于 **303 部中国法律**，提供专业法律解答。\n\n"
                "**使用方法**\n"
                "- 直接输入法律问题\n"
                "- 上传 `.txt` / `.md` / `.csv` 文件，提取文字内容\n"
                "- 支持多轮追问，自动关联上下文\n\n"
                "👇 试试这些问题：",
        actions=[
            cl.Action(name="s", label="草原保护有什么方针？", payload={"q": "草原保护有什么方针？"}),
            cl.Action(name="s", label="醉驾怎么处理？", payload={"q": "醉驾怎么处理？"}),
            cl.Action(name="s", label="试用期最长多久？", payload={"q": "试用期最长多久？"}),
            cl.Action(name="s", label="什么情况下可以解除劳动合同？", payload={"q": "什么情况下可以解除劳动合同？"}),
            cl.Action(name="s", label="侵犯著作权要承担什么责任？", payload={"q": "侵犯著作权要承担什么责任？"}),
        ],
    ).send()


@cl.action_callback("s")
async def on_sample(action: cl.Action):
    q = action.payload["q"]
    await cl.Message(content=q, author="User").send()
    await answer(q)


@cl.on_message
async def on_message(msg: cl.Message):
    question = msg.content
    files = []

    if msg.elements:
        for el in msg.elements:
            if isinstance(el, cl.File):
                try:
                    p = Path(el.path)
                    if p.suffix.lower() in {".txt", ".md", ".csv", ".json", ".log", ".yaml", ".yml"}:
                        text = p.read_text(encoding="utf-8", errors="replace")
                        if len(text) > 8000:
                            text = text[:8000] + "..."
                        files.append(f"【{el.name}】\n{text}")
                except Exception:
                    pass

    if files:
        question = (msg.content or "分析文件涉及的法律问题") + "\n\n--- 上传文件 ---\n" + "\n".join(files) + "\n---"

    await answer(question, bool(files))


async def answer(question: str, has_files: bool = False):
    pipe = get_pipeline()
    mem = cl.user_session.get("memory")

    if cl.user_session.get("first_q") == "":
        cl.user_session.set("first_q", question)

    history = mem.get_history_str()
    msg = cl.Message(content="")
    await msg.send()

    try:
        t0 = time.time()
        sq = question.split("--- 上传文件 ---")[0].strip()[:800] if has_files else question

        rw = pipe.rewriter(sq, history)
        matched = pipe.fetcher.keyword_lookup(rw) or pipe.law_matcher(rw, top_k=3)
        ranked = pipe.ranker(rw, pipe.fetcher(matched), top_k=10, law_filter=matched)
        stage_ms = round((time.time() - t0) * 1000)

        await msg.stream_token(f"🔎 涉及法律：" + "、".join(matched) + ("  [已读文件]" if has_files else "") + "\n\n")

        t1 = time.time()
        tokens = []

        def cb(d):
            tokens.append(d)

        llm = pipe.qa.stream(question, ranked, callback=cb)
        answer_ms = round((time.time() - t1) * 1000)

        for t in tokens:
            await msg.stream_token(t)

        if ranked:
            parts = ["\n\n---\n### 参考法条\n"]
            for i, (doc, s) in enumerate(ranked, 1):
                parts.append(
                    f"{i}. **《{doc.metadata['law_title']}》{doc.metadata['article_no']}** (相似度 {s:.3f})\n"
                    f"   {doc.metadata['text']}\n"
                )
            await msg.stream_token("\n".join(parts))

        await msg.stream_token(
            f"\n---\n检索 {stage_ms}ms · 回答 {answer_ms}ms · 总计 {answer_ms + stage_ms}ms"
        )

        mem.record(question=question, answer="".join(tokens), matched_laws=matched, ranked_articles=ranked)
        cl.user_session.set("memory", mem)

    except Exception as e:
        await msg.stream_token(f"\n\n出错了：{e}")

    await msg.update()


@cl.on_chat_end
async def end():
    cl.user_session.set("memory", SessionMemory())
    cl.user_session.set("first_q", "")
