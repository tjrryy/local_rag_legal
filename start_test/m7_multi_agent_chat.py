"""
M7｜三 Agent 组合 + 多轮上下文
==============================

学习目标：
  - 理解为什么 RAG 系统要拆成多个 Agent（关注点分离）
  - 看到"滚动摘要 + 最近 N 轮"的多轮上下文管理是怎么写的
  - 跑一次 multi-turn demo，看 context 真的被传递了

设计：
  ┌─────────────────────────────────────────────────────────┐
  │  MultiAgentChat (Orchestrator)                          │
  │                                                         │
  │   user_query ──► RetrieverAgent ──► articles            │
  │                  │                     │                │
  │                  │              ┌──────┴──────┐         │
  │                  │              ▼             ▼         │
  │                  │   ConversationMemory   QAAgent       │
  │                  │        │              │    │         │
  │                  │        └──────────────┘    │         │
  │                  │              │             │         │
  │                  │              ▼             │         │
  │                  │      SummarizerAgent       │         │
  │                  │      (压缩历史)            │         │
  │                  └────────────────────────────┘         │
  └─────────────────────────────────────────────────────────┘

使用：
  1) 设置 DEEPSEEK_API_KEY
  2) python3 m3_legal_agents.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

# ---- 常量 ----
PERSIST_DIR = Path(__file__).parent / "vector_db"
COLLECTION = "legal_articles"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 多轮参数：保留最近 2 轮原文 + 滚动摘要
MAX_RECENT_TURNS = 2
K_RETRIEVE = 5


# ============================================================
#  Agent 1: 检索 Agent
# ============================================================

class RetrieverAgent:
    """
    职责：把用户问题变成 Top-K 法条。
    隐藏的细节：bge 指令前缀、ChromaDB 调用、结果格式化。
    """

    def __init__(self, k: int = K_RETRIEVE):
        print(f"[INIT] RetrieverAgent: 加载 ChromaDB ...")
        if not PERSIST_DIR.exists():
            raise FileNotFoundError(
                f"{PERSIST_DIR}/ 不存在，请先运行: python3 m4_chroma.py --build"
            )
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.db = Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION,
        )
        self.k = k
        print(f"[INIT] RetrieverAgent: 集合大小 = {self.db._collection.count()}")

    def __call__(self, query: str) -> str:
        """
        返回格式化的法条字符串（可以直接塞进 LLM prompt）。
        """
        # bge query 前缀
        prefixed = BGE_QUERY_PREFIX + query
        docs = self.db.similarity_search(prefixed, k=self.k)

        if not docs:
            return "（未检索到相关法条）"

        parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("law_title", "")
            artno = doc.metadata.get("article_no", "")
            text = doc.metadata.get("text", doc.page_content)
            parts.append(f"{i}. 《{title}》{artno}\n   {text}")
        return "\n\n".join(parts)


# ============================================================
#  Agent 2: 问答 Agent
# ============================================================

QA_PROMPT = """你是中国法律领域的智能助手。请严格根据【法条参考】和【对话历史】回答用户的最新问题。

要求：
1. 必须先引用法条原文，再做解释
2. 引用时用「《法律名》第X条」的格式
3. 如果【法条参考】中没有相关内容，请直接回答："现有法条中未直接规定该问题"
4. 如果用户的指代不明（如"它"、"那个规定"），请结合【对话历史】理解

【对话历史】
{history}

【法条参考】
{context}

【最新问题】
{question}

【你的回答】"""


class QAAgent:
    """
    职责：基于「法条 + 历史」生成答案。
    关键：prompt 里有 3 个槽位：history / context / question
    """

    def __init__(self):
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("未设置 DEEPSEEK_API_KEY 环境变量")
        self.llm = ChatOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=api_key,
            model="deepseek-chat",
            temperature=0.1,
        )

    def __call__(self, question: str, context: str, history: str) -> str:
        prompt = QA_PROMPT.format(
            history=history or "（无）",
            context=context,
            question=question,
        )
        return self.llm.invoke(prompt).content


# ============================================================
#  Agent 3: 总结 Agent
# ============================================================

SUMMARY_PROMPT = """你是对话摘要助手。请把「当前摘要」和「新增一轮对话」合并成更新后的摘要。

要求：
1. 保留：用户问过哪些法律主题、得到过哪些关键答案、引用过哪些法条
2. 删去：寒暄、重复内容
3. 控制在 200 字以内

【当前摘要】
{current_summary}

【新增对话】
Q: {question}
A: {answer}

【更新后的摘要】"""


class SummarizerAgent:
    """
    职责：把老的对话轮次压缩成一段摘要。
    触发时机：ConversationMemory 里的"原文"轮次超过 MAX_RECENT_TURNS。
    """

    def __init__(self):
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("未设置 DEEPSEEK_API_KEY 环境变量")
        self.llm = ChatOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=api_key,
            model="deepseek-chat",
            temperature=0.0,
        )

    def __call__(self, current_summary: str, question: str, answer: str) -> str:
        prompt = SUMMARY_PROMPT.format(
            current_summary=current_summary or "（无）",
            question=question,
            answer=answer,
        )
        return self.llm.invoke(prompt).content.strip()


# ============================================================
#  ConversationMemory: 多轮上下文管理
# ============================================================

@dataclass
class Turn:
    """一轮对话：用户问 + Agent 答。"""
    question: str
    answer: str


@dataclass
class ConversationMemory:
    """
    滚动摘要 + 最近 N 轮原文 的上下文管理器。

    数据结构：
      - summary: 1 段压缩后的早期对话
      - recent:  最近 N 轮 (Turn) 原文

    get_context() 返回拼好的 prompt 上下文段。
    """
    max_recent: int = MAX_RECENT_TURNS
    summary: str = ""
    recent: list[Turn] = field(default_factory=list)
    summarizer: SummarizerAgent = None  # type: ignore[assignment]

    def add_turn(self, question: str, answer: str) -> None:
        self.recent.append(Turn(question, answer))
        # 超过 N 轮 → 把最老的"原文"压成摘要
        if len(self.recent) > self.max_recent:
            oldest = self.recent.pop(0)
            self.summary = self.summarizer(
                current_summary=self.summary,
                question=oldest.question,
                answer=oldest.answer,
            )

    def get_context(self) -> str:
        """把摘要 + 最近轮次拼成可塞进 prompt 的字符串。"""
        parts = []
        if self.summary:
            parts.append(f"【历史摘要】\n{self.summary}")
        if self.recent:
            lines = []
            for t in self.recent:
                lines.append(f"Q: {t.question}\nA: {t.answer}")
            parts.append("【最近对话】\n" + "\n\n".join(lines))
        return "\n\n".join(parts) if parts else ""


# ============================================================
#  MultiAgentChat: 编排者
# ============================================================

class MultiAgentChat:
    """
    把 3 个 Agent 串起来，chat() 是主入口。
    """

    def __init__(self):
        print("=" * 60)
        print("[BOOT] 初始化三 Agent 系统 ...")
        self.retriever = RetrieverAgent()
        self.summarizer = SummarizerAgent()
        self.qa = QAAgent()
        # 注意：ConversationMemory 需要 summarizer，所以后建
        self.memory = ConversationMemory(summarizer=self.summarizer)
        print("[BOOT] 全部 Agent 就绪。输入 'q' 退出。")
        print("=" * 60)

    def chat(self, user_query: str) -> str:
        """单轮：query → answer，并更新 memory。"""
        print(f"\n[USER] {user_query}")

        # Step 1: 检索
        print("[1/3] RetrieverAgent: 检索中 ...")
        context = self.retriever(user_query)
        preview = context.replace("\n", " ")[:100]
        print(f"        → 召回 {len(context)//200} 段法条（预览: {preview}...）")

        # Step 2: 取历史
        history = self.memory.get_context()
        if history:
            print(f"[2/3] Memory: 已携带历史 ({len(history)} 字符)")
        else:
            print(f"[2/3] Memory: 首轮对话，无历史")

        # Step 3: 生成
        print("[3/3] QAAgent: 生成回答中 ...")
        answer = self.qa(user_query, context, history)
        print(f"[AGENT] {answer}\n")

        # 更新 memory
        self.memory.add_turn(user_query, answer)
        return answer

    def interactive(self) -> None:
        """REPL 主循环。"""
        while True:
            try:
                q = input("你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[BYE]")
                return
            if not q or q.lower() in {"q", "quit", "exit"}:
                print("[BYE]")
                return
            self.chat(q)


# ============================================================
#  演示
# ============================================================

def demo():
    """
    跑一个 4 轮的 demo，验证多轮上下文确实传递了。
    设计：第 2 轮的"它"依赖第 1 轮；第 3 轮"刚才那部法律"依赖第 2 轮。
    """
    print("=" * 60)
    print("[DEMO] 多轮上下文演示（4 轮）")
    print("=" * 60)

    chat = MultiAgentChat()

    # 第 1 轮：引入"草原法"这个主题
    chat.chat("国家对草原保护有什么方针？")
    # 第 2 轮：指代"它" = 草原法
    chat.chat("它第三条具体说了什么？")
    # 第 3 轮：指代"那部法律" = 草原法；问违法责任
    chat.chat("违反这部法律的法律责任是什么？")
    # 第 4 轮：问相似法律
    chat.chat("还有哪些法律有类似规定？")

    print("\n" + "=" * 60)
    print("[DEMO DONE] 查看上面的 4 轮回答，验证：")
    print("  - 第 2 轮 '它第三条' → LLM 应能知道'它' = 草原法")
    print("  - 第 3 轮 '这部法律' → LLM 应能接续上下文")
    print("  - 第 4 轮 '类似规定' → LLM 应基于已读法条做归纳")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        MultiAgentChat().interactive()
