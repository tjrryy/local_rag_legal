"""
M6｜完整 RAG 链路：query → 检索 → DeepSeek LLM → 答案
========================================================

学习目标：
  - 看懂 RAG prompt 模板为什么这么写
  - 理解 "检索" 和 "生成" 是解耦的两步
  - 看到 LLM 在 prompt 里"读"到法条后才能正确回答

使用：
  1) 先确保 DEEPSEEK_API_KEY 已设置：
     export DEEPSEEK_API_KEY="sk-..."

  2) 确保 ChromaDB 索引已建（用 m4_chroma.py --build）

  3) 跑：
     python3 m6_rag.py -q "单位拖欠工资怎么办？"
     python3 m6_rag.py -q "醉驾怎么处理？"

也可以用本地 Ollama 替代（见 --backend ollama）：
  python3 m6_rag.py -q "..." --backend ollama --model qwen2.5:7b
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---- 常量 ----
PERSIST_DIR = Path(__file__).parent / "vector_db"
COLLECTION = "legal_articles"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# ---- Prompt 模板 ----
# 这是 RAG 系统的"灵魂"：怎么把检索到的法条塞进 LLM 的输入
# 关键设计：
#   1. 明确角色（"法律助手"）+ 任务（"仅基于法条"）
#   2. 提供【法条参考】区块，让 LLM "看见"证据
#   3. 要求"先引用后回答"——减少幻觉
#   4. 兜底："未直接规定"——避免硬编
PROMPT_TEMPLATE = """你是中国法律领域的智能助手。请严格根据下面【法条参考】回答用户问题。

要求：
1. 必须先引用法条原文，再做解释
2. 引用时用「《法律名》第X条」的格式
3. 如果【法条参考】中没有任何与问题相关的内容，请直接回答："现有法条中未直接规定该问题"

【法条参考】
{context}

【用户问题】
{question}

【你的回答】
"""


# ---- 1. 准备 retriever 和 LLM ----

def build_retriever() -> Chroma:
    """加载 ChromaDB，转成 retriever。"""
    if not PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"{PERSIST_DIR}/ 不存在，请先运行: python3 m4_chroma.py --build"
        )
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )
    # 关键：把 query 加 bge 前缀再送给 retriever
    # 这里用一个 wrapper 函数来实现
    def retriever_with_prefix(query: str) -> list:
        return db.similarity_search(BGE_QUERY_PREFIX + query, k=5)
    return retriever_with_prefix  # type: ignore[return-value]


def build_llm(backend: str, model: str):
    """构造 LLM。支持 deepseek (云端) / ollama (本地) / openai 兼容。"""
    if backend == "deepseek":
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "未设置 DEEPSEEK_API_KEY 环境变量。\n"
                "申请地址: https://platform.deepseek.com/\n"
                "设置方法: export DEEPSEEK_API_KEY='sk-...'"
            )
        return ChatOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=api_key,
            model=model or "deepseek-chat",
            temperature=0.1,        # 法律场景：低温度，更稳
        )
    elif backend == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            model=model or "qwen2.5:7b",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    elif backend == "qwen":  # 通义千问，OpenAI 兼容协议
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise EnvironmentError("未设置 DASHSCOPE_API_KEY 环境变量")
        return ChatOpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            model=model or "qwen-plus",
            temperature=0.1,
        )
    elif backend == "openai":  # OpenAI 官方
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("未设置 OPENAI_API_KEY 环境变量")
        return ChatOpenAI(
            api_key=api_key,
            model=model or "gpt-4o-mini",
            temperature=0.1,
        )
    else:
        raise ValueError(f"unknown backend: {backend}")


# ---- 2. 构造 RAG 链 ----

def format_docs(docs) -> str:
    """把 retriever 返回的 Document 列表拼成 LLM 能吃的字符串。"""
    if not docs:
        return "（无相关法条）"
    lines = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("law_title", "")
        artno = doc.metadata.get("article_no", "")
        text = doc.metadata.get("text", doc.page_content)
        lines.append(f"{i}. 《{title}》{artno}\n   {text}")
    return "\n\n".join(lines)


def build_rag_chain(retriever, llm):
    """
    组装 RAG chain（LangChain Expression Language）：

    user_input -> retriever -> format_docs -> prompt -> llm -> str
    """
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ---- 3. 主入口 ----

def main():
    parser = argparse.ArgumentParser(description="M6 RAG：query → 检索 → LLM 回答")
    parser.add_argument("-q", "--query", required=True, help="用户问题")
    parser.add_argument("--backend", default="deepseek",
                        choices=["deepseek", "ollama", "qwen", "openai"],
                        help="LLM 后端")
    parser.add_argument("--model", default="",
                        help="模型名（默认 deepseek-chat / qwen2.5:7b）")
    args = parser.parse_args()

    print(f"[INFO] 加载 retriever ...")
    retriever = build_retriever()
    print(f"[INFO] 加载 LLM ({args.backend}) ...")
    llm = build_llm(args.backend, args.model)
    print(f"[INFO] 构造 RAG chain ...")
    chain = build_rag_chain(retriever, llm)
    print()

    print("=" * 60)
    print(f"Q: {args.query}")
    print("=" * 60)

    t0 = time.time()
    answer = chain.invoke(args.query)
    dt = time.time() - t0

    print(answer)
    print()
    print("=" * 60)
    print(f"(总耗时: {dt:.1f} s)")


if __name__ == "__main__":
    main()
