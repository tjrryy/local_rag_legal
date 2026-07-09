"""
M4｜把法条装进 ChromaDB（持久化 + 检索）
==========================================

学习目标：
  - 学会用 LangChain 的 Document / VectorStore 抽象
  - 理解为什么 ChromaDB 是"向量 + 原文 + metadata"一体
  - 看到"第二次跑只要 1 秒，第一次跑 80 秒"的差异

文件产出：
  - ./vector_db/    （ChromaDB 持久化目录）
  - m4_chroma.py    （本脚本）

跑法：
  # 1) 第一次：构建索引（耗时 ~80s）
  python3 m4_chroma.py --build

  # 2) 之后任何时候：直接检索
  python3 m4_chroma.py --query "单位拖欠工资怎么办" --k 5
  python3 m4_chroma.py --query "草原保护方针" --k 3
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ---- 常量 ----
DATA_DIR = Path(__file__).parent.parent / "law_clearnerdata"
PERSIST_DIR = Path(__file__).parent / "vector_db"
COLLECTION = "legal_articles"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
ARTICLE_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+条")


# ---- 1. 把 JSON 读成 LangChain Document 列表 ----

def load_documents(data_dir: Path) -> list[Document]:
    """
    关键概念：Document = page_content（编码进向量的文本） + metadata（保留原信息）
    """
    docs: list[Document] = []
    for fp in sorted(data_dir.glob("laws_dataset_*.json")):
        for law in json.loads(fp.read_text(encoding="utf-8")):
            title = (law.get("title") or "").strip()
            for art in law.get("articles", []):
                art = art.strip()
                if not art:
                    continue
                m = ARTICLE_PATTERN.match(art)
                article_no = m.group(0) if m else ""
                # page_content 会进 embedding；加上「《法律名》第X条」前缀提升召回
                page_content = f"《{title}》{article_no}　{art}"
                # metadata 不会进 embedding，但可以跟着文档一起存
                docs.append(Document(
                    page_content=page_content,
                    metadata={
                        "law_title": title,
                        "article_no": article_no,
                        "source_file": law.get("source_file", ""),
                        "text": art,  # 原始正文，检索后直接读
                    },
                ))
    return docs


# ---- 2. 构建索引 ----

def build_index(docs: list[Document], embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    把所有 Document 编码、写入 ChromaDB。
    Chroma 会自动用 embeddings 把 page_content 转成向量存起来。
    """
    print(f"[INFO] 准备构建索引，共 {len(docs)} 条 ...")
    t0 = time.time()
    # Chroma.from_documents：把 docs 一次性喂进去，内部会调 embeddings 编码
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,                # 注意：新版用 `embedding` 不是 `embedding_function`
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION,
        collection_metadata={"hnsw:space": "cosine"},  # 余弦距离
    )
    print(f"[INFO] 构建完成，耗时 {time.time() - t0:.1f} s")
    print(f"[INFO] 持久化目录: {PERSIST_DIR}/")
    return db


# ---- 3. 加载已有索引 ----

def load_index(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """从磁盘加载，不重新编码。"""
    if not PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"{PERSIST_DIR}/ 不存在，请先运行: python3 m4_chroma.py --build"
        )
    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )


# ---- 4. 检索 ----

def search(db: Chroma, query: str, k: int = 5) -> list[dict]:
    """
    similarity_search_with_score 返回 (Document, distance) 列表。
    ChromaDB 的 distance 在 cosine 下 ∈ [0, 2]，
    我们的余弦相似度 = 1 - distance / 2，∈ [-1, 1]。
    """
    # bge 官方建议：query 端加指令前缀
    q = "为这个句子生成表示以用于检索相关文章：" + query
    raw = db.similarity_search_with_score(q, k=k)
    results = []
    for rank, (doc, dist) in enumerate(raw, 1):
        results.append({
            "rank": rank,
            "law_title": doc.metadata.get("law_title", ""),
            "article_no": doc.metadata.get("article_no", ""),
            "text": doc.metadata.get("text", ""),
            "source_file": doc.metadata.get("source_file", ""),
            "distance": float(dist),
            "score": 1.0 - float(dist) / 2.0,
        })
    return results


# ---- 5. 展示 ----

def render(query: str, results: list[dict]) -> str:
    lines = [f"\nQ: {query}", "-" * 72]
    for r in results:
        snippet = r["text"]
        if len(snippet) > 100:
            snippet = snippet[:100] + "..."
        lines.append(
            f"[{r['rank']}] score={r['score']:.4f}  "
            f"《{r['law_title']}》{r['article_no']}\n     {snippet}"
        )
    return "\n".join(lines)


# ---- 主入口 ----

def main():
    parser = argparse.ArgumentParser(description="M4 ChromaDB 索引 + 检索")
    parser.add_argument("--build", action="store_true",
                        help="构建索引（首次或数据更新时）")
    parser.add_argument("--query", "-q", help="检索问题")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0,
                        help="只编码前 N 条（调试用，0=全部）")
    args = parser.parse_args()

    # 加载 embedding 模型（这一步必须）
    print(f"[INFO] 加载模型 {MODEL_NAME} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    if args.build:
        docs = load_documents(DATA_DIR)
        if args.limit > 0:
            docs = docs[: args.limit]
        build_index(docs, embeddings)

    # 加载索引
    db = load_index(embeddings)
    print(f"[INFO] 集合大小: {db._collection.count()}")

    if args.query:
        t0 = time.time()
        results = search(db, args.query, args.k)
        dt = (time.time() - t0) * 1000
        print(render(args.query, results))
        print(f"\n(检索耗时: {dt:.1f} ms)")
    elif not args.build:
        parser.print_help()


if __name__ == "__main__":
    main()
