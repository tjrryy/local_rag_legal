"""
demo/test_pipeline_no_llm.py
============================
不调 LLM 的离线测试：只验 FAISS 检索 + 法条过滤这段。

用法：
  python3 demo/test_pipeline_no_llm.py
"""

import sys
from pathlib import Path

# 让 pipeline.py 能 import
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import (
    build_embeddings,
    INDEX_DIR,
    ArticleFetcher,
    DATA_DIR,
)
from langchain_community.vectorstores import FAISS
import numpy as np


def main():
    print("=" * 60)
    print("[TEST] 加载 FAISS 索引，验证两阶段检索")
    print("=" * 60)

    embeddings = build_embeddings("ollama", "bge-m3")
    law_db = FAISS.load_local(
        str(INDEX_DIR / "law_names"), embeddings,
        allow_dangerous_deserialization=True,
    )
    article_db = FAISS.load_local(
        str(INDEX_DIR / "articles"), embeddings,
        allow_dangerous_deserialization=True,
    )
    fetcher = ArticleFetcher(DATA_DIR)

    print(f"[INFO] law_names index size: {law_db.index.ntotal}")
    print(f"[INFO] articles index size: {article_db.index.ntotal}")
    print()

    test_queries = [
        "草原保护有什么方针？",
        "数据泄露怎么处理？",
        "中医药怎么管理？",
    ]

    for q in test_queries:
        print(f"\nQ: {q}")

        # Stage 2: 法律名匹配
        prefixed = "为这个句子生成表示以用于检索相关文章：" + q
        laws = law_db.similarity_search(prefixed, k=3)
        law_titles = [d.metadata["law_title"] for d in laws]
        print(f"  [Stage 2] 命中法律: {law_titles}")

        # Stage 3: 取这些法律的全部法条
        candidates = fetcher(law_titles)
        print(f"  [Stage 3] 候选法条数: {len(candidates)}")

        # Stage 4: 精排
        cand_vecs = embeddings.embed_documents([c.page_content for c in candidates])
        q_vec = embeddings.embed_query(prefixed)
        scores = np.dot(cand_vecs, q_vec)
        top_idx = np.argsort(-scores)[:3]
        print(f"  [Stage 4] Top-3:")
        for i in top_idx:
            d = candidates[i]
            print(f"     {scores[i]:.3f}  《{d.metadata['law_title']}》"
                  f"{d.metadata['article_no']}")


if __name__ == "__main__":
    main()