"""
自动标注法条级 ground truth
===========================
对每条 query，在期望法律内部用向量相似度找到最相关的 Top-3 法条，
写入 expected_articles 列。

用法：
  python demo/eval/annotate_ground_truth.py \
    --test-csv test_set_100.csv \
    --output-csv test_set_100_annotated.csv \
    --embed-backend ollama --embed-model bge-m3
"""

from __future__ import annotations

import argparse, csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pipeline import (
    BGE_QUERY_PREFIX, DATA_DIR, INDEX_DIR, ArticleFetcher, build_embeddings,
)


def annotate(
    test_csv: str,
    output_csv: str,
    embed_backend: str = "ollama",
    embed_model: str = "bge-m3",
    top_articles: int = 3,
) -> None:
    embeddings = build_embeddings(embed_backend, embed_model)
    fetcher = ArticleFetcher(DATA_DIR)

    with open(test_csv, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(f"标注 {len(rows)} 条 query...")

    for i, row in enumerate(rows, 1):
        query = (row.get("query") or "").strip()
        expected_raw = (row.get("expected_law") or "").strip()
        expected_laws = [l.strip() for l in expected_raw.split("/") if l.strip()]

        if not query or not expected_laws:
            row["expected_articles"] = ""
            row["expected_rewrite"] = row.get("expected_rewrite", "")
            continue

        # 只取第一个期望法律的法条（大部分 query 只有一个期望法律）
        primary_law = expected_laws[0]
        candidates = fetcher([primary_law])

        if not candidates:
            row["expected_articles"] = ""
            row["expected_rewrite"] = row.get("expected_rewrite", "")
            print(f"  [{i}/{len(rows)}] {query[:30]}... → 未找到法条")
            continue

        # 向量相似度排序，取 top-N
        q_vec = np.array(embeddings.embed_query(BGE_QUERY_PREFIX + query))
        cand_texts = [c.page_content for c in candidates]
        cand_vecs = np.array(embeddings.embed_documents(cand_texts))
        scores = np.dot(cand_vecs, q_vec)
        top_idx = np.argsort(-scores)[:top_articles]

        article_nos = []
        for idx in top_idx:
            art_no = candidates[idx].metadata.get("article_no", "")
            if art_no:
                article_nos.append(art_no)

        row["expected_articles"] = "; ".join(article_nos)
        row["expected_rewrite"] = row.get("expected_rewrite", "")

        print(f"  [{i}/{len(rows)}] {query[:30]}... → {article_nos[:3]} "
              f"(scores: {[round(scores[j], 3) for j in top_idx[:3]]})")

    # 写回 CSV
    new_columns = list(rows[0].keys())
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n标注完成 → {output_csv}")
    print(f"新增列: expected_articles, expected_rewrite")


def main():
    parser = argparse.ArgumentParser(description="自动标注法条级 ground truth")
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--embed-backend", default="ollama", choices=["hf", "ollama"])
    parser.add_argument("--embed-model", default="bge-m3")
    parser.add_argument("--top-articles", type=int, default=3)
    args = parser.parse_args()

    annotate(
        test_csv=args.test_csv,
        output_csv=args.output_csv,
        embed_backend=args.embed_backend,
        embed_model=args.embed_model,
        top_articles=args.top_articles,
    )


if __name__ == "__main__":
    main()
