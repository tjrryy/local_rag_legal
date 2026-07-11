"""
法条级召回评测
==============
评测 Stage 4 精排是否把相关法条排到了前面。

指标：
  Recall@K  — 期望法条在 Top-K 中的比例
  MRR       — 第一个期望法条的倒数排名
  NDCG@K    — 考虑排序位置的归一化折扣累积增益
  Hit@K     — 至少命中一条期望法条的概率

输入 CSV 必须包含 expected_articles 列（由 annotate_ground_truth.py 生成）。

用法：
  python demo/eval/eval_recall.py \
    --test-csv test_set_100_annotated.csv \
    --embed-backend ollama --embed-model bge-m3 \
    --output-csv eval_recall.csv
"""

from __future__ import annotations

import argparse, csv, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pipeline import (
    BGE_QUERY_PREFIX, DATA_DIR, INDEX_DIR, ArticleFetcher, build_embeddings,
)


def ndcg_at_k(relevance: list[float], k: int) -> float:
    """计算 NDCG@K。relevance 按结果排序排列，1.0=相关，0.0=不相关。"""
    rel = np.array(relevance[:k], dtype=float)
    if rel.sum() == 0:
        return 0.0
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(rel)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


class ArticleRecallEvaluator:
    """法条级召回评测器。"""

    def __init__(self, embed_backend: str = "ollama", embed_model: str = "bge-m3"):
        print("[RECALL] 加载 embedding + FAISS + fetcher ...")
        self.embeddings = build_embeddings(embed_backend, embed_model)

        from langchain_community.vectorstores import FAISS
        law_dir = INDEX_DIR / "law_names"
        if not law_dir.exists():
            raise FileNotFoundError(f"索引不存在: {law_dir}")
        self.law_db = FAISS.load_local(
            str(law_dir), self.embeddings, allow_dangerous_deserialization=True,
        )
        self.fetcher = ArticleFetcher(DATA_DIR)
        print(f"[RECALL] 就绪\n")

    def evaluate_one(
        self,
        query: str,
        expected_articles: list[str],
        top_k_values: tuple[int, ...] = (5, 10, 20),
    ) -> dict[str, Any]:
        """
        跑 Stage 2+3+4，计算法条级指标。

        expected_articles: ["第三条", "第四条", "第三十八条"]
        """
        # Stage 2: 法律名匹配
        t0 = time.time()
        qp = BGE_QUERY_PREFIX + query
        law_results = self.law_db.similarity_search(qp, k=3)
        matched_laws = [doc.metadata["law_title"] for doc in law_results]
        t_stage2 = time.time() - t0

        # Stage 3: 取候选法条
        candidates = self.fetcher(matched_laws)

        # Stage 4: 精排
        t0 = time.time()
        cand_texts = [c.page_content for c in candidates]
        cand_vecs = self.embeddings.embed_documents(cand_texts)
        q_vec = self.embeddings.embed_query(qp)
        scores = np.dot(cand_vecs, q_vec)
        ranked_idx = np.argsort(-scores)
        t_stage4 = time.time() - t0

        # --- 计算指标 ---
        max_k = max(top_k_values)
        ranked_articles = []
        for idx in ranked_idx[:max_k]:
            ranked_articles.append(
                candidates[idx].metadata.get("article_no", "")
            )

        expected_set = set(expected_articles)

        result: dict[str, Any] = {
            "query": query,
            "matched_laws": "; ".join(matched_laws),
            "candidate_count": len(candidates),
            "stage2_s": round(t_stage2, 3),
            "stage4_s": round(t_stage4, 3),
            "top10_articles": "; ".join(ranked_articles[:10]),
        }

        # Recall@K
        for k in top_k_values:
            hits = sum(1 for art in ranked_articles[:k] if art in expected_set)
            result[f"recall@{k}"] = round(hits / len(expected_set), 4) if expected_set else 0.0

        # Hit@K
        for k in top_k_values:
            hit = any(art in expected_set for art in ranked_articles[:k])
            result[f"hit@{k}"] = 1 if hit else 0

        # MRR
        mrr = 0.0
        for i, art in enumerate(ranked_articles, 1):
            if art in expected_set:
                mrr = 1.0 / i
                break
        result["mrr"] = round(mrr, 4)

        # NDCG@10
        relevance = [1.0 if art in expected_set else 0.0 for art in ranked_articles[:10]]
        result["ndcg@10"] = round(ndcg_at_k(relevance, 10), 4)

        # 期望法条的最高排名
        best_rank = -1
        worst_rank = -1
        for i, art in enumerate(ranked_articles, 1):
            if art in expected_set:
                if best_rank == -1:
                    best_rank = i
                worst_rank = i
        result["expected_best_rank"] = best_rank
        result["expected_worst_rank"] = worst_rank

        return result

    def evaluate_csv(
        self, test_csv: str, output_csv: str
    ) -> tuple[list[dict], list[dict]]:
        with open(test_csv, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        if "expected_articles" not in rows[0]:
            print("[RECALL] CSV 缺少 expected_articles 列，请先运行 annotate_ground_truth.py")
            return [], []

        detail: list[dict] = []
        for i, row in enumerate(rows, 1):
            query = (row.get("query") or "").strip()
            raw = (row.get("expected_articles") or "").strip()
            expected = [a.strip() for a in raw.split(";") if a.strip()]

            if not query or not expected:
                detail.append({"id": row.get("id", i), "query": query,
                               "error": "缺少 expected_articles"})
                continue

            print(f"  [{i}/{len(rows)}] {query[:50]}...")
            r = self.evaluate_one(query, expected)
            r["id"] = row.get("id", i)
            r["category"] = row.get("category", "")
            r["difficulty"] = row.get("difficulty", "")
            r["expected_law"] = row.get("expected_law", "")
            r["expected_articles"] = raw
            detail.append(r)

        _write_csv(output_csv, detail)
        print(f"\n[RECALL] → {output_csv}")

        summary = _summarize(detail)
        summary_path = output_csv.replace(".csv", "_summary.csv")
        _write_csv(summary_path, summary)
        print(f"[RECALL] → {summary_path}")

        return detail, summary


def _summarize(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for gk in ["category", "difficulty"]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[r.get(gk, "unknown")].append(r)

        for key in ["ALL"] + sorted(groups.keys()):
            subset = rows if key == "ALL" else groups[key]
            if not subset:
                continue
            n = len(subset)
            s = {"group_by": gk, "group": key, "count": n}
            for m in ["recall@5", "recall@10", "recall@20", "hit@5", "hit@10",
                      "hit@20", "mrr", "ndcg@10"]:
                vals = [r.get(m, 0) for r in subset if m in r]
                s[f"avg_{m}"] = round(sum(vals) / len(vals), 4) if vals else 0
            ranks = [r["expected_best_rank"] for r in subset
                     if r.get("expected_best_rank", -1) > 0]
            s["avg_best_rank"] = round(sum(ranks) / len(ranks), 2) if ranks else -1
            out.append(s)
    return out


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="法条级召回评测")
    p.add_argument("--test-csv", required=True)
    p.add_argument("--output-csv", default="eval_recall.csv")
    p.add_argument("--embed-backend", default="ollama", choices=["hf", "ollama"])
    p.add_argument("--embed-model", default="bge-m3")
    args = p.parse_args()

    ev = ArticleRecallEvaluator(args.embed_backend, args.embed_model)
    ev.evaluate_csv(args.test_csv, args.output_csv)


if __name__ == "__main__":
    main()
