"""
改写效果评测
============
评测 Stage 1 Query Rewriter 的质量。

三个指标：
  1. 检索增益 — 改写前后法条召回率的变化（Recall@10 提升多少）
  2. 指代消解 — 多轮对话中指代词是否被替换为具体法律名
  3. 改写保真 — 单轮无指代时改写是否保留了原意（不改不该改的）

用法：
  python demo/eval/eval_rewrite.py \
    --test-csv test_set_100_annotated.csv \
    --embed-backend ollama --embed-model bge-m3 \
    --llm-backend ollama --llm-model qwen2.5:7b \
    --output-csv eval_rewrite.csv
"""

from __future__ import annotations

import argparse, csv, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pipeline import (
    BGE_QUERY_PREFIX, DATA_DIR, INDEX_DIR,
    ArticleFetcher, build_embeddings, build_llm, QueryRewriter,
)


class RewriteEvaluator:
    """改写效果评测器。"""

    def __init__(
        self,
        embed_backend: str = "ollama",
        embed_model: str = "bge-m3",
        llm_backend: str = "ollama",
        llm_model: str = "qwen2.5:7b",
    ):
        print("[REWRITE] 加载组件 ...")
        self.embeddings = build_embeddings(embed_backend, embed_model)

        from langchain_community.vectorstores import FAISS
        self.law_db = FAISS.load_local(
            str(INDEX_DIR / "law_names"), self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.fetcher = ArticleFetcher(DATA_DIR)

        self.llm = build_llm(llm_backend, llm_model)
        self.rewriter = QueryRewriter(self.llm)
        print("[REWRITE] 就绪\n")

    def _quick_recall(
        self, query: str, expected_articles: set[str], top_k: int = 20
    ) -> dict:
        """快速跑 Stage 2-4，返回召回指标。"""
        laws = self.law_db.similarity_search(
            BGE_QUERY_PREFIX + query, k=3
        )
        matched = [doc.metadata["law_title"] for doc in laws]
        candidates = self.fetcher(matched)

        if not candidates:
            return {"recall@10": 0.0, "mrr": 0.0, "laws": matched, "articles": []}

        cand_texts = [c.page_content for c in candidates]
        cand_vecs = self.embeddings.embed_documents(cand_texts)
        q_vec = self.embeddings.embed_query(BGE_QUERY_PREFIX + query)
        scores = np.dot(cand_vecs, q_vec)
        idx = np.argsort(-scores)[:top_k]
        ranked = [candidates[i].metadata.get("article_no", "") for i in idx]

        hits = sum(1 for a in ranked[:10] if a in expected_articles)
        recall = round(hits / len(expected_articles), 4) if expected_articles else 0.0

        mrr = 0.0
        for i, a in enumerate(ranked, 1):
            if a in expected_articles:
                mrr = 1.0 / i
                break

        return {"recall@10": recall, "mrr": round(mrr, 4),
                "laws": matched, "top10_articles": ranked[:10]}

    def evaluate_one(
        self, query: str, history: str, expected_articles: set[str],
        multi_turn: bool = False,
    ) -> dict[str, Any]:
        """评测单条 query 的改写效果。"""
        t0 = time.time()
        try:
            rewritten = self.rewriter(query, history)
        except Exception as e:
            rewritten = query
        rewrite_time = time.time() - t0

        # 检索：原 query
        orig = self._quick_recall(query, expected_articles)

        # 检索：改写后
        rewr = self._quick_recall(rewritten, expected_articles)

        # 增益
        recall_gain = round(rewr["recall@10"] - orig["recall@10"], 4)
        mrr_gain = round(rewr["mrr"] - orig["mrr"], 4)

        # 保真度：改写不应改变原意（简单的字符重叠做代理）
        orig_chars = set(query)
        rewr_chars = set(rewritten)
        fidelity = round(len(orig_chars & rewr_chars) / max(len(orig_chars), 1), 4)

        return {
            "query": query,
            "rewritten": rewritten,
            "is_multi_turn": multi_turn,
            "orig_recall@10": orig["recall@10"],
            "orig_mrr": orig["mrr"],
            "orig_laws": "; ".join(orig["laws"]),
            "rewr_recall@10": rewr["recall@10"],
            "rewr_mrr": rewr["mrr"],
            "rewr_laws": "; ".join(rewr["laws"]),
            "recall_gain": recall_gain,
            "mrr_gain": mrr_gain,
            "fidelity": fidelity,
            "rewrite_time_s": round(rewrite_time, 3),
            "rewriter_changed": rewritten != query,
        }

    def evaluate_csv(
        self, test_csv: str, output_csv: str, limit: int = 0,
    ) -> tuple[list[dict], list[dict]]:
        with open(test_csv, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if limit > 0:
            rows = rows[:limit]

        detail: list[dict] = []
        multi_hist: dict[str, list[dict]] = {}

        for i, row in enumerate(rows, 1):
            query = (row.get("query") or "").strip()
            raw_arts = (row.get("expected_articles") or "").strip()
            expected = set(a.strip() for a in raw_arts.split(";") if a.strip())
            group = row.get("multi_turn_group", "").strip()
            is_multi = bool(group)

            if not query or not expected:
                detail.append({"id": row.get("id", i), "query": query,
                               "error": "缺少 expected_articles"})
                continue

            # 多轮历史
            history = ""
            if group and group in multi_hist:
                turns = multi_hist[group][-6:]
                history = "\n".join(
                    f"Q: {t['q']}\nA: {t.get('a', '')}" for t in turns
                )

            print(f"  [{i}/{len(rows)}] {query[:50]}...")
            r = self.evaluate_one(query, history, expected, multi_turn=is_multi)
            r["id"] = row.get("id", i)
            r["category"] = row.get("category", "")
            r["difficulty"] = row.get("difficulty", "")
            r["multi_turn_group"] = group
            detail.append(r)

            # 更新多轮上下文
            if group:
                multi_hist.setdefault(group, []).append(
                    {"q": query, "a": r["rewritten"]}
                )

        _write_csv(output_csv, detail)
        print(f"\n[REWRITE] → {output_csv}")

        summary = _summarize(detail)
        sp = output_csv.replace(".csv", "_summary.csv")
        _write_csv(sp, summary)
        print(f"[REWRITE] → {sp}")

        return detail, summary


def _summarize(rows: list[dict]) -> list[dict]:
    """按多轮/单轮 + category 汇总。"""
    out: list[dict] = []
    # 按多轮 vs 单轮
    for label, pred in [("多轮指代", lambda r: r.get("is_multi_turn")),
                         ("单轮", lambda r: not r.get("is_multi_turn"))]:
        subset = [r for r in rows if pred(r)]
        if not subset:
            continue
        n = len(subset)
        s = {"group": label, "count": n}
        for m in ["orig_recall@10", "rewr_recall@10", "recall_gain",
                  "orig_mrr", "rewr_mrr", "mrr_gain", "fidelity"]:
            vals = [r[m] for r in subset if m in r]
            s[f"avg_{m}"] = round(sum(vals) / len(vals), 4) if vals else 0
        s["rewrite_changed_pct"] = round(
            sum(1 for r in subset if r.get("rewriter_changed")) / n, 4
        )
        out.append(s)

    # 按 category
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get("category", "unknown")].append(r)
    for key in sorted(groups.keys()):
        subset = groups[key]
        n = len(subset)
        s = {"group": key, "count": n}
        for m in ["recall_gain", "mrr_gain", "fidelity"]:
            vals = [r[m] for r in subset if m in r]
            s[f"avg_{m}"] = round(sum(vals) / len(vals), 4) if vals else 0
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
    p = argparse.ArgumentParser(description="改写效果评测")
    p.add_argument("--test-csv", required=True)
    p.add_argument("--output-csv", default="eval_rewrite.csv")
    p.add_argument("--embed-backend", default="ollama", choices=["hf", "ollama"])
    p.add_argument("--embed-model", default="bge-m3")
    p.add_argument("--llm-backend", default="ollama", choices=["deepseek", "ollama"])
    p.add_argument("--llm-model", default="qwen2.5:7b")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用）")
    args = p.parse_args()

    ev = RewriteEvaluator(args.embed_backend, args.embed_model,
                          args.llm_backend, args.llm_model)
    # 如果有限制，只取前 N 条
    if args.limit > 0:
        import tempfile
        with open(args.test_csv, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        subset = rows[:args.limit]
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="utf-8-sig", delete=False
        )
        w = csv.DictWriter(tmp, fieldnames=subset[0].keys())
        w.writeheader()
        w.writerows(subset)
        tmp.close()
        ev.evaluate_csv(tmp.name, args.output_csv)
    else:
        ev.evaluate_csv(args.test_csv, args.output_csv)


if __name__ == "__main__":
    main()
