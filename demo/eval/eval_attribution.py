"""
分阶段归因评测
==============
对每条 query 跑完整管道，定位失败发生在哪个阶段。

归因逻辑：
  Stage 1 失败 — 改写后与原问题语义偏离（多轮指代未消解）
  Stage 2 失败 — 期望法律不在 Top-3 匹配中
  Stage 4 失败 — 法律匹配对了，但期望法条不在 Top-10 精排结果中
  Stage 5 失败 — 法条召回正确，但答案未引用/引用错误
  OK          — 所有阶段通过

输出：
  归因明细表：每条 query 的失败阶段 + 中间结果
  归因汇总：各阶段的失败率分布

用法：
  python demo/eval/eval_attribution.py \
    --test-csv test_set_100_annotated.csv \
    --embed-backend ollama --embed-model bge-m3 \
    --llm-backend ollama --llm-model qwen2.5:7b \
    --output-csv eval_attribution.csv
"""

from __future__ import annotations

import argparse, csv, sys, time, re
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pipeline import (
    BGE_QUERY_PREFIX, DATA_DIR, INDEX_DIR,
    ArticleFetcher, build_embeddings, build_llm,
    QueryRewriter, LawNameMatcher, ArticleRanker, QAAgent,
)


def _extract_articles_from_answer(answer: str) -> set[str]:
    """从答案中提取引用的法条号。"""
    pattern = re.compile(r"第\s*([零〇一二三四五六七八九十百千万\d]+)\s*条")
    found = set()
    for m in pattern.finditer(answer):
        found.add("第" + m.group(1) + "条")
    return found


class AttributionEvaluator:
    """分阶段归因评测器。"""

    def __init__(
        self,
        embed_backend: str = "ollama",
        embed_model: str = "bge-m3",
        llm_backend: str = "ollama",
        llm_model: str = "qwen2.5:7b",
    ):
        print("[ATTR] 初始化全管道 ...")
        self.embeddings = build_embeddings(embed_backend, embed_model)

        from langchain_community.vectorstores import FAISS
        law_db = FAISS.load_local(
            str(INDEX_DIR / "law_names"), self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.law_matcher = LawNameMatcher(law_db)
        self.fetcher = ArticleFetcher(DATA_DIR)
        self.ranker = ArticleRanker(self.embeddings)

        self.llm = build_llm(llm_backend, llm_model)
        self.rewriter = QueryRewriter(self.llm)
        self.qa = QAAgent(self.llm)
        print("[ATTR] 就绪\n")

    def diagnose_one(
        self,
        query: str,
        history: str,
        expected_laws: list[str],
        expected_articles: set[str],
        multi_turn: bool = False,
    ) -> dict[str, Any]:
        """
        跑完整管道，返回逐阶段诊断结果。
        """
        expected_law_set = set(expected_laws)
        failure_stage: str | None = None
        failure_detail: str = ""

        # ── Stage 1: 改写 ──
        t0 = time.time()
        try:
            rewritten = self.rewriter(query, history)
        except Exception:
            rewritten = query
        t1 = time.time() - t0

        # 改写偏离检测：多轮场景下改写如果没变化，说明指代没消解
        s1_ok = True
        if multi_turn and rewritten == query:
            s1_ok = False
            failure_stage = "S1_rewrite"
            failure_detail = "多轮指代未消解，改写后与原 query 相同"

        # ── Stage 2: 法律匹配 ──
        t0 = time.time()
        matched = self.law_matcher(rewritten, top_k=3)
        t2 = time.time() - t0

        matched_set = set(matched)
        law_hit = bool(expected_law_set & matched_set)

        if not s1_ok:
            pass  # 已在 Stage 1 判定失败
        elif not law_hit:
            failure_stage = "S2_law_match"
            failure_detail = f"期望法律 {expected_laws} 不在匹配结果 {matched} 中"

        # ── Stage 3+4: 法条精排 ──
        candidates = self.fetcher(matched)
        t0 = time.time()
        ranked = self.ranker(rewritten, candidates, top_k=10)
        t4 = time.time() - t0

        ranked_articles = set()
        for doc, _ in ranked:
            ranked_articles.add(doc.metadata.get("article_no", ""))

        article_hit = bool(expected_articles & ranked_articles)

        if failure_stage:
            pass
        elif not article_hit:
            failure_stage = "S4_article_rank"
            failure_detail = (
                f"法律匹配正确 ({matched})，但期望法条 {expected_articles} "
                f"不在 Top-10 中"
            )

        # 精排中期望法条的最高排名
        candidates_all = self.fetcher(matched)
        cand_texts = [c.page_content for c in candidates_all]
        cand_vecs = self.embeddings.embed_documents(cand_texts)
        q_vec = self.embeddings.embed_query(BGE_QUERY_PREFIX + rewritten)
        scores = np.dot(cand_vecs, q_vec)
        full_idx = np.argsort(-scores)
        best_full_rank = -1
        for rank, idx in enumerate(full_idx, 1):
            art_no = candidates_all[idx].metadata.get("article_no", "")
            if art_no in expected_articles:
                best_full_rank = rank
                break

        # ── Stage 5: 答案 ──
        t0 = time.time()
        try:
            raw = self.qa(query, ranked)
            answer = raw if isinstance(raw, str) else str(raw)
        except Exception as e:
            answer = f"[LLM 失败: {e}]"
        t5 = time.time() - t0

        cited = _extract_articles_from_answer(answer)

        if failure_stage:
            pass
        elif expected_articles and not (expected_articles & cited):
            failure_stage = "S5_answer"
            failure_detail = (
                f"法条召回正确（排名={best_full_rank}），"
                f"但答案未引用期望法条 {expected_articles}，"
                f"实际引用 {cited}"
            )

        if not failure_stage:
            failure_stage = "OK"
            failure_detail = "所有阶段通过"

        return {
            "query": query,
            "rewritten": rewritten,
            "answer": answer,
            "expected_laws": "; ".join(expected_laws),
            "expected_articles": "; ".join(sorted(expected_articles)),
            "matched_laws": "; ".join(matched),
            "matched_articles": "; ".join(sorted(ranked_articles)),
            "cited_articles": "; ".join(sorted(cited)),
            "failure_stage": failure_stage,
            "failure_detail": failure_detail,
            "best_full_rank": best_full_rank,
            "rewrite_changed": rewritten != query,
            "t_s1_rewrite": round(t1, 3),
            "t_s2_match": round(t2, 3),
            "t_s4_rank": round(t4, 3),
            "t_s5_answer": round(t5, 3),
            "total_s": round(t1 + t2 + t4 + t5, 3),
        }

    def evaluate_csv(
        self, test_csv: str, output_csv: str, limit: int = 0
    ) -> tuple[list[dict], list[dict]]:
        with open(test_csv, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        if limit > 0:
            rows = rows[:limit]

        detail: list[dict] = []
        multi_hist: dict[str, list[dict]] = {}

        for i, row in enumerate(rows, 1):
            query = (row.get("query") or "").strip()
            expected_raw = (row.get("expected_law") or "").strip()
            expected_laws = [l.strip() for l in expected_raw.split("/") if l.strip()]
            raw_arts = (row.get("expected_articles") or "").strip()
            expected_arts = set(a.strip() for a in raw_arts.split(";") if a.strip())
            group = row.get("multi_turn_group", "").strip()
            is_multi = bool(group)

            if not query or not expected_laws:
                continue

            history = ""
            if group and group in multi_hist:
                turns = multi_hist[group][-6:]
                history = "\n".join(
                    f"Q: {t['q']}\nA: {t.get('a', '')}" for t in turns
                )

            print(f"  [{i}/{len(rows)}] {query[:50]}...")
            r = self.diagnose_one(
                query, history, expected_laws, expected_arts, multi_turn=is_multi
            )
            r["id"] = row.get("id", i)
            r["category"] = row.get("category", "")
            r["difficulty"] = row.get("difficulty", "")
            r["multi_turn_group"] = group
            detail.append(r)

            if group:
                multi_hist.setdefault(group, []).append(
                    {"q": query, "a": r.get("answer", "")}
                )

        _write_csv(output_csv, detail)
        print(f"\n[ATTR] → {output_csv}")

        summary = _summarize(detail)
        sp = output_csv.replace(".csv", "_summary.csv")
        _write_csv(sp, summary)
        print(f"[ATTR] → {sp}")

        return detail, summary


def _summarize(rows: list[dict]) -> list[dict]:
    out: list[dict] = []

    # 全局分布
    stage_counts = defaultdict(int)
    for r in rows:
        stage_counts[r.get("failure_stage", "unknown")] += 1
    total = len(rows)
    s = {"group": "ALL", "count": total}
    for stage in ["OK", "S1_rewrite", "S2_law_match", "S4_article_rank", "S5_answer"]:
        cnt = stage_counts.get(stage, 0)
        s[f"{stage}_count"] = cnt
        s[f"{stage}_pct"] = round(cnt / total, 4) if total else 0
    out.append(s)

    # 按 category
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get("category", "unknown")].append(r)
    for key in sorted(groups.keys()):
        subset = groups[key]
        n = len(subset)
        sc = defaultdict(int)
        for r in subset:
            sc[r.get("failure_stage", "unknown")] += 1
        s2 = {"group": key, "count": n}
        for stage in ["OK", "S1_rewrite", "S2_law_match",
                      "S4_article_rank", "S5_answer"]:
            cnt = sc.get(stage, 0)
            s2[f"{stage}_count"] = cnt
            s2[f"{stage}_pct"] = round(cnt / n, 4) if n else 0
        out.append(s2)

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
    p = argparse.ArgumentParser(description="分阶段归因评测")
    p.add_argument("--test-csv", required=True)
    p.add_argument("--output-csv", default="eval_attribution.csv")
    p.add_argument("--embed-backend", default="ollama", choices=["hf", "ollama"])
    p.add_argument("--embed-model", default="bge-m3")
    p.add_argument("--llm-backend", default="ollama", choices=["deepseek", "ollama"])
    p.add_argument("--llm-model", default="qwen2.5:7b")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    args = p.parse_args()

    ev = AttributionEvaluator(
        args.embed_backend, args.embed_model,
        args.llm_backend, args.llm_model,
    )
    ev.evaluate_csv(args.test_csv, args.output_csv, limit=args.limit)


if __name__ == "__main__":
    main()
