"""
demo/test_retrieval.py
======================
快速检索测试脚本 — 不调 LLM，只测 Stage 2-4 的检索质量。

用法：
  python demo/test_retrieval.py                        # 跑全部 100 条（仅检索）
  python demo/test_retrieval.py --limit 10             # 前 10 条
  python demo/test_retrieval.py --limit 3 --with-llm  # 含 LLM 回答（慢）
  python demo/test_retrieval.py --multi-turn           # 测试多轮指代消解
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ---- 路径 ----
DEMO_DIR = Path(__file__).parent
PROJECT_ROOT = DEMO_DIR.parent
TEST_SET_PATH = PROJECT_ROOT / "test_set_100.json"


def load_test_set(limit: int = None) -> list[dict]:
    data = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    return data[:limit] if limit else data


def main():
    parser = argparse.ArgumentParser(description="快速检索测试")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--with-llm", action="store_true", help="包含 LLM 回答（慢）")
    parser.add_argument("--multi-turn", action="store_true", help="启用多轮上下文")
    parser.add_argument("--no-rewrite", action="store_true", help="跳过 Stage 1 改写")
    parser.add_argument("--output", type=str, default=None, help="结果输出 JSON 路径")
    args = parser.parse_args()

    from pipeline import LegalRAGPipeline

    queries = load_test_set(args.limit)
    n = len(queries)
    print(f"[LOAD] {n} 条 query\n")

    pipeline = LegalRAGPipeline(
        embed_backend="ollama",
        llm_backend="ollama",
        embed_model="bge-m3",
        llm_model="deepseek-r1:7b",
    )

    results = []
    multi_hist: dict[str, list[dict]] = {}

    for i, q in enumerate(queries, 1):
        qid = q["id"]
        query_text = q["query"]
        cat = q.get("category", "")
        exp_law = q.get("expected_law", "")
        group = q.get("multi_turn_group")

        print(f"[{i}/{n}] #{qid} [{cat}] {query_text[:70]}")

        # 多轮上下文
        history = ""
        if args.multi_turn and group and group in multi_hist:
            lines = []
            for t in multi_hist[group][-6:]:
                lines.append(f"Q: {t['q']}")
                lines.append(f"A: {t.get('a', '')}")
            history = "\n".join(lines)

        t0 = time.time()

        # Stage 1: 改写
        if args.no_rewrite:
            rewritten = query_text
        else:
            try:
                rewritten = pipeline.rewriter(query_text, history)
            except Exception:
                rewritten = query_text

        # Stage 2: 匹配法律
        matched = pipeline.law_matcher(rewritten, top_k=3)

        # Stage 3: 取候选法条
        candidates = pipeline.fetcher(matched)

        # Stage 4: 精排
        ranked = pipeline.ranker(rewritten, candidates, top_k=10)

        # Stage 5: LLM 回答（可选）
        answer = ""
        if args.with_llm:
            try:
                answer = pipeline.qa(query_text, ranked)
            except Exception as e:
                answer = f"[LLM 失败: {e}]"

        elapsed = time.time() - t0

        # 输出
        print(f"  改写: {rewritten[:60]}")
        print(f"  法律: {matched}")
        top_articles = [(doc.metadata.get("law_title", ""),
                         doc.metadata.get("article_no", ""),
                         round(score, 3))
                        for doc, score in ranked[:5]]
        for j, (law, art, score) in enumerate(top_articles, 1):
            print(f"  {j}. [{score:.3f}] 《{law}》{art}")
        if answer:
            print(f"  回答: {answer[:120]}...")
        print(f"  耗时: {elapsed:.1f}s")

        # 更新多轮历史
        if args.multi_turn and group:
            multi_hist.setdefault(group, []).append({"q": query_text, "a": answer})

        results.append({
            "id": qid,
            "query": query_text,
            "category": cat,
            "expected_law": exp_law,
            "rewritten": rewritten,
            "matched_laws": matched,
            "top5_articles": [{"law": law, "article": art, "score": s}
                              for law, art, s in top_articles],
            "answer": answer[:500] if answer else "(skipped)",
            "time_s": round(elapsed, 1),
        })

    # 汇总
    print(f"\n{'='*60}")
    print(f"汇总: {n} 条 | 平均 {sum(r['time_s'] for r in results)/n:.1f}s/条")
    law_hit = sum(1 for r in results if r["matched_laws"]) / n
    print(f"法律命中率: {law_hit:.1%}")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
