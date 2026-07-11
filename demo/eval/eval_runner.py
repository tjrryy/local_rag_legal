"""
评测统一入口
============
输入 CSV → 标注 ground truth → 三条评测链路 → 输出 CSV

链路：
  recall     — 法条级召回评测（Recall@K / MRR / NDCG）
  rewrite    — 改写效果评测（检索增益 / 指代消解）
  attribution — 分阶段归因（定位失败阶段：S1/S2/S4/S5）
  full       — 全部跑一遍

用法：
  # 1. 先标注 ground truth
  python demo/eval/eval_runner.py annotate \
    --test-csv test_set_100.csv

  # 2. 跑召回评测（不需要 LLM）
  python demo/eval/eval_runner.py recall \
    --test-csv test_set_100_annotated.csv

  # 3. 跑改写评测 + 归因（需要 LLM）
  python demo/eval/eval_runner.py rewrite \
    --test-csv test_set_100_annotated.csv --limit 10

  # 4. 一键全跑
  python demo/eval/eval_runner.py full \
    --test-csv test_set_100.csv
"""

from __future__ import annotations

import argparse, csv, sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_TEST_CSV = PROJECT_ROOT / "test_set_100.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval_results"


def cmd_annotate(args):
    from annotate_ground_truth import annotate
    out = args.output_csv or str(
        Path(args.test_csv).parent / "test_set_100_annotated.csv"
    )
    annotate(args.test_csv, out, args.embed_backend, args.embed_model, args.top_articles)


def cmd_recall(args):
    from eval_recall import ArticleRecallEvaluator
    ev = ArticleRecallEvaluator(args.embed_backend, args.embed_model)
    ev.evaluate_csv(args.test_csv, args.output_csv or _out("eval_recall"))


def cmd_rewrite(args):
    from eval_rewrite import RewriteEvaluator
    ev = RewriteEvaluator(args.embed_backend, args.embed_model,
                          args.llm_backend, args.llm_model)
    ev.evaluate_csv(args.test_csv, args.output_csv or _out("eval_rewrite"),
                    limit=args.limit)


def cmd_attribution(args):
    from eval_attribution import AttributionEvaluator
    ev = AttributionEvaluator(args.embed_backend, args.embed_model,
                              args.llm_backend, args.llm_model)
    ev.evaluate_csv(args.test_csv, args.output_csv or _out("eval_attribution"),
                    limit=args.limit)


def _out(name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(DEFAULT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    return str(DEFAULT_OUTPUT_DIR / f"{name}_{ts}.csv")


def main():
    p = argparse.ArgumentParser(description="法律 RAG 评测系统")
    sub = p.add_subparsers(dest="command", required=True)

    # annotate
    pa = sub.add_parser("annotate", help="自动标注法条级 ground truth")
    pa.add_argument("--test-csv", default=str(DEFAULT_TEST_CSV))
    pa.add_argument("--output-csv", default=None)
    pa.add_argument("--embed-backend", default="ollama", choices=["hf", "ollama"])
    pa.add_argument("--embed-model", default="bge-m3")
    pa.add_argument("--top-articles", type=int, default=3)

    # recall
    pr = sub.add_parser("recall", help="法条级召回评测")
    pr.add_argument("--test-csv", required=True)
    pr.add_argument("--output-csv", default=None)
    pr.add_argument("--embed-backend", default="ollama", choices=["hf", "ollama"])
    pr.add_argument("--embed-model", default="bge-m3")

    # rewrite
    pw = sub.add_parser("rewrite", help="改写效果评测")
    pw.add_argument("--test-csv", required=True)
    pw.add_argument("--output-csv", default=None)
    pw.add_argument("--embed-backend", default="ollama", choices=["hf", "ollama"])
    pw.add_argument("--embed-model", default="bge-m3")
    pw.add_argument("--llm-backend", default="ollama", choices=["deepseek", "ollama"])
    pw.add_argument("--llm-model", default="qwen2.5:7b")
    pw.add_argument("--limit", type=int, default=0)

    # attribution
    pat = sub.add_parser("attribution", help="分阶段归因评测")
    pat.add_argument("--test-csv", required=True)
    pat.add_argument("--output-csv", default=None)
    pat.add_argument("--embed-backend", default="ollama", choices=["hf", "ollama"])
    pat.add_argument("--embed-model", default="bge-m3")
    pat.add_argument("--llm-backend", default="ollama", choices=["deepseek", "ollama"])
    pat.add_argument("--llm-model", default="qwen2.5:7b")
    pat.add_argument("--limit", type=int, default=0)

    # full
    pf = sub.add_parser("full", help="一键全跑：标注 + 召回 + 改写 + 归因")
    pf.add_argument("--test-csv", default=str(DEFAULT_TEST_CSV))
    pf.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    pf.add_argument("--embed-backend", default="ollama", choices=["hf", "ollama"])
    pf.add_argument("--embed-model", default="bge-m3")
    pf.add_argument("--llm-backend", default="ollama", choices=["deepseek", "ollama"])
    pf.add_argument("--llm-model", default="qwen2.5:7b")
    pf.add_argument("--limit", type=int, default=0)

    args = p.parse_args()

    if args.command == "annotate":
        cmd_annotate(args)
    elif args.command == "recall":
        cmd_recall(args)
    elif args.command == "rewrite":
        cmd_rewrite(args)
    elif args.command == "attribution":
        cmd_attribution(args)
    elif args.command == "full":
        # 先标注（已有文件则跳过）
        annotated_csv = str(
            Path(args.test_csv).parent / "test_set_100_annotated.csv"
        )
        if Path(annotated_csv).exists():
            print(f"[SKIP] 标注文件已存在: {annotated_csv}")
        else:
            from annotate_ground_truth import annotate
            annotate(args.test_csv, annotated_csv, args.embed_backend, args.embed_model)

        # 如果有限制，生成子集 CSV
        recall_csv = annotated_csv
        if args.limit > 0:
            import tempfile
            with open(annotated_csv, "r", encoding="utf-8-sig") as f:
                all_rows = list(csv.DictReader(f))
            sub = all_rows[:args.limit]
            recall_csv = str(Path(args.test_csv).parent / f"test_set_{args.limit}.csv")
            with open(recall_csv, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sub[0].keys())
                w.writeheader()
                w.writerows(sub)
            print(f"[LIMIT] 子集 {args.limit} 条 → {recall_csv}")

        # 召回（不需要 LLM）
        args_recall = argparse.Namespace(
            test_csv=recall_csv, output_csv=None,
            embed_backend=args.embed_backend, embed_model=args.embed_model,
        )
        cmd_recall(args_recall)

        # 改写（需要 LLM）
        args_rewrite = argparse.Namespace(
            test_csv=recall_csv, output_csv=None,
            embed_backend=args.embed_backend, embed_model=args.embed_model,
            llm_backend=args.llm_backend, llm_model=args.llm_model, limit=args.limit,
        )
        cmd_rewrite(args_rewrite)

        # 归因（需要 LLM）
        args_attr = argparse.Namespace(
            test_csv=recall_csv, output_csv=None,
            embed_backend=args.embed_backend, embed_model=args.embed_model,
            llm_backend=args.llm_backend, llm_model=args.llm_model, limit=args.limit,
        )
        cmd_attribution(args_attr)

        print(f"\n{'='*60}")
        print(f"[DONE] 全部评测完成 → {args.output_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
