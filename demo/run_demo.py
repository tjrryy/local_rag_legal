"""
demo/run_demo.py
================
5 阶段管道的 CLI 入口。

使用：
  # 交互式 REPL
  python3 demo/run_demo.py

  # 单条
  python3 demo/run_demo.py -q "单位拖欠工资怎么办？"

  # 多轮（演示改写效果）
  python3 demo/run_demo.py
  你> 国家对草原保护有什么方针？
  你> 它第三条具体说了什么？
  你> 违反这部法律怎么办？
  你> q
"""

from __future__ import annotations

import argparse
import os
import sys

from pipeline import LegalRAGPipeline


def render_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for t in history[-6:]:  # 只带最近 6 轮
        lines.append(f"Q: {t['q']}")
        lines.append(f"A: {t['a']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="5 阶段法律问答 demo")
    parser.add_argument("-q", "--query", help="单条查询；不给就进 REPL")
    parser.add_argument(
        "--embed-backend", default="hf", choices=["hf", "ollama"],
        help="embedding 后端（必须与 build_indexes 时一致）",
    )
    parser.add_argument(
        "--llm-backend", default="deepseek", choices=["deepseek", "ollama"],
        help="LLM 后端",
    )
    parser.add_argument("--top-laws", type=int, default=3, help="召回法律数")
    parser.add_argument("--top-articles", type=int, default=10, help="精排法条数")
    parser.add_argument("--embed-model", default="", help="覆盖默认 embedding 模型")
    parser.add_argument("--llm-model", default="", help="覆盖默认 LLM 模型")
    parser.add_argument("--no-rewrite", action="store_true",
                        help="跳过 Stage 1 Query Rewriter（排查用）")
    args = parser.parse_args()

    # 加载管道
    pipeline = LegalRAGPipeline(
        embed_backend=args.embed_backend,
        llm_backend=args.llm_backend,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
    )

    # 单条模式
    if args.query:
        result = pipeline.run_with_trace(args.query, no_rewrite=args.no_rewrite)
        return

    # REPL 模式
    print("=" * 60)
    print("[REPL] 输入问题，输入 q / quit / exit 退出")
    print("       多轮上下文会自动累积（用于改写）")
    print("=" * 60)

    history: list[dict] = []
    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[BYE]")
            return
        if not q:
            continue
        if q.lower() in {"q", "quit", "exit"}:
            print("[BYE]")
            return

        history_str = render_history(history)
        result = pipeline.run_with_trace(q, history_str)
        # 累积历史
        history.append({"q": q, "a": result.answer})
        if len(history) > 10:
            history = history[-10:]


if __name__ == "__main__":
    main()