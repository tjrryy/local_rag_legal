"""
5 阶段法律问答 demo CLI。
支持单条 / REPL，支持流式输出（--stream）。
"""

import argparse
import time
import sys
from pathlib import Path

# 兼容 python demo/run_demo.py 调用
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import (  # noqa: E402
    LegalRAGPipeline,
    BGE_QUERY_PREFIX,
)


def render_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for t in history[-5:]:
        lines.append(f'"Q: {t["q"]}"')
        lines.append(f'"A: {t["a"]}"')
    return "\n".join(lines)


def repl_stream(pipeline: LegalRAGPipeline):
    """
    REPL 模式：Stage 1-4 走 pipeline，
    Stage 5 走 qa.stream() 流式打印（每个 token 出现即打印）。
    """
    print("=" * 60)
    print("[REPL] 输入问题，输入 q / quit / exit 退出")
    print("       多轮上下文会自动累积（用于改写）")
    print("       Stage 5 流式输出：答案边生成边显示")
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

        # Stage 1: 改写
        print(f"\n[Stage 1] 改写...")
        t0 = time.time()
        rewritten = pipeline.rewriter(q, history_str)
        print(f"      → {rewritten}  ({time.time()-t0:.0f}s)")

        # Stage 2: 法律名
        print(f"[Stage 2] 匹配法律...")
        t0 = time.time()
        kw_matched = pipeline._keyword_law_match(rewritten)
        if kw_matched:
            matched = kw_matched
            print(f"      用关键词匹配: {matched}")
        else:
            matched = pipeline.law_matcher(rewritten, top_k=3)
        print(f"      命中 {len(matched)} 部: {matched}  ({time.time()-t0:.0f}s)")

        # Stage 3: 取法条
        print(f"[Stage 3] 拉候选法条...")
        t0 = time.time()
        candidates = pipeline.fetcher(matched)
        print(f"      候选 {len(candidates)} 条  ({time.time()-t0:.0f}s)")

        # Stage 4: 精排
        print(f"[Stage 4] 精排 top-10...")
        t0 = time.time()
        ranked = pipeline.ranker(rewritten, candidates, top_k=10, law_filter=matched)
        print(f"      完成  ({time.time()-t0:.0f}s)")
        for i, (doc, score) in enumerate(ranked, 1):
            print(f"      {i}. {score:.3f}  《{doc.metadata.get('law_title','')}》"
                  f"{doc.metadata.get('article_no','')}")

        # Stage 5: 流式打印
        print(f"\n[Stage 5] 回答（流式）:")
        print("   ", end="", flush=True)
        sys.stdout.flush()
        t0 = time.time()

        def _chunk_print(delta: str):
            """每个 token 直接写到 stdout，无缓冲感"""
            sys.stdout.write(delta)
            sys.stdout.flush()

        llm_meta = pipeline.qa.stream(q, ranked, callback=_chunk_print)
        total_ms = (time.time() - t0) * 1000
        tok_s = llm_meta.get("tokens", 0) / (total_ms / 1000) if total_ms > 0 else 0
        print()  # 换行
        print(f"\n   [LLM 耗时 {total_ms:.0f}ms  TTFT={llm_meta.get('ttft_ms',0):.0f}ms  "
              f"速度={tok_s:.1f} tok/s")

        answer = llm_meta.get("text", "")

        # 累积历史
        history.append({"q": q, "a": answer})
        if len(history) > 10:
            history = history[-10:]


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
    parser.add_argument("--stream", action="store_true",
                        help="REPL 模式启用 Stage 5 流式输出")
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
        if args.stream:
            # 流式单条：手动走各阶段，Stage 5 流式输出
            result = pipeline.run(args.query)
            matched = result.matched_laws
            candidates = result.candidate_articles
            ranked = result.final_articles
            print(f"\n[Stage 1] 改写: {result.rewritten_query}")
            print(f"[Stage 2] 命中: {matched}")
            for i, (doc, score) in enumerate(ranked, 1):
                print(f"  {i}. {score:.3f} 《{doc.metadata.get('law_title','')}》{doc.metadata.get('article_no','')}")
            print(f"\n[Stage 5] 回答（流式）:")
            print("   ", end="", flush=True)
            t0 = time.time()

            def _cp(d):
                sys.stdout.write(d)
                sys.stdout.flush()
            pipeline.qa.stream(args.query, ranked, callback=_cp)
            print()
        else:
            result = pipeline.run_with_trace(args.query, no_rewrite=args.no_rewrite)
        return

    # REPL 模式
    if args.stream:
        repl_stream(pipeline)
        return

    print("=" * 60)
    print("[REPL] 输入问题，输入 q / quit / exit 退出")
    print("       多轮上下文会自动累积（用于改写）")
    print("       （加 --stream 开启流式输出）")
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
