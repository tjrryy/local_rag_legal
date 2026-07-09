"""
M5｜Benchmark：批量跑标准 query，报告性能 + 相似度
====================================================

学习目标：
  - 学会用"一组标准 query"评估检索质量
  - 看懂 avg latency / avg top-1 / avg top-K score 这 3 个数
  - 养成"先评估再优化"的习惯（而不是拍脑袋换模型）

使用：
  python3 m5_benchmark.py            # 跑内置 8 条
  python3 m5_benchmark.py --k 10     # 改 Top-K
  python3 m5_benchmark.py --quiet    # 不打印每条详情，只看汇总
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = Path(__file__).parent / "vector_db"
COLLECTION = "legal_articles"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 8 条覆盖不同法律领域：劳动 / 行政 / 刑事 / 数据 / 知识产权 等
BENCHMARK = [
    "单位拖欠工资，劳动者该怎么办？",
    "醉驾在法律上如何处理？",
    "国家对草原保护的方针是什么？",
    "网络运营者发生数据泄露应当如何处置？",
    "公民个人信息被泄露可以请求哪些救济？",
    "中医药国家如何保护和扶持？",
    "农产品质量安全由哪个部门监管？",
    "行政处罚的种类有哪些？",
]


def load_db(embeddings: HuggingFaceEmbeddings) -> Chroma:
    if not PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"{PERSIST_DIR}/ 不存在，请先运行: python3 m4_chroma.py --build"
        )
    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--quiet", action="store_true",
                        help="只打印汇总，不打印每条详情")
    args = parser.parse_args()

    print(f"[INFO] 加载模型 {MODEL_NAME} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
    db = load_db(embeddings)
    print(f"[INFO] 集合大小: {db._collection.count()}, Top-K = {args.k}\n")

    latencies: list[float] = []
    top1_scores: list[float] = []
    topk_scores: list[float] = []

    for q in BENCHMARK:
        # 计时
        t0 = time.time()
        results = db.similarity_search_with_score(
            BGE_QUERY_PREFIX + q, k=args.k
        )
        dt = (time.time() - t0) * 1000  # ms

        scores = [1.0 - d / 2.0 for _, d in results]
        top1_scores.append(scores[0])
        topk_scores.append(sum(scores) / len(scores))
        latencies.append(dt)

        if not args.quiet:
            print(f"Q: {q}")
            print(f"   {dt:6.1f} ms | top1={scores[0]:.4f} | top{args.k}_avg={sum(scores)/len(scores):.4f}")
            # 打印第 1 条命中
            doc, _ = results[0]
            title = doc.metadata.get("law_title", "")
            artno = doc.metadata.get("article_no", "")
            print(f"   → top1: 《{title}》{artno}")
            print()

    # 汇总
    n = len(BENCHMARK)
    print("=" * 60)
    print(f"[汇总] 样本数 = {n}, K = {args.k}")
    print(f"  avg latency     : {sum(latencies)/n:6.1f} ms")
    print(f"  avg top-1 score : {sum(top1_scores)/n:.4f}")
    print(f"  avg top-{args.k} score : {sum(topk_scores)/n:.4f}")
    print()
    print("判读参考（经验值，仅供参考）：")
    print("  - top-1 score > 0.7  → 命中度好（找到的法条和问题语义很接近）")
    print("  - top-1 score 0.5~0.7 → 勉强（找到相关但不够精确）")
    print("  - top-1 score < 0.5  → 偏题（可能 query 太抽象 / 跨域 / 用词非法律语）")
    print("  - latency < 100ms (CPU) → 工程上够用")


if __name__ == "__main__":
    main()
