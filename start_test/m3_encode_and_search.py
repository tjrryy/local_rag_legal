"""
M3｜全量编码 + 手写 Top-K 检索
================================

学习目标：
  - 看到"22K 条法条 → (22482, 512) 的矩阵"长什么样
  - 用 numpy 矩阵乘法一次性算"query vs 全部法条"的相似度
  - 用 np.argpartition + 排序取 Top-K

为什么这步很有教学价值：
  - 向量库的"向量检索"底层就是这 3 行代码
  - 只是库帮你做了：分批、持久化、近似加速、过滤等工程优化
  - 这一步跑通后，下一步接 ChromaDB 几乎是机械替换

跑法：
  python3 m3_encode_and_search.py
  python3 m3_encode_and_search.py --query "单位拖欠工资怎么办" --k 5
  python3 m3_encode_and_search.py --query "草原保护" --k 3
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---- 常量 ----
DATA_DIR = Path(__file__).parent.parent / "law_clearnerdata"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 用「第X条」开头作为 chunk 边界标记
ARTICLE_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+条")

# 为了和后续 ChromaDB 一致，每条 page_content 前缀加上「《法律名》第X条」
PAGE_PREFIX = "《{title}》{article_no}　{text}"


# ---- 1. 读取 JSON，把每条法条做成 (text, meta) ----

def load_articles(data_dir: Path) -> list[dict]:
    """
    返回 list[dict]，每条 dict 有：
      - text: 给模型吃的文本（带前缀）
      - raw:  法条原文
      - law_title: 法律名
      - article_no: 第X条
      - source_file: 来源
    """
    out = []
    files = sorted(data_dir.glob("laws_dataset_*.json"))
    for fp in files:
        laws = json.loads(fp.read_text(encoding="utf-8"))
        for law in laws:
            title = (law.get("title") or "").strip()
            for art in law.get("articles", []):
                art = art.strip()
                if not art:
                    continue
                m = ARTICLE_PATTERN.match(art)
                article_no = m.group(0) if m else ""
                out.append({
                    "text": PAGE_PREFIX.format(
                        title=title, article_no=article_no, text=art
                    ),
                    "raw": art,
                    "law_title": title,
                    "article_no": article_no,
                    "source_file": law.get("source_file", ""),
                })
    return out


# ---- 2. 批量编码 ----

def encode_all(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 128,
) -> np.ndarray:
    """
    把 N 条文本编码成 (N, 512) 的 numpy 矩阵。
    model.encode 自带进度条（设 show_progress_bar=True）。
    """
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,    # 归一化 → 后面用点积就是余弦
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vecs.astype("float32")    # faiss/chroma 都吃 float32


# ---- 3. 手写 Top-K ----

def topk_search(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    metas: list[dict],
    k: int = 5,
) -> list[dict]:
    """
    给定 query 向量 (512,)，在 doc_vecs (N, 512) 里找 Top-K 最相似的。
    相似度 = query · doc（点积，因为两边都 L2 归一化）
    返回 list[dict]，每个包含 meta + 相似度分数
    """
    # 矩阵乘法：query (1,512) @ (512,N) → (1,N)
    # 等价于对每条 doc 算点积
    scores = (doc_vecs @ query_vec).astype("float32")

    # argpartition 比 argsort 快：只把 Top-K 排到前面
    topk_idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
    # 真正的相似度排序
    topk_idx = topk_idx[np.argsort(-scores[topk_idx])]

    results = []
    for idx in topk_idx:
        results.append({
            **metas[idx],
            "score": float(scores[idx]),
            "rank": len(results) + 1,
        })
    return results


# ---- 4. 主流程 ----

def main():
    parser = argparse.ArgumentParser(description="M3 全量编码 + Top-K 检索")
    parser.add_argument("--query", "-q", default="单位拖欠工资，劳动者该怎么办？")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0,
                        help="只编码前 N 条（调试用，0=全部）")
    args = parser.parse_args()

    # 1) 加载法条
    print(f"[INFO] 加载 {DATA_DIR} ...")
    articles = load_articles(DATA_DIR)
    if args.limit > 0:
        articles = articles[: args.limit]
    print(f"[INFO] 法条总数: {len(articles)}")

    # 2) 加载模型
    print(f"[INFO] 加载模型 {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    # 3) 全量编码
    t0 = time.time()
    texts = [a["text"] for a in articles]
    doc_vecs = encode_all(model, texts, batch_size=args.batch_size)
    print(f"[INFO] 编码完成: shape={doc_vecs.shape}, 耗时 {time.time() - t0:.1f} s")

    # 4) 编码 query
    q_vec = model.encode(
        [BGE_QUERY_PREFIX + args.query],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0].astype("float32")

    # 5) 检索
    t0 = time.time()
    results = topk_search(q_vec, doc_vecs, articles, k=args.k)
    dt = (time.time() - t0) * 1000

    # 6) 打印
    print()
    print("=" * 72)
    print(f"Q: {args.query}")
    print(f"检索耗时: {dt:.1f} ms")
    print("=" * 72)
    for r in results:
        snippet = r["raw"]
        if len(snippet) > 100:
            snippet = snippet[:100] + "..."
        print(f"[{r['rank']}] score={r['score']:.4f}  "
              f"《{r['law_title']}》{r['article_no']}")
        print(f"    {snippet}")
        print()


if __name__ == "__main__":
    main()
