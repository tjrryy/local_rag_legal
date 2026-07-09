"""
M2｜第一条 Embedding + 手算余弦相似度
======================================

学习目标：
  - 看到 embedding 把一句话变成 384 个浮点数
  - 自己用 numpy 写一遍余弦相似度，不调任何库函数
  - 直观感受"语义相近 → 相似度高"

为什么手写：
  - 你以后调 LangChain / ChromaDB 时，这些函数都藏在底层
  - 手算过一次，余弦相似度对你就再也不是黑盒

跑法：python3 m2_first_embedding.py
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# 模型第一次运行会自动下载到 ~/.cache/huggingface/
# 95MB，bge 系列中文 SOTA、384 维、跑得动 CPU
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# bge 官方建议：query 前面加这个前缀，会让短问句召回更好
# 但 documents 端不加，保持原文
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    手算余弦相似度：
        sim = (a · b) / (||a|| * ||b||)

    因为我们下面会做 L2 归一化（让 ||a|| = ||b|| = 1），
    所以实际就是 sim = a · b，点积就行。
    这里保留通用公式，让你看清"归一化"到底省了什么。
    """
    dot = float(np.dot(a, b))                # a · b
    norm_a = float(np.linalg.norm(a))        # ||a||
    norm_b = float(np.linalg.norm(b))        # ||b||
    return dot / (norm_a * norm_b + 1e-12)   # +1e-12 防 0 除


def main():
    # ---- 1. 加载模型 ----
    # 这一步是整个项目最慢的：第一次要下载 + 加载权重
    print(f"[INFO] 加载模型 {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"[INFO] 模型最大输入长度: {model.max_seq_length}")
    print()

    # ---- 2. 准备 3 个文本：1 条法条 + 2 个 query ----
    # 这是《劳动合同法》第三十条，工资相关
    article = (
        "第三十条　用人单位应当按照劳动合同约定和国家规定，向劳动者及时足额支付劳动报酬。"
        "用人单位拖欠或者未足额支付劳动报酬的，劳动者可以依法向人民法院申请支付令，"
        "人民法院应当依法发出支付令。"
    )

    # query 1: 语义相关
    q_related = "单位拖欠工资，劳动者该怎么办？"
    # query 2: 语义无关（草原保护）
    q_unrelated = "国家保护草原有什么方针？"

    # ---- 3. 编码 ----
    # documents 端：不加前缀，原文入
    doc_vec = model.encode(
        [article],
        normalize_embeddings=True,    # 关键：L2 归一化，让点积 = 余弦
        show_progress_bar=False,
    )[0]

    # query 端：bge 官方建议加前缀
    q_related_vec = model.encode(
        [BGE_QUERY_PREFIX + q_related],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    q_unrelated_vec = model.encode(
        [BGE_QUERY_PREFIX + q_unrelated],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # ---- 4. 看看向量长什么样 ----
    print(f"[INFO] 向量维度: {doc_vec.shape}")              # 期望 (384,)
    print(f"[INFO] 向量前 8 维: {doc_vec[:8]}")
    print(f"[INFO] 向量模长 (应≈1.0): {float(np.linalg.norm(doc_vec)):.6f}")
    print()

    # ---- 5. 算相似度 ----
    sim_related = cosine_sim(doc_vec, q_related_vec)
    sim_unrelated = cosine_sim(doc_vec, q_unrelated_vec)

    print("=" * 60)
    print(f"法条: {article[:30]}...")
    print()
    print(f"Q1: {q_related}")
    print(f"  余弦相似度 = {sim_related:.4f}    ← 应该比较高")
    print()
    print(f"Q2: {q_unrelated}")
    print(f"  余弦相似度 = {sim_unrelated:.4f}    ← 应该比较低")
    print("=" * 60)

    # ---- 6. 验证"归一化后点积 = 余弦" ----
    print()
    print(f"验证: doc·q1 (点积) = {float(np.dot(doc_vec, q_related_vec)):.4f}")
    print(f"      cosine_sim(手算)  = {sim_related:.4f}")
    print("      → 归一化后两者相等，库内部就这么干的。")


if __name__ == "__main__":
    main()
