"""
demo/eval/ — 法律 RAG 评测模块

  L1: eval_retrieval.py   — 检索质量（法律名 Recall / MRR / Precision）
  L2: eval_citation.py    — 引用准确性（幻觉检测 / 引用验证）
  eval_runner.py          — 统一入口（读 CSV → 跑评测 → 输出 CSV）
"""
