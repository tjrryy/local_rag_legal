"""
M1｜读懂 laws_dataset_*.json 的结构
====================================

学习目标：
  - 学会用 pathlib + json 读文件
  - 看清「法律」和「法条」在 JSON 里的样子
  - 知道 chunk 粒度 = 单条法条

不引入任何第三方包，只用 Python 标准库。
跑法：python3 m1_explore_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

# ---- 1. 找到数据文件 ----
# 路径 = 当前脚本所在目录的上一级 / law_clearnerdata / laws_dataset_*.json
# pathlib.Path 是比 os.path 更现代的文件路径操作方式
DATA_DIR = Path(__file__).parent.parent / "law_clearnerdata"

# glob 找所有匹配的文件，按文件名排序（保证每次跑顺序一样）
files = sorted(DATA_DIR.glob("laws_dataset_*.json"))
print(f"[INFO] 找到 {len(files)} 个 JSON 文件")
print(f"[INFO] 第一个文件: {files[0].name}")
print()

# ---- 2. 读第一个文件，看顶层结构 ----
# json.load(文件对象) → Python 对象（list / dict / ...）
raw_text = files[0].read_text(encoding="utf-8")
data = json.loads(raw_text)

# 顶层是什么类型？
print(f"[INFO] 顶层类型: {type(data).__name__}")        # 期望: list
print(f"[INFO] 法律条数: {len(data)}")                    # 期望: 几十到几百
print()

# ---- 3. 拿第一部法律看字段 ----
law = data[0]
print(f"[INFO] 第一部法律的字段:")
for key, value in law.items():
    # 故意截断长字符串，避免打印刷屏
    if isinstance(value, str) and len(value) > 60:
        preview = value[:60] + "..."
    else:
        preview = value
    print(f"  {key}: {type(value).__name__}  =  {preview!r}")
print()

# ---- 4. 看 articles 数组里的样子 ----
articles = law["articles"]
print(f"[INFO] 这部法律共有 {len(articles)} 条法条")
print(f"[INFO] 第一条:")
print(f"  {articles[0]!r}")
print()

# ---- 5. 跨所有文件做一次统计 ----
total_articles = 0
total_laws = 0
for fp in files:
    laws = json.loads(fp.read_text(encoding="utf-8"))
    total_laws += len(laws)
    for law in laws:
        total_articles += len(law.get("articles", []))

print("=" * 50)
print(f"[总结] 跨 {len(files)} 个 JSON：")
print(f"       法律数:  {total_laws}")
print(f"       法条总数: {total_articles}")
print(f"       预计 chunk 数 = 法条数 = {total_articles}")
print("=" * 50)
