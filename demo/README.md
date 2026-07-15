# demo/ - 5 阶段法律问答管道

完整说明见仓库根目录 [README.md](../README.md)。这里只保留 demo 内部的文件索引和最小命令。

## 文件

| 文件 | 作用 |
|---|---|
| `chainlit_app.py` | Chainlit Web 前端（流式问答 + 文件上传） |
| `pipeline.py` | 5 阶段管道核心逻辑 |
| `run_demo.py` | CLI 入口（REPL / 单条） |
| `run_eval.py` | 100 条自动化评测 |
| `build_indexes.py` | 一次性建 2 个 FAISS 索引（法律名 + 法条） |
| `test_pipeline_no_llm.py` | 不调 LLM 的离线两段检索测试 |
| `indexes/` | FAISS 持久化（git ignore） |

## 快速启动

```bash
# Web 前端
chainlit run demo/chainlit_app.py

# CLI
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  python3 demo/run_demo.py \
  --embed-backend ollama --embed-model bge-m3 \
  --llm-backend ollama --llm-model qwen2.5:7b
```

## 5 阶段

```
[1] Query Rewriter  → qwen2.5 把"它"还原成具体法律名
[2] Law Name Matcher → FAISS 在 303 部法律名里找 top-3
[3] Article Fetcher  → 内存字典取这 3 部法律的全部法条
[4] Article Ranker   → bge-m3 批量 embed + 余弦 Top-10
[5] QA Agent         → qwen2.5 按 prompt 写答案 + 引用
```
