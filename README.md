# 基于本地大语言模型的法律法规智能问答系统

## 项目概述

本地化的中国法律法规问答助手，支持**流式输出**、**多轮对话**、**实体抽取增强检索**、**HyDE 可选链路**。已通过端到端验证：303 部法律 / 22,482 条法条 / 两段 FAISS 检索 / Ollama 本地模型（bge-m3 + qwen2.5:7b）。

## 核心技术栈

| 类别 | 选型 | 状态 |
|---|---|---|
| Embedding | Ollama + bge-m3（1024 维）| ✅ |
| 向量索引 | FAISS 两段（法律名 + 法条）| ✅ |
| LLM | Ollama + qwen2.5:7b | ✅ |
| 流式输出 | SSE，token by token 实时推送 | ✅ |
| 多轮会话 | SessionMemory（进程内存，退出即清）| ✅ |
| Web 前端 | Chainlit（ChatGPT 风格，原生流式） | ✅ |
| 文件上传 | 支持 .txt / .md / .csv 提取文字问答 | ✅ |
| 实体抽取 | Stage 1.75：keyword_lookup() 字典命中 < 1ms | ✅ |
| HyDE | Stage 1.5：LLM 先生成假设回答增强检索（可选 `--hyde`）| ✅ |
| 自动化评测 | 100 条带标注测试集 + run_eval.py | ✅ |
| CLI | REPL + `--stream` 流式 | ✅ |

## 快速开始

```bash
# 1) 装依赖
pip install chainlit

# 2) 启动 Web 前端
chainlit run demo/chainlit_app.py
# → 浏览器打开 http://localhost:8501

# 3) CLI REPL（流式）
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  python3 demo/run_demo.py --stream \
  --embed-backend ollama --embed-model bge-m3 \
  --llm-backend ollama --llm-model qwen2.5:7b

# 4) CLI 单条
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  python3 demo/run_demo.py --stream \
  -q "草原保护有什么方针？" \
  --embed-backend ollama --embed-model bge-m3 \
  --llm-backend ollama --llm-model qwen2.5:7b

# 5) 自动化评测 100 条
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  python3 demo/run_eval.py --limit 100
```

## 5 阶段管道

```
用户问题
   ↓
[1] Query Rewriter     qwen2.5 LLM 改写，还原"它/那部/第三条"
   ↓
[1.5] HyDE（可选 --hyde）  LLM 先生成假设法条回答，增强检索
   ↓
[1.75] 实体抽取          keyword_lookup() 字典命中法律名（<1ms）
   ↓
[2] Law Name Matcher   FAISS 在 303 部法律名里找 top-3
   ↓
[3] Article Fetcher   内存字典 O(1) 取 3 部法律的全部法条
   ↓
[4] Article Ranker    FAISS 粗排 + bge-m3 精排 Top-10
   ↓
[5] QA Agent         qwen2.5 按 prompt 写答案 + 引用原文

最终答案（带法条编号引用 + 流式输出）
```

## 性能数据

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 1_rewrite | ~600 ms | qwen2.5:7b |
| 1.75 实体抽取 | < 1 ms | 纯字典命中 |
| 2_match_laws | ~150 ms | bge-m3 + FAISS |
| 4_rank | ~800 ms | FAISS + bge-m3 精排 |
| 5_answer | ~3-6 s | qwen2.5:7b，流式输出 TTFT ~5s |
| **总耗时** | **~6-9 s** | 含模型冷启动 |

## 项目结构

```
local_rag_legal/
├── README.md
├── PROJECT.md                   # 技术文档
├── law_clearnerdata/            # 数据（303 部法律 / 22,482 条法条）
├── start_test/                  # 探索脚本 m1~m7
├── public/                      # 前端静态资源（CSS / JS）
├── .chainlit/                   # Chainlit 配置
└── demo/
    ├── chainlit_app.py          # Chainlit Web 前端（ChatGPT 风格）
    ├── pipeline.py              # 5 阶段核心逻辑
    ├── run_demo.py              # CLI 入口（--stream）
    ├── run_eval.py              # 100 条自动评测
    ├── eval_set.json            # 100 条带标注测试集
    ├── session_memory.py        # 会话短期记忆
    ├── build_indexes.py         # 建 FAISS 索引
    ├── test_pipeline_no_llm.py  # 离线检索测试
    └── indexes/                 # FAISS 持久化（git ignore）
```

## 引用格式

指导老师：徐继伟老师
