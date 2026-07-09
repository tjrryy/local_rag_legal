# 基于本地大语言模型的法律法规智能问答系统

## 项目概述

本项目构建一个基于本地大语言模型的法律法规智能问答系统，结合 RAG（检索增强生成）技术与多 Agent 编排，支持多轮交互、按法律名精确定位法条、自动引用原文作答。已通过端到端验证：303 部法律 / 22,482 条法条 / 两阶段 FAISS 检索 / Ollama 本地 LLM（bge-m3 + qwen2.5:7b）跑通完整 5 阶段管道。

## 项目背景

法律领域文档量大、引用关系严格，单纯的 LLM 容易幻觉。本项目以 303 部法律法规、22,000+ 条法条为基础，自建嵌入索引，把检索从“全库 Top-K”改为“先找法律 → 再找法条”两段式，强约束回答在正确法律内，最后用本地 LLM 生成带法条引用的答案。

## 核心技术栈

| 类别 | 选型 | 状态 |
|---|---|---|
| Embedding | Ollama + bge-m3（1024 维）/ HuggingFace BAAI/bge-small-zh-v1.5 | ✅ 已跑通 |
| 向量索引 | FAISS（两段：法律名 + 法条） | ✅ 全量已建 |
| LLM | Ollama + qwen2.5:7b / DeepSeek API | ✅ Ollama 已跑通 |
| Embedding / LLM 切换 | subprocess + curl 批量端点 | ✅ 已封装 |
| 检索框架 | LangChain FAISS + 自写 ArticleFetcher | ✅ |
| 多轮上下文 | 累积历史 → Stage 1 改写 | ✅ |
| Web 服务 | FastAPI（计划中） | ⏳ |
| Agent 编排 | LangGraph（计划中） | ⏳ |

## 数据统计

- 法律：**303 部**（唯一去重）
- 法条：**22,482 条**
- 覆盖：宪法、民法商法、行政法、刑法、刑事诉讼法、社会法、经济法、资源环境法等
- 数据位置：`law_clearnerdata/laws_dataset_*.json`

## 5 阶段管道（已实现）

```
用户问题
   ↓
[1] Query Rewriter       把"它/那部法律"还原成具体法律名（qwen2.5 LLM）
   ↓
[2] Law Name Matcher     FAISS 在 303 部法律名里找 top-3
   ↓
[3] Article Fetcher      从 3 部法律的全部法条拉候选（in-memory 字典 O(1)）
   ↓
[4] Article Ranker       bge-m3 批量 embed 候选法条 + 余弦相似度 Top-10
   ↓
[5] QA Agent             qwen2.5 LLM 按 prompt 写答案 + 引用《法律名》第X条
   ↓
最终答案（带法条原文引用）
```

每阶段独立可替换，接口都遵循 LangChain / 自写 dataclass 协议。

## 性能（已验证）

测试问题：`草原保护有什么方针？`（303 部法律 / 22,482 条法条 / Ollama 本地）

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 1_rewrite | ~4.0 s | qwen2.5:7b 改写问题 |
| 2_match_laws | ~1.7 s | bge-m3 + FAISS Top-3 |
| 3_fetch | <10 ms | in-memory 字典 |
| 4_rank | ~9.4 s | `/api/embed` 批量 32 条，7 次请求 |
| 5_answer | ~9.0 s | qwen2.5:7b 生成答案 |
| **总耗时** | **~24 s** | 端到端（已含模型冷启动） |

## 快速开始

### 1. 环境准备

```bash
# 1) 安装 Ollama（macOS）
brew install ollama
# 或下载安装：https://ollama.com/download

# 2) 拉模型
ollama pull bge-m3        # 1.2 GB，1024 维中文 embedding
ollama pull qwen2.5:7b    # 4.7 GB，本地 LLM

# 3) 启动 Ollama（自动后台运行）
ollama serve
```

```bash
# 4) 装 Python 依赖
pip install -U \
  langchain langchain-community langchain-core \
  faiss-cpu sentence-transformers numpy
```

### 2. 跑 demo

```bash
cd /Users/icec0re/Desktop/git_submit/local_rag_legal

# 一次性建索引（~5-10 分钟，全量）
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  python3 demo/build_indexes.py \
  --embed-backend ollama --embed-model bge-m3 --reset

# 单条问答
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  python3 demo/run_demo.py \
  -q "草原保护有什么方针？" \
  --embed-backend ollama --embed-model bge-m3 \
  --llm-backend ollama --llm-model qwen2.5:7b

# REPL 多轮
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  python3 demo/run_demo.py \
  --embed-backend ollama --embed-model bge-m3 \
  --llm-backend ollama --llm-model qwen2.5:7b
```

REPL 中输入 `q` / `quit` / `exit` 退出。多轮上下文会自动累积喂给 Stage 1 改写。

### 3. 切到 HuggingFace Embedding（无 Ollama）

```bash
python3 demo/build_indexes.py --embed-backend hf --reset
python3 demo/run_demo.py \
  -q "草原保护有什么方针？" \
  --embed-backend hf \
  --llm-backend ollama --llm-model qwen2.5:7b
```

### 4. 切到 DeepSeek API

```bash
export DEEPSEEK_API_KEY="sk-..."
python3 demo/run_demo.py \
  -q "草原保护有什么方针？" \
  --embed-backend ollama --embed-model bge-m3 \
  --llm-backend deepseek
```

## 测试问题清单

**Level 1｜单段召回**

```bash
for q in \
  "草原保护有什么方针？" \
  "数据泄露怎么处理？" \
  "醉驾在法律上如何处理？" \
  "个人信息被泄露可以请求哪些救济？" \
  "中医医院的管理规定是什么？"
do
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
    python3 demo/run_demo.py -q "$q" \
    --embed-backend ollama --embed-model bge-m3 \
    --llm-backend ollama --llm-model qwen2.5:7b
done
```

**Level 2｜多轮指代改写（REPL 验证）**

```
你> 国家对草原保护有什么方针？
你> 它第三条具体说了什么？
你> 违反这部法律怎么处罚？
你> q
```

**Level 3｜边界场景**

- 抽象问：`什么是连带责任？`
- 跨法律：`员工受伤公司怎么赔？`
- 否定/反问：`草原法第三条没说什么？`

## 后端切换矩阵

| 维度 | 选项 | 切换方式 |
|---|---|---|
| Embedding | bge-m3 / bge-large / BGE-small-zh / M3E | `--embed-backend ollama --embed-model <name>` |
| LLM | qwen2.5:7b / deepseek-chat / gpt-4o-mini | `--llm-backend ollama --llm-model <name>` 或设 `DEEPSEEK_API_KEY` |
| 向量索引 | FAISS | `demo/indexes/{law_names,articles}/` |

## 关键技术细节

### 1. 两段检索 vs 单段

| 维度 | 单段（22K Top-K） | 两段（303 → 200 → 10） |
|---|---|---|
| 召回率 | 中（热门法律挤掉小众） | 高（强约束在正确法律） |
| 精度 | 中 | 高 |
| 适合 | 开放域 | 法规/手册这种强领域 |

### 2. Embedding 批量调用

用 Ollama `/api/embed` 端点（不是 `/api/embeddings`），`BATCH=32`：

- 200 候选 / 32 ≈ 7 次请求
- 单次响应 ~1-2 s
- 相比单条串行：30 s → 9 s（约 3×）

### 3. Sandbox 兼容

当前 sandbox 下 Python HTTP 客户端（`requests` / `urllib`）会被 502，所以 `RobustOllamaEmbeddings` / `RobustOllamaLLM` 都用 `subprocess + curl`。如果你的环境是普通 macOS / Linux，可以直接换回 `requests`，代码更简洁。

### 4. 超长法条截断

`MAX_EMBED_CHARS = 1200`（`demo/build_indexes.py`）：超过 1200 字的法条用前 1200 字做 embedding，metadata 里仍保留全文。这是为了避开 bge-m3 的 4096 上下文限制。`max_article_len` 实际是 24,399 字（民法典某条）。

## 项目结构

```
local_rag_legal/
├── README.md
├── LICENSE
├── law_clearnerdata/                # 数据
│   ├── laws_dataset_*.json         #   303 部法律 / 22,482 条法条
│   ├── laws_dataset_*.csv
│   └── all_articles_*.txt
├── start_test/                      # 学习路径（m1 → m7）
│   ├── m1_explore_data.py
│   ├── m2_first_embedding.py
│   ├── m3_encode_and_search.py
│   ├── m4_chroma.py
│   ├── m5_benchmark.py
│   ├── m6_rag.py
│   ├── m7_multi_agent_chat.py
│   └── m7_memory_test.py
└── demo/                            # 生产路径（5 阶段管道）
    ├── README.md
    ├── build_indexes.py             #   一次性建 FAISS
    ├── pipeline.py                  #   5 阶段核心逻辑
    ├── run_demo.py                  #   CLI 入口
    ├── test_pipeline_no_llm.py      #   离线两段检索测试
    └── indexes/                     #   FAISS 持久化（git ignore）
        ├── law_names/
        └── articles/
```

## 后续计划

- [x] 5 阶段管道端到端跑通
- [x] bge-m3 + qwen2.5:7b 切换为默认（替代 deepseek-r1:7b）
- [x] Embedding 批量调用（30 s → 9 s）
- [ ] FastAPI 包成 `/api/chat`
- [ ] 100+ 条测试集 + 自动评估脚本
- [ ] LangGraph 多 Agent 编排
- [ ] 嵌入模型自动评估（bge-m3 vs bge-large vs M3E）

## 参考文献

- [国家法律法规数据库](https://flk.npc.gov.cn/)
- [Ollama](https://ollama.com/)
- [bge-m3 on HuggingFace](https://huggingface.co/BAAI/bge-m3)
- [qwen2.5 on Ollama](https://ollama.com/library/qwen2.5)
- [LangChain FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss)

## 致谢

指导老师：徐继伟老师

---

*本项目为实习项目，目标是构建一个完整的法律法规智能问答系统。*
