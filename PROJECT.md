# 项目流程与技术文档


---

## 1. 项目目标

做一个**本地化的中国法律法规问答助手**。用户输入一句自然语言问题（例如"醉驾怎么处理"），系统回答并**明确引用法条原文**。

关键约束：

- 数据和模型都在本机跑，**不出云**
- 必须给出**真实法条**而不是 LLM 自己编
- 多轮对话要能维持上下文（"它第三条说了什么"）

## 2. 数据

源数据落在 `law_clearnerdata/`：

| 文件 | 内容 |
|---|---|
| `laws_dataset_20260706_144650.json` | **303 部法律**，每部含 `articles` 列表（每条是一条法条原文） |
| `laws_dataset_20260706_144650.csv` | 同上，CSV 版 |
| `all_articles_20260706_144650.txt` | 把所有法条拼成纯文本，约 ~27 MB |

最终参与检索的**唯一数据**就是那一个 JSON：303 部法律、22,482 条法条。

## 3. 怎么一步步做的（演进路径）

为了不踩坑，我们分**两阶段**推进：

### 第一阶段：`start_test/` —— 探索与试错（m1 ~ m7）

从最基础的"读文件 + 向量化 + 检索"一步步长起来，每个脚本只做一件事：

| 脚本 | 目标 |
|---|---|
| `m1_explore_data.py` | 读 JSON，看清楚 303 部法律长什么样 |
| `m2_first_embedding.py` | 用 HuggingFace 的 `BAAI/bge-small-zh-v1.5` 给法律名编码 |
| `m3_encode_and_search.py` | 给法条编码 + 用 numpy 算点积检索 |
| `m4_chroma.py` | 切到 ChromaDB 跑通向量库，对比自建 vs 框架 |
| `m5_benchmark.py` | 测几个 embedding 模型的速度 + 显存 |
| `m6_rag.py` | 把检索结果塞进 LLM prompt，第一次看到"问答"形态 |
| `m7_multi_agent_chat.py` | 多 Agent + 多轮对话的样例 |
| `m7_memory_test.py` | 验证上下文记忆的持久化 |

这阶段踩出来的几个关键经验：

1. **在 macOS 上用 Ollama 比裸 transformers 顺很多**。transformers + Metal 各种 OpenMP 冲突。
2. **bge-small-zh（512 维）够用，跑得快**，但后面我们升级到 bge-m3（1024 维）。
3. **法条粒度比法律名粒度重要**：用户问"具体怎么处理"，所以每条法条一个向量才对。

### 第二阶段：`demo/` —— 端到端管道（生产级）

把上面零散脚本沉淀成可运行的 5 阶段管道：

```
用户问题
  ↓
[1] QueryRewriter     qwen2.5 LLM，把"它"还原成具体法律名
  ↓
[2] LawNameMatcher    FAISS 在 303 部法律名里找 top-3
  ↓
[3] ArticleFetcher    从 3 部法律的全部法条拉候选
  ↓
[4] ArticleRanker     FAISS 在限定法律里粗排 + bge-m3 精排，取 top-10
  ↓
[5] QAAgent           qwen2.5 按 prompt 写答案 + 引用《法律名》第X条
  ↓
最终答案
```

每个文件夹职责清晰：

| 文件 | 干什么 |
|---|---|
| `demo/build_indexes.py` | 一次性建 2 个 FAISS 索引（法律名 + 法条） |
| `demo/pipeline.py` | 5 阶段核心逻辑，所有 Agent 类都在这里 |
| `demo/run_demo.py` | CLI 入口（单条 / REPL） |
| `demo/test_pipeline_no_llm.py` | 不调 LLM 的离线两段检索测试 |
| `demo/run_eval.py` | 对 `eval_set.json` 跑评测，打召回 / 命中率 / 延迟 |
| `demo/eval_set.json` | 100 条带人工标注的法律问答测试集 |

## 4. 关键技术详解

### 4.1 Embedding：`bge-m3` via Ollama

选 `bge-m3` 的理由：

- 中文法律语料上比 bge-small-zh 略准
- 1024 维向量区分度足够
- **Ollama 提供，Mac 上 Metal 加速**，不需要自己处理 OpenMP / device_map

调用上踩过一个**sandbox 坑**：当前开发环境下 Python `requests` / `urllib` 访问 `localhost:11434` 会被拦（502），但 `curl` 命令行访问没事。所以 `RobustOllamaEmbeddings` 用 **subprocess 调 curl**，绕开 sandbox。生产环境换回 `requests` 即可。

```python
class RobustOllamaEmbeddings:
    def embed_documents(self, texts):
        # 走 /api/embed 批量端点 + subprocess+curl，BATCH=128
        ...
```

另一个经验：**bge-m3 上下文 4096**，但我们最长的法条有 24,000+ 字（民法典某条）。`build_indexes.py` 里加了 `MAX_EMBED_CHARS = 1200`，超过的截到前 1200 字做 embedding，原文仍存 `metadata["text"]` 给阶段 5 引用。

### 4.2 向量索引：FAISS

建两个独立索引：

| 索引 | 数量 | 用来做 |
|---|---|---|
| `indexes/law_names/` | 303 个向量 | Stage 2：在 303 部法律里找 top-3 |
| `indexes/articles/` | 22,482 个向量 | Stage 4 hybrid 粗排 |

实测 `indexes/articles/index.faiss` 是 **92 MB**，加载 + 搜 Top-50 仅 **1.9 ms**。当前是 `IndexFlatL2`（暴力），22K × 1024 维的规模没必要上 HNSW。

### 4.3 LLM：Ollama `qwen2.5:7b`

`deepseek-r1:7b` 是思考型（chain-of-thought）模型，Stage 5 生成一个回答要 ~18 秒。换成 `qwen2.5:7b` 后降到 ~3-9 秒，质量够用。

LLM 端调用也是 `subprocess + curl`，并切到 `stream: True` + `curl --no-buffer`，这样可以**统计首字延迟 (TTFT)**。当前实测：

- 平均端到端 17 s
- 平均 TTFT 4.8 s
- 生成速度 10-14 token/s

### 4.4 5 阶段管道

**Stage 1 Query Rewriter**：单轮问题几乎不消耗（直接用原问），多轮对话里把"它/那部法律"还原成完整法律名。

**Stage 2 LawNameMatcher**：把改写后的问题用 bge-m3 编码，去 `law_names` 索引搜 top-3。另外加了一个**关键词兜底**（`_keyword_law_match`）：如果改写后问题里含有"草原法""网络安全法"等明确实体词，直接跳过 FAISS 走字典查，进一步省时间。

**Stage 3 ArticleFetcher**：纯字典查询，`O(1)`，< 1 ms。把 3 部法律的全部法条作为候选（约 200 条）返给 Stage 4。

**Stage 4 ArticleRanker（hybrid）**：

1. **FAISS 粗排**：限定 `law_filter = matched_laws`，在 22K 法条库里搜 top-50（这条规则由 Stage 2 锁定，杜绝"答错法律"）
2. **bge-m3 精排**：重新编码这 50 条，按点积排序取 top-10

如果只用 FAISS，召回率下降 ~1-2%，但能从 Stage 4 的 ~9 s 压到 ~5 ms。我们保留精排，权衡选了"再快也别损失精度"。

**Stage 5 QAAgent**：prompt 强制 LLM 必须先引用"《法律名》第X条"再解释。这是法律领域最关键的硬约束 —— 不能让 LLM 编。

```text
你是中国法律领域的智能助手。请严格根据下面【法条参考】回答用户问题。
要求：
1. 必须先引用法条原文（用「《法律名》第X条」格式），再做解释
2. 如果多条法条相关，按相关度从高到低引用
3. 如果【法条参考】中没有任何相关内容，请直接回答："现有法条中未直接规定该问题"
```

### 4.5 评测体系

`demo/eval_set.json` 有 100 条，每条结构：

```json
{
  "id": 1,
  "question": "草原保护有什么方针？",
  "expected_answer": "国家对草原实行科学规划...",
  "retrieved_text": "《草原法》第三条：...",
  "model_output": "根据《草原法》第三条...",
  "evaluation_note": "回答准确，引用《草原法》第三条完整原文..."
}
```

`demo/run_eval.py` 跑 100 条，统计三个核心指标：

| 指标 | 当前 | 说明 |
|---|---|---|
| Stage 2 法律名命中率 | **32.7%** | 33/101，期望命中的法律中召回的比例 |
| Stage 4 法条召回率 | **25.0%** | 26/104，期望法条编号出现在 Top10 的比例 |
| Stage 5 答案关键词覆盖率 | **29.8%** | 生成答案覆盖 `expected_answer` 关键词的比例 |
| 平均端到端延迟 | 16.9 s | 100 条总耗时 1700 s，平均值 |
| 平均 TTFT | 4.8 s | Stage 5 LLM 吐出第一个 token 的时间 |

## 5. 性能优化历程

| 阶段 | 改动 | Stage 4 | 总耗时 |
|---|---|---|---|
| 初版 | 暴力 + bge-m3 串行 embed | 30 s | 71 s |
| | 改用 `/api/embed` 批量端点 | 9 s | 28 s |
| | hybrid：FAISS 粗排 + bge-m3 精排 | 0.9 s | 12 s |
| | Stage 5 输入只取 Top-10 法条 | 0.9 s | ~9 s |

后端切换：

- `bge-small-zh-v1.5`（hf） → `bge-m3`（Ollama，1024 维）：质量更好
- `deepseek-r1:7b` → `qwen2.5:7b`：非思考模型，省一半时间
- 索引：保持 `IndexFlatL2`，22K 规模暴力扫描 1.9 ms，没必要换 HNSW

## 6. 局限与未做

实测发现的几条边界：

1. **法律名 FAISS 召回对外行问题敏感**。问"醉驾怎么处理"，Stage 2 召回《海上交通安全法》《公职人员政务处分法》《车船税法》—— 因为 "酒/车" 和这些法律的字面相似度高。需要更聪明的 query 改写或 entity 抽取。

2. **法条级召回在复杂场景下不足**。Stage 4 的 hybrid 严格锁在 Stage 2 命中的法律里（这是设计选择），但 Stage 2 一旦漏掉正确法律，后面就全跑偏。

3. **答案关键词覆盖率 29.8%** 比想象中低。原因是 `eval_set.json` 的 `expected_answer` 写得偏完整、细节多，而模型回答偏简洁（受 prompt 的"只引用相关法条"约束）。这其实是评估方法的问题，不是模型质量问题。

4. **暂未做**：多 Agent 拆解（继续支持 qa_agent）、FastAPI 服务化、Web UI、多 Agent 框架选型实践。

## 7. 怎么跑

```bash
# 1) 装模型
ollama pull bge-m3
ollama pull qwen2.5:7b

# 2) 一次性建索引（5-10 分钟）
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  python3 demo/build_indexes.py --embed-backend ollama --embed-model bge-m3 --reset

# 3) 单条问答
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  python3 demo/run_demo.py -q "草原保护有什么方针？" \
  --embed-backend ollama --embed-model bge-m3 \
  --llm-backend ollama --llm-model qwen2.5:7b

# 4) REPL 多轮
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  python3 demo/run_demo.py \
  --embed-backend ollama --embed-model bge-m3 \
  --llm-backend ollama --llm-model qwen2.5:7b
# 输入 'q' 退出

# 5) 评测 100 条（28 分钟）
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  python3 demo/run_eval.py --no-rewrite
```

## 8. 项目结构

```
local_rag_legal/
├── README.md                 # 快速开始 + 性能数据
├── PROJECT.md                # 本文档（流程 + 技术细节）
├── LICENSE
├── law_clearnerdata/         # 数据（303 部法律 / 22,482 条法条）
├── test_set_100.json         # 评测集（结构化字段版）
├── start_test/               # 学习路径：m1~m7 探索性脚本
│   ├── m1_explore_data.py
│   ├── m2_first_embedding.py
│   ├── m3_encode_and_search.py
│   ├── m4_chroma.py
│   ├── m5_benchmark.py
│   ├── m6_rag.py
│   ├── m7_multi_agent_chat.py
│   └── m7_memory_test.py
└── demo/                     # 生产管道
    ├── README.md
    ├── build_indexes.py      # 一次性建 FAISS 索引
    ├── pipeline.py           # 5 阶段核心
    ├── run_demo.py           # CLI 入口
    ├── test_pipeline_no_llm.py  # 不调 LLM 的检索测试
    ├── run_eval.py           # 自动评测
    ├── eval_set.json         # 100 条带标注测试集
    └── indexes/              # FAISS 持久化（git ignore）
```

## 9. 后续方向

短期改进：

- Stage 2 召回率：在 Question → 法律 检索间加 HyDE（让 LLM 先答一遍再搜）
- 把 `fetch_k` 从 50 提到 100，预期召回率 +5%
- FastAPI 包 `/api/chat`，给前端/Web 用
- 加一个轻量 Web UI

中长期：

- 多 Agent 编排：法律咨询 Agent（多轮澄清 + QA Agent + 总结 Agent）
- 答案缓存：常见 50-100 个问题预生成答案，命中后 < 50ms 返回
- LangGraph 编排 + 可视化 trace

---

*作者：tjrryy。指导老师：徐继伟老师。*
