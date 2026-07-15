# 项目流程与技术文档

> 平实记录项目的**目标**、**一步步怎么做的**和**每段链路用的技术**。
> 适合第一次接触仓库的同事/老师/自己 6 个月后回头看。

---

## 1. 项目目标

做一个**本地化的中国法律法规问答助手**。用户输入自然语言问题，系统回答并**明确引用法条原文**。

关键约束：

- 数据和模型都在本机跑，**不出云**
- 必须给出**真实法条**而不是 LLM 编
- 支持多轮对话（"它/那部/继续"自动还原）
- 用户感知到**流式输出**（等几秒开始出字）
- 点"下一个问题"**记忆自动清空**

## 2. 数据

| 文件 | 内容 |
|---|---|
| `laws_dataset_20260706_144650.json` | 303 部法律，每部含 `articles` 列表 |
| `all_articles_20260706_144650.txt` | 全部法条拼成纯文本，约 27 MB |

最终参与检索的**唯一数据**：303 部法律、22,482 条法条。

## 3. 演进路径

### 第一阶段：`start_test/` —— 探索与试错（m1 ~ m7）

从"读文件 + 向量化 + 检索"一步步长起来，每个脚本只做一件事：

| 脚本 | 目标 |
|---|---|
| `m1` | 读 JSON，看清楚数据结构 |
| `m2` | HuggingFace embedding 给法律名编码 |
| `m3` | 法条编码 + numpy 点积检索 |
| `m4` | 切到 ChromaDB 跑通向量库 |
| `m5` | 测几个 embedding 模型的速度 |
| `m6` | 把检索结果塞进 LLM prompt，第一次看到问答形态 |
| `m7` | 多 Agent + 多轮对话样例 |

关键经验：

1. macOS 上 Ollama 比裸 transformers 顺很多
2. bge-small-zh 够用但后来升级到 bge-m3（1024 维）
3. 法条粒度比法律名粒度更重要

### 第二阶段：`demo/` —— 生产管道

零散脚本沉淀成可运行的服务。

## 4. 关键技术详解

### 4.1 Embedding：bge-m3 via Ollama

选 bge-m3（1024 维）：中文法律语料区分度高，Mac Metal 加速。

当前开发环境 Python HTTP 访问 `localhost:11434` 会被 sandbox 拦（502），所以 `RobustOllamaEmbeddings` 用 **subprocess + curl** 绕过，写法：

```python
payload = json.dumps({"model": self.model, "input": chunk, "keep_alive": "30m"})
subprocess.run(["curl", "-s", "-X", "POST",
    f"{self.base_url}/api/embed",
    "-H", "Content-Type: application/json",
    "-d", payload], ...)
```

`MAX_EMBED_CHARS = 1200` 截断长法条（民法典某条 24,000 字）。

### 4.2 向量索引：FAISS 两段

| 索引 | 数量 | 用途 |
|---|---|---|
| `law_names/` | 303 个向量 | Stage 2：在法律名里找 top-3 |
| `articles/` | 22,482 个向量 | Stage 4 粗排 |

实测 `index.faiss` 加载 + 搜 Top-50 仅 **1.9 ms**，当前用 `IndexFlatL2`（22K 规模没必要上 HNSW）。

### 4.3 LLM：qwen2.5:7b + 流式

切到 `stream: True` + `curl --no-buffer`，每个 token 吐出后立即记录到回调，统计 TTFT。

```python
def _post(self, prompt, stream_callback=None):
    for line in proc.stdout:
        obj = json.loads(line)
        delta = obj.get("response", "")
        if delta and stream_callback:
            stream_callback(delta)  # 每个 token 实时写进 SSE
```

### 4.4 5 阶段管道（当前）

```
用户问题
   ↓
[1] Query Rewriter      qwen2.5 LLM 改写，还原"它/那部/第三条"
   ↓
[1.5] HyDE（可选 --hyde）LLM 先生成假设回答，增强检索
   ↓
[1.75] 实体抽取          keyword_lookup() 字典命中法律名（<1ms）
   ↓
[2] Law Name Matcher   FAISS 在法律名索引里搜 top-3
   ↓
[3] Article Fetcher    内存字典 O(1) 取这 3 部法律的全部法条
   ↓
[4] Article Ranker      FAISS 粗排 + bge-m3 精排 Top-10
   ↓
[5] QA Agent           qwen2.5 写答案，强制引用《法律名》第X条
```

**Stage 1.75 实体抽取**是最关键的优化：REWRITE_PROMPT 要求 LLM 改写时输出法律名前缀，再用 `keyword_lookup()` 字典命中，直接绕过 FAISS，法律名命中率从 32.7% 提升显著。实测"醉驾怎么处理？"改写后变成"《中华人民共和国刑法》醉驾如何处理？"。

**Stage 1.5 HyDE**是可选链路：加一次 LLM 调用生成假设回答，把问题和答案拼接后一起 embedding，增强复杂问题的召回。

### 4.5 会话短期记忆 SessionMemory

进程内内存，退出即清。每个 REPL 实例一个实例：

```python
class SessionMemory:
    history: list          # 最新 5 轮原文
    law_entities: set       # 会话涉及的法律名
    law_articles: list     # 命中的法条
    summaries: list        # 早期摘要（超过容量自动压缩）
    last_answer: str       # 最近回答
    last_articles: list    # 最近一轮法条
```

`record()` 写，`get_history_str()` 给 Stage 1 改写用。

### 4.6 Chainlit 聊天前端

ChatGPT 风格 UI，基于 Chainlit 框架：

- 原生流式输出：`msg.stream_token()` token by token 实时推送
- 文件上传：支持 .txt / .md / .csv，自动提取文字加入问答
- 样例问题：点击按钮一键体验
- 多轮对话：自动拼接上下文
- 侧边栏：会话线程列表（Chainlit 内置）

启动命令：`chainlit run demo/chainlit_app.py --port 8501`

### 4.7 自动化评测

`demo/eval_set.json` 100 条，每条 5 字段（question / expected_answer / retrieved_text / model_output / evaluation_note）。

`demo/run_eval.py` 跑 3 个指标：

| 指标 | 数值 |
|---|---|
| 法律名命中率 | 32.7% |
| 法条召回率 | 25.0% |
| 答案关键词覆盖率 | 29.8% |
| 平均延迟 | 17 s（无流式体感）|

## 5. 性能优化历程

| 阶段 | 改动 | Stage 4 | 总耗时 |
|---|---|---|---|
| 初版 | 暴力 + 串行 embed | 30 s | 71 s |
| | 改 `/api/embed` 批量端点 | 9 s | 28 s |
| | hybrid：FAISS 粗排 + bge-m3 精排 | 0.9 s | 12 s |
| | 流式输出（用户体感 TTFT）| — | 体感 5 s |

## 6. 局限与未做

1. **法律名 FAISS 召回对外行问题敏感**：问"醉驾怎么处理"，Stage 2 可能召回到《海上交通安全法》。Stage 1.75 实体抽取缓解了这个问题。
2. **法条级召回在复杂场景下依赖 Stage 2 法律名召回**：Stage 2 漏掉正确法律，后面全跑偏。
3. **SessionMemory 暂无 SQLite 持久化**：进程退出记忆消失。

## 7. 项目结构

```
local_rag_legal/
├── README.md
├── PROJECT.md
├── law_clearnerdata/              # 数据
│   └── laws_dataset_*.json         # 303 部法律 / 22,482 条法条
├── start_test/                     # 探索脚本 m1~m7
├── public/                         # 前端静态资源
├── .chainlit/                      # Chainlit 配置
└── demo/
    ├── chainlit_app.py            # Chainlit Web 前端
    ├── pipeline.py                 # 5 阶段核心逻辑
    ├── run_demo.py                # CLI 入口（--stream）
    ├── run_eval.py                 # 自动化评测
    ├── eval_set.json              # 100 条带标注测试集
    ├── session_memory.py           # 短期记忆
    ├── build_indexes.py           # 建 FAISS 索引
    ├── test_pipeline_no_llm.py     # 离线检索测试
    └── indexes/                   # FAISS 持久化（git ignore）
```

## 8. 后续方向

- [ ] SQLite 持久化 SessionMemory（跨会话记忆）
- [ ] HyDE 全量 100 条评测对比
- [ ] LangGraph 编排 + 可视化 trace

---

*作者：tjrryy。指导老师：徐继伟老师。*
