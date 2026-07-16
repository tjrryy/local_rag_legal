# 基于本地大语言模型的法律问答系统——中期报告

> 指导老师：徐继伟
> 小组/个人：tjrryy
> 日期：2025-07-16

---

## 一、项目目前的完成进度

### 1.1 数据层

已完成数据采集、清洗、规范化。

- **数据规模**：303 部中国法律、22,482 条法条，来源为 `law_clearnerdata/` 下的 JSON 原始文件
- **数据处理**：按法律粒度拆分法条，构建 `ArticleFetcher` 内存字典（法律名 → 法条列表），O(1) 查询
- **向量索引**：两段 FAISS 索引已持久化到 `demo/indexes/`

| 索引 | 向量维度 | 向量数 | 文件大小 |
|---|---|---|---|
| `law_names/` | 1024 | 303 | ~1.2 MB |
| `articles/` | 1024 | 22,482 | ~92 MB |

### 1.2 检索层

实现了完整的**两段检索链路**：

```
Stage 2：FAISS 在 303 部法律名中搜索 top-3
Stage 3：内存字典取 3 部法律的全部法条（约 200 条）
Stage 4：FAISS 粗排 + bge-m3 精排 Top-10
```

**关键技术选型**：

- Embedding 模型：`bge-m3`（Ollama 本地，1024 维，Mac Metal 加速）
- LLM 模型：`qwen2.5:7b`（Ollama 本地）
- 索引类型：`IndexFlatL2`（22K 规模暴力扫描仅 1.9 ms，无需 HNSW）
- 批量调用：`/api/embed` 端点 + curl subprocess，绕过 sandbox HTTP 限制

### 1.3 生成层

- **Query Rewriter**：qwen2.5:7b 将指代词还原为具体法律名（"它第三条" → "《草原法》第三条"）
- **HyDE（可选）**：Stage 1.5 让 LLM 先生成假设法条回答，增强复杂 query 的检索
- **实体抽取（Stage 1.75）**：`keyword_lookup()` 字典命中法律名，<1ms，直接绕过 FAISS，法律名命中率提升显著
- **QA Agent**：强制引用法条原文（《法律名》第X条）再解释，附流式输出（TTFT ~5s）

### 1.4 对话与 UI

- **SessionMemory**：进程内会话记忆，累积最近 5 轮原文，法律名和法条集合自动去重，退出即清
- **Web UI**：ChainLit 前端，白绿青主题，支持流式输出、多轮追问、文件上传
- **自动化评测**：`demo/eval_set.json`（100 条带标注测试集）+ `run_eval.py` 脚本

### 1.5 完成度总览

| 模块 | 状态 |
|---|---|
| 数据采集与处理 | ✅ 完成 |
| FAISS 两段索引 | ✅ 完成 |
| Embedding 模型调用（bge-m3）| ✅ 完成 |
| LLM 调用（qwen2.5:7b）| ✅ 完成 |
| 两段检索链路 | ✅ 完成 |
| 流式输出 | ✅ 完成 |
| 多轮对话记忆 | ✅ 完成 |
| ChainLit Web UI | ✅ 完成 |
| 自动化评测 | ✅ 完成 |
| FastAPI 服务 | ❌ 已被 ChainLit 替代 |
| LangGraph Agent 编排 | ⏳ 待做 |
| SessionMemory SQLite 持久化 | ⏳ 待做 |
| HyDE 全量 100 条对比评测 | ⏳ 待做 |

---

## 二、Demo 各部分讲解与链路演示

### 2.1 目录结构

```
demo/
├── chainlit_app.py          # ChainLit Web 入口（白绿青主题）
├── pipeline.py             # 5 阶段核心逻辑（HyDE / 实体抽取 / 流式）
├── session_memory.py        # 会话短期记忆
├── run_demo.py             # CLI REPL 入口（--stream）
├── run_eval.py             # 自动化 100 条评测
├── eval_set.json          # 100 条带标注测试集
├── build_indexes.py        # 建 FAISS 索引（一次性）
└── indexes/               # FAISS 持久化（git ignore）
    ├── law_names/          # 303 个法律名向量
    └── articles/           # 22482 条法条向量
```

### 2.2 5 阶段链路详解

```
用户输入
   ↓
[Stage 1] Query Rewriter
  qwen2.5:7b 改写，REWRITE_PROMPT 要求输出《法律名》前缀
  例："醉驾怎么处理？" → "《中华人民共和国刑法》醉驾如何处理？"
  耗时：~600 ms
   ↓
[Stage 1.5] HyDE（可选 --hyde）
  LLM 先生成一段假设法条回答，拼在 query 后一起 embedding
  耗时：+1-2s（额外一次 LLM 调用）
   ↓
[Stage 1.75] 实体抽取
  keyword_lookup() 字典匹配法律名（<1ms）
  例：《中华人民共和国刑法》→ 直接命中，跳过 Stage 2
   ↓
[Stage 2] Law Name Matcher
  bge-m3 编码 query → FAISS 303 法律名向量搜索 top-3
  命中法律：['草原法', '野生动物保护法', '青藏高原生态保护法']
  耗时：~150 ms
   ↓
[Stage 3] Article Fetcher
  内存字典 O(1) 查 3 部法律 → ~200 条法条候选
  耗时：<1 ms
   ↓
[Stage 4] Article Ranker（Hybrid）
  ① FAISS 粗排：法律名限定 filter，在 22K 里搜 top-50（~2 ms）
  ② bge-m3 精排：重编码 50 条 + 点积排序取 top-10（~800 ms）
  耗时：~800 ms
   ↓
[Stage 5] QA Agent
  qwen2.5:7b 生成答案，强制引用《法律名》第X条
  流式输出：SSE chunk 事件逐 token 推送
  耗时：~3-6 s（TTFT ~5s）
   ↓
答案（带法条引用 + 流式逐字出现）
```

### 2.3 流式输出机制

```
后端 pipeline.qa.stream()
   → subprocess + curl stream:True
   → 每个 token 回调 chunk_callback()
   → SSE 事件：data: {"type": "chunk", "delta": "根据"}
   → 前端 fetch + ReadableStream
   → document.getElementById("answer").textContent += delta
   → 用户看到逐字蹦出
```

### 2.4 延迟测算

单次问答完整耗时（qwen2.5:7b 已 warm，bge-m3 已加载）：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| Stage 1_rewrite | ~600 ms | 10% |
| Stage 1.75_entity | <1 ms | ~0% |
| Stage 2_match_laws | ~150 ms | 2% |
| Stage 3_fetch | <1 ms | ~0% |
| Stage 4_rank | ~800 ms | 13% |
| **Stage 5_answer** | **~3-6 s** | **75%** |
| **总计** | **~4-7 s** | 100% |

**结论**：LLM 生成占绝对大头（75%），非 LLM 部分已优化到 < 50ms。进一步加速只能换更小模型（qwen2.5:3b、qwen2.5:1.5b）。

### 2.5 自动化评测结果

100 条测试集，`run_eval.py` 统计：

| 指标 | 数值 |
|---|---|
| Stage 2 法律名命中率 | 32.7% |
| Stage 4 法条召回率 | 25.0% |
| Stage 5 答案关键词覆盖率 | 29.8% |
| 平均端到端延迟 | 17 s（无流式体感）|

---

## 三、后续完成部分

### 3.1 检索层优化

- [ ] **Stage 2 法律名命中率提升**（当前 32.7%）
  - 原因：外行 query（"醉驾怎么处理"）与法律名字面相似度不高
  - 方案：HyDE 全量评测 + 实体词典扩充（法律别名、简称）
  - 预期：从 32.7% 提升至 45-55%
- [ ] **Stage 4 法条精排去掉 bge-m3 重编码**
  - 直接用 FAISS Top-10 当答案（省去 ~800ms），可接受轻微召回损失
- [ ] **并行 Stage 1 + Stage 2**：`ThreadPoolExecutor` 并发执行，改写和匹配同步跑，预期省 400ms

### 3.2 生成层优化

- [ ] **换更小 LLM**：`qwen2.5:3b`（~1.9 GB，预计 Stage 5 从 3s → 1s）
- [ ] **Stage 5 输入截断**：法条全文塞 prompt 太长，只保留 Top-3 法条，预计省 20-30%
- [ ] **答案缓存**：常见问题 hash → 预生成答案，命中后 < 50ms

### 3.3 多轮与记忆

- [ ] **SessionMemory 持久化**：SQLite 存储，跨会话保留法律/法条上下文
- [ ] **自动判断多轮 vs 单轮**：引用词检测（"它/继续/还有"）+ memory 有历史 → 自动走多轮
- [ ] **跨法律追问**：用户从《草原法》追问到《野生动物保护法》，SessionMemory 维护法律切换逻辑

### 3.4 评测与质量

- [ ] **HyDE 全量 100 条对比**：开/关 HyDE 两组对照，找到对哪类问题 HyDE 有效
- [ ] **答案质量评分**：自动评测（BLEU / ROUGE / 法律条文编号匹配率）替代关键词覆盖率
- [ ] **扩充测试集**：当前 100 条，覆盖不足，扩充至 300 条

### 3.5 工程化

- [ ] **LangGraph 编排**：将 5 阶段封装为可观测 DAG，支持 trace
- [ ] **WebSocket 支持**：服务端推送替代 SSE，减少前端轮询
- [ ] **Docker 部署**：`Dockerfile` + `docker-compose` 打包 Ollama + 向量库 + 服务，一行启动

---

## 四、技术栈汇总

| 类别 | 当前选型 |
|---|---|
| Embedding | Ollama + bge-m3（1024 维）|
| 向量索引 | FAISS 两段（法律名 + 法条）|
| LLM | Ollama + qwen2.5:7b |
| 框架 | LangChain（部分）+ 自写 Agent |
| UI | ChainLit（白绿青主题）|
| 评测 | 100 条带标注测试集 + run_eval.py |
| 短期记忆 | SessionMemory（进程内）|
| 流式 | SSE，token by token |

---

*作者：tjrryy。指导老师：徐继伟老师。*
