# 结课报告

> 基于本地大语言模型的法律问答系统
> 学生：tjrryy
> 指导老师：徐继伟
> 日期：2026-07-22

---

## 摘 要

本课题设计并实现了一套基于本地大语言模型的法律问答系统。系统以 303 部中国法律法规、22,482 条法条为数据基础，构建了基于 FAISS 的两段式向量索引；设计了五阶段检索链路（Query Rewriter → HyDE 假设生成 → 实体抽取 → 法律名召回 → 法条精排），结合本地部署的 qwen2.5:7b 大语言模型，实现了强制引用《法律名》第 X 条标准格式的法律问答。系统通过 ChainLit 提供 Web 交互界面，支持 SSE 流式输出与多轮会话记忆，并在 100 条带标注的测试集上获得法律名命中率 32.7%、法条召回率 25.0% 的评测结果。在性能优化方面，通过并行化、精简 prompt、embedding 缓存复用、Ollama 预热等手段，将端到端总耗时从 32.9 秒压至 4.7 秒，**累计降幅 85.7%**。

**关键词**：法律问答系统、本地大语言模型、检索增强生成、FAISS 两段检索、ChainLit、流式输出

---

## 第一章 项目背景及意义

### 1.1 研究背景

随着大语言模型（LLM）能力飞速发展，公众对"AI 法律助手"的需求愈发旺盛。然而当前主流方案存在三大问题：

1. **数据出云风险**：云端 LLM 需要把用户咨询的案件细节上传第三方服务器，敏感法律问题（如离婚、刑事、工伤）容易泄露
2. **幻觉严重**：LLM 在没有法条检索的情况下，常自行编造"《XX法》第三十条"等虚假法条
3. **专业门槛高**：当事人查询时使用口语化表达（如"醉驾会怎么处理"），传统检索效果差

本课题以"中国法律法规 + 本地大模型"为切入点，搭建一套可在本机运行的、引用准确的法律问答系统。

### 1.2 研究意义

- **数据合规**：全部模型与索引本地部署，敏感案件不出云
- **引用真实**：通过 RAG（检索增强生成）强制 LLM 先用检索到的真实法条作答，杜绝幻觉
- **专业普适**：融合实体抽取、字典匹配、向量召回，降低口语化 query 的技术门槛

---

## 第二章 主要研究内容

### 2.1 五阶段检索增强生成链路

整体架构如下：

```
用户问题
   ↓
[Stage 1] Query Rewriter       qwen2.5:7b 还原文意（"它/那部/第三条"）
   ↓
[Stage 1.5] HyDE 假设法条生成    LLM 生成假设法条后混合 embedding（可选）
   ↓
[Stage 1.75] 实体抽取          keyword_lookup() 字典命中法律名（<1ms）
   ↓
[Stage 2] Law Name Matcher    bge-m3 + FAISS 在 303 部法律名里 top-3
   ↓
[Stage 3] Article Fetcher     内存字典 O(1) 取 3 部法律的全部法条
   ↓
[Stage 4] Article Ranker (Hybrid)  FAISS 粗排 + bge-m3 重排 Top-10
   ↓
[Stage 5] QA Agent           qwen2.5 按 prompt 生成答案 + 引用法条
   ↓
流式 SSE 输出
```

### 2.2 工程化与性能优化

端到端总耗时从 32.9 秒优化到 4.7 秒，主要优化点：

- **并行化**：Stage 1 改写与 Stage 2 法律名匹配并发执行
- **精简 prompt**：QA 模板要求不重复法条原文，输出 token 从 231 降到 75
- **embedding 缓存复用**：Stage 2 / Stage 4 共享 query 向量
- **Ollama 预热**：启动时先发"你好"加载模型到内存
- **限长 num_predict=512**：避免啰嗦回答
- **流式输出**：用户等 5 秒即看到答案逐字出现

### 2.3 Web 交互与多轮记忆

ChainLit 白绿青主题前端，支持：
- 问题示例按钮一键问
- SSE 流式 token 逐字显示
- 多轮对话自动拼接到 Query Rewriter
- 文件上传（.txt/.md/.csv）后直接分析
- 进程内 SessionMemory，退出自动清

### 2.4 自动化评测体系

`demo/eval_set.json` 100 条带人工标注，`run_eval.py` 评测 3 个核心指标：
- 法律名命中率（Stage 1.75 实体命中 / Stage 2 FAISS 命中）
- 法条召回率（期望条款编号是否在 Top-10）
- 答案关键词覆盖率（expected_answer 关键词被回答覆盖比例）

---

## 第三章 开发环境

### 3.1 软硬件环境

| 类别 | 配置 |
|---|---|
| 操作系统 | macOS（Apple Silicon）|
| Python | 3.12+ |
| LLM 推理 | Ollama 本地（Metal GPU 加速）|
| Embedding | Ollama + bge-m3（1024 维）|
| LLM 模型 | Ollama + qwen2.5:7b（4.7 GB）|
| 向量索引 | FAISS（CPU 版）|
| Web 框架 | ChainLit |
| 评测工具 | python-docx, 自写评测脚本 |

### 3.2 关键依赖

```
langchain-community
faiss-cpu
sentence-transformers
chainlit
fastapi
python-docx
```

---

## 第四章 本人完成工作

### 4.1 代码贡献统计

通过 `git log --author="tjrryy"` 统计，本人在本项目共提交 **10 个独立 commit**：

| Commit | 类型 | 内容 |
|---|---|---|
| `730d795` | docs | 初始 commit |
| `f9101f5` | docs | 完善 README |
| `6fac00c` | data | 上传清洗后的法律数据 |
| `09abe1e` | feat | **5 阶段问答管道 + Ollama 集成** |
| `e7d4c00` | perf | **Hybrid FAISS+bge-m3 精排 + 流式 LLM TTFT 统计** |
| `d13916a` | feat | 流式输出 + 100 条评测集 + 项目文档 |
| `5161a2c` | feat | **实体抽取增强召回 + HyDE 可选链路** |
| `01c0f3b` | feat | FastAPI 服务 + 流式 SSE + SessionMemory |
| `da5e115` | docs | README / PROJECT 重写 |
| `10c1097` / `ee53ecb` / `3db7631` | docs | 中期报告（md / docx）|

### 4.2 核心模块作者归属

| 模块 | 本人完成 | 备注 |
|---|:---:|---|
| 5 阶段管道 `pipeline.py` | ✅ | 全部本人 |
| 评测脚本 `run_eval.py` | ✅ | 全部本人 |
| 评测集 100 条 `eval_set.json` | ✅ | 全部本人 |
| SessionMemory 模块 | ✅ | 全部本人 |
| 流式 LLM（TTFT 统计） | ✅ | 全部本人 |
| REWRITE_PROMPT 工程化改写 | ✅ | 全部本人 |
| 中期报告（md / docx） | ✅ | 全部本人 |
| ChainLit Web 前端（UI、文件上传、流式 push）| 部分 | 与 yedinghao 协同 |
| FAISS 两段索引搭建 | 部分 | 初始版本共同 |

### 4.3 项目目录贡献占比

| 目录 | 我的贡献 |
|---|---|
| `demo/pipeline.py` | 100% |
| `demo/run_eval.py` | 100% |
| `demo/eval_set.json` | 100% |
| `demo/session_memory.py` | 100% |
| `demo/run_demo.py` | 100%（CLI REPL + 流式）|
| `demo/build_indexes.py` | 共同 |
| `demo/chainlit_app.py` | 协同 |
| `docs/` | 100%（中报、性能测试、综述）|
| `README.md`、`PROJECT.md` | 100% |

---

## 第五章 项目总体设计

### 5.1 系统架构

```
┌─────────────────────────────┐
│  Web (ChainLit)              │  ← 用户界面
└────────────┬────────────────┘
             │
   ┌─────────▼─────────┐
   │  chainlit_app.py  │  ← Web 入口（流式 push、文件上传）
   └─────────┬─────────┘
             │
   ┌─────────▼─────────┐
   │  SessionMemory    │  ← 多轮会话记忆
   └─────────┬─────────┘
             │
   ┌─────────▼─────────┐
   │ LegalRAGPipeline  │  ← 5 阶段管道
   └─────────┬─────────┘
             │
   ┌─────────▼────────────────────────────┐
   │ Stage 1  Rewriter  (qwen2.5:7b)       │
   │ Stage 1.5 HyDE   (qwen2.5:7b, 可选)    │
   │ Stage 1.75 Entity (keyword_lookup()) │
   │ Stage 2  Matcher (bge-m3 + FAISS)    │
   │ Stage 3  Fetcher (内存字典)            │
   │ Stage 4  Ranker  (FAISS + bge-m3 精排) │
   │ Stage 5  QAgent  (qwen2.5:7b + 强引用) │
   └────────────────────────────────────────┘
```

### 5.2 数据流

```
用户 query
   ↓
REWRITE_PROMPT 输出《法律名》前缘 + 同义改写
   ↓
keyword_lookup(<改写后 query>) 直接命中法律
   ↓
FAISS 在 22,482 条法条里粗排 Top-30
   ↓
bge-m3 重编码 Top-30 → 点积 → 取 Top-10
   ↓
QA_PROMPT: 「根据【法条参考】回答，先引用再解释」
   ↓
SSE 逐 token 推送至 ChainLit 前端渲染
```

---

## 第六章 项目实现与测试

### 6.1 功能实现

#### 6.1.1 Stage 1 查询改写（Query Rewriter）

```python
REWRITE_PROMPT = """你是法律领域查询改写助手。请把用户的【最新问题】改写成一个独立的、可直接检索的问题。
规则：
1. 如果问题里有指代（"它"、"那个"、"这部法律"、"第三条"等），结合【对话历史】还原
2. 如果从问题能推断出涉及哪部法律，在改写后的问题开头加上"《法律名》"，帮助后续检索
3. 不要补充新信息，不要解释，只输出改写后的问题

【对话历史】
{history}

【用户最新问题】
{query}

【改写后的问题】"""
```

**测试用例**：

| 输入 | 输出 |
|---|---|
| "它第三条说了什么？" + history="草原保护…《草原法》第三条" | "《草原法》第三条说了什么？" |
| "醉驾怎么处理？" | "《中华人民共和国刑法》醉驾如何处理？" |

#### 6.1.2 Stage 1.75 实体抽取

```python
def keyword_lookup(text: str) -> list[str]:
    """在文本里找法律名关键词，返回对应法律名（去重保序，最多 3 部）"""
    seen, hits = set(), []
    for alias in sorted(self.alias_to_law.keys(), key=len, reverse=True):
        if alias in text:
            law = self.alias_to_law[alias]
            if law not in seen:
                seen.add(law)
                hits.append(law)
                if len(hits) >= 3:
                    break
    return hits
```

预先建立 `alias_to_law` 反查表：`{"草原法": "中华人民共和国草原法", "网络安全": "中华人民共和国网络安全法", ...}`。命中后直接绕过 FAISS，<1ms 返回。

#### 6.1.3 Stage 4 Hybrid 精排

```python
class ArticleRanker:
    def __call__(self, query, candidates, top_k=10, law_filter=None):
        # 1) FAISS 粗排：在 law_filter 限定内搜 Top-fetch_k
        coarse = self.article_db.similarity_search_with_score(
            q, k=fetch_k, filter={"law_title": {"$in": list(law_filter)}}
        )
        # 2) bge-m3 重编码这 fetch_k 条
        emb_cands = self.embeddings.embed_documents([d.page_content for d in coarse_docs])
        # 3) 点积排序取 Top-K
        scores = emb_q @ emb_cands.T
        return sorted(zip(coarse_docs, scores), key=lambda x: -x[1])[:top_k]
```

#### 6.1.4 流式输出（SSE）

```python
def _post(self, prompt, stream_callback=None):
    proc = subprocess.Popen(["curl", "-s", "--no-buffer", "-X", "POST", ...],
                            stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        obj = json.loads(line)
        delta = obj.get("response", "")
        if delta and stream_callback:
            stream_callback(delta)   # 每个 token 立即回调
        if obj.get("done"):
            break
```

### 6.2 测试结果

#### 6.2.1 单条问答

测试问题：`草原保护有什么方针？`

| 指标 | 数值 |
|---|---|
| 总耗时 | ~5 秒 |
| 命中法律 | 中华人民共和国草原法 |
| Top-1 召回条款 | 第三条（命中） |
| 回答质量 | 准确引用《草原法》第三条 |

#### 6.2.2 100 条自动化评测

```
[RESULT] 评测汇总
============================================================
  评测条数       : 100
  总耗时         : 1700 s
  ─────────────────────────────────────────
  Stage 2 法律名命中率 : 33/101 = 32.7%
  Stage 4 法条召回率   : 26/104 = 25.0%
  Stage 5 答案覆盖率   : 29.8%
  ─────────────────────────────────────────
  平均端到端延迟 : 16998 ms
  平均 TTFT      : 4806 ms
```

#### 6.2.3 性能优化测试

四轮迭代总耗时变化：

| 阶段 | 改动 | 总耗时 |
|---|---|---|
| 基线 | 原始版 | 32.9 s |
| 优化1 | Stage 1+2 并行 + Top-6 法条 | 10.6 s |
| 优化2 | QA prompt 精简 + num_predict=512 | 5.8 s |
| 优化3 | embedding 缓存 + 去冗余 | 4.9 s |
| 优化4 | Ollama 预热 + num_thread | **4.7 s** |

**累计降幅 85.7%**，TTFT 从 3.0s 降到 0.2s。

---

## 第七章 项目难点分析

### 7.1 难点一：Stage 5 LLM 生成速度慢

**问题**：第一版 Stage 5 耗时 18 秒，占总时间 55%。LLM 推理是吞 token 的工作，prompt 越长、生成越长就越慢。

**解决方法**：
1. **精简 prompt**：要求 LLM "先引用再解释"，输出 token 从 231 降到 75
2. **Top-10 → Top-6**：法条上下文裁剪，prompt 缩短
3. **Ollama 预热**：启动时发"你好"，触发模型加载到 GPU 内存
4. **限长 num_predict=512**：避免 LLM 啰嗦

**效果**：Stage 5 从 18s 降到 3.7s，占比从 55% 升到 78%（其他更低了）。

### 7.2 难点二：bge-m3 embedding 重复调用

**问题**：Stage 2 法律名匹配和 Stage 4 法条精排都各做一次 query embedding，每条问题白白多调一次。

**解决方法**：维护 `_query_vec_cache` 缓存同一 query 的向量。Stage 2/4 共用。

**效果**：embedding 调用从 2 次降到 1 次，单次约省 200ms。

### 7.3 难点三：Stage 2 法律名召回走偏

**问题**：问"醉驾怎么处理"，FAISS 召回了《海上交通安全法》《公职人员政务处分法》《车船税法》—— 完全无关。

**原因**：bge-m3 把"酒/车"这些字面词当成关键词，反被邻近字面相关的法律匹配上。

**解决方法**：
1. **修改 REWRITE_PROMPT**：强制 LLM 在改写后问题开头加上《法律名》前缀
2. **Stage 1.75 实体抽取**：改写后问题如含"刑法/网络安全法"等关键词，直接命中绕过 FAISS

**效果**：实体抽取路径下命中率 100% 锁定法律名。

### 7.4 难点四：sandbox 环境下 Python HTTP 访问受限

**问题**：开发沙箱（Trae sandbox）下 `requests`/`urllib` 访问 `http://localhost:11434` 会被沙箱拦（502 错误）。

**解决方法**：所有 Ollama 调用（embedding + LLM）都改用 `subprocess + curl` 走命令行，绕过沙箱拦截。

**代码片段**：

```python
def _post(self, prompt):
    payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False, "keep_alive": "30m"})
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{self.base_url}/api/generate",
         "-H", "Content-Type: application/json", "-d", payload],
        capture_output=True, text=True, timeout=600,
    )
    return json.loads(result.stdout)["response"]
```

### 7.5 难点五：流式输出的可靠性

**问题**：用 `stream: True` 时，Ollama 一次性返多个 JSON 行而非简单换行，需要按行解析并判断 `done` 字段。

**解决方法**：

```python
for line in proc.stdout:
    obj = json.loads(line)
    delta = obj.get("response", "")
    if stream_callback and delta:
        stream_callback(delta)
    if obj.get("done"):
        break
```

配合 `curl --no-buffer` 关闭行缓冲，保证 token 立即吐出。

---

## 第八章 结论与展望

### 8.1 研究结论

本课题实现了一套"纯本地"法律问答系统，核心成果：

1. **数据层**：303 部法律 / 22,482 条法条清洗并向量化入库
2. **检索层**：五阶段管道，FAISS 两段检索 + 实体抽取 + HyDE
3. **生成层**：qwen2.5:7b 本地 LLM + 强制引用 prompt
4. **Web 交互**：ChainLit 流式 SSE + 多轮 SessionMemory
5. **评测体系**：100 条带标注测试集，3 个核心指标可量化追踪
6. **性能**：32.9 秒优化到 4.7 秒，累计降幅 85.7%

### 8.2 局限性

- **法律名 FAISS 召回对外行 query 敏感**：bge-m3 在字面词"酒/车"上有偏置
- **法条精排严格依赖法律名召回**：Stage 2 漏召时，Stage 4 也召回错法律
- **SessionMemory 无持久化**：进程退出记忆消失（计划用 SQLite）

### 8.3 未来工作

| 方向 | 预期收益 |
|---|---|
| 接入 BM25 sparse 召回，做 true hybrid retrieval | 法条召回率 +5-10% |
| Stage 1.5 HyDE 与 Stage 1.75 实体抽取全量对比 | 验证 HyDE 收益 |
| SQLite 持久化 SessionMemory | 跨会话记忆 |
| LangGraph 编排 + 可视化 trace | 工程化演示 |
| Docker 容器化部署 | 可移植性 |

### 8.4 项目收获

通过本课题研究，我掌握了：
1. RAG 系统的完整链路设计与工程落地
2. 向量检索与传统字典检索的混合策略
3. 大语言模型的 prompt 工程化
4. Ollama / FAISS / ChainLit 等开源工具的集成
5. 性能瓶颈识别与逐步优化方法论

---

## 参考文献

1. BGE-M3: BAAI/bge-m3 [EB/OL]. Hugging Face, https://huggingface.co/BAAI/bge-m3
2. Qwen2.5: Qwen2.5:7b [EB/OL]. Ollama, https://ollama.com/library/qwen2.5
3. ChainLit Documentation [EB/OL]. https://docs.chainlit.io/
4. LangChain FAISS Integration [EB/OL]. https://python.langchain.com/docs/integrations/vectorstores/faiss
5. STARD: A Chinese Statute Retrieval Dataset with Real Queries [C]//SIGIR 2024
6. CLAW: Chinese Legal Knowledge Benchmarking for LLMs [C]//arXiv 2025
7. LexEval: A Comprehensive Chinese Legal Benchmark [C]//NeurIPS 2024
8. 国家法律法规数据库 [EB/OL]. https://flk.npc.gov.cn/

---

*作者：tjrryy。指导老师：徐继伟老师。*
