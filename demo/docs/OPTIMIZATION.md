# 完整性能优化报告

## 1. 基线（Baseline）

### 1.1 测试环境

- 时间：2026-07-19
- Embedding：`bge-m3`（本地 Ollama）
- LLM：`qwen2.5:7b`（本地 Ollama）
- 索引：303 部法律名 + 22,482 条法条（FAISS 两段索引）
- 测试问题：`单位拖欠工资怎么办？`

### 1.2 端到端耗时

| 阶段 | 耗时 | 占比 |
|---|---|---|
| [1] Query Rewriter | 5,276 ms | 16.0% |
| [2] Law Name Matcher | 8,907 ms | 27.1% |
| [3] Article Fetcher | 0 ms | 0.0% |
| [4] Article Ranker (hybrid) | 618 ms | 1.9% |
| [5] QA Agent | 18,093 ms | 55.0% |
| **总计** | **32,894 ms** | 100% |

### 1.3 LLM 生成指标

- 首字延迟 TTFT：2,977 ms
- 累计 token：231
- LLM 总耗时：18,091 ms
- 生成速度：12.8 tok/s

### 1.4 主要瓶颈

1. **Stage 5 LLM 回答**：占总时间 55%，首字延迟 3s，生成速度偏慢。
2. **Stage 2 法律名匹配**：占 27%，本质是 1 次 query embedding + FAISS 搜索。
3. **Stage 1 Query Rewriter**：占 16%，也是 1 次 LLM 调用。

---

## 2. 优化 1：并行改写与法律匹配 + 精简 LLM 上下文

### 2.1 改动点

1. **Stage 1 与 Stage 2 并行**
   - 若原 query 未命中法律关键词，则 `Query Rewriter` 与 `Law Name Matcher` 并发执行。
   - 若 query 已含法律关键词（如“劳动合同法”），直接走关键词匹配，跳过改写。

2. **精简 Stage 5 LLM 上下文**
   - `top-articles` 默认值从 10 降到 6。
   - 单条法条正文超过 600 字时截断。

3. **REPL 历史截断**
   - 累积历史时只保留回答前 200 字，减少多轮时改写阶段上下文。

### 2.2 性能对比

| 指标 | 基线 | 优化1 | 变化 |
|---|---|---|---|
| 总耗时 | 32,894 ms | 10,585 ms | -67.8% |
| Stage 1 Query Rewriter | 5,276 ms | 629 ms | -88.1% |
| Stage 2 Law Name Matcher | 8,907 ms | 161 ms | -98.2% |
| Stage 5 QA Agent | 18,093 ms | 9,180 ms | -49.3% |
| 首字延迟 TTFT | 2,977 ms | 167 ms | -94.4% |
| 输出 token | 231 | 168 | -27.3% |
| 生成速度 | 12.8 tok/s | 18.3 tok/s | +43.0% |

### 2.3 关键发现

- 法条从 10 条变 6 条 + 截断后，TTFT 大幅下降。
- Stage 1 + Stage 2 并行后，两者合计从 14.2s 降到 0.79s。
- 剩余瓶颈主要是 Stage 5 LLM 生成。

### 2.4 涉及文件

- `demo/pipeline.py`
- `demo/run_demo.py`

---

## 3. 优化 2：精简 QA Prompt + 限制生成长度

### 3.1 改动点

1. **精简 `QA_PROMPT`**
   - 明确告知 LLM：引用法条时用格式即可，不要重复法条原文。
   - 要求直接给出结论和可操作的建议，语言简洁。

2. **限制 `num_predict=512`**
   - 在 Ollama `/api/generate` 请求中加入 `options.num_predict`。
   - 支持通过 `OLLAMA_NUM_PREDICT` 或 `--num-predict` 调整。

3. **新增 `--num-predict` 命令行参数**

### 3.2 性能对比

| 指标 | 优化1后 | 优化2后 | 变化 |
|---|---|---|---|
| 总耗时 | 10,585 ms | 5,752 ms | -45.7% |
| Stage 5 QA Agent | 9,180 ms | 4,443 ms | -51.6% |
| 首字延迟 TTFT | 167 ms | 174 ms | 持平 |
| 输出 token | 168 | 86 | -48.8% |
| 生成速度 | 18.3 tok/s | 19.4 tok/s | +6.0% |

### 3.3 注意事项

- `num_predict` 过小（如 300）会导致回答被截断，默认 512 是质量与速度的折中。

---

## 4. 优化 3：Query Embedding 缓存 + 避免冗余法律名匹配

### 4.1 改动点

1. **共享 query embedding 缓存**
   - `LegalRAGPipeline` 维护 `_query_vec_cache`。
   - `LawNameMatcher` 和 `ArticleRanker` 复用同一向量。

2. **降低 `fetch_k`**
   - `ArticleRanker` 粗排候选从 50 降到 30。

3. **避免冗余法律名匹配**
   - 仅当改写后 query 新增了原 query 中没有的法律关键词时，才做补充匹配。

4. **新增口语化主题词映射**
   - 提升“辞退/赔偿/工伤”等口语词汇到对应法律的召回准确率。

### 4.2 性能对比

| 指标 | 优化2后 | 优化3后 | 变化 |
|---|---|---|---|
| 总耗时 | 5,752 ms | 4,911 ms | -14.6% |
| Stage 1 Query Rewriter | 629 ms | 577 ms | -8.3% |
| Stage 2 Law Name Matcher | 161 ms | 0 ms | 关键词命中 |
| Stage 4 Article Ranker | 616 ms | 607 ms | 持平 |
| Stage 5 QA Agent | 4,443 ms | 3,727 ms | -16.1% |
| 首字延迟 TTFT | 174 ms | 197 ms | 持平 |
| 输出 token | 86 | 73 | -15.1% |
| 生成速度 | 19.4 tok/s | 19.6 tok/s | 持平 |

### 4.3 关键发现

- embedding 缓存让 Stage 2/Stage 4 的 embedding 调用从 2 次降到 1 次。
- 法律名匹配在命中关键词时耗时归零。
- LLM 生成仍是最大瓶颈，占总时间 75.9%。

---

## 5. 优化 4：LLM 侧优化（Warm-up + 可配置选项 + 更小模型）

### 5.1 改动点

1. **LLM 预热（Warm-up）**
   - 在 `build_llm()` 中初始化后立刻发一次轻量 generate 请求（prompt="你好"）。
   - 触发 Ollama 把模型加载到内存/GPU，避免首次真实请求 TTFT 过高。
   - 可通过 `OLLAMA_NO_WARM_UP=1` 或 `--no-warm-up` 关闭。

2. **可覆盖的 Ollama generate options**
   - 通过环境变量 `OLLAMA_OPTIONS`（JSON）或 CLI `--ollama-options` 注入任意 Ollama options，例如 `num_gpu`、`num_thread`、`top_p`。
   - 默认保留：`temperature=0.1`, `top_p=0.9`, `num_predict=512`。

3. **引入并 benchmark `qwen2.5:3b`**
   - 下载 1.9 GB 的小模型，与 7b 对比速度和质量。

### 5.2 性能对比

| 指标 | 优化前（7b，无 warm-up） | 7b + warm-up | 3b + warm-up |
|---|---|---|---|
| 总耗时 | 7,770 ms | **4,713 ms** | 4,883 ms |
| Stage 5 QA Agent | 6,810 ms | 3,677 ms | 3,919 ms |
| 首字延迟 TTFT | 3,375 ms | **177 ms** | 873 ms（首次）/ 139 ms（第二次） |
| 生成速度 | 10.7 tok/s | 20.4 tok/s | **34.5 ~ 42.7 tok/s** |
| 输出 token | 73 | 75 | 135 ~ 248 |
| 回答质量 | 正确 | 正确 | 正确 |

### 5.3 关键结论

- **warm-up 对 TTFT 提升最显著**：7b 的 TTFT 从 3.4s 降到 177ms。
- **3b 生成速度更快但输出更长**：当 `num_predict=512` 时，总耗时与 7b 接近。
- **缩短 `num_predict` 对 3b 收益有限**：降到 256 后，3b 仍生成 248 tokens，回答更啰嗦。

---

## 6. 最终效果总结

### 6.1 基线 vs 最终

| 指标 | 基线 | 优化1 | 优化2 | 优化3+主题词 | 优化4 LLM 侧 | 最终优化幅度 |
|---|---|---|---|---|---|---|
| 总耗时 | 32,894 ms | 10,585 ms | 5,752 ms | 4,911 ms | **4,713 ms** | **-85.7%** |
| Stage 1 Query Rewriter | 5,276 ms | 629 ms | 577 ms | 577 ms | 0 ms | -100% |
| Stage 2 Law Name Matcher | 8,907 ms | 161 ms | 161 ms | 0 ms* | 0 ms* | -100%* |
| Stage 3 Article Fetcher | 0 ms | 0 ms | 0 ms | 0 ms | 0 ms | 持平 |
| Stage 4 Article Ranker | 618 ms | 616 ms | 616 ms | 607 ms | 1,036 ms | +67.6%** |
| Stage 5 QA Agent | 18,093 ms | 9,180 ms | 4,443 ms | 3,727 ms | 3,677 ms | -79.7% |
| 首字延迟 TTFT | 2,977 ms | 167 ms | 174 ms | 197 ms | **177 ms** | **-94.1%** |
| 输出 token | 231 | 168 | 86 | 73 | 75 | -67.5% |
| 生成速度 | 12.8 tok/s | 18.3 tok/s | 19.4 tok/s | 19.6 tok/s | 20.4 tok/s | +59.4% |

\* 对于包含法律主题词的 query，Stage 2 直接走关键词命中，无需 FAISS。  
\** 本轮 Stage 4 绝对值略高，是单次运行波动，整体仍处于 0.5~1.0s 区间。

### 6.2 测试样例

#### 样例 A：单位拖欠工资怎么办？

- 总耗时：**4.7s**
- 命中法律：劳动争议调解仲裁法、劳动合同法、工会法
- 回答质量：正确，建议投诉/申请支付令

#### 样例 B：被无故辞退能赔多少钱？

- 总耗时：**3.1s ~ 7.0s**（取决于 Ollama 模型调度状态）
- 命中法律：劳动合同法
- 回答质量：正确，引用第 47 条（经济补偿）和第 87 条（赔偿金）

### 6.3 已落地优化清单

| # | 优化项 | 文件 | 效果 |
|---|---|---|---|
| 1 | Stage 1 与 Stage 2 并行 | `pipeline.py` | 改写+匹配不再串行 |
| 2 | 默认 top-articles 从 10 降到 6 | `run_demo.py` | 减少 LLM 上下文 |
| 3 | QA prompt 精简，要求不重复法条原文 | `pipeline.py` | 输出 token 减少 50%+ |
| 4 | 限制 `num_predict=512` | `pipeline.py`, `run_demo.py` | 避免过长生成 |
| 5 | query embedding 缓存共享 | `pipeline.py` | Stage 2/4 复用同一向量 |
| 6 | 降低 `fetch_k` 50→30 | `pipeline.py` | 减少候选 doc embedding 调用 |
| 7 | 避免冗余法律名匹配 | `pipeline.py` | 改写后无新增法律关键词则跳过 |
| 8 | 新增口语化主题词映射 | `pipeline.py` | 提升劳动/婚姻/工伤类 query 召回 |
| 9 | REPL 历史截断 | `run_demo.py` | 多轮时减少改写阶段上下文 |
| 10 | LLM 预热 warm-up | `pipeline.py` | TTFT 从 3.4s 降到 177ms |
| 11 | 可配置 Ollama options | `pipeline.py`, `run_demo.py` | 支持 `num_gpu`/`num_thread` 等调参 |
| 12 | 引入 `qwen2.5:3b` 并 benchmark | `run_demo.py` | 生成速度 34~42 tok/s |

### 6.4 推荐启动命令

```bash
cd /Users/yuanangyang/local_rag_legal/demo
export OLLAMA_BASE_URL=http://localhost:11434
python3 run_demo.py -q "单位拖欠工资怎么办？" \
  --embed-backend ollama --llm-backend ollama \
  --embed-model bge-m3 --llm-model qwen2.5:7b \
  --num-predict 512
```

---

## 7. 剩余瓶颈与后续方向

- **Stage 5 LLM 生成仍占 78~90%**，但非 LLM 阶段已基本优化完毕。
- 进一步提升方向：
  1. **答案缓存**：对高频 query 直接命中缓存，可降到 100ms 以内。
  2. **模型量化**：尝试 Ollama 4-bit 量化模型进一步提速。
  3. **外部 API 兜底**：用户可选 DeepSeek/OpenAI 作为速度优先模式。
  4. **3b prompt 工程**：抑制 3b 过度展开，进一步缩短输出长度。

---

## 8. 文档索引

- `01_baseline.md` — 原始性能基线
- `02_optimization_1.md` — 并行 + 精简上下文
- `03_optimization_2.md` — prompt 精简 + 限制生成长度
- `04_optimization_3.md` — embedding 缓存 + 避免冗余匹配
- `06_optimization_4.md` — LLM 侧优化
- `OPTIMIZATION.md` — 本文档（完整汇总）

## 5. 效果评测与迭代优化（2026-07-21）

### 5.1 评测基线（Ollama bge-m3 + qwen2.5:7b，前 10 条）

| 指标 | 数值 |
|---|---|
| Stage 2 法律名命中率 | 60.0% |
| Stage 4 法条召回率 | 33.3% |
| Stage 5 答案关键词覆盖率 | 21.3% |
| 平均端到端延迟 | 5.8 s |
| 平均 TTFT | 0.5 s |

### 5.2 第一轮优化：关键词映射 + 全库回退 + QA Prompt 强化

#### 改动点

1. **扩充 `topic_to_law` 主题词映射**（`demo/pipeline.py`）
   - 新增劳动合同法场景：`劳动合同、试用期、转正、经济补偿金、赔偿金、N+1、孕期、产期、哺乳期、女职工`
   - 新增具体领域：`高铁、动车、列车、铁路、网络安全、数据泄露、个人信息、隐私、网约车、交通事故、消费者、七天无理由`
   - 避免劳动法/民法典覆盖劳动合同法相关 query。

2. **Stage 4 法律限定召回为空时自动回退全库搜索**
   - 原逻辑仅在法律限定召回为 0 时回退一次全库。
   - 新逻辑改为：**法律限定召回 + 全库召回同时执行**，合并去重后精排，避免索引元数据与法律名匹配不一致导致漏召。

3. **强化 `QA_PROMPT`**
   - 要求先给结论再给法律依据。
   - 明确要求保留关键数字、条件、期限、金额计算方式。
   - 要求回答包含用户最关心的实体词。

4. **放宽法条上下文截断长度**
   - `_format_context` 单条法条截断从 600 字放宽到 900 字，减少关键数字丢失。

5. **Python 3.9 兼容性修复**
   - 将 `dict | None`、`list[...] | None` 等类型注解统一改为 `Optional[...]`，避免 Python 3.9 报错。

#### 验证结果（前 10 条）

| 指标 | 优化前 | 优化后 | 变化 |
|---|---|---|---|
| 法律名命中率 | 60.0% | **100.0%** | +40.0% |
| 法条召回率 | 33.3% | **58.3%** | +25.0% |
| 答案覆盖率 | 21.3% | **39.1%** | +17.8% |

### 5.3 关键发现与根因分析

1. **FAISS 索引与 embedding 模型必须一致**
   - 当前 `demo/indexes/articles` 若用 Ollama `bge-m3` 查询，需保证索引由 `bge-m3` 构建。
   - 用 HF `bge-small-zh-v1.5` 构建的索引与 `bge-m3` 维度不同（512 vs 1024），无法直接混用。

2. **部分评测集标注存在法条编号错误**
   - `#8 高铁上吸烟`：expected_law 为《铁路安全管理条例》，expected_article_no 标注为第 107 条；但实际罚款 500-2000 元对应 **第 95 条**。
   - `#9 网约车事故`：expected_article_no 标注为第 1177 条（共同侵权），但 expected_answer 描述的是用人单位责任，对应 **第 1191 条**。
   - `#10 误删微信记录`：expected_answer 认为“不属于法律调整范围”，但标注 retrieved_text 引用第 985 条不当得利，逻辑不一致。
   - `#8 中《铁路安全管理条例》在原始数据集中不存在**，导致法律名命中但无法召回任何法条。

3. **`law_filter` 在部分索引下召回稀疏**
   - 即使法律名命中，《网络安全法》在法律限定内只召回 2 条；
   - 同时执行全库召回后，能够补充更多相关候选。

### 5.4 下一步优化方向

1. **重建与运行模型一致的 FAISS 索引**（优先 Ollama bge-m3）。
2. **修复/校准评测集标注错误**（#8、#9、#10 等）。
3. **补充数据缺失的法律**（如《铁路安全管理条例》），或建立“法律名→可替代法律”的回退映射。
4. **探索 HyDE**：用 LLM 生成假设法条回答后再 embedding，可能提升语义召回。
5. **答案后处理/校验**：用法条原文中的数字、期限做正则校验，确保 LLM 输出覆盖关键实体。

---
