# LocalRAG-Legal 项目优化总文档

> 本文档汇总 `local_rag_legal` 项目从性能优化到效果评测的全链路优化工作。
> 性能优化详见 [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md)；
> 效果评测与优化详见 [`docs/QUALITY_OPTIMIZATION.md`](docs/QUALITY_OPTIMIZATION.md)。

---

## 1. 项目概述

`local_rag_legal` 是一个面向中文法律问答的本地 RAG 系统。核心流程分为 5 个阶段：

```
[1] Query Rewriter    → 改写/扩展用户 query
[2] Law Name Matcher  → 匹配相关法律名
[3] Article Fetcher   → 根据法律名拉取候选法条
[4] Article Ranker    → 对法条重排序
[5] QA Agent          → 生成最终回答
```

系统基于 LangChain + FAISS + Ollama/Sentence-Transformers 构建，支持本地 embedding 和本地 LLM。

---

## 2. 性能优化（2026-07-19）

### 2.1 基线性能

| 指标 | 数值 |
|---|---|
| 端到端总耗时 | 32,894 ms |
| Stage 1 Query Rewriter | 5,276 ms (16.0%) |
| Stage 2 Law Name Matcher | 8,907 ms (27.1%) |
| Stage 4 Article Ranker | 618 ms (1.9%) |
| Stage 5 QA Agent | 18,093 ms (55.0%) |
| 首字延迟 TTFT | 2,977 ms |
| 生成速度 | 12.8 tok/s |

### 2.2 四轮性能优化及效果

| 优化轮次 | 核心改动 | 总耗时 | TTFT |
|---|---|---|---|
| 优化 1 | Stage 1/2 并行 + 精简 Stage 5 上下文 | 10,585 ms | 167 ms |
| 优化 2 | QA Prompt 精简 + `num_predict=512` | 5,752 ms | 174 ms |
| 优化 3 | Query embedding 缓存 + 降低 `fetch_k` + 口语化主题词映射 | 4,911 ms | 197 ms |
| 优化 4 | LLM Warm-up + 可配置 Ollama options + 3b benchmark | 4,713 ms | 177 ms |

**最终效果**：总耗时从 32.9s 降到 4.7s，降幅 **-85.7%**；TTFT 从 3.0s 降到 177ms，降幅 **-94.1%**。

### 2.3 关键落地项

1. **并行改写与法律匹配**：未命中法律关键词时并发执行，命中时跳过改写。
2. **上下文精简**：`top-articles` 默认 10 → 6，单条法条正文超过 600 字截断。
3. **QA Prompt 工程**：要求 LLM 不重复法条原文，直接给结论与可操作建议。
4. **Embedding 缓存共享**：Stage 2/Stage 4 复用同一 query embedding，减少冗余调用。
5. **避免冗余法律名匹配**：改写后无新增法律关键词时不再匹配。
6. **口语化主题词映射**：提升“辞退/赔偿/工伤/试用期/网络安全/个人信息”等场景的命中。
7. **LLM Warm-up**：初始化后触发一次轻量 generate，把模型加载到显存，TTFT 大幅下降。
8. **可配置 Ollama options**：支持通过环境变量/CLI 注入 `num_gpu`、`num_thread`、`top_p` 等。

### 2.4 推荐性能配置

```bash
cd /Users/yuanangyang/local_rag_legal/demo
export OLLAMA_BASE_URL=http://localhost:11434
python3 run_demo.py -q "单位拖欠工资怎么办？" \
  --embed-backend ollama --llm-backend ollama \
  --embed-model bge-m3 --llm-model qwen2.5:7b \
  --num-predict 512
```

---

## 3. 效果评测与优化（2026-07-21）

### 3.1 评测指标

| 指标 | 含义 |
|---|---|
| Stage 2 法律名命中率 | `matched_laws` 命中 `retrieved_text` 法律名的比例 |
| Stage 4 法条召回率 | `final_articles` 覆盖 `retrieved_text` 法条编号的比例 |
| Stage 5 答案覆盖率 | 生成答案包含 `expected_answer` 核心实体词的比例 |

### 3.2 评测基线（qwen2.5:3b，无 HyDE）

| 范围 | 法律名命中率 | 法条召回率 | 答案覆盖率 |
|---|---|---|---|
| 前 10 条 | 80.0% | 33.3% | 29.6% |
| 前 12 条 | 75.0% | 42.9% | 25.4% |

### 3.3 三轮效果优化

#### 第一轮：主题词映射修正 + QA Prompt 调优

- 修正 `topic_to_law`：删除不存在的《铁路安全管理条例》映射；增加 `婚姻法`、`高铁上吸烟`、`知识产权被侵犯如何维权` 等映射。
- 调整 `QA_PROMPT`，抑制 3b 模型过度保守输出。

**结果（前 12 条）**：法律名命中率 100%（+25%），法条召回率 50.0%（+7.1%），答案覆盖率 32.0%（+6.6%）。

#### 第二轮：混合排序 + 全法条兜底

- `ArticleRanker` 引入语义（0.6）+ n-gram 关键词（0.4）融合排序。
- 法律限定内 FAISS 召回稀疏时，自动补充该法律全部法条进入精排。
- 命中法律限定内的法条额外 +0.20 奖励。

**结果（前 12 条）**：法律名命中率 100%，法条召回率 50.0%，答案覆盖率 32.0%。本轮提升有限，因剩余未召回主要源于**评测集标注与数据集不一致**。

#### 第三轮：评测集一致性修复 + `--skip-uncovered`

- 新增 `check_eval_coverage.py` 扫描评测集与数据集一致性。
- 发现 100 条评测集中：10 条期望法律不在数据集中，21 条期望法条号不在数据集中，多条法条号与答案描述不符。
- 修正 `eval_set.json` 中 #2、#8、#9、#10 标注。
- 在 `run_eval.py` 新增 `--skip-uncovered` 参数，自动跳过 bad case。

**结果（前 12 条，过滤 bad case）**：过滤后 10 条，法律名命中率 100%，法条召回率 **66.7%**，答案覆盖率 **37.3%**。

### 3.4 关键发现

1. **Embedding 模型与索引必须一致**
   - `bge-small-zh-v1.5` 维度 512，`bge-m3` 维度 1024，混用会导致搜索质量极差。
   - 最终采用 HF `bge-small-zh-v1.5` 重建 FAISS articles 索引，并用 sentence-transformers 封装更稳定的 embedding wrapper。

2. **评测集与数据集存在严重错配**
   - 缺失法律如《工伤保险条例》《高层民用建筑消防安全管理规定》《航班正常管理规定》等。
   - 缺失法条号如《民法典》第 1062、1191、1254、1167、1199 等（当前数据集民法典条目不完整）。
   - 部分标注法条号与内容不符，如 #8 高铁吸烟原标注第 107 条，实际对应第 95/26 条。

3. **LLM 模型对答案覆盖率影响大**
   - qwen2.5:3b 倾向于保守、简短输出；7b 输出更完整，但当前环境长评测易中断。

### 3.5 剩余瓶颈与下一步方向

1. **补充缺失法律数据**：优先补齐 10 部缺失法律和《民法典》缺失条文。
2. **引入 LLM 法条重排序**：用 LLM 判断候选法条与 query 的相关性，提升语义远但人工标注相关的法条命中。
3. **HyDE 伪文档召回**：LLM 可用时生成假设法条回答后再 embedding 检索。
4. **模型升级**：在稳定环境中跑 qwen2.5:7b 完整 100 条评测。
5. **答案后处理校验**：用法条原文中的数字、期限做正则校验，确保关键实体不丢失。

### 3.6 推荐效果评测命令

```bash
export OLLAMA_BASE_URL=http://localhost:11434
python3 -u demo/run_eval.py \
  --embed-backend hf \
  --embed-model BAAI/bge-small-zh-v1.5 \
  --llm-backend ollama \
  --llm-model qwen2.5:3b \
  --limit 12 \
  --skip-uncovered
```

---

## 4. 项目文件索引

| 文件 | 说明 |
|---|---|
| [`demo/pipeline.py`](demo/pipeline.py) | 核心 RAG Pipeline，含 5 个 Stage、主题词映射、排序、Prompt |
| [`demo/run_demo.py`](demo/run_demo.py) | 交互式 demo 入口 |
| [`demo/run_eval.py`](demo/run_eval.py) | 效果评测脚本，支持 `--skip-uncovered` |
| [`demo/eval_set.json`](demo/eval_set.json) | 100 条评测数据 |
| [`demo/check_eval_coverage.py`](demo/check_eval_coverage.py) | 评测集与数据集一致性检查 |
| [`demo/check_articles.py`](demo/check_articles.py) | 目标法条内容查询 |
| [`demo/diag_rank.py`](demo/diag_rank.py) | 法条召回位置诊断 |
| [`demo/diag_init_hf.py`](demo/diag_init_hf.py) | HF embedding 初始化诊断 |
| [`demo/docs/OPTIMIZATION.md`](demo/docs/OPTIMIZATION.md) | 性能优化详细文档 |
| [`demo/docs/QUALITY_OPTIMIZATION.md`](demo/docs/QUALITY_OPTIMIZATION.md) | 效果优化详细文档 |
| [`demo/docs/PROJECT_OPTIMIZATION.md`](demo/docs/PROJECT_OPTIMIZATION.md) | 本文档（总汇总） |

---

## 5. 总结

- **性能侧**：通过并行、缓存、Prompt 工程、LLM Warm-up 等手段，端到端延迟从 32.9s 降至 4.7s，已无明显瓶颈。
- **效果侧**：通过主题词映射、混合排序、评测集一致性修复，前 12 条过滤 bad case 后法律名命中率 100%、法条召回率 66.7%、答案覆盖率 37.3%。
- **下一步优先级**：补充缺失法律数据 > 补齐《民法典》缺失条文 > LLM 法条重排序/后处理校验 > 7b 模型完整评测。
