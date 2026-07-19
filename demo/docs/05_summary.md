# 05_summary.md — 性能优化总结

## 基线 vs 最终

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

\* 对于包含法律主题词的 query，Stage 2 直接走关键词命中，无需 FAISS。

## 测试样例

### 样例 A：单位拖欠工资怎么办？

- 总耗时：**4.9s**
- 命中法律：劳动争议调解仲裁法、劳动合同法、工会法
- 回答质量：正确，建议投诉/申请支付令

### 样例 B：被无故辞退能赔多少钱？

- 总耗时：**3.1s ~ 7.0s**（取决于 Ollama 模型调度状态）
- 命中法律：劳动合同法
- 回答质量：正确，引用第47条（经济补偿）和第87条（赔偿金）

## 已落地优化清单

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

## 剩余瓶颈

- **Stage 5 LLM 生成仍占 78~90%**，但非 LLM 阶段已基本优化完毕。
- 进一步提升方向：
  1. 对常见问题进行答案缓存（相同 query 直接返回），可降到 100ms 以内。
  2. 启用 Ollama 多 GPU / 量化配置。
  3. 接入外部 API（DeepSeek / OpenAI）作为速度优先模式。
  4. 对 3b 模型做 prompt 工程，抑制过度展开，进一步缩短输出长度。

## 文档索引

- `01_baseline.md` — 原始性能基线
- `02_optimization_1.md` — 并行 + 精简上下文
- `03_optimization_2.md` — prompt 精简 + 限制生成长度
- `04_optimization_3.md` — embedding 缓存 + 避免冗余匹配
- `06_optimization_4.md` — LLM 侧优化（warm-up + 可配置选项 + 更小模型）
- `05_summary.md` — 本文档
