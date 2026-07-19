# 02_optimization_1.md — 优化1：并行改写与法律匹配 + 精简 LLM 上下文

## 改动点

1. **Stage 1 与 Stage 2 并行**
   - 原逻辑：先改写，再法律名匹配，串行执行。
   - 新逻辑：若原 query 未命中法律关键词，则 `Query Rewriter` 与 `Law Name Matcher` 并发执行。
   - 若 query 已含法律关键词（如“劳动合同法”），直接走关键词匹配，跳过改写。

2. **精简 Stage 5 LLM 上下文**
   - `top-articles` 默认值从 10 降到 6。
   - `QAAgent` 最多只把前 6 条法条送入 prompt。
   - 单条法条正文超过 600 字时截断，避免 prompt 膨胀。

3. **REPL 历史截断**
   - 累积历史时只保留回答前 200 字，减少多轮时改写阶段的上下文长度。

## 性能对比

| 指标 | 优化前 | 优化后 | 变化 |
|---|---|---|---|
| 总耗时 | 32,894 ms | 10,585 ms | -67.8% |
| Stage 1 Query Rewriter | 5,276 ms | 629 ms | -88.1% |
| Stage 2 Law Name Matcher | 8,907 ms | 161 ms | -98.2% |
| Stage 4 Article Ranker | 618 ms | 616 ms | 持平 |
| Stage 5 QA Agent | 18,093 ms | 9,180 ms | -49.3% |
| 首字延迟 TTFT | 2,977 ms | 167 ms | -94.4% |
| 输出 token | 231 | 168 | -27.3% |
| 生成速度 | 12.8 tok/s | 18.3 tok/s | +43.0% |

## 关键发现

- **TTFT 大幅下降**是因为减少了 LLM 输入长度（法条从 10 条变 6 条 + 截断）。
- **Stage 1 + Stage 2 并行**后，两者合计从 14.2s 降到 0.79s。
- 当前剩余瓶颈主要是 **Stage 5 LLM 生成（占 86.7%）**，后续优化方向：
  - 使用更快的本地模型（如 `qwen2.5:3b`）
  - 限制 `num_predict` 缩短回答长度
  - 若 GPU 资源允许，启用 Ollama 的并发推理参数

## 代码文件

- `demo/pipeline.py`：`run()` 并行逻辑、`QAAgent` 上下文截断
- `demo/run_demo.py`：`--top-articles` 默认改为 6、REPL 历史截断
