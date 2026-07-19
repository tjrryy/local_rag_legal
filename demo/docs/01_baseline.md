# 01_baseline.md — 性能基线

## 测试环境

- 时间：2026-07-19
- 模型：`bge-m3`（本地 Ollama）+ `qwen2.5:7b`（本地 Ollama）
- 索引：303 部法律名 + 22,482 条法条
- 问题：`单位拖欠工资怎么办？`

## 端到端耗时

| 阶段 | 耗时 | 占比 |
|---|---|---|
| [1] Query Rewriter | 5,276 ms | 16.0% |
| [2] Law Name Matcher | 8,907 ms | 27.1% |
| [3] Article Fetcher | 0 ms | 0.0% |
| [4] Article Ranker (hybrid) | 618 ms | 1.9% |
| [5] QA Agent | 18,093 ms | 55.0% |
| **总计** | **32,894 ms** | 100% |

## LLM 生成指标

- 首字延迟 TTFT：2,977 ms
- 累计 token：231
- LLM 总耗时：18,091 ms
- 生成速度：12.8 tok/s

## 主要瓶颈分析

1. **Stage 5 LLM 回答**：占总时间 55%，其中首字延迟 3s，生成速度 12.8 tok/s 偏慢。
2. **Stage 2 法律名匹配**：占 27%，本质是 1 次 query embedding + FAISS 搜索。
3. **Stage 1 Query Rewriter**：占 16%，也是 1 次 LLM 调用。

## 优化方向

1. **并行 Stage 1 + Stage 2**：改写和法律名匹配可以并发执行。
2. **缩短 LLM prompt / 减少法条上下文**：减少 Stage 5 的输入长度。
3. **降低 top-articles**：从 10 条降到 4-6 条，减少 LLM 处理量。
4. **预热模型 / keep-alive**：`keep_alive=30m` 已设置，确保模型常驻内存。
5. **LLM 参数优化**：`num_predict`、`temperature` 等可适当调低。
