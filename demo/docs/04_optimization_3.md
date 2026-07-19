# 04_optimization_3.md — 优化3：query embedding 缓存 + 避免冗余法律名匹配

## 改动点

1. **共享 query embedding 缓存**
   - 在 `LegalRAGPipeline` 中维护一个 `_query_vec_cache`。
   - `LawNameMatcher` 和 `ArticleRanker` 都复用该缓存。
   - 同一 query 的法律名匹配和法条精排只需要做一次 embedding，避免重复调用 Ollama embedding。

2. **降低 `fetch_k`**
   - `ArticleRanker` 粗排候选从 50 降到 30。
   - 减少精排阶段需要重新编码的候选 doc 数量。

3. **避免冗余法律名匹配**
   - 原逻辑：改写后 query 与原 query 不同时，会用改写后 query 再做一次法律名匹配。
   - 新逻辑：仅当改写后 query 新增了原 query 中没有的法律关键词时，才做补充匹配。
   - 减少一次 FAISS 搜索或关键词扫描。

## 性能对比（同一问题：单位拖欠工资怎么办？）

| 指标 | 优化2后 | 优化3后 | 变化 |
|---|---|---|---|
| 总耗时 | 5,752 ms | 4,911 ms | -14.6% |
| Stage 1 Query Rewriter | 629 ms | 577 ms | -8.3% |
| Stage 2 Law Name Matcher | 161 ms | 0 ms | 触发关键词命中 |
| Stage 4 Article Ranker | 616 ms | 607 ms | 持平 |
| Stage 5 QA Agent | 4,443 ms | 3,727 ms | -16.1% |
| 首字延迟 TTFT | 174 ms | 197 ms | 持平 |
| 输出 token | 86 | 73 | -15.1% |
| 生成速度 | 19.4 tok/s | 19.6 tok/s | 持平 |

## 关键发现

- query embedding 缓存让 Stage 2/Stage 4 的 embedding 调用从 2 次降到 1 次。
- 避免冗余法律名匹配后，该问题下 Stage 2 法律名匹配耗时归零（关键词命中路径）。
- 当前 LLM 生成仍是最大瓶颈，占总时间 75.9%。
- 回答质量保持正确，引用了《劳动争议调解仲裁法》第九条和《劳动合同法》第三十条。

## 代码文件

- `demo/pipeline.py`：`LawNameMatcher` / `ArticleRanker` 共享 query vector cache、降低 fetch_k、避免冗余匹配
