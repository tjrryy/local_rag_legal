# 03_optimization_2.md — 优化2：精简 QA prompt + 限制生成长度

## 改动点

1. **精简 `QA_PROMPT`**
   - 明确告诉 LLM：「引用法条时用格式即可，不要重复法条原文」。
   - 明确要求「直接给出结论和可操作的建议，语言简洁」。
   - 减少 LLM 输入长度，降低 TTFT 和总生成时间。

2. **限制 `num_predict`**
   - 在 Ollama `/api/generate` 请求中加入 `options.num_predict`。
   - 默认值 512，可通过环境变量 `OLLAMA_NUM_PREDICT` 或命令行 `--num-predict` 调整。
   - 避免模型生成过长、啰嗦的回答。

3. **支持命令行参数**
   - `run_demo.py` 新增 `--num-predict` 参数。
   - `LegalRAGPipeline` 初始化前设置 `OLLAMA_NUM_PREDICT` 环境变量。

## 性能对比（同一问题：单位拖欠工资怎么办？）

| 指标 | 优化1后 | 优化2后 | 变化 |
|---|---|---|---|
| 总耗时 | 10,585 ms | 5,752 ms | -45.7% |
| Stage 5 QA Agent | 9,180 ms | 4,443 ms | -51.6% |
| 首字延迟 TTFT | 167 ms | 174 ms | 持平 |
| 输出 token | 168 | 86 | -48.8% |
| 生成速度 | 18.3 tok/s | 19.4 tok/s | +6.0% |

## 关键发现

- prompt 要求「不要重复法条原文」后，LLM 输出 token 从 168 降到 86，回答更 concise。
- Stage 5 从 9.2s 降到 4.4s，总耗时首次进入 6s 以内。
- TTFT 保持 170ms 左右，说明模型预热良好。

## 注意事项

- `num_predict` 过小（如 300）会导致回答被截断，默认 512 是质量与速度的折中。
- 若问题需要详细解释，可在运行时调大 `--num-predict`。

## 代码文件

- `demo/pipeline.py`：`QA_PROMPT` 精简、`RobustOllamaLLM` 支持 `num_predict`
- `demo/run_demo.py`：新增 `--num-predict` 参数
