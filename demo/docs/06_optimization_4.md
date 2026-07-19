# 06_optimization_4.md — LLM 侧优化：Warm-up + 可配置选项 + 更小模型

## 背景

前 3 轮优化后，非 LLM 阶段已基本压到 0~1s，**Stage 5 QA Agent 仍占总耗时 75~90%**。本轮专门从 LLM 推理侧寻找收益。

## 优化内容

### 1. LLM 预热（Warm-up）

Ollama 首次调用模型时存在加载/调度延迟，导致 TTFT 高达 3s+。在 `pipeline.py` 的 `build_llm()` 中增加 `warm_up()`：

- 初始化 LLM 后立即发一次轻量 generate 请求（prompt="你好"）。
- 触发 Ollama 把模型加载到内存/GPU，后续真实请求的 TTFT 大幅降低。
- 可通过 `OLLAMA_NO_WARM_UP=1` 或 `--no-warm-up` 关闭。

### 2. 可覆盖的 Ollama generate options

原实现只暴露 `temperature` 和 `num_predict`。本轮允许通过环境变量 `OLLAMA_OPTIONS`（JSON）或 CLI `--ollama-options` 注入任意 Ollama options，例如：

```bash
python3 demo/run_demo.py -q "..." \
  --ollama-options '{"num_gpu":40,"num_thread":8,"top_p":0.8}'
```

默认保留：`temperature=0.1`, `top_p=0.9`, `num_predict=512`。

### 3. 引入更小模型 `qwen2.5:3b`

本地 `qwen2.5:7b` 生成速度约 10~20 tok/s；`qwen2.5:3b`（1.9 GB）生成速度可达 34~42 tok/s，且回答质量在常见劳动法律 query 上仍可接受。

## 代码改动

| 文件 | 改动 |
|---|---|
| `pipeline.py` | `RobustOllamaLLM` 增加 `options`、`warm_up()`；`build_llm()` 支持 `OLLAMA_OPTIONS` 和 warm-up |
| `run_demo.py` | 新增 `--ollama-options`、`--no-warm-up` CLI 参数 |

## 性能对比

### 同 query：`单位拖欠工资怎么办？`

| 指标 | 优化前（7b，无 warm-up） | 7b + warm-up | 3b + warm-up |
|---|---|---|---|
| 总耗时 | 7,770 ms | **4,713 ms** | 4,883 ms |
| Stage 5 QA Agent | 6,810 ms | 3,677 ms | 3,919 ms |
| 首字延迟 TTFT | **3,375 ms** | **177 ms** | 873 ms（首次）/ 139 ms（第二次） |
| 生成速度 | 10.7 tok/s | 20.4 tok/s | **34.5 ~ 42.7 tok/s** |
| 输出 token | 73 | 75 | 135 ~ 248 |
| 回答质量 | 正确 | 正确 | 正确 |

### 关键结论

- **warm-up 对 TTFT 提升最显著**：7b 的 TTFT 从 3.4s 降到 177ms，总耗时从 7.8s 降到 4.7s。
- **3b 的生成速度更快但输出更长**：当 `num_predict=512` 时，3b 倾向于生成更详细的解释，导致 Stage 5 总耗时与 7b 接近。
- **缩短 `num_predict` 对 3b 收益有限**：降到 256 后，3b 仍生成 248 tokens，耗时反而因没触发截断而增加到 5.8s（回答更啰嗦）。

## 推荐配置

如果追求**最低首字延迟 + 稳定速度**，优先使用：

```bash
python3 demo/run_demo.py -q "单位拖欠工资怎么办？" \
  --embed-backend ollama --llm-backend ollama \
  --embed-model bge-m3 --llm-model qwen2.5:7b \
  --num-predict 512
```

如果硬件资源紧张或追求更高吞吐，可尝试 `qwen2.5:3b` 并配合更严格的 prompt 长度控制。

## 后续方向

1. **答案缓存**：对高频 query 直接命中缓存，跳过 LLM。
2. **模型量化**：尝试 Ollama 4-bit 量化模型进一步提速。
3. **外部 API 兜底**：用户可选 DeepSeek/OpenAI 作为速度优先模式。
