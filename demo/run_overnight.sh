#!/usr/bin/env bash
# 夜间后台任务：等 qwen2.5:7b 下载完成后构建索引并跑 demo 耗时测试
set -euo pipefail

export PATH="/usr/local/bin:$PATH"
export OLLAMA_BASE_URL="http://localhost:11434"

LOG="/tmp/rag_overnight.log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 开始夜间任务"
cd /Users/yuanangyang/local_rag_legal/demo

# 1. 等待 qwen2.5:7b 下载完成
echo "等待 qwen2.5:7b 下载完成..."
while ! ollama list | grep -q "qwen2.5:7b"; do
    sleep 60
    echo "  $(date +'%H:%M:%S') 仍在拉取 qwen2.5:7b..."
done
echo "qwen2.5:7b 已就绪"

# 2. 构建 FAISS 索引
echo "构建 FAISS 索引 (ollama + bge-m3)..."
python3 build_indexes.py --embed-backend ollama --embed-model bge-m3

# 3. 跑单条 demo 测端到端链路耗时
echo "跑单条 demo 测链路耗时..."
python3 run_demo.py -q "单位拖欠工资怎么办？" --embed-backend ollama --llm-backend ollama --embed-model bge-m3 --llm-model qwen2.5:7b

# 4. 跑 10 条 retrieval 测试（仅检索，不含 LLM）
echo "跑 10 条检索测试（不含 LLM）..."
python3 test_retrieval.py --limit 10 --no-rewrite --output /tmp/rag_retrieval_10.json

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 夜间任务完成"
