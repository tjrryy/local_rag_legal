#!/usr/bin/env bash
# 构建法条索引 + 跑 demo 耗时测试
set -euo pipefail

export PATH="/usr/local/bin:$PATH"
export OLLAMA_BASE_URL="http://localhost:11434"

LOG="/tmp/rag_build_and_test.log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 开始构建索引并测试"
cd /Users/yuanangyang/local_rag_legal/demo

# 清理可能卡住的旧 build_indexes 进程
pkill -f "build_indexes.py --embed-backend ollama" 2>/dev/null || true
sleep 2

# 1. 构建 FAISS 两段索引（已有 law_names 会重建）
echo "构建 FAISS 索引 (ollama + bge-m3)..."
python3 build_indexes.py --embed-backend ollama --embed-model bge-m3 --reset

# 2. 跑单条 demo 测端到端链路耗时
echo "跑单条 demo 测链路耗时..."
python3 run_demo.py -q "单位拖欠工资怎么办？" --embed-backend ollama --llm-backend ollama --embed-model bge-m3 --llm-model qwen2.5:7b

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 完成"
