# 基于本地大语言模型的法律法规智能问答系统

## 项目概述

本项目旨在构建一个基于本地大语言模型的法律法规智能问答系统，结合 RAG（检索增强生成）技术与 Agent 任务调度，实现允许多轮交互、任务调用的智能问答功能。

## 项目背景

在大模型快速发展背景下，法律领域对智能问答与信息抽取提出更高要求。面对海量法律法规文档，如何构建自主知识库并结合本地 LLM 进行问答成为重要研究方向。本项目以国家法律法规数据库等公共法律数据库为数据源，通过文档处理构建本地法规知识库，并利用 RAG+Agent 技术构建智能问答系统。

## 技术栈

| 类别 | 选型 |
|------|------|
| LLM | DeepSeek / 通义千问 / LLaMA3（优先国产模型） |
| 部署方式 | Ollama 本地部署 / API 接入 |
| 文档处理 | python-docx |
| 向量数据库 | ChromaDB（开发）/ Qdrant（生产） |
| RAG 框架 | LangChain |
| Agent 框架 | LangGraph（多 Agent 协作） |
| Web 框架 | FastAPI |
| 嵌入模型 | sentence-transformers（中文模型） |

## 项目目标

### 核心任务
1. **部署本地 LLM**：支持 DeepSeek、通义千问等国产模型，Ollama 或 API 接入
2. **数据采集**：编写爬虫/文档处理程序，获取法律法规文本
3. **文档向量化**：清洗、分段、向量化处理
4. **构建 RAG 问答系统**：检索 + 生成
5. **多 Agent 协作**：使用 LangChain/LangGraph 实现 Agent 任务调度
6. **Web 界面**：FastAPI 提供 RESTful API 接口
7. **技术文档**：编写项目文档，提交汇报材料

### 附加要求
- 使用 FastAPI 搭建系统服务端接口，支持 RESTful 调用
- 构建测试数据集：至少 100 条问答样本

## 项目结构

```
.
├── 法律原文/                    # 原始法律文档（.docx）
├── 法律解释修正案/              # 法律解释与修正案
├── dataset/                     # 处理后的数据集
│   ├── laws_dataset_*.json     # 结构化数据
│   ├── laws_dataset_*.csv      # CSV 索引
│   └── all_articles_*.txt      # 完整法条文本
├── crawler.py                   # 文档处理脚本
├── doc_to_dataset.py           # 法条切分工具
├── chromadb_crud.py            # 向量数据库 CRUD 教程
└── README.md                   # 项目说明文档
```

## 数据统计

- 法律原文文档：**300+ 部**
- 覆盖法律类型：宪法、民法商法、行政法规、刑法、刑事诉讼法、劳动法、社会保障法、知识产权法、环境保护法等
- 法条总数：**24,000+ 条**
- 法律解释与修正案：**40+ 份**

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install python-docx chromadb langchain langchain-chroma langchain-community
pip install sentence-transformers fastapi uvicorn
```

### 2. 数据处理

```bash
# 处理法律原文
python3 doc_to_dataset.py "法律原文" -o dataset

# 处理法律解释修正案
python3 doc_to_dataset.py "法律解释修正案" -o dataset_supplement
```

### 3. 构建向量数据库

```python
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 加载嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 加载向量数据库
db = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings
)
```

### 4. 启动 API 服务

```bash
uvicorn api:app --reload
```

## 测试数据集

测试数据集采用以下 JSON 结构：

```json
[
  {
    "question": "我国最低工资标准是多少？",
    "expected_answer": "最低工资标准由各省份制定，例如北京市为每月2320元。",
    "retrieved_text": "根据《最低工资规定》第二条...",
    "model_output": "北京市最低工资目前为2320元。",
    "evaluation_note": "回答基本准确，但未强调地区差异。"
  }
]
```

### 字段说明

| 字段 | 说明 |
|------|------|
| question | 待检索并问答的问题文本 |
| expected_answer | 人工标注的标准答案 |
| retrieved_text | 系统返回的检索段落（可选） |
| model_output | 模型或 Agent 生成的最终回答 |
| evaluation_note | 评价意见或备注 |

## 后续计划

- [ ] 接入 DeepSeek API 进行 RAG 问答验证
- [ ] 构建多 Agent 协作框架
- [ ] 设计 FastAPI 接口
- [ ] 编写 100+ 条测试问答对
- [ ] 评估系统性能
- [ ] 部署本地 LLM 模型（Ollama）

## 参考文献

- [国家法律法规数据库](https://flk.npc.gov.cn/)
- [LangChain 官方文档](https://python.langchain.com/)
- [ChromaDB 官方文档](https://docs.trychroma.com/)
- [FastAPI 官方文档](https://fastapi.tiangolo.com/)

## 致谢

指导老师：徐继伟老师

---

*本项目为实习项目，目标构建一个完整的法律法规智能问答系统。*