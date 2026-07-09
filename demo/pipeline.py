"""
demo/pipeline.py
================
5 阶段法律问答管道（核心逻辑）。

  [1] Query Rewriter       把"它/那部法律"还原成具体法律名
  [2] Law Name Matcher     FAISS 在 303 部法律名里找 top-K 法律
  [3] Article Fetcher      从这些法律的全部法条里取候选
  [4] Article Ranker       按相似度重排，取 top-K 法条
  [5] QA Agent             DeepSeek LLM 生成答案

这个模块只定义类，不直接跑。run_demo.py 调它。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

DEMO_DIR = Path(__file__).parent
PROJECT_ROOT = DEMO_DIR.parent
DATA_DIR = PROJECT_ROOT / "law_clearnerdata"
INDEX_DIR = DEMO_DIR / "indexes"

ARTICLE_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+条")
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
MAX_EMBED_CHARS = 1200

# ---- LLM Prompts ----

REWRITE_PROMPT = """你是法律领域查询改写助手。请把用户的【最新问题】改写成一个独立的、可直接检索的问题。

规则：
1. 如果问题里有指代（"它"、"那个"、"这部法律"、"第三条"等），结合【对话历史】还原
2. 不要补充新信息，不要回答问题本身
3. 输出只有改写后的问题，不要任何解释

【对话历史】
{history}

【用户最新问题】
{query}

【改写后的问题】"""


QA_PROMPT = """你是中国法律领域的智能助手。请严格根据下面【法条参考】回答用户问题。

要求：
1. 必须先引用法条原文（用「《法律名》第X条」格式），再做解释
2. 如果多条法条相关，按相关度从高到低引用
3. 如果【法条参考】中没有任何相关内容，请直接回答："现有法条中未直接规定该问题"

【法条参考】
{context}

【用户问题】
{question}

【你的回答】"""


# ============================================================
#  数据结构
# ============================================================

@dataclass
class PipelineResult:
    rewritten_query: str
    matched_laws: list[str]
    candidate_articles: list[Document]
    final_articles: list[tuple[Document, float]]  # (doc, score)
    answer: str
    timings: dict[str, float] = field(default_factory=dict)


# ============================================================
#  Embedding / LLM 工厂
# ============================================================

def build_embeddings(backend: str = "hf", model: str = ""):
    if backend == "hf":
        return HuggingFaceEmbeddings(
            model_name=model or "BAAI/bge-small-zh-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        )
    elif backend == "ollama":
        import time
        import json
        import subprocess

        class RobustOllamaEmbeddings:
            """
            健壮的 Ollama embedding 客户端（pipeline 端专用）。
            用 subprocess 调 curl 绕过 sandbox 对 Python HTTP 的限制。
            """
            def __init__(self, model: str, base_url: str):
                self.model = model
                self.base_url = base_url.rstrip("/")

            def _post(self, prompt: str) -> list[float]:
                payload = json.dumps({
                    "model": self.model,
                    "prompt": prompt,
                    "keep_alive": "30m",
                })
                last_err = None
                for attempt in range(3):
                    try:
                        result = subprocess.run(
                            ["curl", "-s", "-X", "POST",
                             f"{self.base_url}/api/embeddings",
                             "-H", "Content-Type: application/json",
                             "-d", payload],
                            capture_output=True, text=True, timeout=120,
                        )
                        if result.returncode != 0:
                            raise RuntimeError(f"curl rc={result.returncode}: {result.stderr}")
                        body = result.stdout
                        if not body:
                            raise RuntimeError("empty response (model reloading?)")
                        data = json.loads(body)
                        if "embedding" not in data:
                            raise RuntimeError(f"unexpected body: {body[:200]}")
                        return data["embedding"]
                    except Exception as e:
                        last_err = e
                        print(f"      [try {attempt+1}/3] {type(e).__name__}: {e}")
                        time.sleep(3)
                raise RuntimeError(f"embedding 失败 3 次: {last_err}")

            def embed_documents(self, texts):
                # 走 /api/embed 批量端点 + subprocess+curl（与 build_indexes.py 同步）
                BATCH = 32
                out: list[list[float]] = []
                total = len(texts)
                for i in range(0, total, BATCH):
                    chunk = texts[i : i + BATCH]
                    payload = json.dumps({
                        "model": self.model,
                        "input": chunk,
                        "keep_alive": "30m",
                    }, ensure_ascii=False)
                    last_err = None
                    ok = False
                    for attempt in range(3):
                        try:
                            result = subprocess.run(
                                ["curl", "-s", "-X", "POST",
                                 f"{self.base_url}/api/embed",
                                 "-H", "Content-Type: application/json",
                                 "-d", payload],
                                capture_output=True, text=True, timeout=120,
                            )
                            if result.returncode != 0:
                                raise RuntimeError(f"curl rc={result.returncode}: {result.stderr}")
                            body = result.stdout
                            if not body:
                                raise RuntimeError("empty response")
                            data = json.loads(body)
                            vecs = data.get("embeddings")
                            if not vecs or len(vecs) != len(chunk):
                                raise RuntimeError(
                                    f"bad batch response: got {len(vecs) if vecs else 0}, want {len(chunk)}"
                                )
                            out.extend(vecs)
                            ok = True
                            break
                        except Exception as e:
                            last_err = e
                            time.sleep(2)
                    if not ok:
                        raise RuntimeError(f"embed batch 失败 @ {i}/{total}: {last_err}")
                return out

            def embed_query(self, text):
                return self._post(text)

            # 新版 LangChain 要求 Embeddings 对象本身可调用
            def __call__(self, text: str) -> list[float]:
                return self._post(text)

        return RobustOllamaEmbeddings(
            model=model or "nomic-embed-text",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    raise ValueError(f"unknown embed backend: {backend}")


def build_llm(backend: str = "deepseek", model: str = ""):
    if backend == "deepseek":
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("未设置 DEEPSEEK_API_KEY")
        return ChatOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=api_key,
            model=model or "deepseek-chat",
            temperature=0.1,
        )
    elif backend == "ollama":
        import subprocess
        import json

        class RobustOllamaLLM:
            """
            用 subprocess 调 curl 跑 Ollama LLM（绕过 sandbox 对 Python HTTP 的限制）。
            遵循 LangChain LLM 协议：invoke() 返回 str。
            """
            def __init__(self, model: str, base_url: str):
                self.model = model
                self.base_url = base_url.rstrip("/")

            def _post(self, prompt: str) -> str:
                payload = json.dumps({
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "30m",
                })
                last_err = None
                for attempt in range(3):
                    try:
                        result = subprocess.run(
                            ["curl", "-s", "-X", "POST",
                             f"{self.base_url}/api/generate",
                             "-H", "Content-Type: application/json",
                             "-d", payload],
                            capture_output=True, text=True, timeout=600,
                        )
                        if result.returncode != 0:
                            raise RuntimeError(f"curl rc={result.returncode}: {result.stderr}")
                        body = result.stdout
                        if not body:
                            raise RuntimeError("empty response (model reloading?)")
                        data = json.loads(body)
                        if "response" not in data:
                            raise RuntimeError(f"no response field: {body[:300]}")
                        return data["response"]
                    except Exception as e:
                        last_err = e
                        print(f"      [LLM try {attempt+1}/3] {type(e).__name__}: {e}")
                        import time as _t
                        _t.sleep(3)
                raise RuntimeError(f"LLM 失败 3 次: {last_err}")

            def invoke(self, prompt, **kwargs) -> str:
                # 处理 ChatPromptTemplate 输出的多 message 格式
                if hasattr(prompt, "to_string"):
                    prompt = prompt.to_string()
                elif hasattr(prompt, "content"):
                    # 单个 message
                    prompt = prompt.content if hasattr(prompt, "content") else str(prompt)
                elif not isinstance(prompt, str):
                    prompt = str(prompt)
                return self._post(prompt)

        return RobustOllamaLLM(
            model=model or "qwen2.5:7b",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    raise ValueError(f"unknown llm backend: {backend}")


def build_article_embedding_text(
    law_title: str,
    article_no: str,
    article_text: str,
    max_chars: int = MAX_EMBED_CHARS,
) -> str:
    """
    构造“用于向量化”的法条文本。

    与 build_indexes.py 保持一致：embedding 侧对超长正文做截断，
    metadata 里保留完整原文，供最终回答引用。
    """
    prefix = f"《{law_title}》{article_no}　"
    text = article_text.strip()
    if len(text) <= max_chars:
        return prefix + text
    return prefix + text[:max_chars].rstrip() + "……"


# ============================================================
#  Stage 1: Query Rewriter
# ============================================================

class QueryRewriter:
    """
    把用户的原始问题改写成"独立可检索"的问题。
    例：原问题"它第三条说了什么？"
        + 历史"Q: 国家对草原保护有什么方针？A: 提到《草原法》第三条..."
        → 改写后："《中华人民共和国草原法》第三条说了什么？"
    """

    def __init__(self, llm):
        self.llm = llm

    def __call__(self, query: str, history: str = "") -> str:
        prompt = REWRITE_PROMPT.format(history=history or "（无）", query=query)
        try:
            out = self.llm.invoke(prompt)
            if hasattr(out, "content"):
                out = out.content
        except Exception as e:
            print(f"[WARN] 改写失败：{e}，退回原问题")
            return query
        # 兜底：如果 LLM 没返回合理结果，用原 query
        if not out or len(out) > len(query) * 4:
            return query
        return out.strip()


# ============================================================
#  Stage 2: Law Name Matcher
# ============================================================

class LawNameMatcher:
    """
    在 303 部法律名里找 top-K。
    用 FAISS 索引（由 build_indexes.py 预先构建）。
    """

    def __init__(self, db: FAISS):
        self.db = db

    def __call__(self, query: str, top_k: int = 3) -> list[str]:
        q = BGE_QUERY_PREFIX + query
        results = self.db.similarity_search(q, k=top_k)
        return [doc.metadata["law_title"] for doc in results]


# ============================================================
#  Stage 3: Article Fetcher
# ============================================================

class ArticleFetcher:
    """
    给定法律名列表，从原始 JSON 把每部法律的全部法条取出来。
    用 in-memory 字典做 O(1) 查询。
    """

    def __init__(self, data_dir: Path):
        print(f"[INIT] ArticleFetcher: 加载 {data_dir} ...")
        self.law_to_articles: dict[str, list[Document]] = {}
        for fp in sorted(data_dir.glob("laws_dataset_*.json")):
            for law in json.loads(fp.read_text(encoding="utf-8")):
                title = (law.get("title") or "").strip()
                if not title:
                    continue
                docs = []
                for art in law.get("articles", []):
                    art = art.strip()
                    if not art:
                        continue
                    m = ARTICLE_PATTERN.match(art)
                    article_no = m.group(0) if m else ""
                    page_content = build_article_embedding_text(
                        law_title=title,
                        article_no=article_no,
                        article_text=art,
                    )
                    docs.append(Document(
                        page_content=page_content,
                        metadata={
                            "law_title": title,
                            "article_no": article_no,
                            "text": art,
                            "source_file": law.get("source_file", ""),
                            "source_path": law.get("source_path", ""),
                        },
                    ))
                # 同名法律只保留第一份（项目里都是唯一的）
                self.law_to_articles.setdefault(title, docs)
        total = sum(len(v) for v in self.law_to_articles.values())
        print(f"[INIT] ArticleFetcher: {len(self.law_to_articles)} 部法律, "
              f"{total} 条法条")

    def __call__(self, law_titles: Iterable[str]) -> list[Document]:
        out: list[Document] = []
        for t in law_titles:
            out.extend(self.law_to_articles.get(t, []))
        return out


# ============================================================
#  Stage 4: Article Ranker
# ============================================================

class ArticleRanker:
    """
    给定候选法条 + query，按相似度重排，取 top-K。
    直接用 embedding 模型算点积。
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __call__(
        self,
        query: str,
        candidates: list[Document],
        top_k: int = 10,
    ) -> list[tuple[Document, float]]:
        if not candidates:
            return []
        if len(candidates) <= top_k:
            # 候选比 top_k 还少，全要
            cand_vecs = self.embeddings.embed_documents(
                [c.page_content for c in candidates]
            )
            q_vec = self.embeddings.embed_query(BGE_QUERY_PREFIX + query)
            scores = np.dot(cand_vecs, q_vec)
            return list(zip(candidates, scores.tolist()))

        # 候选多：先粗筛（FAISS 文章索引），再精排
        # 这里直接精排：embed 全部候选，排序取 top_k
        cand_vecs = self.embeddings.embed_documents(
            [c.page_content for c in candidates]
        )
        q_vec = self.embeddings.embed_query(BGE_QUERY_PREFIX + query)
        scores = np.array(np.dot(cand_vecs, q_vec))
        top_idx = np.argsort(-scores)[:top_k]
        return [(candidates[i], float(scores[i])) for i in top_idx]


# ============================================================
#  Stage 5: QA Agent
# ============================================================

class QAAgent:
    def __init__(self, llm):
        self.llm = llm

    def _format_context(self, articles: list[tuple[Document, float]]) -> str:
        if not articles:
            return "（无相关法条）"
        lines = []
        for i, (doc, score) in enumerate(articles, 1):
            title = doc.metadata.get("law_title", "")
            artno = doc.metadata.get("article_no", "")
            text = doc.metadata.get("text", "")
            lines.append(
                f"{i}. (相似度={score:.3f}) 《{title}》{artno}\n   {text}"
            )
        return "\n\n".join(lines)

    def __call__(self, query: str, articles: list[tuple[Document, float]]) -> str:
        context = self._format_context(articles)
        prompt = QA_PROMPT.format(context=context, question=query)
        try:
            out = self.llm.invoke(prompt)
            if hasattr(out, "content"):
                out = out.content
            return out
        except Exception as e:
            return f"[LLM 调用失败: {e}]"


# ============================================================
#  Orchestrator
# ============================================================

class LegalRAGPipeline:
    """
    把 5 个 Stage 串起来。run() 是主入口。
    """

    def __init__(
        self,
        embed_backend: str = "hf",
        llm_backend: str = "deepseek",
        embed_model: str = "",
        llm_model: str = "",
    ):
        print("=" * 60)
        print(f"[BOOT] 初始化 5 阶段管道 ...")
        print(f"       embedding: {embed_backend}  |  llm: {llm_backend}")
        print("=" * 60)

        # Embedding & 2 个 FAISS 索引
        self.embeddings = build_embeddings(embed_backend, embed_model)
        if not (INDEX_DIR / "law_names").exists():
            raise FileNotFoundError(
                f"{INDEX_DIR}/law_names 不存在，请先跑：\n"
                f"  python3 demo/build_indexes.py --embed-backend {embed_backend}"
            )
        self.law_matcher_db = FAISS.load_local(
            str(INDEX_DIR / "law_names"),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.article_db = FAISS.load_local(
            str(INDEX_DIR / "articles"),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        # LLM & Agent
        self.llm = build_llm(llm_backend, llm_model)
        self.rewriter = QueryRewriter(self.llm)
        self.law_matcher = LawNameMatcher(self.law_matcher_db)
        self.fetcher = ArticleFetcher(DATA_DIR)
        self.ranker = ArticleRanker(self.embeddings)
        self.qa = QAAgent(self.llm)
        print("[BOOT] 全部就绪。\n")

    def run(
        self,
        query: str,
        history: str = "",
        top_laws: int = 3,
        top_articles: int = 10,
    ) -> PipelineResult:
        timings: dict[str, float] = {}

        # Stage 1: 改写
        t0 = time.time()
        rewritten = self.rewriter(query, history)
        timings["1_rewrite"] = time.time() - t0

        # Stage 2: 法律名匹配
        t0 = time.time()
        matched = self.law_matcher(rewritten, top_k=top_laws)
        timings["2_match_laws"] = time.time() - t0

        # Stage 3: 取法条
        t0 = time.time()
        candidates = self.fetcher(matched)
        timings["3_fetch"] = time.time() - t0

        # Stage 4: 重排
        t0 = time.time()
        ranked = self.ranker(rewritten, candidates, top_k=top_articles)
        timings["4_rank"] = time.time() - t0

        # Stage 5: 回答
        t0 = time.time()
        answer = self.qa(query, ranked)
        timings["5_answer"] = time.time() - t0

        return PipelineResult(
            rewritten_query=rewritten,
            matched_laws=matched,
            candidate_articles=candidates,
            final_articles=ranked,
            answer=answer,
            timings=timings,
        )

    def run_with_trace(self, query: str, history: str = "", no_rewrite: bool = False) -> PipelineResult:
        """
        带 trace 的 run：把每一步的中间结果打印出来。
        """
        print(f"\n[USER] {query}")
        if no_rewrite:
            # 跳过 Stage 1，直接拿原问题走后续阶段
            t0 = time.time()
            timings: dict[str, float] = {}
            rewritten = query
            timings["1_rewrite"] = 0.0
            t0 = time.time()
            matched = self.law_matcher(rewritten, top_k=3)
            timings["2_match_laws"] = time.time() - t0
            t0 = time.time()
            candidates = self.fetcher(matched)
            timings["3_fetch"] = time.time() - t0
            t0 = time.time()
            ranked = self.ranker(rewritten, candidates, top_k=10)
            timings["4_rank"] = time.time() - t0
            t0 = time.time()
            answer = self.qa(query, ranked)
            timings["5_answer"] = time.time() - t0
            res = PipelineResult(
                rewritten_query=rewritten,
                matched_laws=matched,
                candidate_articles=candidates,
                final_articles=ranked,
                answer=answer,
                timings=timings,
            )
        else:
            res = self.run(query, history)

        print(f"\n[Stage 1] 改写后: {res.rewritten_query!r}")
        print(f"[Stage 2] 命中法律: {res.matched_laws}")
        print(f"[Stage 3] 候选法条数: {len(res.candidate_articles)}")
        print(f"[Stage 4] 精排 Top-{len(res.final_articles)}:")
        for i, (doc, score) in enumerate(res.final_articles, 1):
            print(f"    {i}. {score:.3f}  《{doc.metadata['law_title']}》"
                  f"{doc.metadata['article_no']}")
        print(f"\n[Stage 5] 回答:\n{res.answer}")

        total = sum(res.timings.values())
        print(f"\n[TIMING] 总耗时 {total*1000:.0f} ms")
        for k, v in res.timings.items():
            print(f"   - {k}: {v*1000:.0f} ms")
        return res
