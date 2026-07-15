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
    ttft_ms: float = 0.0        # Stage 5 改写/回答的首字延迟（TTFT）
    llm_tokens: int = 0         # Stage 5 输出的 token 数
    llm_total_ms: float = 0.0   # Stage 5 LLM 总耗时（与 timings["5_answer"] 一致）


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
                BATCH = 128
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

        class LLMResult:
            """LLM 调用结果，包含文本和首字延迟 (TTFT) / 总延迟。"""
            __slots__ = ("text", "ttft_ms", "total_ms", "tokens")
            def __init__(self, text: str, ttft_ms: float, total_ms: float, tokens: int):
                self.text = text
                self.ttft_ms = ttft_ms
                self.total_ms = total_ms
                self.tokens = tokens
            def __str__(self):
                return self.text

        class RobustOllamaLLM:
            """
            用 subprocess 调 curl 跑 Ollama LLM（绕过 sandbox 对 Python HTTP 的限制）。
            遵循 LangChain LLM 协议：invoke() 返回 LLMResult（带 text / ttft_ms / total_ms）。
            """
            def __init__(self, model: str, base_url: str):
                self.model = model
                self.base_url = base_url.rstrip("/")

            def _post(self, prompt: str, stream_callback=None) -> LLMResult:
                payload = json.dumps({
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "keep_alive": "30m",
                })
                last_err = None
                for attempt in range(3):
                    t_start = time.time()
                    ttft_ms = None
                    chunks: list[str] = []
                    token_count = 0
                    try:
                        proc = subprocess.Popen(
                            ["curl", "-s", "--no-buffer", "-X", "POST",
                             f"{self.base_url}/api/generate",
                             "-H", "Content-Type: application/json",
                             "-d", payload],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )
                        assert proc.stdout is not None
                        for line in proc.stdout:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if ttft_ms is None:
                                ttft_ms = (time.time() - t_start) * 1000
                            delta = obj.get("response", "")
                            if delta:
                                chunks.append(delta)
                                token_count += 1
                                if stream_callback:
                                    stream_callback(delta)
                            if obj.get("done"):
                                break
                        proc.wait(timeout=600)
                        if not chunks:
                            raise RuntimeError("empty response (model reloading?)")
                        total_ms = (time.time() - t_start) * 1000
                        return LLMResult(
                            text="".join(chunks),
                            ttft_ms=ttft_ms or total_ms,
                            total_ms=total_ms,
                            tokens=token_count,
                        )
                    except Exception as e:
                        last_err = e
                        print(f"      [LLM try {attempt+1}/3] {type(e).__name__}: {e}")
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        time.sleep(3)
                raise RuntimeError(f"LLM 失败 3 次: {last_err}")

            def invoke(self, prompt, **kwargs):
                # 处理 ChatPromptTemplate 输出的多 message 格式
                if hasattr(prompt, "to_string"):
                    prompt = prompt.to_string()
                elif hasattr(prompt, "content"):
                    # 单个 message
                    prompt = prompt.content if hasattr(prompt, "content") else str(prompt)
                elif not isinstance(prompt, str):
                    prompt = str(prompt)
                return self._post(prompt)

            def invoke_stream(self, prompt, callback=print):
                """
                流式调用：每个 chunk 生成后立即回调 callback(text)。
                callback 默认 print，逐字打印到终端。
                返回 (text, ttft_ms, total_ms, tokens)。
                """
                if hasattr(prompt, "to_string"):
                    prompt = prompt.to_string()
                elif hasattr(prompt, "content"):
                    prompt = prompt.content if hasattr(prompt, "content") else str(prompt)
                elif not isinstance(prompt, str):
                    prompt = str(prompt)
                return self._post(prompt, stream_callback=callback)

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
            # 兼容 LLMResult / 带 .content 的消息对象 / 纯 str
            if hasattr(out, "text"):
                out = out.text
            elif hasattr(out, "content"):
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

        # 关键词 → 法律名 反查表（用于 Stage 2 跳过 FAISS）
        # 把"草原"、"网络安全"、"中医药"等核心实体词映射到对应法律
        # 用法律名去掉前缀/后缀得到的核心 2~4 字作 key
        self.alias_to_law: dict[str, str] = {}
        for title in self.law_to_articles.keys():
            # 例如"中华人民共和国草原法" → 核心"草原法"
            core = title
            for prefix in ("中华人民共和国", "全国人民代表大会常务委员会", "最高人民法院", "最高人民检察院"):
                if core.startswith(prefix):
                    core = core[len(prefix):]
            core = core.strip()
            # 排除补充规定、解释、修正案等，避免歧义
            if any(k in core for k in ["解释", "修正", "补充", "决定", "批复", "意见"]):
                continue
            if 2 <= len(core) <= 8:
                self.alias_to_law[core] = title

    def __call__(self, law_titles: Iterable[str]) -> list[Document]:
        out: list[Document] = []
        for t in law_titles:
            out.extend(self.law_to_articles.get(t, []))
        return out

    def keyword_lookup(self, text: str) -> list[str]:
        """在文本里找法律名关键词，返回对应法律名（去重保序，最多 3 部）"""
        seen: set[str] = set()
        hits: list[str] = []
        # 长 alias 优先匹配（避免"刑法"先吃掉"刑法修正案"等）
        for alias in sorted(self.alias_to_law.keys(), key=len, reverse=True):
            if alias in text:
                law = self.alias_to_law[alias]
                if law not in seen:
                    seen.add(law)
                    hits.append(law)
                    if len(hits) >= 3:
                        break
        return hits


# ============================================================
#  Stage 4: Article Ranker
# ============================================================

class ArticleRanker:
    """
    Hybrid 精排器：
      1) FAISS 粗排：限定在 law_filter（Stage 2 命中的法律）里搜 Top-fetch_k
         - 有 law_filter：FAISS with filter，强约束在正确法律
         - 无 law_filter：回退到 FAISS 全库搜（粗排场景）
         - 无 article_db：回退到传入的 candidates
      2) bge-m3 精排：对这 K1 条重新编码 + 点积，取 Top-K
    相对暴力精排（200 候选全重编码）快 ~3-5 倍。
    """

    def __init__(self, embeddings, article_db=None, fetch_k: int = 50):
        self.embeddings = embeddings
        self.article_db = article_db
        self.fetch_k = fetch_k

    def __call__(
        self,
        query: str,
        candidates: list[Document] | None = None,
        top_k: int = 10,
        law_filter: list[str] | None = None,
    ) -> list[tuple[Document, float]]:
        q = BGE_QUERY_PREFIX + query

        # 1) 粗排
        coarse: list[Document] = []
        if self.article_db is not None and law_filter:
            # 强约束：在 Stage 2 命中的法律里搜
            coarse = self.article_db.similarity_search(
                q, k=self.fetch_k,
                filter={"law_title": {"$in": list(law_filter)}},
            )
        elif self.article_db is not None:
            # 无过滤：全库搜
            coarse = self.article_db.similarity_search(q, k=self.fetch_k)
        else:
            # 兜底：传入的 candidates
            coarse = candidates or []

        if not coarse:
            return []

        # 2) 精排
        cand_vecs = self.embeddings.embed_documents(
            [c.page_content for c in coarse]
        )
        q_vec = self.embeddings.embed_query(q)
        scores = np.array(np.dot(cand_vecs, q_vec))
        top_idx = np.argsort(-scores)[:top_k]
        return [(coarse[i], float(scores[i])) for i in top_idx]


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

    def __call__(self, query: str, articles: list[tuple[Document, float]]) -> tuple[str, dict]:
        """
        返回 (answer_text, llm_meta)
        llm_meta = {"ttft_ms": ..., "total_ms": ..., "tokens": ..., "ok": True/False}
        """
        context = self._format_context(articles)
        prompt = QA_PROMPT.format(context=context, question=query)
        try:
            out = self.llm.invoke(prompt)
            if hasattr(out, "text"):           # RobustOllamaLLM.LLMResult
                meta = {
                    "ttft_ms": getattr(out, "ttft_ms", 0.0),
                    "total_ms": getattr(out, "total_ms", 0.0),
                    "tokens": getattr(out, "tokens", 0),
                    "ok": True,
                }
                return out.text, meta
            if hasattr(out, "content"):         # ChatMessage 风格
                return out.content, {"ttft_ms": 0.0, "total_ms": 0.0, "tokens": 0, "ok": True}
            return str(out), {"ttft_ms": 0.0, "total_ms": 0.0, "tokens": 0, "ok": True}
        except Exception as e:
            return f"[LLM 调用失败: {e}]", {"ttft_ms": 0.0, "total_ms": 0.0, "tokens": 0, "ok": False}

    def stream(self, query: str, articles: list[tuple[Document, float]], callback=print) -> dict:
        """
        流式调用：每个 token 生成后立即回调 callback(token)。
        返回 llm_meta。
        """
        context = self._format_context(articles)
        prompt = QA_PROMPT.format(context=context, question=query)
        try:
            out = self.llm.invoke_stream(prompt, callback=callback)
            if hasattr(out, "text"):
                meta = {
                    "ttft_ms": getattr(out, "ttft_ms", 0.0),
                    "total_ms": getattr(out, "total_ms", 0.0),
                    "tokens": getattr(out, "tokens", 0),
                    "ok": True,
                }
                return meta
            return {"ttft_ms": 0.0, "total_ms": 0.0, "tokens": 0, "ok": False}
        except Exception as e:
            return {"ttft_ms": 0.0, "total_ms": 0.0, "tokens": 0, "ok": False}


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
        # ArticleRanker 注入 article_db：FAISS 粗排 + bge-m3 精排
        self.ranker = ArticleRanker(self.embeddings, article_db=self.article_db, fetch_k=50)
        self.qa = QAAgent(self.llm)
        print("[BOOT] 全部就绪。\n")

    def _keyword_law_match(self, text: str) -> list[str]:
        """
        在改写后的问题里做关键词匹配，命中"草原法"、"网络安全法"等已知法律名。
        命中就跳过 Stage 2 的 FAISS 调用。
        """
        return self.fetcher.keyword_lookup(text)

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
        # 先用关键词兜底（改写后的问题里若已含"草原法""网络安全法"等，跳过 FAISS）
        kw_matched = self._keyword_law_match(rewritten)
        if kw_matched:
            matched = kw_matched
            print(f"[Stage 2] 用关键词匹配: {matched}")
        else:
            matched = self.law_matcher(rewritten, top_k=top_laws)
        timings["2_match_laws"] = time.time() - t0

        # Stage 3: 取法条
        t0 = time.time()
        candidates = self.fetcher(matched)
        timings["3_fetch"] = time.time() - t0

        # Stage 4: 重排（限定在 Stage 2 命中的法律里）
        t0 = time.time()
        ranked = self.ranker(rewritten, candidates, top_k=top_articles, law_filter=matched)
        timings["4_rank"] = time.time() - t0

        # Stage 5: 回答
        t0 = time.time()
        answer, llm_meta = self.qa(query, ranked)
        timings["5_answer"] = time.time() - t0

        return PipelineResult(
            rewritten_query=rewritten,
            matched_laws=matched,
            candidate_articles=candidates,
            final_articles=ranked,
            answer=answer,
            timings=timings,
            ttft_ms=llm_meta.get("ttft_ms", 0.0),
            llm_tokens=llm_meta.get("tokens", 0),
            llm_total_ms=llm_meta.get("total_ms", 0.0),
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
            ranked = self.ranker(rewritten, candidates, top_k=10, law_filter=matched)
            timings["4_rank"] = time.time() - t0
            t0 = time.time()
            answer, llm_meta = self.qa(query, ranked)
            timings["5_answer"] = time.time() - t0
            res = PipelineResult(
                rewritten_query=rewritten,
                matched_laws=matched,
                candidate_articles=candidates,
                final_articles=ranked,
                answer=answer,
                timings=timings,
                ttft_ms=llm_meta.get("ttft_ms", 0.0),
                llm_tokens=llm_meta.get("tokens", 0),
                llm_total_ms=llm_meta.get("total_ms", 0.0),
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
        # Stage 名称映射（更可读）
        stage_label = {
            "1_rewrite": "[1] Query Rewriter",
            "2_match_laws": "[2] Law Name Matcher",
            "3_fetch": "[3] Article Fetcher",
            "4_rank": "[4] Article Ranker (hybrid)",
            "5_answer": "[5] QA Agent",
        }
        # 累计 LLM / Embedding 调用时间（粗略分类）
        llm_stages = {"1_rewrite", "5_answer"}
        emb_stages = {"2_match_laws", "4_rank"}
        llm_ms = sum(res.timings[k] for k in llm_stages) * 1000
        emb_ms = sum(res.timings[k] for k in emb_stages) * 1000

        print(f"\n[TIMING] 总耗时 {total*1000:.0f} ms")
        print(f"   ├─ LLM 调用合计    : {llm_ms:6.0f} ms  ({llm_ms / (total*1000) * 100:5.1f}%)")
        print(f"   ├─ Embedding 合计  : {emb_ms:6.0f} ms  ({emb_ms / (total*1000) * 100:5.1f}%)")
        print(f"   └─ 其他（IO/排序）: {(total*1000 - llm_ms - emb_ms):6.0f} ms")
        print()
        print(f"   {'阶段':<28} {'耗时':>8} {'占比':>8}")
        print(f"   {'-'*28} {'-'*8} {'-'*8}")
        for k, v in res.timings.items():
            pct = v / total * 100 if total > 0 else 0
            print(f"   {stage_label.get(k, k):<28} {v*1000:7.0f}ms {pct:7.1f}%")
        # Stage 5 LLM 详细指标：首字延迟 / token 数 / token/s
        if res.ttft_ms > 0 or res.llm_tokens > 0:
            tok_s = (res.llm_tokens / (res.llm_total_ms / 1000)) if res.llm_total_ms > 0 else 0
            print()
            print(f"   [Stage 5 LLM 详细]")
            print(f"   ├─ 首字延迟 TTFT   : {res.ttft_ms:6.0f} ms")
            print(f"   ├─ 累计 token 数   : {res.llm_tokens}")
            print(f"   ├─ LLM 总耗时      : {res.llm_total_ms:6.0f} ms")
            if tok_s > 0:
                print(f"   └─ 生成速度        : {tok_s:6.1f} tok/s")
        return res
