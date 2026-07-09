"""
demo/build_indexes.py
=====================
一次性脚本：构建 2 个 FAISS 索引。

  1) law_names/  - 303 部法律名的索引（用于"先找法律"）
  2) articles/   - 22482 条法条的索引（用于"再找法条"）

支持两种 embedding 后端：
  - hf      ：HuggingFace BAAI/bge-small-zh-v1.5（默认）
  - ollama  ：本地 Ollama，例如 nomic-embed-text

使用：
  python3 demo/build_indexes.py
  python3 demo/build_indexes.py --embed-backend ollama
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ---- 路径 ----
DEMO_DIR = Path(__file__).parent
PROJECT_ROOT = DEMO_DIR.parent
DATA_DIR = PROJECT_ROOT / "law_clearnerdata"
INDEX_DIR = DEMO_DIR / "indexes"

ARTICLE_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+条")
MAX_EMBED_CHARS = 1200


# ---- 1. 读取所有 laws ----

def load_unique_laws(data_dir: Path) -> list[dict]:
    """
    从 JSON 中提取每部法律的「唯一一份」数据。
    返回 [{title, articles: [...], source_file, source_path}, ...]
    """
    seen: dict[str, dict] = {}
    for fp in sorted(data_dir.glob("laws_dataset_*.json")):
        for law in json.loads(fp.read_text(encoding="utf-8")):
            title = (law.get("title") or "").strip()
            if not title:
                continue
            if title not in seen:
                seen[title] = {
                    "title": title,
                    "articles": law.get("articles", []),
                    "source_file": law.get("source_file", ""),
                    "source_path": law.get("source_path", ""),
                }
    return list(seen.values())


# ---- 2. 构造 Document ----

def make_law_name_doc(law: dict) -> Document:
    """
    把"法律名"做成 Document。
    搜索法律名时只对 title 编码，所以 page_content = title。
    """
    return Document(
        page_content=law["title"],
        metadata={"law_title": law["title"]},
    )


def build_article_embedding_text(
    law_title: str,
    article_no: str,
    article_text: str,
    max_chars: int = MAX_EMBED_CHARS,
) -> str:
    """
    构造“用于向量化”的法条文本。

    Ollama embedding 模型有上下文长度限制。少数法条极长，直接送入会报：
      the input length exceeds the context length

    这里保留：
      - 法律名
      - 条号
      - 正文前 max_chars 个字符

    metadata 里仍然保留完整原文，供最终回答引用。
    """
    prefix = f"《{law_title}》{article_no}　"
    text = article_text.strip()
    if len(text) <= max_chars:
        return prefix + text
    return prefix + text[:max_chars].rstrip() + "……"


def make_article_docs(law: dict) -> list[Document]:
    """
    把一部法律的每条法条做成 Document。
    每条 page_content = "《法律名》第X条　法条正文"。
    """
    docs = []
    for art in law["articles"]:
        art = art.strip()
        if not art:
            continue
        m = ARTICLE_PATTERN.match(art)
        article_no = m.group(0) if m else ""
        page_content = build_article_embedding_text(
            law_title=law["title"],
            article_no=article_no,
            article_text=art,
        )
        docs.append(Document(
            page_content=page_content,
            metadata={
                "law_title": law["title"],
                "article_no": article_no,
                "text": art,
                "source_file": law["source_file"],
                "source_path": law["source_path"],
            },
        ))
    return docs


# ---- 3. Embedding 后端工厂 ----

def build_embeddings(backend: str, model: str = ""):
    """
    factory：返回一个 LangChain Embeddings 对象。
    HF 和 Ollama 都遵循 LangChain Embeddings 协议，下游代码一致。
    """
    if backend == "hf":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model or "BAAI/bge-small-zh-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        )
    elif backend == "ollama":
        import os
        import time
        import json
        import subprocess

        class RobustOllamaEmbeddings:
            """
            健壮的 Ollama embedding 客户端：
              - 用 subprocess 调 curl（绕过 sandbox 对 Python HTTP 的限制）
              - 串行请求 + 自动 retry
            遵循 LangChain Embeddings 协议（embed_documents / embed_query）
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

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                # 用 Ollama 的 /api/embed 批量端点 + subprocess+curl
                # 走 curl 而不是 urllib，是因为 sandbox 下 Python HTTP 不稳
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
                            print(f"      [batch {i//BATCH+1} try {attempt+1}/3] {type(e).__name__}: {e}")
                            time.sleep(2)
                    if not ok:
                        raise RuntimeError(f"embed batch 失败 @ {i}/{total}: {last_err}")
                    done = min(i + BATCH, total)
                    if done % (BATCH * 4) == 0 or done == total:
                        print(f"      encoded {done}/{total}")
                return out

            def embed_query(self, text: str) -> list[float]:
                return self._post(text)

            # 新版 LangChain 要求 Embeddings 对象本身可调用
            def __call__(self, text: str) -> list[float]:
                return self._post(text)

        return RobustOllamaEmbeddings(
            model=model or "nomic-embed-text",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    else:
        raise ValueError(f"unknown embed backend: {backend}")


# ---- 4. 主流程 ----

def main():
    parser = argparse.ArgumentParser(description="构建 demo 用的 FAISS 索引")
    parser.add_argument(
        "--embed-backend", default="hf", choices=["hf", "ollama"],
        help="embedding 后端：hf（HuggingFace BGE）或 ollama",
    )
    parser.add_argument("--embed-model", default="", help="覆盖默认 embedding 模型")
    parser.add_argument("--limit-laws", type=int, default=0,
                        help="只建前 N 部法律的索引（调试用，0=全部）")
    parser.add_argument("--reset", action="store_true",
                        help="删除已有 indexes/ 再重建")
    args = parser.parse_args()

    if args.reset and INDEX_DIR.exists():
        import shutil
        shutil.rmtree(INDEX_DIR)
        print(f"[INFO] 已清空 {INDEX_DIR}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 加载数据
    print(f"[INFO] 加载 {DATA_DIR} ...")
    laws = load_unique_laws(DATA_DIR)
    if args.limit_laws > 0:
        laws = laws[: args.limit_laws]
    print(f"[INFO] 共 {len(laws)} 部法律")

    # 2) Embedding 模型
    print(f"[INFO] 加载 embedding 模型 ({args.embed_backend}) ...")
    embeddings = build_embeddings(args.embed_backend, args.embed_model)

    # 3) 法律名索引
    print(f"[1/2] 构建法律名索引 ...")
    law_name_docs = [make_law_name_doc(l) for l in laws]
    law_name_db = FAISS.from_documents(law_name_docs, embeddings)
    law_name_db.save_local(str(INDEX_DIR / "law_names"))
    print(f"      → 写入 {INDEX_DIR / 'law_names'} ({len(law_name_docs)} 条)")

    # 4) 法条索引
    print(f"[2/2] 构建法条索引 ...")
    article_docs: list[Document] = []
    for law in laws:
        article_docs.extend(make_article_docs(law))
    truncated_count = sum(
        1
        for doc in article_docs
        if doc.page_content.endswith("……")
    )
    if truncated_count:
        print(
            f"      [INFO] {truncated_count} 条超长法条已截断到前 {MAX_EMBED_CHARS} 字用于向量化"
        )
    article_db = FAISS.from_documents(article_docs, embeddings)
    article_db.save_local(str(INDEX_DIR / "articles"))
    print(f"      → 写入 {INDEX_DIR / 'articles'} ({len(article_docs)} 条)")

    print()
    print("[DONE] 索引构建完成。下一步：")
    print("       python3 demo/run_demo.py -q '单位拖欠工资怎么办？'")


if __name__ == "__main__":
    main()
