"""
会话短期记忆（Session Memory）

在当前 REPL 进程内维护结构化记忆，REPL 退出后自动清空。

容量策略：
  - history: 保留最新 5 轮原文
  - summaries: 超过容量后把早期 5 轮合并成一个摘要句
  - REPL 进程退出后所有数据随进程一起销毁（无持久化）
"""

from dataclasses import dataclass, field


@dataclass
class SessionMemory:
    history: list = field(default_factory=list)
    law_entities: set = field(default_factory=set)
    law_articles: list = field(default_factory=list)
    summaries: list = field(default_factory=list)
    last_answer: str = ""
    last_articles: list = field(default_factory=list)
    MAX_HISTORY: int = 5
    MAX_SUMMARIES: int = 10

    def record(self, question, answer, matched_laws, ranked_articles):
        """
        记录一轮问答。
        ranked_articles: list of (Document, score)
        """
        self.history.append({"q": question, "a": answer})
        if len(self.history) > self.MAX_HISTORY:
            self._compress()
        self.last_answer = answer
        self.law_entities.update(matched_laws)
        arts = []
        for doc, _score in ranked_articles[:10]:
            title = doc.metadata.get("law_title", "")
            artno = doc.metadata.get("article_no", "")
            arts.append((title, artno))
        self.last_articles = arts
        self.law_articles.extend(arts)
        if len(self.law_articles) > 200:
            self.law_articles = self.law_articles[-100:]

    def _compress(self):
        oldest = self.history[: self.MAX_HISTORY]
        self.history = self.history[self.MAX_HISTORY:]
        parts = []
        for turn in oldest:
            q = turn["q"][:20].replace("\n", " ").strip()
            a = turn["a"][:40].replace("\n", " ").strip()
            parts.append("Q" + q + " A" + a)
        self.summaries.append("; ".join(parts))
        if len(self.summaries) > self.MAX_SUMMARIES:
            self.summaries = self.summaries[-self.MAX_SUMMARIES:]

    def get_history_str(self):
        """拼成字符串，给 Stage 1 改写 prompt 用。"""
        parts = []
        if self.summaries:
            parts.append("[早期摘要]" + " | ".join(self.summaries))
        if self.history:
            lines = []
            for turn in self.history[-3:]:
                lines.append("Q: " + turn["q"])
                lines.append("A: " + turn["a"][:80])
            parts.append("[近期对话]\n" + "\n".join(lines))
        return "\n".join(parts)

    def get_law_entities(self):
        """返回涉及的法律名列表（保序去重）。"""
        seen, out = set(), []
        for law in self.law_entities:
            if law not in seen:
                seen.add(law)
                out.append(law)
        return out

    def get_last_article(self, article_no):
        for law, art in reversed(self.last_articles):
            if article_no in art:
                return law, art
        for law, art in reversed(self.law_articles):
            if article_no in art:
                return law, art
        return None, None

    def __len__(self):
        return len(self.history)
