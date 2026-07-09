"""
M7 单元测试：不调 API，只验 ConversationMemory 的滚动逻辑
========================================================

这个测试让你看清楚：
  - 摘要怎么"长出来"的
  - 最近 N 轮是怎么滚动的
  - get_context() 输出的格式

跑法：python3 m7_memory_test.py
"""

from m7_multi_agent_chat import ConversationMemory, Turn


class FakeSummarizer:
    """假装是 SummarizerAgent，不调 LLM，把内容拼起来。"""
    def __call__(self, current_summary: str, question: str, answer: str) -> str:
        prefix = (current_summary + " | ") if current_summary else ""
        return prefix + f"Q{hash(question)%1000}:{question[:8]}"


def main():
    mem = ConversationMemory(max_recent=2, summarizer=FakeSummarizer())

    turns = [
        ("国家对草原保护有什么方针？", "根据《草原法》第三条..."),
        ("它第三条具体说了什么？", "《草原法》第三条规定..."),
        ("违反这部法律的法律责任是什么？", "违反草原法..."),
        ("还有哪些类似规定？", "还有《森林法》《土地管理法》..."),
    ]

    for i, (q, a) in enumerate(turns, 1):
        print(f"\n===== 第 {i} 轮 =====")
        mem.add_turn(q, a)
        print(f"summary: {mem.summary[:80]!r}")
        print(f"recent : {[(t.question[:15], t.answer[:15]) for t in mem.recent]}")
        print(f"context: {mem.get_context()[:120]!r}")


if __name__ == "__main__":
    main()
