"""
基于 demo/eval_set.json (100 条) 的自动化评测脚本
================================================================
评估三个核心指标：
  1) Stage 2 法律名命中率  (matched_laws 与 retrieved_text 中法律名匹配)
  2) Stage 4 法条召回率    (final_articles 覆盖 retrieved_text 中法条编号)
  3) 答案关键词覆盖率     (生成答案含 expected_answer 的核心实体词)

用法：
  python3 demo/run_eval.py \
    [--embed-backend ollama --embed-model bge-m3] \
    [--llm-backend ollama --llm-model qwen2.5:7b] \
    [--limit 100] \
    [--no-rewrite]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

# 兼容 python demo/run_eval.py 调用
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import LegalRAGPipeline, DATA_DIR  # noqa: E402


# ---------------- 工具函数 ----------------

# 从 retrieved_text 中提取法律名（如"《中华人民共和国草原法》"）
LAW_PATTERN = re.compile(r"《([^》]+)》")
# 从 retrieved_text 中提取法条编号（"第一百零八条"、"第三条"、"第十五条第二款"）
ARTICLE_PATTERN = re.compile(r"第[一二三四五六七八九十百千零〇0-9]+条(?:第[一二三四五六七八九十百千零〇0-9]+款)?(?:第[一二三四五六七八九十百千零〇0-9]+项)?")
# 法条编号归一化："第一百三十三条" -> "133条"
CN_NUM = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000,
}


def cn_to_int(s: str) -> Optional[int]:
    """把中文数字字符串转成 int，支持 '十''百' 之类。失败返 None。"""
    if not s:
        return None
    # 全阿拉伯数字直接 int
    if s.isdigit():
        return int(s)
    total, cur = 0, 0
    for ch in s:
        if ch not in CN_NUM:
            return None
        n = CN_NUM[ch]
        if n >= 10:
            if cur == 0:
                cur = 1
            total += cur * n
            cur = 0
        else:
            cur = n
    total += cur
    return total


def extract_law_names(text: str) -> list[str]:
    """从文本里提取所有《法律名》。"""
    return [m.group(1) for m in LAW_PATTERN.finditer(text or "")]


def extract_article_numbers(text: str) -> set[int]:
    """从文本里提取所有法条编号（归一化为阿拉伯数字）。"""
    nums: set[int] = set()
    for m in ARTICLE_PATTERN.finditer(text or ""):
        s = m.group(0)
        # 形如 "第三百一十五条"
        cn = s[1:].rstrip("条").rstrip("款").rstrip("项")
        cn = re.sub(r"(第[一二三四五六七八九十百千零〇0-9]+).+$", r"\1", cn)
        # 提取"第X条"中的 X（中文字符串）
        bare = re.sub(r"第", "", s)
        bare = re.sub(r"条.*$", "", bare)
        # 逐字转换
        digits = cn_to_int(bare)
        if digits is not None and digits > 0:
            nums.add(digits)
    return nums


def normalize_law_name(s: str) -> str:
    """去掉前缀/后缀做近似匹配。"""
    s = s.replace("中华人民共和国", "").replace("全国人民代表大会常务委员会", "")
    s = s.replace("最高人民法院", "").replace("最高人民检察院", "")
    s = s.strip()
    # 去掉"修正案""解释""补充"等限定
    for suffix in ["修正案", "解释", "补充", "决定"]:
        if suffix in s:
            s = s.split(suffix)[0]
    return s.strip()


def law_match_score(matched: list[str], expected: list[str]) -> tuple[int, int, list[str]]:
    """
    计算法律名命中。matched 和 expected 中的任意一对（归一化后）相同即视为命中。
    返回 (命中数, 期望命中数, 实际命中的法律名)
    """
    expected_norm = [normalize_law_name(l) for l in expected]
    matched_norm = [normalize_law_name(m) for m in matched]
    hits = []
    for en in expected_norm:
        for mn in matched_norm:
            if en == mn or (en and (en in mn or mn in en)):
                hits.append(mn)
                break
    # 去重保序
    seen, uniq = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h); uniq.append(h)
    return len(uniq), len(set(expected_norm)), uniq


def article_recall(ranked: list, expected_articles: set[int]) -> tuple[int, int]:
    """
    ranked 形如 [(Document, score), ...]，metadata 里有 article_no
    期望法条编号（int）有 N 个，看 Top10 里命中几个
    """
    if not expected_articles:
        return 0, 0
    got = 0
    for doc, _score in ranked[:10]:
        an = doc.metadata.get("article_no", "")
        digits = extract_article_numbers(an)
        for d in digits:
            if d in expected_articles:
                got += 1
                break
    return got, len(expected_articles)


def keyword_coverage(answer: str, expected: str) -> tuple[float, set[str]]:
    """
    计算 expected 里"关键词"被 answer 覆盖的比例。
    关键词定义：2 个汉字以上的非停用词串。
    """
    stopwords = {"的", "和", "或", "依", "据", "应当", "可以", "年", "月", "日",
                 "以", "为", "有", "在", "是", "由", "向", "对", "一", "二", "三", "四", "五",
                 "等", "或者", "以及", "第", "条", "款", "项"}
    # 抽取 expected 中 2-12 字的连续汉字片段作为关键词
    tokens = set()
    for m in re.finditer(r"[\u4e00-\u9fa5]{2,8}", expected):
        t = m.group(0)
        if t in stopwords or t.startswith("第"):
            continue
        tokens.add(t)
    if not tokens:
        return 1.0, set()
    covered = {t for t in tokens if t in (answer or "")}
    return len(covered) / len(tokens), covered


# ---------------- 主评测 ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-backend", default="ollama", choices=["hf", "ollama"])
    parser.add_argument("--embed-model", default="bge-m3")
    parser.add_argument("--llm-backend", default="ollama", choices=["deepseek", "ollama"])
    parser.add_argument("--llm-model", default="qwen2.5:7b")
    parser.add_argument("--limit", type=int, default=100, help="评测条数")
    parser.add_argument("--eval-file", default="demo/eval_set.json")
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument("--hyde", action="store_true", help="启用 HyDE 增强检索")
    parser.add_argument("--skip-uncovered", action="store_true", help="跳过数据集中不存在的法律和法条号（bad case）")
    args = parser.parse_args()

    print(f"[INIT] 评测配置：")
    print(f"  embed: {args.embed_backend}/{args.embed_model}")
    print(f"  llm:   {args.llm_backend}/{args.llm_model}")
    print(f"  评测文件: {args.eval_file}")
    print(f"  评测条数: {args.limit}")
    print(f"  HyDE:   {'启用' if args.hyde else '禁用'}")
    print(f"  跳过未覆盖: {'是' if args.skip_uncovered else '否'}")

    pipeline = LegalRAGPipeline(
        embed_backend=args.embed_backend,
        llm_backend=args.llm_backend,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
        enable_hyde=args.hyde,
    )

    # 加载数据集覆盖信息，用于过滤 bad case
    law_titles: set[str] = set()
    law_articles: dict[str, set[int]] = {}
    article_pat = re.compile(r"^第([一二三四五六七八九十百千零〇0-9]+)条")
    for fp in sorted(DATA_DIR.glob("laws_dataset_*.json")):
        for law in json.loads(fp.read_text(encoding="utf-8")):
            title = (law.get("title") or "").strip()
            if not title:
                continue
            law_titles.add(title)
            arts: set[int] = set()
            for art in law.get("articles", []):
                art = art.strip()
                if not art:
                    continue
                m = article_pat.match(art)
                if m:
                    n = cn_to_int(m.group(1))
                    if n is not None:
                        arts.add(n)
            law_articles.setdefault(title, arts)

    def normalize_title(s: str) -> str:
        for t in law_titles:
            if s in t or t in s:
                return t
        return s

    items = json.loads(Path(args.eval_file).read_text(encoding="utf-8"))
    items = items[: args.limit]
    skipped_ids: list[int] = []
    if args.skip_uncovered:
        filtered: list[dict] = []
        for item in items:
            rt = item.get("retrieved_text", "")
            expected_laws = extract_law_names(rt)
            expected_articles = extract_article_numbers(rt)
            # 检查法律是否存在
            matched_title = None
            for law in expected_laws:
                matched_title = normalize_title(law)
                if matched_title in law_titles:
                    break
            if expected_laws and matched_title not in law_titles:
                skipped_ids.append(item["id"])
                continue
            # 检查法条号是否存在
            missing_art = any(
                art_no not in law_articles.get(matched_title, set())
                for art_no in expected_articles
            )
            if expected_articles and missing_art:
                skipped_ids.append(item["id"])
                continue
            filtered.append(item)
        items = filtered
        print(f"  过滤后条数: {len(items)} (跳过 bad case: {len(skipped_ids)})")

    # 累计指标
    n_law_hit, n_law_expected = 0, 0
    n_art_hit, n_art_expected = 0, 0
    n_ans_cov_sum = 0.0
    n_total_latency_ms = 0.0
    n_ttft_sum = 0.0

    # 错误样本
    low_law_misses: list[tuple[int, str, list[str], list[str]]] = []
    low_art_misses: list[tuple[int, str, set[int], list[int]]] = []

    t_start = time.time()
    for i, item in enumerate(items, 1):
        q = item["question"]
        expected_laws = extract_law_names(item.get("retrieved_text", ""))
        expected_articles = extract_article_numbers(item.get("retrieved_text", ""))

        try:
            res = pipeline.run_with_trace(q, no_rewrite=args.no_rewrite) if args.no_rewrite else pipeline.run(q)
        except Exception as e:
            print(f"[{i:3d}] ERR: {e}")
            continue

        # Stage 2: 法律名
        law_hit, law_tot, hit_names = law_match_score(res.matched_laws, expected_laws)
        n_law_hit += law_hit
        n_law_expected += law_tot
        if law_tot > 0 and law_hit < law_tot:
            low_law_misses.append((item["id"], q, expected_laws, res.matched_laws))

        # Stage 4: 法条编号召回
        art_hit, art_tot = article_recall(res.final_articles, expected_articles)
        n_art_hit += art_hit
        n_art_expected += art_tot
        if art_tot > 0 and art_hit == 0:
            top10_nums = []
            for doc, _ in res.final_articles[:10]:
                top10_nums.extend(extract_article_numbers(doc.metadata.get("article_no", "")))
            low_art_misses.append((item["id"], q, expected_articles, top10_nums))

        # Stage 5: 答案关键词覆盖率
        cov, _ = keyword_coverage(res.answer, item["expected_answer"])
        n_ans_cov_sum += cov

        # 延迟
        n_total_latency_ms += sum(res.timings.values()) * 1000
        n_ttft_sum += res.ttft_ms

        if i % 10 == 0 or i == len(items):
            elapsed = time.time() - t_start
            print(f"[{i:3d}/{len(items)}] avg="
                  f"law={n_law_hit/max(n_law_expected,1):.2f} "
                  f"art={n_art_hit/max(n_art_expected,1):.2f} "
                  f"ans={n_ans_cov_sum/i:.2f} "
                  f"latency={(n_total_latency_ms/i):.0f}ms "
                  f"ttft={n_ttft_sum/i:.0f}ms "
                  f"total={elapsed:.0f}s")

    total_elapsed = time.time() - t_start

    # 最终汇总
    print("\n" + "=" * 60)
    print("[RESULT] 评测汇总")
    print("=" * 60)
    n = len(items)
    print(f"  评测条数       : {n}")
    if skipped_ids:
        print(f"  跳过 bad case  : {len(skipped_ids)} 条 (ids: {skipped_ids})")
    print(f"  总耗时         : {total_elapsed:.0f} s")
    print(f"  ─────────────────────────────────────────")
    print(f"  Stage 2 法律名命中率 : {n_law_hit}/{n_law_expected} "
          f"= {n_law_hit/max(n_law_expected,1)*100:.1f}%")
    print(f"  Stage 4 法条召回率   : {n_art_hit}/{n_art_expected} "
          f"= {n_art_hit/max(n_art_expected,1)*100:.1f}%")
    print(f"  Stage 5 答案覆盖率   : {n_ans_cov_sum/n*100:.1f}%")
    print(f"  ─────────────────────────────────────────")
    print(f"  平均端到端延迟 : {n_total_latency_ms/n:.0f} ms")
    print(f"  平均 TTFT      : {n_ttft_sum/n:.0f} ms")

    # 错误样本打印
    if low_law_misses:
        print(f"\n[FAIL] 法律名召回失败 {len(low_law_misses)} 条（前 5）：")
        for id_, q, exp, got in low_law_misses[:5]:
            print(f"  #{id_}: {q[:30]}...")
            print(f"     期望: {exp}")
            print(f"     实际: {got}")
    if low_art_misses:
        print(f"\n[FAIL] 法条编号完全未召回 {len(low_art_misses)} 条（前 5）：")
        for id_, q, exp, got in low_art_misses[:5]:
            print(f"  #{id_}: {q[:30]}...")
            print(f"     期望条号: {sorted(exp)}")
            print(f"     Top10 召回: {sorted(got)[:10]}")


if __name__ == "__main__":
    main()