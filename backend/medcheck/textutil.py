"""文本工具：药名归一化、句子切分、精确引文校验。

这些函数全是确定性的——幻觉防线的一半在这里，另一半在提示词里。
"""

from __future__ import annotations

import re

# 常见盐基/剂型后缀：归一化时去掉，避免 "WARFARIN SODIUM" 匹配不上 "warfarin"
SALT_SUFFIXES: tuple[str, ...] = (
    "sodium", "hydrochloride", "hcl", "potassium", "calcium", "magnesium",
    "sulfate", "sulphate", "bitartrate", "tartrate", "citrate", "maleate",
    "mesylate", "besylate", "tosylate", "succinate", "fumarate", "phosphate",
    "acetate", "nitrate", "bromide", "chloride", "iodide", "oxide", "carbonate",
    "bicarbonate", "gluconate", "lactate", "oxalate", "stearate", "palmitate",
    "benzoate", "salicylate", "dihydrate", "monohydrate", "anhydrous",
    "hydrobromide", "hemihydrate", "decanoate", "enanthate", "valerate",
)

# 药名里无意义的通用词，参与匹配会误伤
STOPWORDS: frozenset[str] = frozenset(
    {
        "and", "or", "with", "the", "of", "for", "plus", "tablet", "tablets",
        "capsule", "capsules", "oral", "solution", "suspension", "injection",
        "cream", "gel", "ointment", "extended", "release", "delayed", "er",
        "sr", "xr", "ir", "mg", "ml", "mcg", "iu", "usp", "us", "drug",
        "medicine", "medication", "hydrochloride", "sodium",
    }
)

# 中文药名 → 英文通用名（只放演示/常用药，未知药名交给 openFDA 自己匹配）
ZH_TO_EN: dict[str, str] = {
    "华法林": "warfarin",
    "布洛芬": "ibuprofen",
    "阿司匹林": "aspirin",
    "乙酰水杨酸": "aspirin",
    "对乙酰氨基酚": "acetaminophen",
    "扑热息痛": "acetaminophen",
    "头孢地尼": "cefdinir",
    "头孢": "cefdinir",
    "阿莫西林": "amoxicillin",
    "克拉维酸": "amoxicillin and clavulanate potassium",
    "二甲双胍": "metformin",
    "奥美拉唑": "omeprazole",
    "氯吡格雷": "clopidogrel",
    "萘普生": "naproxen",
    "舍曲林": "sertraline",
    "曲马多": "tramadol",
    "泼尼松": "prednisone",
    "强的松": "prednisone",
    "阿托伐他汀": "atorvastatin",
    "左甲状腺素": "levothyroxine",
    "甲氨蝶呤": "methotrexate",
    "西地那非": "sildenafil",
    "万艾可": "sildenafil",
    "硝酸甘油": "nitroglycerin",
    "辛伐他汀": "simvastatin",
    "胺碘酮": "amiodarone",
    "地高辛": "digoxin",
    "呋塞米": "furosemide",
    "赖诺普利": "lisinopril",
    "氨氯地平": "amlodipine",
    "美托洛尔": "metoprolol",
    "阿普唑仑": "alprazolam",
    "地西泮": "diazepam",
    "安定": "diazepam",
    "氟西汀": "fluoxetine",
    "帕罗西汀": "paroxetine",
    "文拉法辛": "venlafaxine",
    "加巴喷丁": "gabapentin",
    "普瑞巴林": "pregabalin",
    "左氧氟沙星": "levofloxacin",
    "环丙沙星": "ciprofloxacin",
    "阿奇霉素": "azithromycin",
    "红霉素": "erythromycin",
    "克拉霉素": "clarithromycin",
    "利福平": "rifampin",
    "氟康唑": "fluconazole",
    "泼尼松龙": "prednisolone",
    "螺内酯": "spironolactone",
    "氢氯噻嗪": "hydrochlorothiazide",
    "别嘌醇": "allopurinol",
    "秋水仙碱": "colchicine",
    "碳酸锂": "lithium carbonate",
    "丙戊酸": "valproic acid",
    "卡马西平": "carbamazepine",
    "苯妥英": "phenytoin",
    "苯巴比妥": "phenobarbital",
    "胰岛素": "insulin",
    "格列本脲": "glyburide",
    "西咪替丁": "cimetidine",
    "雷尼替丁": "ranitidine",
    "多潘立酮": "domperidone",
    "昂丹司琼": "ondansetron",
    "氯苯那敏": "chlorpheniramine",
    "苯海拉明": "diphenhydramine",
    "氯雷他定": "loratadine",
    "西替利嗪": "cetirizine",
    "泼尼松滴眼液": "prednisolone",
}


def drop_salt(name: str) -> str:
    """去掉盐基后缀，但只在安全的情况下。

    规则（确定性、可解释）：只有当"去掉盐基词后只剩一个实词、且该词长度 >= 5"时才去掉。
    - "Warfarin Sodium"      -> "warfarin"    ✅（主词足够长，是真正的药名）
    - "Lithium Carbonate"    -> "lithium"     ✅（恰好也是有效查询名）
    - "Morphine Sulfate"     -> "morphine"    ✅
    - "Sodium Chloride"      -> 原样保留       ✅（去掉后只剩一个盐基词，无意义）
    """
    words = [w for w in (name or "").split() if w]
    if len(words) < 2:
        return name
    kept = [w for w in words if w.lower() not in SALT_SUFFIXES]
    if len(kept) == 1 and len(kept[0]) >= 5 and len(kept) != len(words):
        return kept[0]
    return name


def resolve_drug_name(name: str) -> tuple[str, bool]:
    """返回 (查询串, 是否由内建中文药名字典命中)。

    第二个值很重要：字典命中 = 我们**确实**认出了这个药名；
    没命中 = 只是原样拿去查 openFDA，不代表它是个药。
    """
    raw = (name or "").strip()
    if not raw:
        return "", False
    if re.search(r"[\u4e00-\u9fff]", raw):
        cleaned = re.sub(r"[\s（(].*$", "", raw)
        for zh, en in sorted(ZH_TO_EN.items(), key=lambda kv: -len(kv[0])):
            if zh in cleaned or cleaned in zh:
                return en, True
    return normalize_query(raw), False


# 中文里高频出现的"非药名"片段：出现在片段里就说明这不是一个药名
_ZH_STOP_FRAGMENTS: tuple[str, ...] = (
    "注意事项", "注意", "什么", "怎么", "怎样", "可以", "能不能", "一起吃",
    "冲突", "相互", "作用", "副作用", "说明", "用法", "用量", "剂量", "帮我",
    "查一下", "查询", "请问", "你好", "医生", "药师", "医院", "吃", "喝",
    "服用", "同时", "和", "与", "的", "吗", "呢", "了", "我", "你", "他",
    "安全", "危险", "风险", "影响", "一起", "两", "个",
)


def is_probable_drug_name(candidate: str) -> bool:
    """粗判一段文本"像不像一个药名"。

    目的不是精确识别，而是**宁可反问用户、也不要瞎猜一个药名去查**。
    """
    c = (candidate or "").strip()
    if not c:
        return False
    if re.search(r"[\u4e00-\u9fff]", c):
        if any(frag in c for frag in _ZH_STOP_FRAGMENTS):
            return False
        # 中文药名很少超过 12 个字；超长基本是一句话
        return len(c) <= 12
    if len(c) < 3 or len(c) > 48:
        return False
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9\s\-']*$", c):
        return False
    words = c.split()
    if len(words) > 4:
        return False
    # 纯数字/纯剂量残留
    if not any(ch.isalpha() for ch in c):
        return False
    return True


def normalize_query(name: str) -> str:
    """把用户输入变成能喂给 openFDA 的查询串。"""
    raw = (name or "").strip()
    if not raw:
        return ""
    # 中文药名先查字典（按长度倒序，优先匹配长词，如"阿莫西林克拉维酸"）
    if re.search(r"[\u4e00-\u9fff]", raw):
        cleaned = re.sub(r"[\s（(].*$", "", raw)
        for zh, en in sorted(ZH_TO_EN.items(), key=lambda kv: -len(kv[0])):
            if zh in cleaned or cleaned in zh:
                return en
    # 去掉常见剂型/规格噪声
    cleaned = re.sub(r"[（(].*?[)）]", " ", raw)
    cleaned = re.sub(
        r"\d+(\.\d+)?\s*(mg|ml|mcg|ug|g|iu|%|毫克|微克|克|片|粒)",
        " ",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"[\d.]+$", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -,")
    return drop_salt(cleaned) or raw


def name_stems(name: str) -> set[str]:
    """抽取可用于匹配的词干集合。

    "WARFARIN SODIUM" -> {"warfarin", "warfarin sodium"}
    """
    if not name:
        return set()
    low = re.sub(r"[^a-z0-9\s\-]", " ", name.lower())
    low = re.sub(r"\s+", " ", low).strip()
    if not low:
        return set()
    stems: set[str] = set()
    words = [w for w in low.split() if w]
    if low:
        stems.add(low)
    # 去掉盐基后的主词
    kept = [w for w in words if w not in {s for s in SALT_SUFFIXES}]
    if kept:
        stems.add(" ".join(kept))
        if len(kept) > 1:
            stems.add(" ".join(kept[:2]))
    # 单个词，长度 >= 5 才留（防止 "and" / "er" 这类误伤）
    for w in words:
        if len(w) >= 5 and w not in STOPWORDS:
            stems.add(w)
    return {s for s in stems if len(s) >= 4 and s not in STOPWORDS}


# 段内小节标题（"7.4 Antibiotics and Antifungals"）也应该切句，
# 否则整张表格 + 下一节会被当成"一句话"，引文会长到没法读。
SECTION_MARKER = re.compile(r"(?<=[0-9])\s+(?=[0-9]{1,2}(?:\.[0-9]{1,2}){1,2}\s+[A-Z])")
CLAUSE_BREAK = re.compile(r"(?<=[a-z0-9\)\]])\s+(?=(?:Table|Figure)\s+[0-9])")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """按句切分，返回每句在原文中的 (start, end)。

    切分规则刻意保守：只在 ". "、"; "、"\\n" 之后断句，避免把
    "5 mg." 这种剂量切碎导致引文不再是原文子串。另外额外在小节标题
    （"7.4 Antibiotics ..."）与表格标题（"Table 3: ..."）前断句。
    返回的每个 span 都保证是原文的**连续子串**。
    """
    if not text:
        return []
    cuts: list[int] = [0]
    for inner in (SECTION_MARKER, CLAUSE_BREAK):
        for m in inner.finditer(text):
            cuts.append(m.start())
    for i, ch in enumerate(text[:-1]):
        if ch == "\n" or (ch in ".;" and text[i + 1] in " \n\t"):
            cuts.append(i + 1)
    cuts = sorted({c for c in cuts if 0 <= c <= len(text)})
    if cuts[-1] != len(text):
        cuts.append(len(text))

    out: list[tuple[int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        s, e = a, b
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s and len(text[s:e]) >= 8:
            out.append((s, e))
    return out


def find_sentences(text: str, patterns: list[str], limit: int = 3) -> list[str]:
    """返回原文中命中任一 pattern 的整句（原样返回，保证可逐字校验）。"""
    return [s for s, _ in find_sentence_spans(text, patterns, limit)]


def compile_terms(terms: list[str]) -> list[re.Pattern[str]]:
    """把词干/关键词编译成"词首匹配"的正则（warfarin 能命中 warfarin sodium）。"""
    out: list[re.Pattern[str]] = []
    for term in terms or []:
        term = (term or "").strip()
        if len(term) < 4:
            continue
        try:
            out.append(re.compile(r"\b" + re.escape(term) + r"\w*", re.I))
        except re.error:
            continue
    return out


def clip_around(text: str, index: int, budget: int) -> str:
    """在 index 附近取一段**原文子串**（保留命中词），并尽量对齐到词边界。

    用来说明书表格那种超长"句子"：引用必须仍然逐字可核，但要短到能读。
    """
    if len(text) <= budget:
        return text
    left = max(0, index - budget // 2)
    right = min(len(text), left + budget)
    left = max(0, right - budget)
    # 对齐到最近的空格，避免从词中间切断
    if left > 0:
        space = text.find(" ", left, min(index, left + 40))
        if space != -1:
            left = space + 1
    if right < len(text):
        space = text.rfind(" ", max(index, right - 40), right)
        if space != -1:
            right = space
    return text[left:right].strip()


def find_sentence_spans(
    text: str, patterns: list[str], limit: int = 3, budget: int = 0
) -> list[tuple[str, tuple[int, int]]]:
    """同上，但同时返回句子在原文中的位置（用于登记"句子级证据"）。

    budget > 0 时，超长句子会被裁成命中词附近的窗口（仍是原文子串）。
    """
    if not text or not patterns:
        return []
    compiled = compile_terms(patterns)
    if not compiled:
        return []
    hits: list[tuple[str, tuple[int, int]]] = []
    seen: set[str] = set()
    for start, end in sentence_spans(text):
        sentence = text[start:end]
        match = next((c.search(sentence) for c in compiled if c.search(sentence)), None)
        if match is None:
            continue
        key = sentence.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        if budget and len(sentence) > budget:
            clipped = clip_around(sentence, match.start(), budget)
            offset = sentence.find(clipped)
            if offset >= 0:
                hits.append((clipped, (start + offset, start + offset + len(clipped))))
            else:
                hits.append((sentence, (start, end)))
        else:
            hits.append((sentence, (start, end)))
        if len(hits) >= limit:
            break
    return hits


def is_verbatim(quote: str, sources: list[str]) -> bool:
    """引文必须是某段官方原文的**逐字子串**（允许空白归一化）。

    这是最硬的一道防线：模型编造的话不可能在这里通过。
    """
    if not quote or not quote.strip():
        return False
    needle = collapse_ws(quote)
    if len(needle) < 8:
        return False
    for src in sources:
        if needle in collapse_ws(src):
            return True
    return False


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clip(text: str, limit: int) -> str:
    text = collapse_ws(text)
    return text if len(text) <= limit else text[:limit].rstrip() + " …"
