"""中英双语标签层。

设计：**内部判定值保持英文常量**（critical / high / low / unknown），
只在渲染时映射成人类语言。这样等级比较、单调约束的代码完全不需要知道语言。

中文是主语言（完整报告），英文用于 `main_sample.py` 的简洁版输出。
"""

from __future__ import annotations

from .models import Severity, Verdict

# ---------------------------------------------------------------- 结论档（四档）

VERDICT_ZH: dict[str, str] = {
    Verdict.CRITICAL: "危险",
    Verdict.HIGH: "注意",
    Verdict.LOW: "未发现直接冲突证据",
    Verdict.UNKNOWN: "证据不足",
}

VERDICT_EN: dict[str, str] = {
    Verdict.CRITICAL: "DANGER",
    Verdict.HIGH: "CAUTION",
    Verdict.LOW: "NO DIRECT CONFLICT FOUND",
    Verdict.UNKNOWN: "UNKNOWN",
}

# 中文简洁标签（只有简洁版用；LOW 显示为「安全」，但**永远**要带边界说明）
VERDICT_SHORT_ZH: dict[str, str] = {
    Verdict.CRITICAL: "危险",
    Verdict.HIGH: "注意",
    Verdict.LOW: "安全",
    Verdict.UNKNOWN: "证据不足",
}

# 一句话解释每一档到底在说什么（英文版必须把"安全"的边界一起说掉）
VERDICT_GLOSS_EN: dict[str, str] = {
    Verdict.CRITICAL:
        "Official labeling contains a serious, explicit statement about this combination.",
    Verdict.HIGH:
        "Official labeling flags this combination in warnings or interaction sections.",
    Verdict.LOW:
        "No direct interaction is recorded in the official labels retrieved. "
        "Absence of a record is NOT proof of safety.",
    Verdict.UNKNOWN:
        "Insufficient official data to assess this combination at all.",
}

# ★ 这一档永远不能只印"安全"两个字——必须带边界说明
VERDICT_QUALIFIER_ZH = "（不等于已证实安全）"
VERDICT_QUALIFIER_EN = "(safety not established)"

SEVERITY_ZH: dict[str, str] = {
    Severity.CRITICAL: "严重",
    Severity.HIGH: "重要",
    Severity.MEDIUM: "关注",
    Severity.INFO: "提示",
}

SEVERITY_EN: dict[str, str] = {
    Severity.CRITICAL: "Critical",
    Severity.HIGH: "High",
    Severity.MEDIUM: "Medium",
    Severity.INFO: "Info",
}


def verdict_label(
    verdict: str, lang: str = "zh", with_qualifier: bool = False, short: bool = False
) -> str:
    """取结论档的显示名。

    short=True 时中文用"安全"这个短标签（简洁版用），
    但只要命中 LOW 档，with_qualifier 必须同时为 True —— 由调用方保证。
    """
    if lang == "en":
        label = VERDICT_EN.get(verdict, verdict)
        return f"{label} {VERDICT_QUALIFIER_EN}" if with_qualifier else label
    table = VERDICT_SHORT_ZH if short else VERDICT_ZH
    label = table.get(verdict, verdict)
    return f"{label}{VERDICT_QUALIFIER_ZH}" if with_qualifier else label


def severity_label(severity: str, lang: str = "zh") -> str:
    table = SEVERITY_EN if lang == "en" else SEVERITY_ZH
    return table.get(severity, severity)


def verdict_gloss(verdict: str, lang: str = "zh") -> str:
    """给"安全"档补上边界说明；其余档返回空串。"""
    if verdict != Verdict.LOW:
        return ""
    if lang == "en":
        return VERDICT_GLOSS_EN[Verdict.LOW]
    return "在已取得的官方说明书中没有查到该组合的直接记载。**说明书未提及不等于已证实安全。**"


# ---------------------------------------------------------------- 报告文案

STRINGS: dict[str, dict[str, str]] = {
    "title": {"zh": "用药冲突核查报告", "en": "Drug Interaction Check Report"},
    "pair": {"zh": "核查组合", "en": "Pair checked"},
    "result": {"zh": "核查结果", "en": "Result"},
    "generated_by_llm": {"zh": "DeepSeek 模型组织语言", "en": "written by DeepSeek"},
    "generated_by_template": {"zh": "确定性模板引擎", "en": "deterministic template engine"},
    "escalation_header": {"zh": "需立即采取的行动（人工介入）", "en": "Action required (human review)"},
    "analysis_header": {
        "zh": "一、详细分析（逐条对应官方说明书原文）",
        "en": "1. Analysis (each item tied to official label text)",
    },
    "no_findings": {
        "zh": "确定性筛查未命中任何规则条目。这不代表「安全」，只代表说明书里没有被本工具规则覆盖的记载。",
        "en": "No rule fired. This does NOT mean the pair is safe; it only means the labels do not "
              "contain statements this tool checks for.",
    },
    "sorted_hint": {"zh": "按严重程度排列（先看排在前面的）：", "en": "Sorted by severity (read from the top):"},
    "suggested_action": {"zh": "建议动作", "en": "Suggested action"},
    "brief_header": {
        "zh": "二、两种药品的简略说明书（只列安全相关内容）",
        "en": "2. Brief safety sheets (safety-relevant sections only)",
    },
    "gap_header": {"zh": "三、本次核查的边界与资料缺口", "en": "3. Scope and data gaps"},
    "sources_header": {"zh": "四、出处清单（可逐条核对）", "en": "4. Evidence list (verbatim, verifiable)"},
    "no_sources": {
        "zh": "（本次没有可引用原文——因为一份官方说明书都没取到。）",
        "en": "(No quotable text this run: no official label could be retrieved.)",
    },
    "meta_header": {"zh": "五、数据来源与流程", "en": "5. Data source and pipeline"},
    "generic_name": {"zh": "通用名", "en": "Generic"},
    "brand_name": {"zh": "商品名", "en": "Brand"},
    "active_ingredient": {"zh": "有效成分", "en": "Active ingredient"},
    "label_version": {"zh": "说明书版本", "en": "Label version"},
    "not_registered": {"zh": "未登记", "en": "not listed"},
    "no_label": {"zh": "未能取得官方说明书", "en": "Official label not retrieved"},
    "no_label_tail": {
        "zh": "因此本节没有可引用的安全要点。",
        "en": "so no safety points can be quoted for this drug.",
    },
    "traceback": {"zh": "溯源", "en": "Source"},
    "original_text": {"zh": "原文", "en": "Quote"},
    "retrieved_at": {"zh": "检索于", "en": "retrieved"},
    "label_missing_sections": {
        "zh": "的说明书缺少以下安全章节：",
        "en": "label has no section for:",
    },
    "otc_note": {
        "zh": "（该标签可能为简化/OTC 标签）。",
        "en": "(likely a simplified/OTC label).",
    },
}


def t(key: str, lang: str = "zh") -> str:
    entry = STRINGS.get(key)
    if not entry:
        return key
    return entry.get(lang) or entry.get("zh", key)


# ---------------------------------------------------------------- 升级闸门文案

GATE_EN: dict[str, str] = {
    "critical": (
        "**DANGER — do not act on your own.** Tell the prescriber or pharmacist about this "
        "combination (with doses and timing) before you take it. Do not add, reduce, or stop "
        "any medication on your own until they answer."
    ),
    "high": (
        "**CAUTION.** Ask a pharmacist or prescriber about this combination before your next "
        "dose, and read the flagged points in the official label."
    ),
    "unknown": (
        "**UNKNOWN — information is insufficient.** No conclusion can be drawn for this pair. "
        "Ask a pharmacist or prescriber directly; do not treat this as clearance."
    ),
    "duplicate": (
        "**DUPLICATE ACTIVE INGREDIENT.** The same ingredient is supplied by both products — "
        "a concrete double-dosing risk. Verify the prescription with the prescriber now."
    ),
    "missing": (
        "**DATA GAP.** No official label was retrieved for: {drugs}. This check does NOT cover "
        "them. Confirm with a pharmacist or prescriber before relying on this report."
    ),
}

DISCLAIMER_EN = """\
> **Important**: this report was generated automatically from FDA official labeling
> (openFDA drug label). It states only what the official documents do and do not contain.
> **It is not medical advice and does not replace a prescriber or pharmacist.**
> Whether to combine these drugs, adjust doses, or stop a medication must be decided by your
> prescriber or pharmacist, taking into account your doses, duration, kidney and liver function,
> and every other medication you take. Seek care immediately for warning signs
> (bleeding, rash, breathing difficulty, severe dizziness).
"""
