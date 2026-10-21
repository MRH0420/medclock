"""数据结构：证据、标签、筛查结果、报告草稿。

这里最重要的两件事：

1. `Evidence` —— 每一条给用户看的原文都带 id（"ev_001"）和出处。
   报告里任何医学陈述都必须携带 evidence id，没有 id 的就只能升级为
   "说明书未记载"（见 Agent.md §5.6 的铁律）。

2. `Verdict` 的**单调约束**：确定性筛查算出的风险等级是下限，
   LLM 只允许往上修正，不允许往下淡化。这是消灭幻觉的硬约束之一。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------- 安全字段表

# ★ 直接照搬 api_test.py 的 SAFETY_FIELDS（严格筛查口径），只做了两点增强：
#   1) 合并 openFDA 里同义但不同名的字段（warnings / warnings_and_cautions /
#      boxed_warning 在 OTC 与 Rx 标签里叫法不同）；
#   2) 增加 use_in_specific_populations、information_for_patients，
#      这两节在处方药标签里承载了大量安全信息。
SAFETY_FIELDS: dict[str, list[str]] = {
    "核心警告": [
        "boxed_warning",
        "warnings",
        "warnings_and_cautions",
        "contraindications",
        "do_not_use",
    ],
    "使用前咨询/停用": [
        "ask_doctor",
        "ask_doctor_or_pharmacist",
        "stop_use",
        "when_using",
        "precautions",
    ],
    "特殊人群": [
        "pregnancy_or_breast_feeding",
        "pregnancy",
        "nursing_mothers",
        "pediatric_use",
        "geriatric_use",
        "keep_out_of_reach_of_children",
        "use_in_specific_populations",
    ],
    "不良反应与相互作用": [
        "adverse_reactions",
        "drug_interactions",
        "overdosage",
        "drug_abuse_and_dependence",
    ],
    "其他相关": [
        "active_ingredient",
        "purpose",
        "dosage_and_administration",
        "storage_and_handling",
        "information_for_patients",
    ],
}

ALL_SAFETY_FIELDS: tuple[str, ...] = tuple(
    f for fields in SAFETY_FIELDS.values() for f in fields
)
FIELD_TO_CATEGORY: dict[str, str] = {
    f: cat for cat, fields in SAFETY_FIELDS.items() for f in fields
}

# 命中这些字段 = 说明书明确点名了对方药品 → 权重最高
DIRECT_MENTION_FIELDS: tuple[str, ...] = (
    "boxed_warning",
    "contraindications",
    "do_not_use",
    "drug_interactions",
    "warnings",
    "warnings_and_cautions",
    "precautions",
    "ask_doctor",
    "ask_doctor_or_pharmacist",
    "stop_use",
    "information_for_patients",
)

FIELD_LABELS_ZH: dict[str, str] = {
    "boxed_warning": "黑框警告",
    "warnings": "警告",
    "warnings_and_cautions": "警告与注意事项",
    "contraindications": "禁忌症",
    "do_not_use": "禁用情形",
    "ask_doctor": "使用前咨询医生",
    "ask_doctor_or_pharmacist": "使用前咨询医生或药师",
    "stop_use": "停药情形",
    "when_using": "使用期间注意",
    "precautions": "注意事项",
    "pregnancy_or_breast_feeding": "妊娠或哺乳",
    "pregnancy": "妊娠",
    "nursing_mothers": "哺乳期",
    "pediatric_use": "儿童用药",
    "geriatric_use": "老年用药",
    "keep_out_of_reach_of_children": "儿童防护",
    "use_in_specific_populations": "特殊人群用药",
    "adverse_reactions": "不良反应",
    "drug_interactions": "药物相互作用",
    "overdosage": "过量",
    "drug_abuse_and_dependence": "滥用与依赖",
    "active_ingredient": "有效成分",
    "purpose": "用途",
    "dosage_and_administration": "用法用量",
    "storage_and_handling": "储存",
    "information_for_patients": "患者须知",
}


# ---------------------------------------------------------------- 风险等级


class Verdict:
    """风险等级（四档）。值本身是英文常量，渲染时按语言映射（见 i18n.py）。

    顺序即保守程度：UNKNOWN < LOW < HIGH < CRITICAL。

    ★ "未发现直接冲突证据"（LOW）刻意**不叫"安全"**：官方说明书未提及不等于
      已证实安全，所以这一档在任何输出里都必须带上这条边界说明。
    """

    CRITICAL = "critical"      # 危险：禁忌/黑框点名对方，或有效成分重复
    HIGH = "high"              # 注意：相互作用/警告章节点名，或缺少相互作用章节
    LOW = "low"                # 见标题：未发现直接冲突证据（★ 永远带边界说明）
    UNKNOWN = "unknown"        # 未知：资料不足，无法判断

    ORDER: tuple[str, ...] = (UNKNOWN, LOW, HIGH, CRITICAL)
    RANK: dict[str, int] = {v: i for i, v in enumerate(ORDER)}

    # 旧命名 / 中文字面量的兼容映射
    LEGACY_ALIASES: dict[str, str] = {
        "冲突风险高": CRITICAL,
        "需要特别注意": HIGH,
        "未发现直接冲突证据": LOW,
        "证据不足": UNKNOWN,
        "安全": LOW,
    }

    @classmethod
    def normalize(cls, value: str) -> str:
        """把旧的中文档位名（或任何别名）映射到新的英文常量。"""
        if value in cls.RANK:
            return value
        return cls.LEGACY_ALIASES.get(value, cls.UNKNOWN)

    @classmethod
    def worse(cls, *verdicts: str) -> str:
        """取最保守（等级最高）的一个。"""
        best = cls.UNKNOWN
        for raw in verdicts:
            v = cls.normalize(raw)
            if cls.RANK[v] > cls.RANK[best]:
                best = v
        return best

    @classmethod
    def at_least(cls, candidate: str, floor: str) -> str:
        """单调约束：candidate 不允许低于 floor。"""
        cand = cls.normalize(candidate)
        fl = cls.normalize(floor)
        return cand if cls.RANK[cand] >= cls.RANK[fl] else fl


class Severity:
    """单条发现的分量（与结论档分开：一个 🔴 发现也可以出现在 🟡 结论里）。"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    INFO = "info"

    RANK = {INFO: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3}
    LABEL_ZH = {INFO: "提示", MEDIUM: "关注", HIGH: "重要", CRITICAL: "严重"}
    LABEL_EN = {INFO: "Info", MEDIUM: "Medium", HIGH: "High", CRITICAL: "Critical"}


# ---------------------------------------------------------------- 证据


@dataclass
class Evidence:
    """一条可溯源的官方原文引文。"""

    evidence_id: str
    drug: str
    section: str                 # openFDA 字段名，如 drug_interactions
    section_zh: str               # 中文字段名
    quote: str                    # ★ 官方原文（可在 label 全文中逐字命中）
    source: str = "openFDA drug label (FDA SPL)"
    url: str = ""
    retrieved_at: str = ""
    confidence: str = "official"  # official | derived | user_stated

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def short(self, limit: int = 200) -> str:
        q = " ".join(self.quote.split())
        return q if len(q) <= limit else q[:limit] + "…"


# ---------------------------------------------------------------- 药品标签


@dataclass
class LabelRecord:
    """一次成功的官方标签查询结果（已按 SAFETY_FIELDS 抽取）。"""

    query: str                       # 用户输入的原始药名
    matched: bool                    # 是否命中官方标签
    strategy: str = ""               # 命中策略：generic_name / brand_name / ...
    status: int = 0                  # HTTP 状态码
    message: str = ""                # 未命中原因
    name: str = ""                   # 展示名（品牌名或通用名）
    generic_names: list[str] = field(default_factory=list)
    brand_names: list[str] = field(default_factory=list)
    active_ingredients: list[str] = field(default_factory=list)
    label_id: str = ""
    set_id: str = ""
    effective_time: str = ""         # 说明书版本日期
    url: str = ""
    retrieved_at: str = ""
    fetched_from: str = ""           # network | cache | fixture
    # 字段名 -> 官方原文（列表）
    fields: dict[str, list[str]] = field(default_factory=dict)
    # 该药的药理类别原始字符串（来自 openFDA NDC/label 的 pharm_class，或内置兜底表）
    pharm_classes: list[str] = field(default_factory=list)
    pharm_class_source: str = ""
    # 被跳过的空字段（说明"该标签没写这一节"，这本身是要告诉用户的信息）
    present_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)  # 安全字段中缺失的
    evidence: dict[str, Evidence] = field(default_factory=dict)  # evidence_id -> Evidence

    @property
    def display(self) -> str:
        if self.generic_names:
            return self.generic_names[0].title()
        return self.name or self.query

    @property
    def stems(self) -> set[str]:
        from .textutil import name_stems

        out: set[str] = set()
        for n in list(self.generic_names) + list(self.brand_names) + [self.query]:
            out |= name_stems(n)
        return out

    def coverage(self) -> dict[str, bool]:
        return {
            "has_interaction_section": "drug_interactions" in self.fields,
            "has_boxed_warning": "boxed_warning" in self.fields,
            "has_contraindications": "contraindications" in self.fields,
            "has_warnings": bool(
                {"warnings", "warnings_and_cautions"} & set(self.fields)
            ),
            "has_pregnancy": bool(
                {"pregnancy", "pregnancy_or_breast_feeding", "nursing_mothers"}
                & set(self.fields)
            ),
        }

    def evidence_list(self) -> list[Evidence]:
        return [self.evidence[k] for k in sorted(self.evidence)]


# ---------------------------------------------------------------- 筛查结果


@dataclass
class Finding:
    """一条确定性筛查发现（规则命中，不是模型想出来的）。

    中英双份文案都在这里：**规则的措辞只有这一处定义**，
    渲染时按语言取，避免两套话术漂移。
    """

    rule_id: str
    severity: str            # Severity.*
    title: str
    detail: str
    evidence_ids: list[str] = field(default_factory=list)
    action: str = ""
    title_en: str = ""
    detail_en: str = ""
    action_en: str = ""

    def localized(self, lang: str = "zh") -> dict[str, str]:
        if lang == "en":
            return {
                "title": self.title_en or self.title,
                "detail": self.detail_en or self.detail,
                "action": self.action_en or self.action,
            }
        return {"title": self.title, "detail": self.detail, "action": self.action}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Note:
    """一条边界/限制说明（中英各一份）。"""

    zh: str
    en: str = ""

    def text(self, lang: str = "zh") -> str:
        if lang == "en":
            return self.en or self.zh
        return self.zh


@dataclass
class PairScreen:
    """两药配对筛查结果。"""

    drug_a: str
    drug_b: str
    verdict_floor: str = Verdict.UNKNOWN   # ★ LLM 不得低于这个等级
    findings: list[Finding] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    limitations: list[Note] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "drug_a": self.drug_a,
            "drug_b": self.drug_b,
            "verdict_floor": self.verdict_floor,
            "findings": [f.to_dict() for f in self.findings],
            "coverage": self.coverage,
            "limitations": [n.zh for n in self.limitations],
        }

    def add_note(self, zh: str, en: str = "") -> None:
        self.limitations.append(Note(zh=zh, en=en))


# ---------------------------------------------------------------- 报告草稿


@dataclass
class ReportDraft:
    """交给渲染器的结构化报告（LLM 或模板产出，格式一致）。"""

    verdict: str
    headline: str
    one_line_conclusion: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    brief_a: dict[str, Any] = field(default_factory=dict)
    brief_b: dict[str, Any] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    generated_by: str = "template"   # template | llm
    warnings: list[str] = field(default_factory=list)  # 校验器给出的提示
