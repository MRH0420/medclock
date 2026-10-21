"""确定性筛查引擎（不涉及任何模型调用）。

分工说明（这是本项目防幻觉的核心设计）：

| 谁 | 负责什么 | 为什么 |
|---|---|---|
| **本模块（代码）** | 判定"官方说明书里到底写了什么" | 确定性、可复现、可审计，演示不会翻车 |
| **LLM** | 把已核实的证据组织成人话、给出行动建议 | 语言能力强，但不许新增医学事实 |

也就是说：**风险等级由代码算，措辞由模型写。**
模型只能把代码给出的等级往上修正（更保守），永远不能往下淡化。
"""

from __future__ import annotations

import re
from typing import Any

from .models import (
    DIRECT_MENTION_FIELDS,
    FIELD_LABELS_ZH,
    SAFETY_FIELDS,
    Evidence,
    Finding,
    LabelRecord,
    Note,
    PairScreen,
    Severity,
    Verdict,
)
from .safety_fields import category_of
from .textutil import compile_terms, find_sentence_spans, name_stems, sentence_spans

# 发现项 → 它能把结论档抬到哪一档（两者分开，方便"严重发现但结论不是危险"这种情形）
SEVERITY_TO_VERDICT: dict[str, str] = {
    Severity.CRITICAL: Verdict.CRITICAL,
    Severity.HIGH: Verdict.HIGH,
    Severity.MEDIUM: Verdict.HIGH,   # 只在"患者须知"里被点名 → 仍属需要留意的组合
    Severity.INFO: Verdict.UNKNOWN,
}

# 命中"整句"最长保留多少字符（表格型长句会围绕命中词裁剪，仍是原文子串）
MENTION_BUDGET = 420

# 命中这些字段的"点名"最严重
_CRITICAL_MENTION_FIELDS = ("boxed_warning", "contraindications", "do_not_use")
# 面向处方者的章节 → 重要
_HIGH_MENTION_FIELDS = (
    "drug_interactions",
    "warnings",
    "warnings_and_cautions",
    "precautions",
    "stop_use",
    "ask_doctor",
    "ask_doctor_or_pharmacist",
)
# 只出现在"患者须知"这类科普章节里的点名 → 关注（不抬到"重要"）
_MEDIUM_MENTION_FIELDS = ("information_for_patients", "when_using", "purpose")

# Explicit co-use statements with a potentially fatal outcome must not be
# downgraded merely because they appear in the drug-interactions section.
_CO_USE_TERMS = re.compile(
    r"\b(?:concurrent(?:ly)?|concomitant(?:ly)?|co[- ]?administration|"
    r"co[- ]?administered|together)\b",
    re.IGNORECASE,
)
_FATAL_OUTCOME_TERMS = re.compile(
    r"\b(?:fatal|life[- ]threatening|death|致命|危及生命)\b",
    re.IGNORECASE,
)
_FIELD_LABELS_EN = {
    "boxed_warning": "Boxed warning",
    "warnings": "Warnings",
    "warnings_and_cautions": "Warnings and precautions",
    "contraindications": "Contraindications",
    "do_not_use": "Do not use",
    "drug_interactions": "Drug interactions",
    "precautions": "Precautions",
    "stop_use": "Stop use",
    "ask_doctor": "Ask a doctor",
    "ask_doctor_or_pharmacist": "Ask a doctor or pharmacist",
    "information_for_patients": "Patient information",
    "when_using": "When using",
    "purpose": "Purpose",
}


def _field_label_en(field_name: str) -> str:
    return _FIELD_LABELS_EN.get(field_name, field_name.replace("_", " ").capitalize())


def _compile_stems(stems: set[str]) -> list[re.Pattern[str]]:
    """词干集合 → 已编译正则（供需要复用正则的场景）。"""
    return compile_terms(sorted(stems, key=len, reverse=True))


def _field_evidence_ids(record: LabelRecord, field_name: str) -> list[str]:
    return [
        ev_id
        for ev_id, ev in sorted(record.evidence.items())
        if ev.section == field_name
    ]


def _register(
    register, holder: LabelRecord, field_name: str, sentence: str, fallback_ids: list[str]
) -> str:
    """把"命中的整句"登记成句子级证据；没有 registrar 时退回字段级证据。"""
    if register is not None:
        try:
            return register(holder, field_name, sentence)
        except Exception:  # noqa: BLE001 - 登记失败不能让筛查崩掉
            pass
    return fallback_ids[0] if fallback_ids else ""


def find_mentions(
    record: LabelRecord,
    terms: list[str],
    limit: int = 4,
    register=None,
) -> list[dict[str, Any]]:
    """在标签的安全字段里找命中 terms 的整句原文。

    ★ 重要：检索的是**完整官方原文**（record.fields），不是被截断过的入库引文。
      否则长章节（例如华法林的 drug_interactions 有上万字符）后半段的
      「点名」会被截断吃掉，造成假阴性。
    """
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for field_name, texts in record.fields.items():
        fallback_ids = _field_evidence_ids(record, field_name)
        for text in texts:
            for sentence, _span in find_sentence_spans(
                text, terms, limit=limit, budget=MENTION_BUDGET
            ):
                key = sentence.strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    {
                        "section": field_name,
                        "section_zh": FIELD_LABELS_ZH.get(field_name, field_name),
                        "category": category_of(field_name),
                        "sentence": sentence,
                        "evidence_id": _register(
                            register, record, field_name, sentence, fallback_ids
                        ),
                    }
                )
                if len(hits) >= limit:
                    return hits
    return hits


def verify_term(
    term: str,
    record_a: LabelRecord,
    record_b: LabelRecord,
    limit: int = 3,
    register=None,
) -> dict[str, Any]:
    """★ 给 Agent 用的"共享类别/共同风险"核验器。

    Agent 可以提出假设（"这两个药是不是都提到出血风险？"），
    但**结论只能来自这里返回的官方原文**。没有原文命中 → 该假设作废。
    """
    terms = sorted(name_stems(term) or {term.lower()}, key=len, reverse=True)
    if not terms:
        return {"term": term, "error": "term 太短或无效，无法检索。"}
    a = find_mentions(record_a, terms, limit, register=register)
    b = find_mentions(record_b, terms, limit, register=register)
    return {
        "term": term,
        "found_in": {
            "A": record_a.display,
            "B": record_b.display,
        },
        "matches_a": a,
        "matches_b": b,
        "shared_in_both": bool(a) and bool(b),
        "note": (
            "只列出官方说明书原文命中句。若某一侧为空，说明该侧说明书未出现这个词，"
            "不得据此推断'没有该风险'。"
        ),
    }


def _coverage_of(record: LabelRecord) -> dict[str, Any]:
    cov = record.coverage()
    return {
        "drug": record.display,
        "queried_as": record.query,
        "matched": record.matched,
        "label_effective_time": record.effective_time,
        "n_safety_fields": len(record.fields),
        **cov,
    }


def screen_class_mention(
    holder: LabelRecord,
    partner: LabelRecord,
    partner_classes: list[tuple[str, str]],
    partner_class_source: str,
    register=None,
) -> list[Finding]:
    """★ 规则 R-CLASS-001 —— "说明书按类别描述、未点名本药" 的检测。

    背景（实测得出）：说明书经常只写药物类别而不写具体药名。逐字核对过的例子：
      · Sertraline 警告：Concomitant use of aspirin, nonsteroidal anti-inflammatory
        drugs (NSAIDs) ... may increase this risk   ← 全文没有 "ibuprofen"
      · Methotrexate 相互作用：与 nonsteroidal anti-inflammatory drugs (NSAIDs) 合用
      · Lithium 相互作用：NSAID decrease renal blood flow ... increased serum lithium
    这三份标签都**没有**出现对方药名，所以纯药名匹配必然漏检。

    判定（全部落在官方原文上）：
      ① 对方药的药理类别来自官方接口（openFDA NDC/label 的 pharm_class）或内置兜底表；
      ② 该类别的名称/常见缩写**逐字出现在本方标签的安全章节里**；
      ③ 命中的整句登记为句子级证据，报告里逐字引用。

    ★ 边界：这只说明"说明书按类别描述了这类风险"，**不等于**已确认本药与之相互作用，
      也不等于两药一定不能合用。所有输出都带这句限定。
    """
    findings: list[Finding] = []
    if not (holder.matched and partner.matched) or not partner_classes:
        return findings

    hits: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for phrase, display in partner_classes:
        for hit in find_mentions(holder, [phrase], limit=2, register=register):
            key = (hit["section"], display)
            if key in seen:
                continue
            seen.add(key)
            hit["phrase"] = phrase
            hit["display"] = display
            hits.append(hit)
    if not hits:
        return findings

    # ★ 按**展示名**聚合：同一类别的多个检索变体（NSAID / nonsteroidal...）
    #   只出一条 finding，避免同一件事重复三遍。
    by_class: dict[str, dict[str, Any]] = {}
    for hit in hits:
        slot = by_class.setdefault(
            hit["display"],
            {"sections": [], "sections_en": [], "sentences": [], "evidence_ids": [],
             "phrases": []},
        )
        if hit["section_zh"] not in slot["sections"]:
            slot["sections"].append(hit["section_zh"])
        if hit["section"] not in slot["sections_en"]:
            slot["sections_en"].append(hit["section"])
        if hit["sentence"] not in slot["sentences"]:
            slot["sentences"].append(hit["sentence"])
        if hit["evidence_id"] and hit["evidence_id"] not in slot["evidence_ids"]:
            slot["evidence_ids"].append(hit["evidence_id"])
        if hit["phrase"] not in slot["phrases"]:
            slot["phrases"].append(hit["phrase"])

    for display, slot in by_class.items():
        sections_zh = "、".join(f"【{s}】" for s in slot["sections"][:3])
        findings.append(
            Finding(
                rule_id="R-CLASS-001",
                severity=Severity.MEDIUM,
                title=(
                    f"{holder.display} 的官方说明书按**类别**提到 "
                    f"{partner.display} 所属的「{display}」"
                ),
                detail=(
                    f"{partner.display} 经官方数据登记为「{display}」"
                    f"（来源：{partner_class_source or '官方药品目录'}）。"
                    f"{holder.display} 的说明书在 {sections_zh} 中描述了这一**类别**的风险。"
                    "**注意：说明书是按类别描述的，并未点名本药**——所以这条既不等于已确认"
                    "两者的相互作用，也不等于不能合用；它说明的是「这份说明书认为该类别需要"
                    "提醒」，而本药属于该类别。"
                ),
                evidence_ids=slot["evidence_ids"][:3],
                action=(
                    f"问药师一句：「{holder.display} 的说明书提到{display}这个类别，"
                    f"我同时吃 {partner.display}，需要额外注意什么？」"
                ),
                title_en=(
                    f"{holder.display} label names the **class** "
                    f"\u201c{display}\u201d that {partner.display} belongs to"
                ),
                detail_en=(
                    f"{partner.display} is registered under the pharmacologic class "
                    f"\u201c{display}\u201d (source: {partner_class_source or 'official drug listing'}). "
                    f"The {holder.display} label describes the risk for that **class** in "
                    f"{', '.join(slot.get('sections_en') or slot['sections'][:3])}. "
                    "**The label describes the class, not this specific drug** — so this is neither "
                    "confirmation of an interaction nor a statement that the pair must not be used; "
                    "it means the label flags the class and this drug belongs to it."
                ),
                action_en=(
                    f"Ask the pharmacist: \u201cThe {holder.display} label mentions {display} — "
                    f"what should I watch for while I am also taking {partner.display}?\u201d"
                ),
            )
        )
    return findings


def screen_pair(
    record_a: LabelRecord, record_b: LabelRecord, register=None,
    classes_a: list[tuple[str, str]] | None = None,
    classes_b: list[tuple[str, str]] | None = None,
    class_source_a: str = "",
    class_source_b: str = "",
) -> PairScreen:
    """严格筛查两药配对，输出发现清单 + **风险等级下限**。

    `register(record, field_name, sentence) -> evidence_id` 用于把命中的整句
    登记成句子级证据（由 agent 层注入，本模块不依赖证据库实现）。

    `classes_a/classes_b` 是两药各自的**药理类别候选列表**（`(检索短语, 展示名)`），
    来自官方 pharm_class 数据；缺省表示"没拿到类别数据"，此时类别检查会记为覆盖缺口。
    """
    screen = PairScreen(drug_a=record_a.query, drug_b=record_b.query)
    findings: list[Finding] = []
    limitations: list[str] = []

    problems: list[str] = []

    def note(zh: str, en: str = "") -> None:
        limitations.append(Note(zh=zh, en=en))

    # --- 数据可用性 ---------------------------------------------------
    for rec in (record_a, record_b):
        if not rec.matched:
            problems.append(
                f"未能取得 “{rec.query}” 的官方说明书（{rec.message or '未找到'}），"
                "因此无法判断它与另一种药的相互作用。"
            )
    missing_a = [f for f in record_a.missing_fields] if record_a.matched else []
    missing_b = [f for f in record_b.missing_fields] if record_b.matched else []

    for rec, missing in ((record_a, missing_a), (record_b, missing_b)):
        if not rec.matched:
            continue
        elif "drug_interactions" in missing:
            findings.append(
                Finding(
                    rule_id="R-DATA-002",
                    severity=Severity.HIGH,
                    title=f"{rec.display} 的官方说明书中**没有药物相互作用章节**",
                    detail=(
                        f"该标签共提供 {len(rec.fields)} 个安全字段，其中不含 drug_interactions。"
                        "常见于非处方药（OTC）标签与部分简化标签：**缺少这一节不代表安全**，"
                        "只代表这份官方文件没有可引用的相互作用结论。"
                    ),
                    evidence_ids=[],
                    action="把这一节缺失当作「信息缺口」告知使用者，并请药师在配药系统里复核。",
                    title_en=f"{rec.display} label has **no drug-interaction section**",
                    detail_en=(
                        f"This label provides {len(rec.fields)} safety sections, none of which is "
                        "drug_interactions — common for OTC and simplified labels. **A missing "
                        "section is not evidence of safety**; it only means this official document "
                        "contains no quotable interaction statement."
                    ),
                    action_en=(
                        "Treat the missing section as an information gap and ask a pharmacist to "
                        "screen the pair in the dispensing system."
                    ),
                )
            )

    # --- 规则 R-DIRECT-001：说明书点名对方药品 ------------------------
    a_stems = {s for n in record_b.generic_names + record_b.brand_names for s in name_stems(str(n))}
    a_stems |= name_stems(record_b.query)
    b_stems = {s for n in record_a.generic_names + record_a.brand_names for s in name_stems(str(n))}
    b_stems |= name_stems(record_a.query)

    direct_pairs = (
        (record_a, record_b, sorted(a_stems, key=len, reverse=True)),
        (record_b, record_a, sorted(b_stems, key=len, reverse=True)),
    )
    for holder, partner, terms in direct_pairs:
        if not holder.matched or not terms:
            continue
        hits = find_mentions(holder, terms, limit=6, register=register)
        if not hits:
            continue
        # 同一个药可能被多个章节点名 → 合并成一条发现（严重度取最高）
        merged: dict[str, dict[str, Any]] = {}
        for hit in hits:
            slot = merged.setdefault(
                hit["section"],
                {
                    "section": hit["section"],
                    "section_zh": hit["section_zh"],
                    "sentences": [],
                    "evidence_ids": [],
                },
            )
            slot["sentences"].append(hit["sentence"])
            if hit["evidence_id"]:
                slot["evidence_ids"].append(hit["evidence_id"])

        sections = list(merged)
        explicit_severe_co_use = any(
            _CO_USE_TERMS.search(sentence) and _FATAL_OUTCOME_TERMS.search(sentence)
            for hit in hits
            for sentence in [hit["sentence"]]
        )
        if set(sections) & set(_CRITICAL_MENTION_FIELDS) or explicit_severe_co_use:
            severity = Severity.CRITICAL
        elif set(sections) & set(_HIGH_MENTION_FIELDS):
            severity = Severity.HIGH
        else:
            severity = Severity.MEDIUM
        problems.append(SEVERITY_TO_VERDICT[severity])

        section_zh_list = "、".join(
            f"【{merged[s]['section_zh']}】" for s in sections[:3]
        )
        section_en_list = ", ".join(
            f"[{_field_label_en(merged[s]['section'])}]" for s in sections[:3]
        )
        evidence_ids: list[str] = []
        for slot in merged.values():
            for ev_id in slot["evidence_ids"]:
                if ev_id and ev_id not in evidence_ids:
                    evidence_ids.append(ev_id)
        findings.append(
            Finding(
                rule_id="R-DIRECT-001",
                severity=severity,
                title=(
                    f"{holder.display} 的官方说明书在 {section_zh_list} 中"
                    f"明确提到 {partner.display}"
                ),
                detail=(
                    "命中章节："
                    + "；".join(
                        f"{merged[s]['section_zh']}（{len(merged[s]['sentences'])} 处提及）"
                        for s in sections
                    )
                    + "。原文见下方引用（逐字可核对）。**「说明书提到对方药品」不等于"
                    "「两者一定有害」，它意味着这是说明书作者认为需要提醒读者的组合。**"
                ),
                evidence_ids=evidence_ids[:4],
                action="不要自行调整剂量或停用；先把这个组合告诉开方医生或药师。",
                title_en=(
                    f"{holder.display} label explicitly names {partner.display} in "
                    f"{section_en_list}"
                ),
                detail_en=(
                    "The official label reports a fatal or life-threatening outcome with "
                    "concurrent use. Review the quoted wording and drug class with a clinician; "
                    "the label does not predict an individual's outcome."
                    if explicit_severe_co_use
                    else (
                        "Sections that mention the other drug: "
                        + "; ".join(
                            f"{_field_label_en(merged[s]['section'])} ({len(merged[s]['sentences'])} mention(s))"
                            for s in sections
                        )
                        + ". Verbatim quotes are printed below. A label naming the other drug "
                        "does not by itself prove the pair is harmful."
                    )
                ),
                action_en=(
                    "Do not use this combination until a prescriber or pharmacist has reviewed "
                    "it. Do not stop or change prescribed doses without their advice."
                    if explicit_severe_co_use
                    else "Do not change or stop doses on your own; tell the prescriber or pharmacist "
                    "about this combination first."
                ),
            )
        )

    # --- 规则 R-DUP-001：有效成分重复 ---------------------------------
    ing_a = {i.strip().lower() for i in record_a.active_ingredients if i.strip()}
    ing_b = {i.strip().lower() for i in record_b.active_ingredients if i.strip()}
    shared_ing = sorted(ing_a & ing_b)
    if shared_ing:
        problems.append(Verdict.CRITICAL)
        findings.append(
            Finding(
                rule_id="R-DUP-001",
                severity=Severity.CRITICAL,
                title="两种药含有**相同有效成分**：" + "、".join(shared_ing),
                detail=(
                    "openFDA 官方标签登记的 substance_name 完全重合。"
                    "同时服用等于在不知情的情况下加大同一成分的剂量，"
                    "这是最容易被忽略、后果又最明确的一类「重复用药」。"
                ),
                evidence_ids=[],
                action="立刻联系开方医生或药师确认是否属于重复用药，不要自行叠加服用。",
                title_en="Both products contain the **same active ingredient**: "
                         + ", ".join(shared_ing),
                detail_en=(
                    "openFDA lists identical substance_name values for both products. Taking them "
                    "together raises the dose of that ingredient without you realising it — the "
                    "most easily missed and most concrete form of duplicate dosing."
                ),
                action_en=(
                    "Contact the prescriber or pharmacist now to confirm whether this is duplicate "
                    "dosing. Do not stack them on your own."
                ),
            )
        )

    # --- 规则 R-CLASS-001：说明书按"类别"描述（未点名本药） -------------
    # 方向：本方说明书提到对方所属的类别（A 的标签 × B 的类别；B 的标签 × A 的类别）
    class_findings = (
        screen_class_mention(record_a, record_b, classes_b or [], class_source_b, register=register)
        + screen_class_mention(record_b, record_a, classes_a or [], class_source_a, register=register)
    )
    if class_findings:
        findings.extend(class_findings)
        problems.append(Verdict.HIGH)
        note(
            "「类别级提示」来自官方药理类别数据："
            + "；".join(
                f"{rec.display}={src or '无数据'}"
                for rec, src in ((record_a, class_source_a), (record_b, class_source_b))
                if src
            )
            + "。它说明的是**说明书按类别描述了这类风险、但未点名本药**，"
            "既不代表已确认相互作用，也不代表不能合用。",
            "Class-level findings come from official pharmacologic-class data ("
            + "; ".join(
                f"{rec.display}={src or 'no data'}"
                for rec, src in ((record_a, class_source_a), (record_b, class_source_b))
                if src
            )
            + "). They mean **the label describes the risk by class without naming this drug** — "
            "neither confirmation of an interaction nor a prohibition.",
        )

    # 覆盖缺口必须单独说清楚：某侧没有类别数据 = 那个方向的类别检查根本没做成
    # （★ 这一条与上面是否有命中无关，不能因为"另一侧命中了"就不提）
    if record_a.matched and record_b.matched:
        missing_side = []
        if not classes_a:
            missing_side.append(record_a.display)
        if not classes_b:
            missing_side.append(record_b.display)
        if missing_side:
            note(
                "本次**未能取得** " + "、".join(missing_side) + " 的药理类别数据，"
                "因此以该药为对象的「说明书按类别描述」检查**未被覆盖**。",
                "No pharmacologic-class data was available for "
                + ", ".join(missing_side)
                + ", so the class-level check targeting that drug could NOT be covered.",
            )

    # --- 规则 R-BOXED-001：黑框警告（与配对无关，但必须提示） ----------
    # 注意：这条**不抬高配对风险等级**——黑框警告是该药自身的属性，
    # 直接决定"这两种药能不能一起用"的应该是相互作用证据。
    for rec in (record_a, record_b):
        if rec.matched and "boxed_warning" in rec.fields:
            ev_ids = [
                ev_id
                for ev_id, ev in sorted(rec.evidence.items())
                if ev.section == "boxed_warning"
            ]
            findings.append(
                Finding(
                    rule_id="R-BOXED-001",
                    severity=Severity.MEDIUM,
                    title=f"{rec.display} 带有 FDA **黑框警告**（最高级别警示）",
                    detail=(
                        "该药本身存在需要医患共同关注的严重风险，与配对结论无关，"
                        "但阅读本报告时不应忽略。它**不构成**「这两种药不能一起用」的依据，"
                        "也不代表两者一定冲突。"
                    ),
                    evidence_ids=ev_ids[:1],
                    action="了解黑框警告内容；出现其中描述的症状立即就医。",
                    title_en=f"{rec.display} carries an FDA **boxed warning** (the strongest warning)",
                    detail_en=(
                        "This drug has a serious risk that patients and clinicians must know about. "
                        "It is unrelated to the pairing result, but should not be ignored. It does "
                        "**not** mean the two drugs cannot be used together."
                    ),
                    action_en=(
                        "Read the boxed warning; seek care immediately if you notice the symptoms "
                        "it describes."
                    ),
                )
            )

    # --- 规则 R-DATA-003：两药都没有相互作用章节 -----------------------
    both_missing_interaction = (
        record_a.matched
        and record_b.matched
        and "drug_interactions" in missing_a
        and "drug_interactions" in missing_b
    )
    if both_missing_interaction:
        problems.append(Verdict.UNKNOWN)
        note(
            "两份官方标签都不含药物相互作用章节，因此**本次核查无法对「是否冲突」给出结论**。",
            "Neither official label contains a drug-interaction section, so **this check cannot "
            "conclude anything about the combination**.",
        )

    # --- 规则 R-COVER-001：说明书记载范围提示 --------------------------
    if record_a.matched and record_b.matched:
        note(
            "官方说明书通常只列出经过充分研究的相互作用，"
            "**未提及 ≠ 已证实安全**；临床判断还需结合剂量、疗程、肝肾功能与同服的其他药物。",
            "Official labels usually list only well-studied interactions. **Not mentioned does not "
            "mean proven safe.** Clinical judgement also depends on dose, duration, kidney and "
            "liver function, and every other medication taken.",
        )

    # ---------------------------------------------------------------- 汇总
    auto_floor = Verdict.worse(*problems) if problems else Verdict.UNKNOWN
    if not record_a.matched or not record_b.matched:
        auto_floor = Verdict.worse(auto_floor, Verdict.UNKNOWN)
    elif not problems:
        # 两份标签都有相互作用章节、且没有任何命中 → 才允许落到最低档
        auto_floor = Verdict.LOW

    screen.findings = sorted(
        findings, key=lambda f: -Severity.RANK.get(f.severity, 0)
    )  # 🔴 严重优先展示：先看最要紧的
    screen.verdict_floor = auto_floor
    screen.coverage = {
        "A": _coverage_of(record_a),
        "B": _coverage_of(record_b),
        "fields_checked": {
            cat: fields for cat, fields in SAFETY_FIELDS.items()
        },
    }
    screen.limitations = limitations
    return screen


# ---------------------------------------------------------------- 简略说明书

_BRIEF_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("禁忌/禁用", ("contraindications", "do_not_use")),
    ("核心警告", ("boxed_warning", "warnings", "warnings_and_cautions")),
    ("用药前须知/停用", ("ask_doctor", "ask_doctor_or_pharmacist", "stop_use", "when_using", "precautions")),
    ("特殊人群", (
        "pregnancy_or_breast_feeding", "pregnancy", "nursing_mothers",
        "pediatric_use", "geriatric_use", "use_in_specific_populations",
        "keep_out_of_reach_of_children",
    )),
    ("相互作用", ("drug_interactions",)),
    ("常见不良反应", ("adverse_reactions",)),
    ("用法用量要点", ("dosage_and_administration",)),
    ("储存", ("storage_and_handling",)),
)


def brief_sheet(record: LabelRecord, max_sentences: int = 2) -> dict[str, Any]:
    """生成"够用就好"的简略说明书骨架：只保留安全相关内容。

    ★ 返回的每一条都带 evidence_id；拿不到原文的条目一律标注"未记载"，
    绝不用模型的记忆去补。
    """
    sheet: dict[str, Any] = {
        "drug_input": record.query,
        "drug_name": record.display,
        "generic_name": "、".join(record.generic_names) or "未登记",
        "brand_name": "、".join(record.brand_names) or "未登记",
        "active_ingredient": "、".join(record.active_ingredients) or "未登记",
        "label_effective_time": record.effective_time or "未登记",
        "label_url": record.url,
        "matched": record.matched,
        "sections": [],
        "missing": [],
    }
    if not record.matched:
        sheet["missing"] = ["全部：未取得官方说明书"]
        return sheet

    for label_zh, field_names in _BRIEF_SECTIONS:
        entries: list[dict[str, Any]] = []
        for field_name in field_names:
            ev_ids = [
                ev_id
                for ev_id, ev in sorted(record.evidence.items())
                if ev.section == field_name
            ]
            for ev_id in ev_ids:
                ev = record.evidence[ev_id]
                for sentence in _first_sentences(ev.quote, max_sentences):
                    entries.append(
                        {
                            "field": field_name,
                            "field_zh": FIELD_LABELS_ZH.get(field_name, field_name),
                            "text": sentence,
                            "evidence_id": ev_id,
                        }
                    )
        if entries:
            sheet["sections"].append({"section": label_zh, "items": entries[:3]})
        else:
            sheet["missing"].append(label_zh)
    return sheet


def _first_sentences(text: str, limit: int) -> list[str]:
    out: list[str] = []
    for start, end in sentence_spans(text):
        sentence = text[start:end].strip()
        if len(sentence) < 12:
            continue
        out.append(sentence)
        if len(out) >= limit:
            break
    if not out and text.strip():
        out.append(text.strip()[:300])
    return out


def evidence_index(evidence: list[Evidence], limit_quote: int = 400) -> list[dict[str, Any]]:
    """报告末尾的"原文出处"清单。"""
    rows = []
    for ev in evidence:
        quote = " ".join(ev.quote.split())
        if len(quote) > limit_quote:
            quote = quote[:limit_quote].rstrip() + " …"
        rows.append(
            {
                "evidence_id": ev.evidence_id,
                "drug": ev.drug,
                "section": ev.section,
                "section_zh": ev.section_zh,
                "quote": quote,
                "url": ev.url,
                "retrieved_at": ev.retrieved_at,
            }
        )
    return rows
