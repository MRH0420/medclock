"""报告撰写 + 校验 + 渲染。

分工：
- `ReportWriter`：把结构化证据喂给 DeepSeek，拿回 JSON 报告草稿。
- `validate_draft`：★ 幻觉防线。逐条检查引用是否存在、引文是否逐字命中、
  风险等级有没有被下调、有没有出现"安全/可以一起吃"这类越界措辞。
  不通过的草稿直接丢弃，改走模板。
- `render_markdown`：把草稿渲染成最终报告（模型和模板走同一个渲染器，
  所以"降级"不会改变报告的结构与信息量，只是措辞更朴素）。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .i18n import (
    DISCLAIMER_EN,
    SEVERITY_EN,
    SEVERITY_ZH,
    VERDICT_EN,
    VERDICT_GLOSS_EN,
    VERDICT_QUALIFIER_EN,
    VERDICT_QUALIFIER_ZH,
    VERDICT_ZH,
    severity_label,
    t,
    verdict_gloss,
    verdict_label,
)
from .config import Settings
from .drug_classes import strip_class_suffix
from .llm_client import DeepSeekClient, LLMError, _extract_json
from .models import (
    Evidence,
    LabelRecord,
    PairScreen,
    ReportDraft,
    Severity,
    Verdict,
)
from .prompts import (
    DISCLAIMER,
    REPORT_SYSTEM_PROMPT,
    REPORT_SYSTEM_PROMPT_EN,
    build_report_user_prompt,
)
from .screener import brief_sheet, evidence_index
from .textutil import collapse_ws, is_verbatim, sentence_spans

CITE_RE = re.compile(r"\[(ev_\d{3,})\]")


class ReportValidationError(RuntimeError):
    pass


# ---------------------------------------------------------------- 校验


FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    ("可以一起吃", "越界结论：不得替医生判断联用安全性"),
    ("可以一起吃", "越界结论"),
    ("放心吃", "越界结论"),
    ("绝对安全", "越界结论"),
    ("没有问题", "越界结论"),
    ("没有冲突", "越界结论：只能陈述'说明书未记载'"),
    ("不会冲突", "越界结论"),
    ("无需担心", "越界结论"),
    ("建议停药", "越界建议：不得建议停药"),
    ("建议您停药", "越界建议"),
    ("改成", "越界建议：不得给出剂量/换药建议"),
    ("替代医生", "越界表述"),
)

SAFE_QUALIFIERS = (
    "未发现", "未记载", "没有记载", "不足以", "不等于", "不代表",
    "说明书未", "本次核查", "不构成", "不能说明", "无法确认", "未提及",
)


def _scan_text_forbidden(text: str, where: str, problems: list[str]) -> None:
    if not text:
        return
    for pat, why in FORBIDDEN_PATTERNS:
        idx = text.find(pat)
        if idx < 0:
            continue
        window = text[max(0, idx - 30) : idx + len(pat) + 30]
        if any(q in window for q in SAFE_QUALIFIERS):
            continue
        problems.append(f"{where} 出现越界措辞「{pat}」（{why}）：…{collapse_ws(window)}…")


def validate_draft(
    draft: ReportDraft,
    *,
    registry: dict[str, Evidence],
    sources: dict[str, list[str]],
    screen: PairScreen,
    require_citations: bool = True,
) -> list[str]:
    """返回问题列表；空列表 = 通过。任何一条硬问题都会让调用方丢弃草稿。"""
    problems: list[str] = []

    # 1) 风险等级单调性
    if draft.verdict not in Verdict.RANK:
        problems.append(f"verdict 取值非法：{draft.verdict!r}")
    else:
        if Verdict.RANK[draft.verdict] < Verdict.RANK[screen.verdict_floor]:
            problems.append(
                f"风险等级被下调：模型给出「{draft.verdict}」，"
                f"低于确定性筛查的下限「{screen.verdict_floor}」——这是不允许的。"
            )

    # 2) 引用必须存在
    cited = set(CITE_RE.findall(json.dumps(draft.findings, ensure_ascii=False)))
    cited |= set(draft.evidence_ids)
    unknown = sorted(c for c in cited if c not in registry)
    if unknown:
        problems.append(f"引用了不存在的 evidence_id：{', '.join(unknown)}")

    # 3) 引用必须存在（每条 finding 的 evidence_ids）
    for i, finding in enumerate(draft.findings):
        missing = [x for x in (finding.get("evidence_ids") or []) if x not in registry]
        if missing:
            problems.append(f"findings[{i}] 引用不存在的 evidence_id：{', '.join(missing)}")

    # 4) analysis 里必须出现引用
    if require_citations:
        for i, finding in enumerate(draft.findings):
            analysis = str(finding.get("analysis") or "")
            ids = CITE_RE.findall(analysis) or list(finding.get("evidence_ids") or [])
            if not ids:
                problems.append(
                    f"findings[{i}]「{finding.get('title', '')[:24]}」没有任何 evidence_id 引用。"
                )
            bad = [x for x in ids if x not in registry]
            if bad:
                problems.append(f"findings[{i}] 引用不存在：{', '.join(bad)}")

    # 5) 引文必须逐字命中官方原文
    for i, finding in enumerate(draft.findings):
        for quote in finding.get("sentences") or []:
            text = str(quote.get("text") or "")
            ev_id = str(quote.get("evidence_id") or "")
            if not text.strip():
                continue
            pool = sources.get(ev_id) or [ev.quote for ev in registry.values()]
            if not is_verbatim(text, pool):
                problems.append(
                    f"findings[{i}] 的引文无法在官方原文中逐字匹配（疑似转述或编造）："
                    f"“{collapse_ws(text)[:80]}…”"
                )

    # 6) 简略说明书的分节文字也不许出现越界措辞
    for key in ("brief_a", "brief_b"):
        brief = draft.__dict__.get(key) or {}
        _scan_text_forbidden(str(brief.get("one_liner") or ""), f"{key}.one_liner", problems)
        for sec in brief.get("sections") or []:
            _scan_text_forbidden(str(sec.get("text") or ""), f"{key}.sections", problems)

    for i, finding in enumerate(draft.findings):
        _scan_text_forbidden(str(finding.get("analysis") or ""), f"findings[{i}].analysis", problems)
        _scan_text_forbidden(str(finding.get("action") or ""), f"findings[{i}].action", problems)
    _scan_text_forbidden(draft.one_line_conclusion, "one_line_conclusion", problems)
    _scan_text_forbidden(draft.headline, "headline", problems)

    # 7) 发现项必须与确定性筛查一一对应（不许删项）
    if len(draft.findings) < len(screen.findings):
        problems.append(
            f"findings 数量少于确定性筛查结果（{len(draft.findings)} < {len(screen.findings)}），"
            "模型不得删减已命中的规则。"
        )

    return problems


# ---------------------------------------------------------------- 撰写


class ReportWriter:
    def __init__(self, settings: Settings, client: DeepSeekClient | None = None) -> None:
        self.s = settings
        self.client = client or DeepSeekClient(settings)

    # -------------------------------------------------------- 证据打包

    @staticmethod
    def evidence_block(registry: dict[str, Evidence], max_quote: int = 1100) -> str:
        lines: list[str] = []
        for ev_id in sorted(registry):
            ev = registry[ev_id]
            quote = collapse_ws(ev.quote)
            if len(quote) > max_quote:
                quote = quote[:max_quote].rstrip() + " …"
            lines.append(
                f"{ev_id} | {ev.drug} | {ev.section_zh}({ev.section}) | "
                f"出处 {ev.url}\n    原文：{quote}"
            )
        return "\n".join(lines) if lines else "（无）"

    @staticmethod
    def screen_block(screen: PairScreen) -> str:
        rows: list[str] = []
        for i, f in enumerate(screen.findings, 1):
            rows.append(
                f"{i}. [{f.severity}] {f.title}\n"
                f"   规则：{f.rule_id} | 证据：{', '.join(f.evidence_ids) or '（无原文，属结构性提示）'}\n"
                f"   说明：{collapse_ws(f.detail)}\n"
                f"   建议动作：{f.action}"
            )
        if not rows:
            rows.append("（确定性规则未命中任何条目）")
        return "\n".join(rows)

    @staticmethod
    def coverage_block(screen: PairScreen) -> str:
        out: list[str] = []
        for key in ("A", "B"):
            c = screen.coverage.get(key) or {}
            out.append(
                f"- {key}：{c.get('drug')}（查询名 {c.get('queried_as')}）"
                f"命中={c.get('matched')} 安全字段数={c.get('n_safety_fields')} "
                f"相互作用章节={c.get('has_interaction_section')} "
                f"黑框警告={c.get('has_boxed_warning')} "
                f"禁忌={c.get('has_contraindications')} "
                f"说明书版本={c.get('label_effective_time') or '未登记'}"
            )
        for note in screen.limitations:
            out.append(f"- 限制：{note}")
        return "\n".join(out)

    # -------------------------------------------------------- JSON 修复

    def _repair_json(
        self, broken: str, error: Exception, lang: str
    ) -> dict[str, Any] | None:
        """把一段坏 JSON 交给模型修好。失败返回 None（调用方按原错误处理）。

        注意：修复只允许改**结构**，内容仍然要过后面所有的引用校验 ——
        所以这不是"放宽标准"，只是少一次无谓的降级。
        """
        if not broken.strip():
            return None
        ask = (
            "The following text was meant to be a single JSON object but failed to parse "
            f"({error}). Reply with the corrected COMPLETE JSON object only — same content, "
            "no markdown fences, no commentary.\n\n"
            if lang == "en"
            else "下面这段文本本应是一个 JSON 对象，但解析失败"
            f"（{error}）。请只回复修正后的**完整 JSON 对象**：内容保持不变，不要 markdown 代码块，"
            "不要任何解释。\n\n"
        )
        try:
            reply = self.client.chat(
                [{"role": "user", "content": ask + broken}],
                json_mode=True,
                temperature=0.0,
            )
            return _extract_json(reply.get("content") or "")
        except (LLMError, json.JSONDecodeError, ValueError):
            return None

    # -------------------------------------------------------- 主流程

    def write(
        self,
        *,
        record_a: LabelRecord,
        record_b: LabelRecord,
        screen: PairScreen,
        registry: dict[str, Evidence],
        sources: dict[str, list[str]],
        lang: str = "zh",
    ) -> ReportDraft:
        user_prompt = build_report_user_prompt(
            drug_a=record_a.display,
            drug_b=record_b.display,
            evidence_block=self.evidence_block(registry),
            screen_block=self.screen_block(screen),
            coverage_block=self.coverage_block(screen),
            floor=screen.verdict_floor,
            lang=lang,
        )
        messages = [
            {
                "role": "system",
                "content": REPORT_SYSTEM_PROMPT_EN if lang == "en" else REPORT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": user_prompt},
        ]

        last_error = ""
        for attempt in range(2):
            try:
                reply = self.client.chat(messages, json_mode=True, temperature=0.0)
                raw_text = reply.get("content") or ""
                try:
                    raw = _extract_json(raw_text)
                except (json.JSONDecodeError, ValueError) as exc:
                    # ★ 一次性 JSON 修复：模型偶尔会返回被截断或转义错的 JSON，
                    #   直接丢掉整份草稿太浪费（内容本身可能完全正确）。
                    repaired = self._repair_json(raw_text, exc, lang)
                    if repaired is None:
                        raise
                    raw = repaired
            except (LLMError, json.JSONDecodeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The previous output could not be parsed as JSON ({exc}). "
                            "Output ONE valid JSON object only."
                            if lang == "en"
                            else f"上一次输出无法解析为 JSON（{exc}）。请只输出一个 JSON 对象。"
                        ),
                    }
                )
                continue

            draft = _draft_from_json(raw, screen, sources=sources)
            problems = validate_draft(
                draft, registry=registry, sources=sources, screen=screen
            )
            if not problems:
                draft.generated_by = "llm"
                return draft

            last_error = "；".join(problems[:3])
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your report failed validation:\n- "
                        + "\n- ".join(problems[:6])
                        + "\nFix these and output the COMPLETE JSON again. Remember: only verbatim "
                        "sentences from the evidence list; never lower the risk tier; never add a "
                        "medical fact."
                        if lang == "en"
                        else "你的报告没有通过校验，问题如下：\n- "
                        + "\n- ".join(problems[:6])
                        + "\n请修正后**重新输出完整 JSON**。"
                        "注意：只能用证据清单里的原文整句；不得下调风险等级；不得新增医学事实。"
                    ),
                }
            )

        raise ReportValidationError(last_error or "模型输出未通过校验")


def _draft_from_json(
    raw: dict[str, Any],
    screen: PairScreen,
    sources: dict[str, list[str]] | None = None,
) -> ReportDraft:
    findings = raw.get("findings")
    if not isinstance(findings, list):
        findings = []
    sources = sources or {}
    clean_findings: list[dict[str, Any]] = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        # ★ 逐字过滤：对不上原文的"引文"直接丢掉（渲染时改用证据库里的原文），
        #   整句话都不许它进报告。
        sentences: list[dict[str, str]] = []
        for s in f.get("sentences") or []:
            if not isinstance(s, dict):
                continue
            text = str(s.get("text") or "").strip()
            ev_id = str(s.get("evidence_id") or "").strip()
            if not text:
                continue
            pool = sources.get(ev_id) or []
            if not is_verbatim(text, pool):
                continue
            sentences.append({"text": text, "evidence_id": ev_id})
        clean_findings.append(
            {
                "title": str(f.get("title") or "").strip(),
                "severity": str(f.get("severity") or Severity.MEDIUM).strip(),
                "analysis": str(f.get("analysis") or "").strip(),
                "action": str(f.get("action") or "").strip(),
                "evidence_ids": [
                    str(x) for x in (f.get("evidence_ids") or []) if isinstance(x, (str, int))
                ],
                "sentences": sentences,
            }
        )

    def brief(key: str) -> dict[str, Any]:
        b = raw.get(key) or {}
        if not isinstance(b, dict):
            b = {}
        sections = []
        for sec in b.get("sections") or []:
            if isinstance(sec, dict):
                sections.append(
                    {
                        "section": str(sec.get("section") or "").strip(),
                        "text": str(sec.get("text") or "").strip(),
                    }
                )
        return {
            "drug_name": str(b.get("drug_name") or "").strip(),
            "one_liner": str(b.get("one_liner") or "").strip(),
            "sections": sections,
        }

    verdict = str(raw.get("verdict") or Verdict.UNKNOWN).strip()
    verdict = Verdict.at_least(verdict, screen.verdict_floor)  # ★ 单调约束（硬编码，不只靠提示词）

    return ReportDraft(
        verdict=verdict,
        headline=str(raw.get("headline") or "").strip(),
        one_line_conclusion=str(raw.get("one_line_conclusion") or "").strip(),
        findings=clean_findings,
        brief_a=brief("brief_a"),
        brief_b=brief("brief_b"),
        evidence_ids=[str(x) for x in (raw.get("evidence_ids") or [])],
        limitations=[str(x) for x in (raw.get("limitations") or []) if str(x).strip()],
        generated_by="llm",
    )


# ---------------------------------------------------------------- 无 Key 的模板引擎


def build_template_draft(
    *,
    record_a: LabelRecord,
    record_b: LabelRecord,
    screen: PairScreen,
    registry: dict[str, Evidence],
    lang: str = "zh",
    notice: str = "",
) -> ReportDraft:
    """确定性模板报告：结构、引用、原文全部与 LLM 版一致，只是措辞更朴素。"""
    en = lang == "en"
    findings: list[dict[str, Any]] = []
    for f in screen.findings:
        # 句子级证据：直接把命中整句原文摆出来（模板不转述，只呈现）
        # ★ 只用"句子边界内的完整原文"，绝不加省略号——否则会在逐字校验里被判成转述。
        sentences = []
        for ev_id in f.evidence_ids:
            ev = registry.get(ev_id)
            if ev is None:
                continue
            sentences.append({"text": _verbatim_excerpt(ev.quote, 600), "evidence_id": ev_id})
        text = f.localized(lang)
        findings.append(
            {
                "title": text["title"],
                "severity": f.severity,
                "analysis": collapse_ws(text["detail"].replace("**", ""))
                + (" " + " ".join(f"[{i}]" for i in f.evidence_ids) if f.evidence_ids else ""),
                "action": text["action"],
                "evidence_ids": list(f.evidence_ids),
                "sentences": sentences,
            }
        )

    verdict = screen.verdict_floor
    pair = f"{record_a.display} × {record_b.display}"
    if en:
        one_line = {
            Verdict.CRITICAL: (
                f"Official labeling contains an explicit, serious statement about the {pair} "
                "combination (see the findings and verbatim quotes below). This tool makes no "
                "medication decision — tell your prescriber or pharmacist before you take it."
            ),
            Verdict.HIGH: (
                f"Official labeling flags the {pair} combination in prescriber-facing sections "
                "(possibly only in one of the two labels). That does not mean it must not be used "
                "together, but it is a pair worth confirming with a doctor or pharmacist."
            ),
            Verdict.LOW: (
                f"No direct statement about {pair} was found in the official labels retrieved. "
                "Labels usually list only well-studied interactions, so **absence of a record is "
                "not proof of safety**."
            ),
            Verdict.UNKNOWN: (
                f"This check could not assess {pair}: the official data needed is missing (see the "
                "gaps section). Until it is resolved and a pharmacist is consulted, do not treat "
                "this pair as confirmed safe."
            ),
        }[verdict]
    elif verdict == Verdict.CRITICAL:
        one_line = (
            f"官方说明书中存在关于 {pair} 组合的**明确、严重级别的记载**（详见下文发现与原文）。"
            "本工具不做用药决定，只负责把说明书原文摆到你面前：请在服用前把这个组合告知开方医生或药师。"
        )
    elif verdict == Verdict.HIGH:
        one_line = (
            f"官方说明书对 {pair} 组合有需要特别留意的记载（可能只出现在其中一份说明书里）。"
            "这不等于「不能一起用」，但属于值得让医生或药师确认的组合。"
        )
    elif verdict == Verdict.LOW:
        one_line = (
            f"在已取得的官方说明书中，**没有发现直接记载 {pair} 相互作用的条文**。"
            "请注意：说明书通常只列出经过充分研究的相互作用，未记载不等于已证实安全。"
        )
    else:
        one_line = (
            f"本次核查**资料不足**，无法对 {pair} 是否冲突给出结论（详见「本次核查的边界与资料缺口」）。"
            "在补齐资料或咨询药师之前，请不要把这个组合当作「已确认安全」。"
        )

    if notice:
        one_line = notice.strip() + "\n\n" + one_line

    headline = (
        f"{pair} — {verdict_label(verdict, lang)}"
        if en
        else f"{pair}：{verdict_label(verdict, lang)}（确定性筛查结果）"
    )
    return ReportDraft(
        verdict=verdict,
        headline=headline,
        one_line_conclusion=one_line,
        findings=findings,
        brief_a=_template_brief(record_a, lang),
        brief_b=_template_brief(record_b, lang),
        evidence_ids=[ev.evidence_id for ev in record_a.evidence_list()]
        + [ev.evidence_id for ev in record_b.evidence_list()],
        limitations=[n.zh for n in screen.limitations],
        generated_by="template",
    )


def _template_brief(record: LabelRecord, lang: str = "zh") -> dict[str, Any]:
    sheet = brief_sheet(record)
    en = lang == "en"
    sections = []
    for block in sheet.get("sections", []):
        texts = []
        for item in block.get("items", [])[:2]:
            texts.append(f"{collapse_ws(item['text'])} [{item['evidence_id']}]")
        if texts:
            sections.append({"section": block["section"], "text": " ".join(texts)})
    missing = sheet.get("missing") or []
    if missing:
        sections.append(
            {
                "section": "Sections absent from label" if en else "说明书未提供",
                "text": (
                    ", ".join(missing)
                    + " — these sections are absent from the label. That is an information gap, "
                    "not a statement of safety."
                    if en
                    else "、".join(missing)
                    + " —— 该说明书里没有这几节内容，属于信息缺口，不代表「安全」。"
                ),
            }
        )
    return {
        "drug_name": sheet.get("drug_name") or record.query,
        "one_liner": (
            (
                f"{sheet.get('drug_name')}：本次核查共取到 {len(record.fields)} 个安全字段。"
                if record.matched
                else f"{record.query}：未取得官方说明书（{record.message or '未找到'}），"
                "无法给出安全要点。"
            )
            if not en
            else (
                f"{sheet.get('drug_name')}: {len(record.fields)} safety sections retrieved."
                if record.matched
                else f"{record.query}: no official label retrieved, so no safety points."
            )
        ),
        "sections": sections,
    }


# ---------------------------------------------------------------- 渲染


SEVERITY_EMOJI = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.INFO: "🔵",
}

VERDICT_EMOJI = {
    Verdict.CRITICAL: "🔴",
    Verdict.HIGH: "🟠",
    Verdict.LOW: "🟢",
    Verdict.UNKNOWN: "⚪",
}


def render_markdown(
    *,
    draft: ReportDraft,
    record_a: LabelRecord,
    record_b: LabelRecord,
    screen: PairScreen,
    registry: dict[str, Evidence],
    lang: str = "zh",
    compact: bool = False,
) -> str:
    """渲染报告。

    lang="zh" 出完整中文报告；lang="en" 出英文版；
    compact=True 时只出一页"简洁版"（结论 + 要点 + 证据编号），供 main_sample.py 使用。
    """
    return (
        _render_compact(draft, record_a, record_b, screen, registry, lang)
        if compact
        else _render_full(draft, record_a, record_b, screen, registry, lang)
    )


def _note_text(note, lang: str) -> str:
    if hasattr(note, "text"):
        return note.text(lang)
    return str(note)


def _render_compact(
    draft: ReportDraft,
    record_a: LabelRecord,
    record_b: LabelRecord,
    screen: PairScreen,
    registry: dict[str, Evidence],
    lang: str,
) -> str:
    """简洁版：一页看完「结论 + 依据 + 该做什么」。英文版只输出英文结论。"""
    en = lang == "en"
    pair = f"{record_a.display} × {record_b.display}"
    lines: list[str] = []

    if en:
        lines.append(f"# Drug interaction check — {pair}")
    else:
        lines.append(f"# 用药冲突核查（简洁版）— {pair}")
    lines.append("")
    lines.append(
        f"## {VERDICT_EMOJI.get(draft.verdict, '')} "
        f"{verdict_label(draft.verdict, lang, with_qualifier=(draft.verdict == Verdict.LOW))}"
    )
    gloss = verdict_gloss(draft.verdict, lang)
    if gloss:
        lines.append("")
        lines.append(f"> {gloss}")
    lines.append("")
    if draft.one_line_conclusion:
        lines.append(draft.one_line_conclusion.strip())
        lines.append("")

    if draft.findings:
        lines.append(f"**{'Findings' if en else '要点'}**")
        lines.append("")
        for i, f in enumerate(draft.findings, 1):
            sev = str(f.get("severity") or Severity.MEDIUM)
            ids = " ".join(f"[{x}]" for x in (f.get("evidence_ids") or []) if x in registry)
            lines.append(
                f"{i}. {SEVERITY_EMOJI.get(sev, '🟡')} {f.get('title', '')} {ids}".rstrip()
            )
        lines.append("")

    if draft.brief_a or draft.brief_b:
        lines.append(f"**{'Brief safety notes' if en else '两种药的安全要点'}**")
        lines.append("")
        for rec in (record_a, record_b):
            cov = rec.coverage()
            bits: list[str] = []
            if cov["has_boxed_warning"]:
                bits.append("boxed warning" if en else "黑框警告")
            if cov["has_contraindications"]:
                bits.append("contraindications" if en else "禁忌症")
            if cov["has_interaction_section"]:
                bits.append("interaction section" if en else "相互作用章节")
            else:
                bits.append(
                    "NO interaction section in label" if en else "说明书无相互作用章节"
                )
            name = rec.display if rec.matched else rec.query
            if en:
                lines.append(f"- **{name}**: {', '.join(bits)}")
            else:
                lines.append(f"- **{name}**：{'、'.join(bits)}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        f"Evidence: {len(registry)} verbatim quotes from {len(screen.findings)} rule hit(s); "
        f"run `{('python main.py') if not en else 'python main.py'}` with the same pair for the full "
        f"report with every quote and source link."
        if en
        else f"证据：{len(registry)} 条官方原文、{len(screen.findings)} 条规则命中；"
             f"同一组合跑 `python main.py` 可看带全部原文与溯源链接的完整报告。"
    )
    lines.append("")
    for note in screen.limitations[:3]:
        lines.append(f"- {_note_text(note, lang)}")
    lines.append("")
    lines.append(
        "> Not medical advice. Generated from FDA official labeling (openFDA drug label). "
        "Whether to combine these drugs must be decided by your prescriber or pharmacist."
        if en
        else "- 本报告不是医嘱，不能替代医生或药师判断。"
    )
    return "\n".join(lines)


def _render_full(
    draft: ReportDraft,
    record_a: LabelRecord,
    record_b: LabelRecord,
    screen: PairScreen,
    registry: dict[str, Evidence],
    lang: str,
) -> str:
    en = lang == "en"
    lines: list[str] = []
    pair = f"{record_a.display} × {record_b.display}"
    verdict_shown = verdict_label(
        draft.verdict, lang, with_qualifier=(draft.verdict == Verdict.LOW)
    )

    lines.append(f"# {t('title', lang)}")
    lines.append("")
    lines.append(f"**{t('pair', lang)}**：{pair}" if not en else f"**{t('pair', lang)}**: {pair}")
    lines.append(
        f"**{t('result', lang)}**：{VERDICT_EMOJI.get(draft.verdict, '')} **{verdict_shown}**"
        f"（{t('generated_by_llm' if draft.generated_by == 'llm' else 'generated_by_template', lang)}）"
        if not en
        else f"**{t('result', lang)}**: {VERDICT_EMOJI.get(draft.verdict, '')} **{verdict_shown}** "
             f"({t('generated_by_llm' if draft.generated_by == 'llm' else 'generated_by_template', lang)})"
    )
    lines.append("")
    gloss = verdict_gloss(draft.verdict, lang)
    if gloss:
        lines.append(f"> {gloss}")
        lines.append("")
    if draft.headline:
        lines.append(f"## {draft.headline}")
        lines.append("")
    if draft.one_line_conclusion:
        lines.append(draft.one_line_conclusion)
        lines.append("")

    # ---- 详细分析
    lines.append(f"## {t('analysis_header', lang)}")
    lines.append("")
    if not draft.findings:
        lines.append(t("no_findings", lang))
        lines.append("")
    else:
        order = {"critical": 0, "high": 1, "medium": 2, "info": 3}
        ordered = sorted(
            enumerate(draft.findings), key=lambda p: order.get(str(p[1].get("severity")), 9)
        )
        lines.append(t("sorted_hint", lang))
        lines.append("")
        for rank, (_orig, f) in enumerate(ordered, 1):
            sev = str(f.get("severity") or Severity.MEDIUM)
            lines.append(
                f"{rank}. {SEVERITY_EMOJI.get(sev, '🟡')} "
                f"[{severity_label(sev, lang)}] {f.get('title', '')}"
            )
        lines.append("")

    for i, f in enumerate(draft.findings, 1):
        sev = str(f.get("severity") or Severity.MEDIUM)
        badge = SEVERITY_EMOJI.get(sev, "🟡")
        head = (
            f"### {i}. {badge} {f.get('title', '')}（{severity_label(sev, lang)}）"
            if not en
            else f"### {i}. {badge} {f.get('title', '')} ({severity_label(sev, lang)})"
        )
        lines.append(head)
        lines.append("")
        analysis = f.get("analysis") or ""
        if analysis:
            lines.append(analysis)
            lines.append("")
        for s in f.get("sentences") or []:
            text = collapse_ws(str(s.get("text") or ""))
            if text:
                lines.append(f"> {text}")
                lines.append(
                    f"> —— [{s.get('evidence_id')}] {_ev_cite(registry, str(s.get('evidence_id') or ''), lang)}"
                    if not en
                    else f"> — [{s.get('evidence_id')}] {_ev_cite(registry, str(s.get('evidence_id') or ''), lang)}"
                )
                lines.append("")
        ids = [x for x in (f.get("evidence_ids") or []) if x in registry]
        if ids and not (f.get("sentences")):
            for ev_id in ids:
                ev = registry[ev_id]
                lines.append(f"> {_shorten(ev.quote, 420)}")
                lines.append(
                    f"> —— [{ev_id}] {ev.drug} · {ev.section_zh} · {ev.url}"
                    if not en
                    else f"> — [{ev_id}] {ev.drug} · {ev.section} · {ev.url}"
                )
                lines.append("")
        if f.get("action"):
            lines.append(
                f"**{t('suggested_action', lang)}**：{f['action']}"
                if not en
                else f"**{t('suggested_action', lang)}**: {f['action']}"
            )
            lines.append("")

    # ---- 简略说明书
    lines.append(f"## {t('brief_header', lang)}")
    lines.append("")
    for tag, rec, brief in (("A", record_a, draft.brief_a), ("B", record_b, draft.brief_b)):
        lines.extend(_render_brief(tag, rec, brief, registry, lang))
        lines.append("")

    # ---- 资料缺口
    lines.append(f"## {t('gap_header', lang)}")
    lines.append("")
    seen_notes: set[str] = set()
    for note in screen.limitations:
        text = _note_text(note, lang)
        if text in seen_notes:
            continue
        seen_notes.add(text)
        lines.append(f"- {text}")
    for note in draft.limitations:
        text = _note_text(note, lang)
        if not text or text in seen_notes:
            continue
        seen_notes.add(text)
        lines.append(f"- {text}")
    for rec in (record_a, record_b):
        if not rec.matched:
            lines.append(
                f"- **{rec.query}**：未取得官方说明书（{rec.message or '未找到'}）。"
                if not en
                else f"- **{rec.query}**: no official label retrieved ({rec.message or 'not found'})."
            )
        elif rec.missing_fields:
            gaps = [
                f for f in ("drug_interactions", "contraindications", "boxed_warning")
                if f in rec.missing_fields
            ]
            if gaps:
                if en:
                    lines.append(
                        f"- **{rec.display}** {t('label_missing_sections', lang)} "
                        + ", ".join(gaps)
                        + " " + t("otc_note", lang)
                    )
                else:
                    lines.append(
                        f"- **{rec.display}** 的说明书缺少以下安全章节："
                        + "、".join(
                            {
                                "drug_interactions": "药物相互作用",
                                "contraindications": "禁忌症",
                                "boxed_warning": "黑框警告",
                            }[g]
                            for g in gaps
                        )
                        + "（该标签可能为简化/OTC 标签）。"
                    )
    lines.append("")

    # ---- 出处清单
    lines.append(f"## {t('sources_header', lang)}")
    lines.append("")
    rows = evidence_index([e for e in registry.values()], limit_quote=360)
    if not rows:
        lines.append(t("no_sources", lang))
    for row in rows:
        if en:
            lines.append(
                f"- **[{row['evidence_id']}]** {row['drug']} · {row['section']}"
                f" · {t('retrieved_at', lang)} {row['retrieved_at']}\n"
                f"  - {t('original_text', lang)}: {row['quote']}\n"
                f"  - {t('traceback', lang)}: {row['url']}"
            )
        else:
            lines.append(
                f"- **[{row['evidence_id']}]** {row['drug']} · {row['section_zh']}"
                f"（`{row['section']}`）· 检索于 {row['retrieved_at']}\n"
                f"  - 原文：{row['quote']}\n"
                f"  - 溯源：{row['url']}"
            )
    lines.append("")

    # ---- 元信息
    lines.append(f"## {t('meta_header', lang)}")
    lines.append("")
    for rec in (record_a, record_b):
        lines.append(
            f"- **{rec.query}** → {rec.display if rec.matched else ('未命中' if not en else 'no match')}"
            f"｜strategy `{rec.strategy or 'n/a'}`｜source `{rec.fetched_from or 'n/a'}`"
            f"｜label version `{rec.effective_time or t('not_registered', lang)}`"
            f"｜safety fields {len(rec.fields)}"
        )
    # 药理类别（"说明书按类别描述"检测的依据，必须让读者看到来源）
    for rec in (record_a, record_b):
        if not rec.pharm_classes:
            lines.append(
                f"- **{rec.display if rec.matched else rec.query}** 的药理类别："
                + ("未取得（类别级检查未覆盖此药）" if not en
                   else "not retrieved (class-level check did not cover this drug)")
            )
            continue
        pretty = "、".join(strip_class_suffix(c) for c in rec.pharm_classes[:8])
        source = rec.pharm_class_source or "openFDA"
        lines.append(
            f"- **{rec.display if rec.matched else rec.query}** 的药理类别"
            f"（来源 `{source}`）：{pretty}"
            if not en
            else f"- **{rec.display if rec.matched else rec.query}** pharmacologic classes "
                 f"(source `{source}`): {pretty}"
        )
    lines.append("- Data source: openFDA drug label API (FDA official SPL labeling)")
    lines.append(
        "- Pipeline: two drug names → official labels → strict SAFETY_FIELDS extraction → "
        "deterministic rules (direct mention / duplicate ingredient / missing section / "
        "boxed warning / pharmacologic-class mention) → evidence-bound writing → verbatim "
        "citation check"
    )
    lines.append(
        "- Pharmacologic class data: openFDA `/drug/ndc.json` (`pharm_class`) with "
        "`/drug/label.json` (`openfda.pharm_class_*`) as fallback and a small curated table "
        "for offline use"
    )
    lines.append("")
    lines.append(DISCLAIMER_EN if en else DISCLAIMER)
    return "\n".join(lines)


def _render_brief(
    tag: str,
    record: LabelRecord,
    brief: dict[str, Any],
    registry: dict[str, Evidence],
    lang: str = "zh",
) -> list[str]:
    en = lang == "en"
    out: list[str] = []
    name = brief.get("drug_name") or record.display
    out.append(f"### {tag}. {name}")
    out.append("")
    if en:
        meta = [
            f"{t('generic_name', lang)}: {'; '.join(record.generic_names) or t('not_registered', lang)}",
            f"{t('brand_name', lang)}: {'; '.join(record.brand_names[:3]) or t('not_registered', lang)}",
            f"{t('active_ingredient', lang)}: {'; '.join(record.active_ingredients) or t('not_registered', lang)}",
            f"{t('label_version', lang)}: {record.effective_time or t('not_registered', lang)}",
        ]
        out.append(" | ".join(meta))
    else:
        meta = [
            f"通用名：{'、'.join(record.generic_names) or '未登记'}",
            f"商品名：{'、'.join(record.brand_names[:3]) or '未登记'}",
            f"有效成分：{'、'.join(record.active_ingredients) or '未登记'}",
            f"说明书版本：{record.effective_time or '未登记'}",
        ]
        out.append("｜".join(meta))
    out.append("")
    if not record.matched:
        if en:
            out.append(
                f"**{t('no_label', lang)}**: {record.message or 'not found'}. "
                f"{t('no_label_tail', lang)}"
            )
        else:
            out.append(
                f"**未能取得官方说明书**：{record.message or '未找到'}。因此本节没有可引用的安全要点。"
            )
        return out
    if brief.get("one_liner"):
        out.append(brief["one_liner"])
        out.append("")
    for sec in brief.get("sections") or []:
        title = sec.get("section") or ""
        text = collapse_ws(sec.get("text") or "")
        if not text:
            continue
        out.append(f"- **{title}**：{text}" if not en else f"- **{title}**: {text}")
    out.append("")
    out.append(f"{t('traceback', lang)}：{record.url}" if not en else f"{t('traceback', lang)}: {record.url}")
    return out


def _ev_cite(registry: dict[str, Evidence], ev_id: str, lang: str = "zh") -> str:
    ev = registry.get(ev_id)
    if not ev:
        return ""
    section = ev.section if lang == "en" else ev.section_zh
    return f"{ev.drug} · {section} · {ev.url}"


def _shorten(text: str, limit: int) -> str:
    text = collapse_ws(text)
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _verbatim_excerpt(text: str, limit: int) -> str:
    """在**句子边界**上截断，结果仍是原文的连续子串（不加省略号）。

    用于任何会被"逐字校验"或"人肉核对"的引文：宁可短，也必须逐字为真。
    """
    flat = collapse_ws(text)
    if len(flat) <= limit:
        return flat
    best = ""
    for start, end in sentence_spans(flat):
        if end > limit:
            break
        best = flat[:end]
    return best or flat[:limit].rstrip()
