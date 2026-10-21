"""Agent 编排层：控制流 + 工具 + 状态 + 边界。

一句话概括这个 Agent 的分工：
> **代码负责"事实"，模型负责"措辞"，边界由硬规则守住。**

流程（每一步都会写进 traces/*.jsonl）：

    ① Observe    解析用户输入里的两个药名（含糊就先反问，不猜）
    ② Retrieve   lookup_drug_label × 2  —— 官方说明书，查不到就如实说
    ③ Detect     screen_pair_interaction —— 确定性规则，算出风险等级下限
    ④ Safety     verdict = max(模型结论, 代码下限)；缺资料则禁止下"无冲突"结论
    ⑤ Write      证据化撰写（LLM 或模板），再做逐字引用校验
    ⑥ Escalate   高风险 → 明确要求人工（医生/药师）介入
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .api_tool import LabelClient
from .config import Settings
from .i18n import GATE_EN
from .llm_client import DeepSeekClient, LLMError, parse_tool_arguments
from .models import Evidence, LabelRecord, PairScreen, ReportDraft, Verdict
from .prompts import AGENT_SYSTEM_PROMPT, LLM_FAILED_NOTICE, NO_KEY_NOTICE
from .reviewer import (
    ReportValidationError,
    ReportWriter,
    build_template_draft,
    render_markdown,
    validate_draft,
)
from .safety_fields import EvidenceRegistry, LabelStore
from .textutil import is_probable_drug_name, normalize_query
from .tool_runtime import ToolRuntime
from .tools import TOOLS
from .trace import Tracer


@dataclass
class AgentResult:
    ok: bool
    report_markdown: str = ""
    verdict: str = ""
    engine: str = ""                 # llm | template | aborted
    escalation: str = ""             # 需要人工介入时的口径
    missing: list[str] = field(default_factory=list)
    run_id: str = ""
    trace_path: str = ""
    notes: list[str] = field(default_factory=list)
    screen: PairScreen | None = None
    records: list[LabelRecord] = field(default_factory=list)


class MedCheckAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.s = settings or Settings()
        self.trace = tracer or Tracer(enabled=self.s.trace)
        self.client = LabelClient(self.s, self.trace)
        self.registry = EvidenceRegistry(self.s)
        self.store = LabelStore(self.s, self.registry)
        self.runtime = ToolRuntime(self.client, self.store)
        self.messages: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self._notice_shown = False
        if not self.s.has_api_key:
            # 总是记录"没有 Key"这件事；是否打印到终端由 analyze() 按引擎决定，
            # 避免 --engine template 时重复刷屏。
            self.notes.append(NO_KEY_NOTICE)
        self._llm_ready = False

    # ------------------------------------------------------------ 终端输出

    def _say(self, text: str) -> None:
        if self.s.verbose:
            print(text, flush=True)

    # ------------------------------------------------------------ ① 解析输入

    def extract_drug_names(self, text: str) -> list[str]:
        """尽量从自由文本里抠出两个药名（先按分隔符，再按已知中文药名）。

        抠不出来就返回空列表 —— 交给 LLM 循环去反问用户，**绝不猜**。
        """
        if not text:
            return []
        import re

        raw = text.strip()
        parts: list[str] = []
        for sep in ("、", ",", "，", " 和 ", " 与 ", "和", "与", " vs ", " vs.", " vs", "×", " x ", "|", "/"):
            if sep in raw:
                parts = [p.strip() for p in raw.split(sep) if p.strip()]
                break
        if not parts:
            # 没有分隔符：试试"药名1 药名2"这种空格分隔
            parts = [p for p in re.split(r"\s{1,}", raw) if p]
        cleaned = [normalize_query(p) for p in parts if p]
        cleaned = [c for c in cleaned if c]
        # 去掉"请帮我查下"这类动词尾巴
        noise = ("查一下", "查询", "帮我", "看看", "冲突", "相互作用", "能不能一起吃", "一起", "吗", "？", "?")
        out = []
        for c in cleaned:
            for n in noise:
                c = c.replace(n, " ")
            c = re.sub(r"\s+", " ", c).strip(" -，,。.")
            if c:
                out.append(c)
        # ★ 只保留"像药名"的片段：宁可少认一个，交给模型反问用户，也不瞎猜
        return [c for c in out if is_probable_drug_name(c)][:2]

    # ------------------------------------------------------------ ②③ 检索与筛查

    def ensure_label(self, drug_name: str, slot: str = "") -> LabelRecord:
        result = self.client.fetch(drug_name)
        record = self.store.build(result)
        if slot:
            self.runtime.set_pair(record, slot)
            # 顺手取药理类别（用于"说明书按类别描述"的检测；有缓存，代价很小）
            info = self.runtime.fetch_classes(
                slot,
                drug_name,
                generic_hint=record.display if record.matched else "",
                substance_hint=" and ".join(record.active_ingredients) if record.matched else "",
            )
            record.pharm_classes = list(info.raw)
            record.pharm_class_source = info.source
        self.trace.log(
            "retrieve",
            {
                "query": drug_name,
                "matched": record.matched,
                "display": record.display,
                "strategy": record.strategy,
                "from": record.fetched_from,
                "safety_fields": len(record.fields),
                "evidence_ids": list(record.evidence),
            },
        )
        return record

    def ensure_pair_screen(self, force: bool = False) -> PairScreen | None:
        if self.runtime.screen is not None and not force:
            return self.runtime.screen
        payload = self.runtime.screen_pair_interaction(force=force)
        if "error" in payload:
            return None
        screen = self.runtime.screen
        self.trace.log(
            "screen",
            {
                "verdict_floor": screen.verdict_floor,
                "rules": [
                    {"rule_id": f.rule_id, "severity": f.severity, "evidence": f.evidence_ids}
                    for f in screen.findings
                ],
                "limitations": screen.limitations,
            },
        )
        return screen

    # ------------------------------------------------------------ ④ 安全闸门

    def safety_gate(
        self,
        screen: PairScreen,
        records: list[LabelRecord],
        lang: str = "zh",
        verdict: str | None = None,
    ) -> list[str]:
        """返回必须触发的升级口径（空列表 = 不需要额外升级）。

        verdict 传入**最终成稿的等级**：模型有可能把等级抬得比代码下限更高
        （单调约束允许），此时升级口径必须跟着升级，否则会出现
        "结论写危险、行动建议却写注意"的自相矛盾。

        ★ 只要存在资料缺口（某个药没查到官方标签），无论等级如何，
          都必须出现"去找人确认"的升级要求——不允许凭残缺资料给结论。
        """
        en = lang == "en"
        gates: list[str] = []
        missing = [r.query for r in records if not r.matched]
        level = Verdict.worse(screen.verdict_floor, verdict or screen.verdict_floor)

        if level == Verdict.CRITICAL:
            gates.append(
                GATE_EN["critical"]
                if en
                else "🔴 **本组合被判定为危险**：请在服用前把这两个药（连同剂量、服用时间）"
                "一起告诉开方医生或药师，由他们决定是否调整方案。在得到明确答复前，"
                "不要自行加服、减量或停药。"
            )
        elif level == Verdict.HIGH:
            gates.append(
                GATE_EN["high"]
                if en
                else "🟠 **本组合需要特别注意**：建议在下次服药前主动咨询药师或医生，"
                "并把官方说明书里被点名的风险点读一遍。"
            )
        elif level == Verdict.UNKNOWN:
            gates.append(
                GATE_EN["unknown"]
                if en
                else "⚪ **资料不足，无法核查**：本次没有拿到足够权威的资料来判断这个组合，"
                "请直接咨询药师或医生——不要因为「没查出问题」就当作可以放心服用。"
            )

        if any(f.rule_id == "R-DUP-001" for f in screen.findings):
            gates.append(
                GATE_EN["duplicate"]
                if en
                else "🔴 **有效成分重复**：同一种成分被两种药重复提供，属于明确的用药风险，"
                "请立即与开药医生核对处方。"
            )

        if missing:
            gates.append(
                GATE_EN["missing"].format(drugs=", ".join(missing))
                if en
                else "⚠️ **资料缺口**：" + "、".join(missing)
                + " 没有查到官方说明书，本次核查**不覆盖**这一项。"
                "请向药师或医生确认该药的信息后重新核查。"
            )
        return gates

    # ------------------------------------------------------------ ⑤ 报告

    def _llm_report(
        self,
        record_a: LabelRecord,
        record_b: LabelRecord,
        screen: PairScreen,
    ) -> ReportDraft:
        if not self.s.has_api_key:
            raise LLMError("未配置 DeepSeek API Key")
        writer = ReportWriter(self.s, DeepSeekClient(self.s))
        return writer.write(
            record_a=record_a,
            record_b=record_b,
            screen=screen,
            registry=self.runtime.registry,
            sources=self.runtime.all_sources(),
            lang=self.s.lang,
        )

    def compose_report(self, record_a: LabelRecord, record_b: LabelRecord) -> AgentResult:
        """从两份标签 + 筛查结果生成最终报告（含兜底路径）。"""
        screen = self.runtime.screen
        if screen is None:
            raise RuntimeError("compose_report 需要先完成配对筛查。")

        # --- 资料缺口：不做"无冲突"结论
        if not record_a.matched or not record_b.matched:
            missing = [r.query for r in (record_a, record_b) if not r.matched]
            screen.verdict_floor = Verdict.worse(screen.verdict_floor, Verdict.UNKNOWN)
            screen.add_note(
                "有药品未取得官方说明书（" + "、".join(missing) + "），"
                "因此**本次核查不能判定「无冲突」**。",
                "No official label was retrieved for: " + ", ".join(missing)
                + ". **This check therefore cannot conclude that the pair is free of conflicts.**",
            )
            self.trace.log("safety_gate", {"gate": "missing_label", "drugs": missing})

        draft: ReportDraft
        engine = self.s.engine
        notes: list[str] = list(self.notes)
        if engine == "llm" and not self.s.has_api_key:
            raise LLMError(
                "engine=llm 要求使用 DeepSeek，但没有配置 API Key。"
                "请设置 DEEPSEEK_API_KEY（或 Agent_test/.env、medcheck/config.py），"
                "或改用 --engine auto / --engine template。"
            )
        if engine == "auto" and not self.s.has_api_key:
            engine = "template"
            notes.append(NO_KEY_NOTICE)

        if engine in ("auto", "llm"):
            try:
                draft = self._llm_report(record_a, record_b, screen)
                self.trace.log(
                    "write",
                    {"engine": "llm", "verdict": draft.verdict,
                     "findings": len(draft.findings), "citations": len(draft.evidence_ids)},
                )
            except (LLMError, ReportValidationError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                if engine == "llm":
                    raise
                notice = LLM_FAILED_NOTICE.format(reason=reason[:200])
                notes.append(notice)
                self._say(notice)
                draft = build_template_draft(
                    record_a=record_a,
                    record_b=record_b,
                    screen=screen,
                    registry=self.runtime.registry,
                    lang=self.s.lang,
                )
                # 模板草稿同样要过校验（保证引用与单调性）
                problems = validate_draft(
                    draft,
                    registry=self.runtime.registry,
                    sources=self.runtime.all_sources(),
                    screen=screen,
                    require_citations=False,
                )
                if problems:
                    self._say("[warn] 模板草稿校验提示：" + "；".join(problems[:2]))
                self.trace.log(
                    "write",
                    {"engine": "template(fallback)", "reason": reason[:200]},
                )
        else:
            draft = build_template_draft(
                record_a=record_a,
                record_b=record_b,
                screen=screen,
                registry=self.runtime.registry,
                lang=self.s.lang,
            )
            self.trace.log("write", {"engine": "template"})

        # --- 硬约束再确认（双保险）
        draft.verdict = Verdict.at_least(draft.verdict, screen.verdict_floor)

        escalation = self.safety_gate(
            screen, [record_a, record_b], lang=self.s.lang, verdict=draft.verdict
        )
        if escalation:
            self.trace.log("escalate", {"gates": escalation})

        report = render_markdown(
            draft=draft,
            record_a=record_a,
            record_b=record_b,
            screen=screen,
            registry=self.runtime.registry,
            lang=self.s.lang,
            compact=self.s.compact,
        )
        lang = self.s.lang
        if escalation and not self.s.compact:
            block = "\n".join(f"> {g}" for g in escalation)
            anchor = "\n## " + (
                "1. Analysis" if lang == "en" else "一、详细分析"
            )
            heading = (
                "Action required (human review)" if lang == "en" else "需立即采取的行动（人工介入）"
            )
            if anchor in report:
                report = report.replace(
                    anchor, f"\n## {heading}\n\n{block}\n{anchor}", 1
                )
            else:
                # 简洁版没有分析小节：把升级口径直接插在标题之后
                report = report.replace("\n\n", f"\n\n> {block}\n\n", 1)
        title = "# " + ("Drug Interaction Check Report" if lang == "en" else "用药冲突核查报告")
        engine_label = draft.generated_by
        if draft.generated_by == "llm":
            engine_label = f"llm:{self.s.model}"
        header = (
            f"> run `{self.trace.run_id}` | engine `{engine_label}` | "
            f"source openFDA drug label (FDA official labeling) | "
            f"retrieved {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            if lang == "en"
            else f"> 运行标识 `{self.trace.run_id}`｜引擎 `{engine_label}`"
            f"｜数据源 openFDA drug label（FDA 官方说明书）"
            f"｜检索时间 {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        if report.startswith(title + "\n"):
            report = report.replace(title + "\n", title + "\n\n" + header, 1)

        self.trace.log(
            "finish",
            {
                "verdict": draft.verdict,
                "engine": draft.generated_by,
                "evidence_count": len(self.runtime.registry),
                "network_calls": self.client.network_calls,
            },
        )
        return AgentResult(
            ok=True,
            report_markdown=report,
            verdict=draft.verdict,
            engine=draft.generated_by,
            escalation="\n".join(escalation),
            run_id=self.trace.run_id,
            trace_path=str(self.trace.path or ""),
            notes=notes,
            screen=screen,
            records=[record_a, record_b],
        )

    # ------------------------------------------------------------ 主入口

    def analyze(self, drug_a: str, drug_b: str) -> AgentResult:
        """确定性的两步主路径：查两份官方说明书 → 筛查 → 出报告。

        这条路径不依赖模型（引擎=template 时也完全可用），
        所以现场网络抖动或 Key 失效时，核心功能仍然在。
        """
        self._say(f"[Observe] 识别到两个药名：{drug_a} / {drug_b}")
        self.trace.log("observe", {"drug_a": drug_a, "drug_b": drug_b, "mode": "pair"})
        if self.notes and not self._notice_shown and self.s.engine != "template":
            self._notice_shown = True
            self._say(self.notes[0])

        record_a = self.ensure_label(drug_a, slot="A")
        self._say(
            f"[Tool] lookup_drug_label({drug_a}) → "
            + (f"命中 {record_a.display}（{record_a.fetched_from}）" if record_a.matched else f"未命中：{record_a.message[:80]}")
        )
        record_b = self.ensure_label(drug_b, slot="B")
        self._say(
            f"[Tool] lookup_drug_label({drug_b}) → "
            + (f"命中 {record_b.display}（{record_b.fetched_from}）" if record_b.matched else f"未命中：{record_b.message[:80]}")
        )

        screen = self.ensure_pair_screen(force=True)
        if screen is None:
            return AgentResult(
                ok=False,
                engine="aborted",
                missing=[r.query for r in (record_a, record_b) if not r.matched],
                run_id=self.trace.run_id,
                trace_path=str(self.trace.path or ""),
                notes=["两份不同的官方说明书未能同时取得，无法进行配对筛查。"],
                records=[record_a, record_b],
            )
        self._say(
            f"[Detect] screen_pair_interaction → 风险等级下限 {screen.verdict_floor}，"
            f"命中规则 {len(screen.findings)} 条"
        )

        result = self.compose_report(record_a, record_b)
        self._say(f"[Safety] 最终风险等级：{result.verdict}（引擎 {result.engine}）")
        return result

    def run(self, user_input: str | None = None, max_steps: int | None = None) -> AgentResult:
        """完整 Agent 入口。

        - 输入里能直接识别出两个药名 → 走确定性主路径（快、稳、便宜）。
        - 识别不出（只有一个药名 / 名称含糊 / 是闲聊）→ 交给模型用工具循环处理，
          必要时向用户反问，而不是硬猜一个药名出来。
        """
        steps = max_steps or self.s.max_steps
        if user_input:
            self.messages.append({"role": "user", "content": user_input})
            names = self.extract_drug_names(user_input)
            if len(names) >= 2:
                return self.analyze(names[0], names[1])
            self._say(
                "[Observe] 未能从输入中确定两个药名，改为让模型用工具确认（不足就反问用户）。"
            )
            self.trace.log("observe", {"input": user_input, "extracted": names, "mode": "llm_loop"})
        return self._tool_loop(steps)

    # ------------------------------------------------------------ 工具循环

    def _tool_loop(self, max_steps: int) -> AgentResult:
        if not self.s.has_api_key:
            raise LLMError(
                "未配置 DeepSeek API Key，无法进行开放式对话。\n"
                "请设置环境变量 DEEPSEEK_API_KEY，或在 Agent_test/.env 里填写，"
                "或直接改 medcheck/config.py；也可以直接用两个药名作为参数运行"
                "（例如：python main.py --drug-a 华法林 --drug-b 布洛芬），"
                "那条路径不依赖模型。"
            )
        client = DeepSeekClient(self.s)
        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}, *self.messages]

        for step in range(1, max_steps + 1):
            reply = client.chat(messages, tools=TOOLS)
            tool_calls = reply.get("tool_calls") or []
            if not tool_calls:
                # 模型没有调用工具：可能是反问用户，或直接给了文字结论（后者不被接受）
                content = (reply.get("content") or "").strip()
                self.trace.log("model_text", {"step": step, "text": content[:400]})
                if self.runtime.screen is not None and len(self.runtime.pair) >= 2:
                    return self.compose_report(
                        self.runtime.pair[0], self.runtime.pair[1]
                    )
                return AgentResult(
                    ok=False,
                    engine="aborted",
                    run_id=self.trace.run_id,
                    trace_path=str(self.trace.path or ""),
                    notes=[
                        "信息不足，需要用户补充。",
                        content,
                    ],
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": reply.get("content") or "",
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                args = parse_tool_arguments(tc["arguments"])
                payload, summary = self.runtime.execute_with_slot(tc["name"], args)
                self._say(f"[Tool] {summary}")
                self.trace.log(
                    "tool_call",
                    {"step": step, "tool": tc["name"], "args": args, "result": summary},
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": payload}
                )

            if self.runtime.screen is not None:
                records = self.runtime.pair
                if len(records) >= 2:
                    self._say("[Detect] 资料齐备，进入确定性筛查与报告撰写。")
                    return self.compose_report(records[0], records[1])

        return AgentResult(
            ok=False,
            engine="aborted",
            run_id=self.trace.run_id,
            trace_path=str(self.trace.path or ""),
            notes=[f"达到工具调用步数上限（{max_steps}），为避免乱跑已停止。"],
        )


def evidence_cards(evidence: list[Evidence], limit: int = 8) -> str:
    """备用：把证据列表渲染成人读文本（调试用）。"""
    rows = []
    for ev in evidence[:limit]:
        rows.append(f"[{ev.evidence_id}] {ev.drug} · {ev.section_zh}\n    {ev.short(160)}")
    return "\n".join(rows) or "（无）"


__all__ = ["MedCheckAgent", "AgentResult", "evidence_cards"]
