"""工具执行器：模型只能通过这里触碰世界，返回的一切都是可引用的官方原文。"""

from __future__ import annotations

import json
from typing import Any

from .api_tool import LabelClient
from .models import FIELD_LABELS_ZH, SAFETY_FIELDS, Evidence, LabelRecord, PairScreen
from .safety_fields import LabelStore, category_of
from .screener import find_mentions, screen_pair, verify_term
from .textutil import collapse_ws, name_stems


class ToolRuntime:
    """持有一次会话的全部状态：客户端、证据库、已查标签、筛查结果。"""

    def __init__(self, client: LabelClient, store: LabelStore) -> None:
        self.client = client
        self.store = store
        self.screen: PairScreen | None = None
        self.screen_calls = 0
        self.tool_calls: list[str] = []
        # 本次要核查的两个"槽位"（A / B）。刻意不做去重：
        # 用户完全可能查"同一个药 + 同一个药"来确认重复用药。
        self.pair: list[LabelRecord] = []
        # 槽位 → DrugClassInfo（药理类别数据，来自官方 pharm_class 或内置兜底表）
        self.classes: dict[str, Any] = {}

    # ------------------------------------------------------------ 便捷访问

    @property
    def registry(self) -> dict[str, Evidence]:
        return self.store.registry._items  # noqa: SLF001 - 同一包内，刻意为之

    def class_candidates(self, slot: str) -> list[tuple[str, str]]:
        info = self.classes.get(slot)
        return info.candidates() if info is not None else []

    def class_source(self, slot: str) -> str:
        info = self.classes.get(slot)
        return (info.source if info is not None else "") or ""

    def set_pair(self, record: LabelRecord, slot: str = "") -> LabelRecord:
        """登记一个待核查药品。slot="A"/"B" 会覆盖对应槽位。"""
        if slot == "A" and self.pair:
            self.pair[0] = record
        elif slot == "B":
            while len(self.pair) < 2:
                self.pair.append(record)
            self.pair[1] = record
        elif record not in self.pair:
            self.pair.append(record)
        self.screen = None  # 配对变了，之前的筛查结果作废
        return record

    def fetch_classes(
        self, slot: str, drug_name: str, generic_hint: str = "", substance_hint: str = ""
    ) -> Any:
        """取该槽位药品的药理类别（带缓存 / fixtures / 内置兜底）。"""
        info = self.client.fetch_drug_class(drug_name, generic_hint, substance_hint)
        self.classes[slot] = info
        return info

    def matched_records(self) -> list[LabelRecord]:
        return [r for r in self.pair if r.matched]

    def distinct_matched(self) -> list[LabelRecord]:
        return self.matched_records()

    def sources_for(self, ev_id: str) -> list[str]:
        """该证据的"可核对原文"：字段完整原文 + 该证据自身引文。

        用完整原文做池子，是为了让"句子级证据"（整句引用）能逐字校验通过。
        """
        ev = self.registry.get(ev_id)
        if not ev:
            return []
        for rec in self.store._cache.values():  # noqa: SLF001
            if ev.section not in rec.fields:
                continue
            if ev.drug and rec.display and ev.drug != rec.display:
                continue
            out = list(rec.fields.get(ev.section) or [])
            out.append(ev.quote)
            return out
        return [ev.quote]

    def all_sources(self) -> dict[str, list[str]]:
        return {ev_id: self.sources_for(ev_id) for ev_id in self.registry}

    # ------------------------------------------------------------ 工具实现

    def lookup_drug_label(self, drug_name: str, slot: str = "") -> dict[str, Any]:
        result = self.client.fetch(drug_name)
        record = self.store.build(result)
        if slot:
            self.set_pair(record, slot)
        payload: dict[str, Any] = {
            "query": record.query,
            "matched": record.matched,
            "display_name": record.display,
            "generic_name": record.generic_names,
            "brand_name": record.brand_names[:3],
            "active_ingredient": record.active_ingredients,
            "match_strategy": record.strategy,
            "fetched_from": record.fetched_from,
            "label_effective_time": record.effective_time,
            "source_url": record.url,
            "retrieved_at": record.retrieved_at,
            "safety_fields": {
                field: {
                    "section_zh": FIELD_LABELS_ZH.get(field, field),
                    "category": category_of(field),
                    "evidence_id": next(
                        (i for i, ev in sorted(record.evidence.items()) if ev.section == field),
                        "",
                    ),
                    "preview": collapse_ws(" ".join(texts))[:160],
                }
                for field, texts in record.fields.items()
            },
            "present_safety_fields": record.present_fields,
            "missing_safety_fields": record.missing_fields,
            "message": record.message,
        }
        if not record.matched:
            payload["instruction"] = (
                "查不到就是查不到：请在结论中如实说明无法核查，"
                "建议用户向药师/医生确认，不要凭记忆补充该药的信息。"
            )
        return payload

    def fetch_label_section(
        self, drug_name: str, section: str | None = None, category: str | None = None
    ) -> dict[str, Any]:
        record = self.store.get_cached(drug_name)
        if record is None:
            return {"error": f"尚未查询过 “{drug_name}”，请先调用 lookup_drug_label。"}
        if not record.matched:
            return {"error": f"“{drug_name}” 没有可用的官方说明书。"}

        wanted: list[str] = []
        if section:
            wanted = [section]
        elif category:
            wanted = list(SAFETY_FIELDS.get(category, []))
        else:
            return {"error": "需要提供 section 或 category。"}

        out: dict[str, Any] = {"drug": record.display, "sections": []}
        for field in wanted:
            if field not in record.fields:
                out["sections"].append(
                    {"section": field, "present": False, "note": "该说明书没有这一节。"}
                )
                continue
            ev_id = next(
                (i for i, ev in sorted(record.evidence.items()) if ev.section == field), ""
            )
            out["sections"].append(
                {
                    "section": field,
                    "section_zh": FIELD_LABELS_ZH.get(field, field),
                    "present": True,
                    "evidence_id": ev_id,
                    "text": collapse_ws(" ".join(record.fields[field])),
                }
            )
        return out

    def search_label(
        self, drug_name: str, term: str, limit: int = 3
    ) -> dict[str, Any]:
        """在某个药品的**全部安全章节**里搜一个词，返回命中整句与所在章节。

        用途：确认"这个药是不是也提到 XX 风险"，或核对某个具体条文。
        """
        record = self.store.get_cached(drug_name)
        if record is None:
            return {"error": f"尚未查询过 “{drug_name}”，请先调用 lookup_drug_label。"}
        if not record.matched:
            return {"error": f"“{drug_name}” 没有可用的官方说明书。"}
        terms = sorted(name_stems(term) or {term.lower()}, key=len, reverse=True)
        hits = find_mentions(record, terms, limit=limit, register=self.register_sentence)
        return {
            "drug": record.display,
            "term": term,
            "found": bool(hits),
            "hits": hits,
            "note": "未命中只代表该说明书的这些安全章节里没有出现这个词，不代表没有该风险。",
        }

    def register_sentence(self, record: LabelRecord, field_name: str, sentence: str) -> str:
        """把命中的整句原文登记成句子级证据（供筛选用）。"""
        ev = self.store.registry.add(
            drug=record.display,
            section=field_name,
            quote=sentence,
            url=record.url,
            retrieved_at=record.retrieved_at,
        )
        return ev.evidence_id

    def screen_pair_interaction(self, force: bool = False) -> dict[str, Any]:
        """对两个槽位的药品做确定性配对筛查。

        ★ 即使某个药查不到官方标签，也照样出筛查结果（等级会落到「证据不足」）——
          绝不因为"缺资料"就直接报"没冲突"。
        """
        if len(self.pair) < 2:
            return {
                "error": (
                    "需要两个药品才能做配对筛查（当前只有 "
                    f"{len(self.pair)} 个）。请分别调用一次 lookup_drug_label。"
                    "如果用户只给了一个药名，请先向用户确认另一个药名。"
                )
            }
        record_a, record_b = self.pair[0], self.pair[1]
        if self.screen is None or force:
            self.screen = screen_pair(
                record_a,
                record_b,
                register=self.register_sentence,
                classes_a=self.class_candidates("A"),
                classes_b=self.class_candidates("B"),
                class_source_a=self.class_source("A"),
                class_source_b=self.class_source("B"),
            )
            self.screen_calls += 1
        payload = self.screen.to_dict()
        payload["order_note"] = (
            f"A = {record_a.display if record_a.matched else record_a.query}（查询名 {record_a.query}）"
            f"（matched={record_a.matched}），"
            f"B = {record_b.display if record_b.matched else record_b.query}（查询名 {record_b.query}）"
            f"（matched={record_b.matched}）"
        )
        payload["instruction"] = (
            "最终报告的 verdict 必须 >= verdict_floor（只能更保守）。"
            "findings 必须与本结果一一对应，不得增删。"
        )
        return payload

    def verify_term(self, term: str) -> dict[str, Any]:
        if len(self.pair) < 2:
            return {"error": "需要两个药品才能做交叉核验，请先完成两次 lookup_drug_label。"}
        return verify_term(
            term, self.pair[0], self.pair[1], register=self.register_sentence
        )

    # ------------------------------------------------------------ 分发

    def execute(self, name: str, args: dict[str, Any]) -> tuple[str, str]:
        """返回 (给模型看的 JSON 字符串, 给终端看的简短摘要)。"""
        try:
            if name == "lookup_drug_label":
                drug = str(args.get("drug_name") or "").strip()
                if not drug:
                    return json.dumps({"error": "缺少 drug_name"}, ensure_ascii=False), "参数缺失"
                payload = self.lookup_drug_label(drug)
                summary = (
                    f"lookup({drug}) -> "
                    + (f"命中 {payload['display_name']}" if payload["matched"] else "未命中")
                )
            elif name == "screen_pair_interaction":
                payload = self.screen_pair_interaction()
                summary = "screen_pair() -> " + (
                    payload.get("error")
                    or f"floor={payload['verdict_floor']}, findings={len(payload['findings'])}"
                )
            elif name == "fetch_label_section":
                payload = self.fetch_label_section(
                    str(args.get("drug_name") or ""),
                    args.get("section"),
                    args.get("category"),
                )
                summary = f"fetch_section({args.get('drug_name')}, {args.get('section') or args.get('category')})"
            elif name == "verify_term":
                payload = self.verify_term(str(args.get("term") or ""))
                summary = f"verify_term({args.get('term')}) -> shared={payload.get('shared_in_both')}"
            elif name == "search_term_in_label":
                payload = self.search_label(
                    str(args.get("drug_name") or ""),
                    str(args.get("term") or ""),
                    int(args.get("limit") or 3),
                )
                summary = (
                    f"search_label({args.get('drug_name')}, {args.get('term')}) -> "
                    f"{len(payload.get('hits') or [])} hits"
                )
            else:
                return (
                    json.dumps({"error": f"未知工具 {name}"}, ensure_ascii=False),
                    f"未知工具 {name}",
                )
        except Exception as exc:  # noqa: BLE001 - 工具异常不能炸掉整个循环
            payload = {"error": f"{type(exc).__name__}: {exc}"}
            summary = f"{name} 异常：{exc}"
        self.tool_calls.append(name)
        return json.dumps(payload, ensure_ascii=False), summary

    def execute_with_slot(self, name: str, args: dict[str, Any]) -> tuple[str, str]:
        """按调用次数自动分配 A/B 槽位（专供工具循环里的 lookup_drug_label）。"""
        if name == "lookup_drug_label":
            slot = ""
            if len(self.pair) < 2:
                slot = "A" if not self.pair else "B"
            payload = self.lookup_drug_label(str(args.get("drug_name") or "").strip(), slot)
            summary = (
                f"lookup({args.get('drug_name')}) -> "
                + (f"命中 {payload['display_name']}" if payload["matched"] else "未命中")
            )
            self.tool_calls.append(name)
            return json.dumps(payload, ensure_ascii=False), summary
        return self.execute(name, args)
