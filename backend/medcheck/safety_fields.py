"""按 SAFETY_FIELDS 严格抽取安全章节，并把每一段官方原文登记成证据。

为什么单独一个模块：**字段口径必须只有一处定义**。
api_test.py 探明了"查什么"，这里负责"查到之后怎么变成可引用的证据"。
"""

from __future__ import annotations

from .api_tool import LabelFetchResult, label_source_url
from .config import Settings
from .models import (
    ALL_SAFETY_FIELDS,
    FIELD_LABELS_ZH,
    FIELD_TO_CATEGORY,
    SAFETY_FIELDS,
    Evidence,
    LabelRecord,
)
from .textutil import clip, collapse_ws, sentence_spans


def as_text_list(value: object) -> list[str]:
    """openFDA 的字段值有的是 str，有的是 list[str]，统一成 list[str]。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items()]
    return [str(value)]


def extract_safety_fields(record: dict) -> dict[str, list[str]]:
    """只保留 SAFETY_FIELDS 里的字段——其余（临床药理、说明书排版区…）一律丢弃。

    对应 api_test.py 的 extract_safety()：这是"严格筛查"的第一层含义，
    即**我们只根据官方安全章节说话**。
    """
    out: dict[str, list[str]] = {}
    for field_name in ALL_SAFETY_FIELDS:
        texts = as_text_list(record.get(field_name))
        if texts:
            out[field_name] = texts
    return out


def _safe_truncate_at_sentence(text: str, limit: int) -> str:
    """截断但保证结果仍是原文的连续子串（句子边界优先）。"""
    text = collapse_ws(text)
    if len(text) <= limit:
        return text
    best = ""
    for start, end in sentence_spans(text):
        if end > limit:
            break
        best = text[:end]
    if best and len(best) >= limit * 0.4:
        return best
    return text[:limit].rstrip()


class EvidenceRegistry:
    """全局证据登记处：给出稳定的 ev_XXX 编号。"""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._items: dict[str, Evidence] = {}
        self._counter = 0

    def add(
        self, drug: str, section: str, quote: str, url: str, retrieved_at: str
    ) -> Evidence:
        self._counter += 1
        ev_id = f"ev_{self._counter:03d}"
        item = Evidence(
            evidence_id=ev_id,
            drug=drug,
            section=section,
            section_zh=FIELD_LABELS_ZH.get(section, section),
            quote=quote,
            url=url,
            retrieved_at=retrieved_at,
        )
        self._items[ev_id] = item
        return item

    def get(self, ev_id: str) -> Evidence | None:
        return self._items.get(ev_id)

    def get_many(self, ids: list[str]) -> list[Evidence]:
        return [self._items[i] for i in ids if i in self._items]

    def all(self) -> list[Evidence]:
        return [self._items[k] for k in sorted(self._items)]

    def by_drug(self, drug: str) -> list[Evidence]:
        return [e for e in self.all() if e.drug == drug]


class LabelStore:
    """把 LabelFetchResult 变成带证据的 LabelRecord，并在会话内缓存。"""

    def __init__(self, settings: Settings, registry: EvidenceRegistry) -> None:
        self.s = settings
        self.registry = registry
        self._cache: dict[str, LabelRecord] = {}

    def key(self, drug_name: str) -> str:
        return (drug_name or "").strip().lower()

    def get_cached(self, drug_name: str) -> LabelRecord | None:
        return self._cache.get(self.key(drug_name))

    def build(self, result: LabelFetchResult, classes=None) -> LabelRecord:
        """把查询结果变成带证据的 LabelRecord。classes = DrugClassInfo（可空）。"""
        cached = self.get_cached(result.query)
        if cached is not None and cached.matched == result.matched:
            # 类别数据是后补的：如果第一次构建时还没有，这里补上
            if classes is not None and not cached.pharm_classes:
                cached.pharm_classes = list(classes.raw)
                cached.pharm_class_source = classes.source
            return cached

        if not result.matched:
            record = LabelRecord(
                query=result.query,
                matched=False,
                status=result.status,
                message=result.message,
                retrieved_at=result.retrieved_at,
                fetched_from=result.fetched_from,
                url=result.url,
                pharm_classes=list(classes.raw) if classes else [],
                pharm_class_source=classes.source if classes else "",
            )
            self._cache[self.key(result.query)] = record
            return record

        raw = result.label or {}
        openfda = raw.get("openfda") or {}
        generic_names = [str(x) for x in (openfda.get("generic_name") or [])]
        brand_names = [str(x) for x in (openfda.get("brand_name") or [])]
        ingredients = [str(x) for x in (openfda.get("substance_name") or [])]
        display = (brand_names or generic_names or [result.normalized])[0]

        fields = extract_safety_fields(raw)
        url = label_source_url(raw)

        record = LabelRecord(
            query=result.query,
            matched=True,
            strategy=result.strategy,
            status=result.status,
            message=result.message,
            name=display,
            generic_names=generic_names,
            brand_names=brand_names,
            active_ingredients=ingredients,
            label_id=str(raw.get("id") or ""),
            set_id=str(raw.get("set_id") or ""),
            effective_time=str(raw.get("effective_time") or ""),
            url=url,
            retrieved_at=result.retrieved_at,
            fetched_from=result.fetched_from,
            fields={},
            present_fields=[],
            missing_fields=[],
            pharm_classes=list(classes.raw) if classes else [],
            pharm_class_source=classes.source if classes else "",
        )

        present: list[str] = []
        missing: list[str] = []
        for field_name in ALL_SAFETY_FIELDS:
            texts = fields.get(field_name)
            if not texts:
                missing.append(field_name)
                continue
            present.append(field_name)
            clipped: list[str] = []
            for text in texts:
                body = _safe_truncate_at_sentence(text, self.s.evidence_quote_limit)
                if body:
                    clipped.append(body)
            record.fields[field_name] = clipped
            full_quote = _safe_truncate_at_sentence(
                " ".join(clipped), self.s.evidence_quote_limit
            )
            ev = self.registry.add(
                drug=display,
                section=field_name,
                quote=full_quote,
                url=url,
                retrieved_at=result.retrieved_at,
            )
            record.evidence[ev.evidence_id] = ev

        record.present_fields = present
        record.missing_fields = missing
        self._cache[self.key(result.query)] = record
        return record

    def label_sources(self, record: LabelRecord) -> list[str]:
        """该药品所有官方原文（用于引文逐字校验）。"""
        out: list[str] = []
        for texts in record.fields.values():
            out.extend(texts)
        return out


def category_of(field_name: str) -> str:
    return FIELD_TO_CATEGORY.get(field_name, "其他相关")


def field_summary(record: LabelRecord, limit: int = 0) -> dict[str, list[str]]:
    """给模型的字段清单：字段名 -> 中文名 + 库存引文 id。"""
    out: dict[str, list[str]] = {}
    for field_name, texts in record.fields.items():
        ids = [
            ev_id
            for ev_id, ev in record.evidence.items()
            if ev.section == field_name
        ]
        out[field_name] = [FIELD_LABELS_ZH.get(field_name, field_name), *ids]
    return out


def clipped(text: str, limit: int = 240) -> str:
    return clip(text, limit)
