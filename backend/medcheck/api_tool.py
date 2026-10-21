"""API Tool —— 唯一被允许访问外部世界的地方。

三件事：
1. **只查官方来源**：openFDA `drug/label.json`（= FDA 官方 SPL 说明书原文）。
   没有开放搜索引擎，没有内容农场（见 Agent.md §6.2：开放搜索在医疗场景是陷阱）。
2. **三级缓存**：network → cache/ → fixtures/。演示可以 `--offline` 全本地，
   结构上不依赖网络。
3. **绝不猜**：查不到就返回 `matched=False` + 原因，让上层去升级，
   而不是让模型"想一个答案"。

注意：这里刻意绕过系统代理（trust_env=False 的等价实现），
因为环境变量里的 127.0.0.1:7890 代理在演示机上经常是死的。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .drug_classes import DrugClassInfo, record_matches_drug
from .textutil import name_stems, normalize_query
CACHE_SCHEMA = "medcheck/openfda-label-cache@1"
CLASS_CACHE_SCHEMA = "medcheck/drug-class-cache@1"

# 药理类别接口：openFDA NDC 端点的类别覆盖率远高于 label 端点
# （实测：label 端点对 sertraline/warfarin/clopidogrel 等完全没有类别数据）
NDC_API_URL = "https://api.fda.gov/drug/ndc.json"

# 查询策略：从最精确到最宽松，命中即停
STRATEGIES: tuple[tuple[str, str], ...] = (
    ("generic_name", 'openfda.generic_name:"{q}"'),
    ("brand_name", 'openfda.brand_name:"{q}"'),
    ("substance_name", 'openfda.substance_name:"{q}"'),
    ("full_text", "{q}"),
)

EXACT_STRATEGIES: tuple[tuple[str, str], ...] = STRATEGIES[:3]


class ApiToolError(RuntimeError):
    """工具层面的硬失败（网络断、超时、响应不是 JSON）。"""


@dataclass
class LabelFetchResult:
    """一次查询的结果：要么 label，要么一个说得清的原因。"""

    query: str
    normalized: str
    url: str
    fetched_from: str = ""            # network | cache | fixture
    strategy: str = ""
    label: dict[str, Any] | None = None
    status: int = 0
    message: str = ""
    tried: list[str] = field(default_factory=list)
    retrieved_at: str = ""

    @property
    def matched(self) -> bool:
        return self.label is not None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _slug(name: str) -> str:
    out = []
    for ch in (name or "").lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-") or "unknown"


def build_opener() -> urllib.request.OpenerDirector:
    """显式禁用代理：环境里的 HTTP_PROXY 常常指向一个没开的本地端口。"""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


class LabelClient:
    """openFDA 官方标签查询客户端（带缓存 / fixtures / 离线模式）。"""

    def __init__(
        self,
        settings: Settings,
        trace=None,
        exact_only: bool = False,
        skip_local: bool = False,
    ) -> None:
        self.s = settings
        self.opener = build_opener()
        self.trace = trace
        self.network_calls = 0
        # 抓 fixtures 时用：只走精确名字匹配，避免全文检索把组合药/别的剂型抓进来
        self.strategies = EXACT_STRATEGIES if exact_only else STRATEGIES
        # 抓 fixtures 时用：跳过 cache/ 与 fixtures/，强制回到网络取最新数据
        self.skip_local = skip_local

    # ------------------------------------------------------------ 对外主入口

    def fetch(self, drug_name: str, limit: int = 5) -> LabelFetchResult:
        normalized = normalize_query(drug_name)
        url_base = self.s.label_api_url
        retrieved_at = _now()

        if not normalized:
            return LabelFetchResult(
                query=drug_name, normalized=normalized, url=url_base,
                status=0, message="药名为空，无法查询。", retrieved_at=retrieved_at,
            )

        # ① 本地缓存（含 fixtures）
        if not self.skip_local and (self.s.use_cache or self.s.offline):
            hit = self._read_local(normalized)
            if hit is not None:
                label, origin, raw = hit
                return LabelFetchResult(
                    query=drug_name, normalized=normalized,
                    url=raw.get("url", url_base), fetched_from=origin,
                    strategy=raw.get("strategy", "local"),
                    label=label, status=raw.get("status", 200),
                    tried=raw.get("tried", []),
                    retrieved_at=raw.get("retrieved_at", retrieved_at),
                )

        if self.s.offline:
            return LabelFetchResult(
                query=drug_name, normalized=normalized, url=url_base,
                status=0, message=(
                    f"离线模式：本地没有 {normalized} 的说明书缓存。"
                    "请联网跑一次（去掉 --offline）把它抓进 cache/，"
                    "然后就能离线复现。本 Agent 不会凭记忆作答。"
                ),
                retrieved_at=retrieved_at,
            )

        # ② 网络查询（唯一的出网路径）
        last_message = ""
        tried: list[str] = []
        for strategy, template in self.strategies:
            search = template.format(q=normalized)
            tried.append(f"{strategy}: {search}")
            try:
                status, payload = self._http_get(search, limit)
            except ApiToolError as exc:
                last_message = f"网络查询失败：{exc}"
                break
            if status == 200 and payload.get("results"):
                best, why = pick_best_label(payload["results"], normalized)
                if best is not None:
                    result = LabelFetchResult(
                        query=drug_name, normalized=normalized, url=url_base,
                        fetched_from="network", strategy=strategy, label=best,
                        status=200, tried=tried, retrieved_at=retrieved_at,
                        message=why,
                    )
                    if self.s.write_cache:
                        self._write_cache(normalized, result)
                    if self.trace:
                        self.trace.log("tool_call", {
                            "tool": "lookup_drug_label",
                            "query": drug_name,
                            "strategy": strategy,
                            "found": True,
                            "records": len(payload["results"]),
                        })
                    return result
            last_message = (
                f"openFDA 在 `{strategy}` 策略下没有找到 “{normalized}” "
                f"（HTTP {status}）。"
            )

        # ③ 网络失败/查不到：可选退回 fixtures
        if self.s.fixture_fallback:
            hit = self._read_local(normalized, force_fixture=True)
            if hit is not None:
                label, origin, raw = hit
                return LabelFetchResult(
                    query=drug_name, normalized=normalized,
                    url=raw.get("url", url_base), fetched_from=origin,
                    strategy=raw.get("strategy", "fixture"),
                    label=label, status=raw.get("status", 200), tried=tried,
                    retrieved_at=raw.get("retrieved_at", retrieved_at),
                    message="网络不可用，已退回本地预抓取的官方说明书快照。",
                )

        return LabelFetchResult(
            query=drug_name, normalized=normalized, url=url_base, status=0,
            message=last_message or f"未找到 “{normalized}” 的官方标签。",
            tried=tried, retrieved_at=retrieved_at,
        )

    # ------------------------------------------------------------ 药理类别

    def fetch_drug_class(
        self, drug_name: str, generic_hint: str = "", substance_hint: str = ""
    ) -> DrugClassInfo:
        """查询一个药品的**药理类别**（用于"说明书按类别描述"的检测）。

        三级来源，先命中先用：本地缓存 → fixtures 快照 → 官方接口。
        接口策略：NDC 端点（覆盖最好，实测 28/32 常见药有数据）→
              label 端点的 openfda.pharm_class_*（兜底）→
              内置 CURATED_CLASSES（离线/接口无数据时的最后一道）。
        """
        normalized = normalize_query(drug_name)
        if not normalized:
            return DrugClassInfo(drug=normalized, source="none")

        hint = (generic_hint or "").strip()
        hint = "" if hint.lower() == normalized.lower() else hint

        # ① 本地（cache/classes/ 与 fixtures/pharmacologic_classes.json）
        hit = self._read_class_local(normalized, hint)
        if hit is not None:
            return hit

        if self.s.offline:
            # 离线模式：只认本地数据 + 内置兜底表，绝不发请求
            return self._curated_class(normalized, "curated(offline)")

        # ② 官方接口：NDC —— 逐个查询变体尝试，命中即停。
        #    变体来源：归一化名 / 去连字符 / **复方成分的其它排列**。
        #    实例：trimethoprim-sulfamethoxazole 在 NDC 里登记为
        #          "Sulfamethoxazole and Trimethoprim"，顺序相反，只按原词序查会 404。
        raw: set[str] = set()
        source = ""
        for query in self._class_query_variants(normalized, hint, substance_hint, drug_name):
            try:
                status, payload = self._http_get_json(
                    NDC_API_URL, f'generic_name:"{query}"', limit=100
                )
            except ApiToolError:
                continue
            if status != 200:
                continue
            kept = 0
            for rec in payload.get("results", []):
                generic = str(rec.get("generic_name") or "")
                classes = rec.get("pharm_class") or []
                if not classes or not record_matches_drug(generic, query):
                    continue
                raw |= {str(c) for c in classes}
                kept += 1
            if kept:
                source = f"openFDA ndc ({kept} 个产品, 查询 {query!r})"
                break

        # ③ 兜底：label 端点自带的 pharm_class
        if not raw:
            for query in self._class_query_variants(normalized, hint, substance_hint, drug_name):
                try:
                    status, payload = self._http_get(query, limit=20)
                except ApiToolError:
                    continue
                if status != 200:
                    continue
                for rec in payload.get("results", []):
                    openfda = rec.get("openfda") or {}
                    for key in ("pharm_class_epc", "pharm_class_moa",
                                "pharm_class_cs", "pharm_class_pe"):
                        raw |= {str(x) for x in (openfda.get(key) or [])}
                if raw:
                    source = "openFDA label (openfda.pharm_class_*)"
                    break

        info = DrugClassInfo(
            drug=hint or normalized,
            raw=sorted(raw),
            source=source or "none",
            retrieved_at=_now(),
        )
        if info.is_empty:
            # 接口也没数据 → 内置兜底表
            curated = self._curated_class(normalized, "curated(fallback)")
            if not curated.is_empty:
                return curated
            if hint and hint.lower() in self._curated_keys:
                return self._curated_class(hint.lower(), "curated(fallback)")
            return info
        if self.s.write_cache:
            self._write_class_cache(normalized, info)
        return info

    def _curated_class(self, normalized: str, source: str) -> DrugClassInfo:
        from .drug_classes import CURATED_CLASSES

        key = normalized.lower().strip()
        return DrugClassInfo(
            drug=normalized,
            raw=list(CURATED_CLASSES.get(key, ())),
            source=source if key in CURATED_CLASSES else "none",
            retrieved_at=_now(),
        )

    @property
    def _curated_keys(self) -> frozenset[str]:
        from .drug_classes import CURATED_CLASSES

        return frozenset(CURATED_CLASSES)

    @staticmethod
    def _class_query_variants(
        normalized: str, hint: str, substance_hint: str = "", raw_name: str = ""
    ) -> list[str]:
        """生成 NDC 类别查询的候选写法（去重、保序）。

        必要性（都是实测踩到的）：
        · openFDA 的 generic_name 对复方药的**词序**敏感：
          `trimethoprim-sulfamethoxazole` 查不到，`sulfamethoxazole and trimethoprim` 可以。
        · `normalize_query` 会为"查说明书"的目的去掉盐基词（lithium carbonate → lithium），
          但类别数据是登记在**完整通用名**下的，所以必须把原始输入也作为候选。
        """
        out: list[str] = []

        def add(value: str) -> None:
            v = re.sub(r"\s+", " ", (value or "").strip())
            if v and v.lower() not in {x.lower() for x in out}:
                out.append(v)

        add(normalized)
        add(normalized.replace("-", " "))
        add(raw_name)
        add(raw_name.replace("-", " "))
        if hint:
            add(hint)
            add(hint.replace("-", " "))
        if substance_hint:
            add(substance_hint)
            add(substance_hint.replace("-", " "))
        # 复方成分的其它排列（含 "and" 连接的形式），以及去盐基后的完整组合
        for base in (raw_name, normalized, hint, substance_hint):
            parts = [p for p in re.split(r"[-\s]+", (base or "").strip()) if len(p) > 3]
            if len(parts) >= 2:
                add(" and ".join(reversed(parts)))
                add(" ".join(reversed(parts)))
                add(" and ".join(parts))
        return out

    def _read_class_local(
        self, normalized: str, hint: str = ""
    ) -> DrugClassInfo | None:
        keys = [normalized]
        if hint:
            keys.append(hint.lower())
        # 运行期缓存
        path = self.s.cache_dir / "classes" / f"{_slug(normalized)}.json"
        for candidate, origin in (
            (path, "cache"),
            (self.s.fixtures_dir / "pharmacologic_classes.json", "fixture"),
        ):
            if not candidate.exists():
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if origin == "cache":
                raw = data.get("raw") or []
                if raw and str(data.get("drug") or "").lower() in {
                    normalized.lower(), hint.lower(), ""
                }:
                    return DrugClassInfo(
                        drug=data.get("drug") or normalized,
                        raw=[str(x) for x in raw],
                        source=f"cache ({data.get('source') or 'openFDA'})",
                        retrieved_at=data.get("retrieved_at", ""),
                    )
            else:
                drugs = data.get("drugs") or {}
                for key in keys:
                    entry = drugs.get(key) or drugs.get(_slug(key))
                    if entry and entry.get("raw"):
                        return DrugClassInfo(
                            drug=entry.get("drug") or normalized,
                            raw=[str(x) for x in entry["raw"]],
                            source=f"fixture ({entry.get('source') or 'openFDA'})",
                            retrieved_at=entry.get("retrieved_at", ""),
                        )
        return None

    def _write_class_cache(self, normalized: str, info: DrugClassInfo) -> None:
        path = self.s.cache_dir / "classes" / f"{_slug(normalized)}.json"
        payload = {
            "schema": CLASS_CACHE_SCHEMA,
            "query": normalized,
            "drug": info.drug,
            "raw": info.raw,
            "source": info.source,
            "retrieved_at": info.retrieved_at,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass

    # ------------------------------------------------------------ HTTP

    def _http_get_json(
        self, base_url: str, search: str, limit: int = 1
    ) -> tuple[int, dict[str, Any]]:
        """通用 openFDA JSON 查询（label / ndc 端点共用）。"""
        params = urllib.parse.urlencode({"search": search, "limit": limit})
        url = f"{base_url}?{params}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "MedCheck-Agent/1.0 (hackathon; official-label-only)",
                "Accept": "application/json",
            },
        )
        self.network_calls += 1
        try:
            with self.opener.open(req, timeout=self.s.request_timeout) as resp:
                body = resp.read().decode("utf-8", "replace")
                return resp.status, json.loads(body)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # openFDA 用 404 表示"没有匹配记录"
                return 404, {}
            raise ApiToolError(f"HTTP {exc.code} {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise ApiToolError(f"无法连接 {base_url}（{exc.reason}）") from exc
        except TimeoutError as exc:
            raise ApiToolError(f"请求超时（>{self.s.request_timeout}s）") from exc
        except json.JSONDecodeError as exc:
            raise ApiToolError(f"响应不是合法 JSON：{exc}") from exc

    def _http_get(self, search: str, limit: int) -> tuple[int, dict[str, Any]]:
        return self._http_get_json(self.s.label_api_url, search, limit)

    # ------------------------------------------------------------ 本地缓存 / fixtures

    def _read_local(
        self, normalized: str, force_fixture: bool = False
    ) -> tuple[dict[str, Any], str, dict[str, Any]] | None:
        candidates: list[tuple[Path, str]] = []
        if not force_fixture:
            candidates.append((self.s.cache_dir / f"{_slug(normalized)}.json", "cache"))
        candidates.append((self.s.fixtures_dir / f"{_slug(normalized)}.json", "fixture"))
        for path, origin in candidates:
            if not path.exists():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            label = raw.get("label")
            if isinstance(label, dict) and label:
                return label, origin, raw
        # fixtures 里可能用索引名存（例如按通用名/别名存）
        aliases = self._fixture_alias_map()
        if normalized in aliases:
            path = self.s.fixtures_dir / aliases[normalized]
            if path.exists():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    label = raw.get("label")
                    if isinstance(label, dict) and label:
                        return label, "fixture", raw
                except (OSError, json.JSONDecodeError):
                    pass
        return None

    def _fixture_alias_map(self) -> dict[str, str]:
        index_path = self.s.fixtures_dir / "index.json"
        if not index_path.exists():
            return {}
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        alias: dict[str, str] = {}
        for filename, meta in (data.get("drugs") or {}).items():
            for key in meta.get("aliases", []):
                alias[str(key).lower()] = filename
        return alias

    def _write_cache(self, normalized: str, result: LabelFetchResult) -> None:
        path = self.s.cache_dir / f"{_slug(normalized)}.json"
        payload = {
            "schema": CACHE_SCHEMA,
            "query": result.query,
            "normalized": normalized,
            "url": result.url,
            "strategy": result.strategy,
            "status": result.status,
            "tried": result.tried,
            "retrieved_at": result.retrieved_at,
            "label": result.label,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except OSError:
            pass  # 缓存写失败不该影响主流程


# ---------------------------------------------------------------- 记录选择


def _text_of(record: dict[str, Any], field_name: str) -> str:
    val = record.get(field_name)
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    if isinstance(val, str):
        return val
    return ""


def pick_best_label(
    records: list[dict[str, Any]], normalized: str
) -> tuple[dict[str, Any] | None, str]:
    """在若干条官方标签记录里挑一条最适合做安全筛查的。

    启发式（全部可解释、可复现）：
    ① 名字真的对得上（词干匹配），否则弃用——防止 openFDA 全文检索跑偏；
    ② **单方优先**：查 metformin 要拿 "METFORMIN HYDROCHLORIDE"，
       而不是 "SITAGLIPTIN AND METFORMIN" 这类复方（复方标签会让配对筛查串味）；
    ③ 优先口服/全身给药剂型（外用霜剂的标签根本没有相互作用章节）；
    ④ 安全字段覆盖多者优先；
    ⑤ 同样好就取版本最新的。
    """
    if not records:
        return None, "openFDA 没有返回任何记录。"

    query_words = _word_set(normalized)
    query_stems = name_stems(normalized) or {normalized.lower()}

    def extra_moieties(openfda: dict[str, Any]) -> int:
        """候选里"多出来的有效成分"个数：0 = 单方，1 = 二联复方，以此类推。"""
        generic = openfda.get("generic_name") or []
        if generic:
            return min(
                len(_word_set(str(n)) - query_words) for n in generic
            )
        subs = openfda.get("substance_name") or []
        if subs:
            return max(0, len(subs) - 1)
        return 0

    scored: list[tuple[tuple, dict[str, Any], str]] = []
    for rec in records:
        openfda = rec.get("openfda") or {}
        names = list(openfda.get("generic_name") or []) + list(
            openfda.get("brand_name") or []
        ) + list(openfda.get("substance_name") or [])
        rec_stems: set[str] = set()
        for n in names:
            rec_stems |= name_stems(str(n))
        name_score = 2 if query_stems & rec_stems else 0
        if not rec_stems:
            name_score = 1  # 没有 openfda 名字信息，不能直接判死

        extras = extra_moieties(openfda)

        from .models import ALL_SAFETY_FIELDS

        present = [f for f in ALL_SAFETY_FIELDS if rec.get(f)]
        coverage = len(present)

        route_text = _text_of(rec, "dosage_and_administration") + _text_of(
            rec, "spl_product_data_elements"
        ) + _text_of(rec, "description")
        topical = 1 if any(
            kw in route_text.lower()
            for kw in ("topical", "apply to the affected", "cream", "gel", "ointment")
        ) else 0
        oral_bonus = 1 if any(
            kw in route_text.lower()
            for kw in ("oral", "swallow", "by mouth", "tablet", "capsule")
        ) else 0
        route_score = oral_bonus - topical

        version = str(rec.get("effective_time") or "")
        scored.append(
            ((name_score, -extras, coverage, route_score, version), rec, ", ".join(present))
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    best_key, best_rec, present = scored[0]
    if best_key[0] == 0:
        return None, (
            f"openFDA 有记录，但没有一条的通用名/品牌名与 “{normalized}” 对得上，"
            "为避免张冠李戴，本 Agent 拒绝使用这些记录。"
        )
    n_fields = len(present.split(",")) if present else 0
    why = (
        f"共 {len(records)} 条候选记录，已排除复方制剂（多出实词 {abs(best_key[1])} 个），"
        f"采用安全字段覆盖 {n_fields} 个的一条。"
    )
    return best_rec, why


def _word_set(name: str) -> set[str]:
    import re

    from .textutil import SALT_SUFFIXES, STOPWORDS

    low = re.sub(r"[^a-z0-9\s]", " ", (name or "").lower())
    return {
        w
        for w in low.split()
        if w and w not in STOPWORDS and w not in SALT_SUFFIXES and len(w) > 1
    }


def label_source_url(record: dict[str, Any]) -> str:
    """给证据留一个人类可点的溯源链接（DailyMed 官方说明书页面）。"""
    set_id = str(record.get("set_id") or "").strip()
    if set_id:
        return f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"
    label_id = str(record.get("id") or "").strip()
    if label_id:
        return f"https://api.fda.gov/drug/label.json?search=id:{label_id}"
    return "https://api.fda.gov/drug/label.json"
