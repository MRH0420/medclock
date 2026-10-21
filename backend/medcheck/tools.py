"""工具定义（OpenAI / DeepSeek function calling 格式）。

工具少而精，并且**每一个都只做它能负责的事**：
- lookup_drug_label   → 取官方说明书（唯一的出网口）
- screen_pair_interaction → 确定性规则筛查，输出风险等级下限
- fetch_label_section → 读某一节的官方原文（模型按需深入，而不是一次性灌满上下文）
- verify_term         → 核验一个词是否真的出现在两份说明书里（防"共享类别"幻觉）

注意这里**没有** "web_search"，也**没有** "add_finding / set_verdict" 这类工具：
模型不能自己写结论，它只能读证据、然后写报告文字（见 Agent.md §6.2、§5.6）。
"""

from __future__ import annotations

from .models import SAFETY_FIELDS

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_drug_label",
            "description": (
                "查询某个药品的 FDA 官方说明书（openFDA drug/label API），"
                "并只抽取安全相关章节（黑框警告、禁忌、警告与注意事项、相互作用、"
                "特殊人群等）。用户输入的是中文或商品名也可以直接传。"
                "返回 matched=false 表示查不到——查不到就是查不到，不要凭空补。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "药品名称，可以是通用名、商品名或中文名。",
                    }
                },
                "required": ["drug_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "screen_pair_interaction",
            "description": (
                "对两份已取得的官方说明书做确定性配对筛查：①互相点名检测 "
                "②有效成分重复检测 ③相互作用章节缺失检测 ④黑框警告提示。"
                "返回 findings（规则命中，带 evidence_id）与 verdict_floor（风险等级下限）。"
                "最终报告的风险等级**不得低于** verdict_floor，只能更保守。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_label_section",
            "description": (
                "读取某份说明书里某个安全章节的完整官方原文（带 evidence_id），"
                "用于确认细节，例如相互作用章节到底怎么写的。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string", "description": "药名（须先 lookup 过）"},
                    "section": {
                        "type": "string",
                        "description": "章节名（openFDA 字段名）",
                        "enum": [f for fields in SAFETY_FIELDS.values() for f in fields],
                    },
                    "category": {
                        "type": "string",
                        "description": "可选：按类别取该药品该类别的所有章节",
                        "enum": list(SAFETY_FIELDS.keys()),
                    },
                },
                "required": ["drug_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_term",
            "description": (
                "核验一个英文术语（如 bleeding、NSAID、QT、serotonin、CYP3A4）"
                "是否真的出现在两份说明书里，返回命中句原文。"
                "**任何关于“两药共享某类风险/同一药理类别”的说法都必须先由本工具确认。**"
                "没有命中原文 → 该说法作废，不得写进报告。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {"type": "string", "description": "要核验的英文术语"}
                },
                "required": ["term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_term_in_label",
            "description": (
                "在某个药品的**全部安全章节**里搜一个英文词（如 bleeding、NSAID、"
                "anticoagulant、renal），返回命中整句与所在章节。"
                "用来核对某个具体条文，或确认单侧说明书是否也提到某个风险。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string", "description": "药名（须先 lookup 过）"},
                    "term": {"type": "string", "description": "要搜索的英文词"},
                    "limit": {"type": "integer", "description": "最多返回几条命中（默认 3）"},
                },
                "required": ["drug_name", "term"],
            },
        },
    },
]

TOOL_NAMES: tuple[str, ...] = tuple(t["function"]["name"] for t in TOOLS)
