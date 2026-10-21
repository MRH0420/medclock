"""药品"药理类别"查询与类别匹配。

为什么需要它：
说明书经常**只按药物类别**描述相互作用，而不是点名具体药名。实测例子（逐字核对过）：

  Sertraline 警告：Concomitant use of aspirin, nonsteroidal anti-inflammatory drugs
                   (NSAIDs), other antiplatelet drugs, warfarin ... may increase this risk
  Methotrexate 相互作用：concomitant administration of methotrexate ... and nonsteroidal
                   anti-inflammatory drugs (NSAIDs)
  Lithium 相互作用：NSAID decrease renal blood flow, resulting in ... increased serum lithium

这三份标签里**都没有出现 "ibuprofen"**，所以纯药名匹配必然漏检。
本模块负责回答"这个药属于什么类别"，再由 screener 去对方标签里找该类别的表述。

数据来源（全部来自 FDA 官方接口，可溯源）：
  ① openFDA `/drug/ndc.json` 的 `pharm_class`（覆盖率最高，实测 28/32 常见药有数据）
  ② openFDA `/drug/label.json` 的 `openfda.pharm_class_*`（部分标签带，作为兜底）
  ③ 内置 CURATED_CLASSES：常用药的人工兜底（离线可用、可在演示机上复现）

★ 边界：类别匹配**不等于**相互作用结论。它只说明"对方说明书按类别描述了这类风险"，
  本药是否受影响、影响多大，仍需药师结合剂量与患者情况判断。所有输出都必须带这句限定。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ------------------------------------------------------------------ 词表

# 剂型/连接词：判断"是不是同一个药"时忽略
DOSE_FORM_WORDS: frozenset[str] = frozenset(
    """and or with the of for plus tablet tablets capsule capsules oral solution suspension
    injection cream gel ointment extended release delayed er sr xr ir mg ml mcg iu usp drug
    medicine medication film coated chewable softgel liquid drops spray powder granule packet
    effervescent concentrate anhydrous hcl""".split()
)

# 盐基/离子詞：判断"是不是同一个药"时**算作同一成分**（不能当额外成分）
SALT_WORDS: frozenset[str] = frozenset(
    """hydrochloride sodium potassium calcium magnesium sulfate sulphate bitartrate tartrate
    citrate maleate mesylate besylate tosylate succinate fumarate phosphate acetate nitrate
    bromide chloride iodide oxide carbonate bicarbonate gluconate lactate oxalate stearate
    palmitate benzoate salicylate dihydrate monohydrate hydrobromide hemihydrate decanoate
    enanthate valerate""".split()
)

# 太宽泛、不能用于"类别点名"判定的类别（会把无关的药也匹配上）
BROAD_CLASS_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"^allergens", r"dietary proteins", r"plant proteins", r"seed storage proteins",
        r"cell-mediated immunity", r"increased igg", r"bee venoms", r"^copper",
        r"^cations", r"^divalent", r"venous blood", r"^antidote for",
        r"increased histamine release", r"skin barrier activity",
    )
)
# 单词类别里允许参与匹配的例外（这些单词本身就是明确的药理类别，
# 而且其它说明书确实会用它们来点名风险，例如 tetracycline 的标签写 "calcium"）
SINGLE_WORD_CLASS_ALLOW: frozenset[str] = frozenset(
    {"macrolides", "quinolones", "fluoroquinolones", "benzodiazepines", "nitrates",
     "thyroxine", "catecholamines", "purines", "nucleosides", "diuretics", "statins",
     "anticoagulants", "corticosteroids", "calcium", "potassium"}
)

# 常见缩写别名：类别键 → 说明书里可能出现的其他写法
ACRONYM_ALIASES: dict[str, tuple[str, ...]] = {
    # 归一化后的键（空格分隔的小写）→ 别名
    "nonsteroidal anti inflammatory drug": (
        "nsaid", "nsaids", "nonsteroidal anti-inflammatory", "non-steroidal anti-inflammatory",
        "nonsteroidal anti inflammatory",
    ),
    "serotonin reuptake inhibitor": (
        "ssri", "ssris", "selective serotonin reuptake", "serotonin reuptake inhibitors",
    ),
    "monoamine oxidase inhibitor": ("maoi", "maois", "monoamine oxidase inhibitors"),
    "ace inhibitor": ("ace inhibitor", "ace inhibitors", "acei",
                      "angiotensin converting enzyme inhibitor"),
    "hmg coa reductase inhibitor": ("statin", "statins", "hmg-coa reductase", "hmg coa reductase"),
    "proton pump inhibitor": ("ppi", "ppis", "proton-pump inhibitor", "proton pump inhibitors"),
    "phosphodiesterase 5 inhibitor": ("pde5", "pde-5", "phosphodiesterase type 5",
                                      "phosphodiesterase-5"),
    "p2y12 platelet inhibitor": ("p2y12", "thienopyridine"),
    "vitamin k antagonist": ("coumarin", "vka", "vitamin k antagonists"),
    "beta blocker": ("beta blocker", "beta-blocker", "beta blockers",
                     "beta adrenergic blocking", "β-blocker"),
    "beta2 agonist": ("beta2 agonist", "beta-2 agonist", "beta2-adrenergic", "short-acting beta"),
    "calcium channel blocker": ("calcium channel blocker", "calcium channel antagonists",
                                "calcium-channel blocker"),
    "aldosterone antagonist": ("aldosterone antagonist", "potassium-sparing diuretic",
                              "potassium sparing diuretic", "mineralocorticoid receptor antagonist"),
    "cardiac glycoside": ("cardiac glycoside", "cardiac glycosides", "digitalis"),
    "nitrate": ("nitrate", "nitrates", "nitroglycerin"),
    "xanthine oxidase inhibitor": ("xanthine oxidase inhibitor", "xanthine oxidase inhibitors"),
    "purine antimetabolite": ("purine antimetabolite", "purine analogue", "azathioprine"),
    "folate antagonist": ("folate antagonist", "folate analog", "methotrexate",
                          "folic acid antagonist"),
    "benzodiazepine": ("benzodiazepine", "benzodiazepines", "benzo"),
    "opioid agonist": ("opioid agonist", "opioids", "opioid analgesic", "narcotic"),
    "potassium supplement": ("potassium supplement", "potassium supplements",
                             "potassium chloride", "potassium salt", "potassium compounds"),
    "macrolide": ("macrolide", "macrolides"),
    "thyroid hormone": ("thyroid hormone", "levothyroxine", "thyroxine"),
    "mood stabilizer": ("mood stabilizer", "lithium"),
    "fluoroquinolone": ("fluoroquinolone", "fluoroquinolones"),
    "quinolone": ("quinolone", "quinolones"),
    "rifamycin": ("rifamycin", "rifamycins", "rifampin"),
    "antiarrhythmic": ("antiarrhythmic", "amiodarone"),
}

# ------------------------------------------------------------------ 内置兜底表
# 仅用于：网络不可用、且该药没有缓存类别时。来源为公开的药理学常识分类，
# 与 FDA 接口给出的 EPC/MoA 命名保持一致（便于与接口数据合并去重）。
CURATED_CLASSES: dict[str, tuple[str, ...]] = {
    "warfarin": ("Vitamin K Antagonist [EPC]", "Vitamin K Inhibitors [MoA]"),
    "aspirin": ("Nonsteroidal Anti-inflammatory Drug [EPC]", "Cyclooxygenase Inhibitors [MoA]",
                "Antiplatelet Agent [EPC]"),
    "ibuprofen": ("Nonsteroidal Anti-inflammatory Drug [EPC]", "Cyclooxygenase Inhibitors [MoA]"),
    "naproxen": ("Nonsteroidal Anti-inflammatory Drug [EPC]", "Cyclooxygenase Inhibitors [MoA]"),
    "acetaminophen": ("Analgesic [EPC]", "Antipyretic [EPC]"),
    "sertraline": ("Serotonin Reuptake Inhibitor [EPC]", "Serotonin Uptake Inhibitors [MoA]"),
    "fluoxetine": ("Serotonin Reuptake Inhibitor [EPC]", "Serotonin Uptake Inhibitors [MoA]"),
    "paroxetine": ("Serotonin Reuptake Inhibitor [EPC]", "Serotonin Uptake Inhibitors [MoA]"),
    "phenelzine": ("Monoamine Oxidase Inhibitor [EPC]", "Monoamine Oxidase Inhibitors [MoA]"),
    "lithium carbonate": ("Mood Stabilizer [EPC]",),
    "methotrexate": ("Folate Analog Metabolic Inhibitor [EPC]",
                     "Folic Acid Metabolism Inhibitors [MoA]"),
    "azathioprine": ("Purine Antimetabolite [EPC]", "Nucleic Acid Synthesis Inhibitors [MoA]"),
    "allopurinol": ("Xanthine Oxidase Inhibitor [EPC]", "Xanthine Oxidase Inhibitors [MoA]"),
    "simvastatin": ("HMG-CoA Reductase Inhibitor [EPC]",
                    "Hydroxymethylglutaryl-CoA Reductase Inhibitors [MoA]"),
    "atorvastatin": ("HMG-CoA Reductase Inhibitor [EPC]",
                     "Hydroxymethylglutaryl-CoA Reductase Inhibitors [MoA]"),
    "clarithromycin": ("Macrolide Antimicrobial [EPC]", "Macrolides [CS]",
                       "Cytochrome P450 3A4 Inhibitors [MoA]"),
    "erythromycin": ("Macrolide Antimicrobial [EPC]", "Macrolides [CS]",
                     "Cytochrome P450 3A4 Inhibitors [MoA]"),
    "ciprofloxacin": ("Fluoroquinolone Antibacterial [EPC]", "Cytochrome P450 1A2 Inhibitors [MoA]"),
    "levofloxacin": ("Fluoroquinolone Antibacterial [EPC]",),
    "tizanidine": ("Central alpha-2 Adrenergic Agonist [EPC]", "Adrenergic alpha2-Agonists [MoA]"),
    "clopidogrel": ("P2Y12 Platelet Inhibitor [EPC]", "P2Y12 Receptor Antagonists [MoA]"),
    "omeprazole": ("Proton Pump Inhibitor [EPC]", "Cytochrome P450 2C19 Inhibitors [MoA]"),
    "lisinopril": ("Angiotensin Converting Enzyme Inhibitor [EPC]",
                   "Angiotensin-converting Enzyme Inhibitors [MoA]"),
    "spironolactone": ("Aldosterone Antagonist [EPC]", "Aldosterone Antagonists [MoA]",
                       "Potassium-sparing Diuretic [EPC]"),
    "albuterol": ("beta2-Adrenergic Agonist [EPC]", "Adrenergic beta2-Agonists [MoA]"),
    "propranolol": ("beta-Adrenergic Blocker [EPC]", "Adrenergic beta-Antagonists [MoA]"),
    "metoprolol": ("beta-Adrenergic Blocker [EPC]", "Adrenergic beta-Antagonists [MoA]"),
    "verapamil": ("Calcium Channel Blocker [EPC]", "Calcium Channel Antagonists [MoA]"),
    "amlodipine": ("Calcium Channel Blocker [EPC]", "Calcium Channel Antagonists [MoA]"),
    "digoxin": ("Cardiac Glycoside [EPC]", "Cardiac Glycosides [CS]"),
    "amiodarone": ("Antiarrhythmic [EPC]", "Cytochrome P450 3A4 Inhibitors [MoA]",
                   "Cytochrome P450 2C9 Inhibitors [MoA]"),
    "metformin": ("Biguanide [EPC]", "Biguanides [CS]"),
    "levothyroxine": ("l-Thyroxine [EPC]", "Thyroxine [CS]"),
    "oxycodone": ("Opioid Agonist [EPC]", "Full Opioid Agonists [MoA]"),
    "alprazolam": ("Benzodiazepine [EPC]", "Benzodiazepines [CS]"),
    "diazepam": ("Benzodiazepine [EPC]", "Benzodiazepines [CS]"),
    "nitroglycerin": ("Nitrate Vasodilator [EPC]", "Nitrates [CS]"),
    "sildenafil": ("Phosphodiesterase 5 Inhibitor [EPC]", "Phosphodiesterase 5 Inhibitors [MoA]"),
    "potassium chloride": ("Potassium Salt [EPC]", "Potassium Compounds [CS]"),
    "calcium carbonate": ("Calcium [CS]", "Antacid [EPC]"),
    "rifampin": ("Rifamycin Antibacterial [EPC]", "Rifamycins [CS]"),
    "amoxicillin": ("Penicillin-class Antibacterial [EPC]", "Penicillins [CS]"),
    "cefdinir": ("Cephalosporin Antibacterial [EPC]", "Cephalosporins [CS]"),
    "trimethoprim-sulfamethoxazole": (
        "Sulfonamide Antibacterial [EPC]", "Dihydrofolate Reductase Inhibitors [MoA]",
    ),
    "tramadol": ("Opioid Agonist [EPC]", "Serotonin Reuptake Inhibitor [EPC]"),
    "prednisone": ("Corticosteroid [EPC]", "Corticosteroid Hormone Receptor Agonists [MoA]"),
}

# ------------------------------------------------------------------ 数据结构


@dataclass
class DrugClassInfo:
    """一个药品的药理类别信息（全部来自官方接口或内置兜底表）。"""

    drug: str
    raw: list[str] = field(default_factory=list)      # 原始字符串（含 [EPC]/[MoA] 后缀）
    source: str = ""                                  # ndc | label | curated | cache | fixture | none
    retrieved_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "drug": self.drug,
            "raw": self.raw,
            "source": self.source,
            "retrieved_at": self.retrieved_at,
        }

    @property
    def is_empty(self) -> bool:
        return not self.raw

    def candidates(self) -> list[tuple[str, str]]:
        """可用于"类别点名"检索的 (检索短语, 展示名) 列表。

        过滤规则：去掉 [EPC] 类后缀 → 排除过于宽泛的类别 → 单实词类别需要白名单。
        同一展示名只保留一次（同样一个类别在 EPC/MoA/CS 里会重复出现）。
        """
        out: list[tuple[str, str]] = []
        seen_canon: set[str] = set()
        emitted: set[str] = set()
        for entry in self.raw:
            core = strip_class_suffix(entry)
            if not core:
                continue
            if any(p.search(core) for p in BROAD_CLASS_PATTERNS):
                continue
            words = [w for w in re.split(r"[\s\-]+", core) if w]
            if len(words) < 2 and normalize_class_key(core) not in SINGLE_WORD_CLASS_ALLOW:
                continue
            canon, pretty = canonical_display(core)
            if canon in seen_canon:
                continue                      # ★ 同一个类别（不同注册名）只出一条
            seen_canon.add(canon)

            # 检索短语 = 原始写法 + 归一化写法 + 常见缩写/连字符变体，命中任一即算
            phrases = [core.lower(), canon]
            phrases += list(ACRONYM_ALIASES.get(canon, ()))
            for phrase in phrases:
                p = re.sub(r"\s+", " ", phrase).strip()
                if not p or p in emitted:
                    continue
                emitted.add(p)
                out.append((p, pretty))
        return out


# ------------------------------------------------------------------ 工具函数


def strip_class_suffix(entry: str) -> str:
    """'Nonsteroidal Anti-inflammatory Drug [EPC]' → 'Nonsteroidal Anti-inflammatory Drug'"""
    return re.sub(r"\s*\[[A-Za-z]{2,4}\]\s*$", "", (entry or "").strip()).strip()


def normalize_class_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()


def name_tokens(name: str) -> set[str]:
    low = re.sub(r"[^a-z0-9\s]", " ", (name or "").lower())
    return {t for t in low.split() if len(t) > 2 and t not in DOSE_FORM_WORDS}


def record_matches_drug(product_generic_name: str, query: str) -> bool:
    """判断某条 NDC 产品记录是不是"我们查的那个药"（排除复方与同名异物）。

    规则：产品名与查询名有实词交集，且产品名里**没有多余的成分词**
    （盐基/离子词不算多余成分）。
    """
    want = name_tokens(query)
    got = name_tokens(product_generic_name)
    if not want or not got or not (want & got):
        return False
    extra = {t for t in got - want if t not in SALT_WORDS}
    return not extra


# 同一类别的不同 registry 写法（EPC/MoA/CS 各自命名不同）→ 归一到同一个键
CANONICAL_CLASS: dict[str, str] = {
    "non steroidal": "nonsteroidal anti inflammatory drug",
    "nonsteroidal anti inflammatory drug": "nonsteroidal anti inflammatory drug",
    "anti inflammatory agents non steroidal": "nonsteroidal anti inflammatory drug",
    "serotonin reuptake inhibitor": "serotonin reuptake inhibitor",
    "serotonin uptake inhibitors": "serotonin reuptake inhibitor",
    "monoamine oxidase inhibitor": "monoamine oxidase inhibitor",
    "monoamine oxidase inhibitors": "monoamine oxidase inhibitor",
    "potassium salt": "potassium supplement",
    "potassium compounds": "potassium supplement",
    "thyroxine": "thyroid hormone",
    "l thyroxine": "thyroid hormone",
    "aldosterone antagonist": "aldosterone antagonist",
    "aldosterone antagonists": "aldosterone antagonist",
    "p2y12 platelet inhibitor": "p2y12 platelet inhibitor",
    "p2y12 receptor antagonists": "p2y12 platelet inhibitor",
    "vitamin k antagonist": "vitamin k antagonist",
    "vitamin k inhibitors": "vitamin k antagonist",
    "cardiac glycoside": "cardiac glycoside",
    "cardiac glycosides": "cardiac glycoside",
    "nitrate vasodilator": "nitrate",
    "nitrates": "nitrate",
    "benzodiazepine": "benzodiazepine",
    "benzodiazepines": "benzodiazepine",
    "opioid agonist": "opioid agonist",
    "full opioid agonists": "opioid agonist",
    "macrolide antimicrobial": "macrolide",
    "macrolides": "macrolide",
    "fluoroquinolone antibacterial": "fluoroquinolone",
    "fluoroquinolones": "fluoroquinolone",
    "quinolone antimicrobial": "quinolone",
    "quinolones": "quinolone",
    "rifamycin antibacterial": "rifamycin",
    "rifamycins": "rifamycin",
    "proton pump inhibitor": "proton pump inhibitor",
    "proton pump inhibitors": "proton pump inhibitor",
    "hmg coa reductase inhibitor": "hmg coa reductase inhibitor",
    "hydroxymethylglutaryl coa reductase inhibitors": "hmg coa reductase inhibitor",
    "phosphodiesterase 5 inhibitor": "phosphodiesterase 5 inhibitor",
    "phosphodiesterase 5 inhibitors": "phosphodiesterase 5 inhibitor",
    "angiotensin converting enzyme inhibitor": "ace inhibitor",
    "angiotensin converting enzyme inhibitors": "ace inhibitor",
    "calcium channel blocker": "calcium channel blocker",
    "calcium channel antagonists": "calcium channel blocker",
    "xanthine oxidase inhibitor": "xanthine oxidase inhibitor",
    "xanthine oxidase inhibitors": "xanthine oxidase inhibitor",
    "purine antimetabolite": "purine antimetabolite",
    "folate analog metabolic inhibitor": "folate antagonist",
    "folic acid metabolism inhibitors": "folate antagonist",
    "beta adrenergic blocker": "beta blocker",
    "adrenergic beta antagonists": "beta blocker",
    "beta2 adrenergic agonist": "beta2 agonist",
    "adrenergic beta2 agonists": "beta2 agonist",
    "central alpha 2 adrenergic agonist": "alpha2 agonist",
    "adrenergic alpha2 agonists": "alpha2 agonist",
    "mood stabilizer": "mood stabilizer",
    "antiarrhythmic": "antiarrhythmic",
}


def canonical_display(core: str) -> tuple[str, str]:
    """把类别的各种注册名归一：(canonical_key, 最规范的展示名)。"""
    key = normalize_class_key(core)
    canon = CANONICAL_CLASS.get(key)
    if canon:
        # 展示名取"人读起来最清楚"的那个写法
        pretty = {
            "nonsteroidal anti inflammatory drug": "Nonsteroidal Anti-inflammatory Drug (NSAID)",
            "serotonin reuptake inhibitor": "Serotonin Reuptake Inhibitor (SSRI)",
            "monoamine oxidase inhibitor": "Monoamine Oxidase Inhibitor (MAOI)",
            "potassium supplement": "Potassium Supplement / Potassium Salt",
            "thyroid hormone": "Thyroid Hormone (Thyroxine)",
            "ace inhibitor": "Angiotensin Converting Enzyme (ACE) Inhibitor",
            "hmg coa reductase inhibitor": "HMG-CoA Reductase Inhibitor (Statin)",
            "proton pump inhibitor": "Proton Pump Inhibitor (PPI)",
            "phosphodiesterase 5 inhibitor": "Phosphodiesterase 5 (PDE5) Inhibitor",
            "beta blocker": "Beta-Adrenergic Blocker",
            "beta2 agonist": "Beta2-Adrenergic Agonist",
            "alpha2 agonist": "Central Alpha-2 Adrenergic Agonist",
            "calcium channel blocker": "Calcium Channel Blocker",
            "aldosterone antagonist": "Aldosterone Antagonist / Potassium-sparing Diuretic",
            "vitamin k antagonist": "Vitamin K Antagonist (Warfarin class)",
            "nitrate": "Nitrate Vasodilator",
            "opioid agonist": "Opioid Agonist",
            "macrolide": "Macrolide Antimicrobial",
            "fluoroquinolone": "Fluoroquinolone Antibacterial",
            "quinolone": "Quinolone Antimicrobial",
            "rifamycin": "Rifamycin Antibacterial",
            "benzodiazepine": "Benzodiazepine",
            "cardiac glycoside": "Cardiac Glycoside",
            "p2y12 platelet inhibitor": "P2Y12 Platelet Inhibitor",
            "xanthine oxidase inhibitor": "Xanthine Oxidase Inhibitor",
            "purine antimetabolite": "Purine Antimetabolite",
            "folate antagonist": "Folate Antagonist / Antimetabolite",
            "mood stabilizer": "Mood Stabilizer",
            "antiarrhythmic": "Antiarrhythmic",
        }.get(canon)
        return canon, pretty or core
    return key, core


def class_reference_block(infos: dict[str, Any], lang: str = "zh") -> dict[str, Any]:
    """给模型看的类别参考块（写进报告提示词的数据，不是结论）。"""
    out: dict[str, Any] = {}
    for slot, info in infos.items():
        out[slot] = {
            "drug": info.drug,
            "source": info.source,
            "classes": [{"phrase": p, "display": d} for p, d in info.candidates()],
        }
    return out
