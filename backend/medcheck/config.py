"""运行期配置：路径、缓存策略、DeepSeek 连接参数。

设计原则：**离线优先**。所有官方说明书都可以先抓到本地 fixtures/ 里，
演示时用 --offline 走本地，结构上不依赖网络（见 Agent.md §6.4）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 路径

PKG_DIR = Path(__file__).resolve().parent
ROOT_DIR = PKG_DIR.parent  # backend/
CACHE_DIR = PKG_DIR / "cache"
FIXTURES_DIR = PKG_DIR / "fixtures"
TRACE_DIR = ROOT_DIR / ".data" / "medcheck_traces"

# ---------------------------------------------------------------- 凭据

# ★ 用户需要在这里填自己的 DeepSeek API Key（也可以放到环境变量 / .env 文件里）
DEEPSEEK_API_KEY_PLACEHOLDER = ""

# 空字符串、None、"sk-xxx"、"YOUR_API_KEY" 之类都视为"没填"
_PLACEHOLDER_MARKERS = (
    "sk-xxx", "your_api_key", "your-api-key", "yourkey", "placeholder",
    "replace_me", "replace-me", "todo", "<", "填入", "你的",
)


def read_deepseek_api_key() -> str:
    """按优先级读取 DeepSeek Key：环境变量 > Agent_test/.env > config.py 常量。

    取不到就返回空字符串——调用方必须处理"没有 Key"的情况，
    而不是带着空 Key 去发请求（那只会得到一个看不懂的 401）。
    """
    value = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if not value:
        value = _read_dotenv(DEEPSEEK_API_KEY_PLACEHOLDER)
    if _is_placeholder(value):
        return ""
    return value


def _is_placeholder(value: str) -> bool:
    if not value:
        return True
    low = value.lower()
    return any(marker in low for marker in _PLACEHOLDER_MARKERS)


def _read_dotenv(fallback: str) -> str:
    for path in (ROOT_DIR / ".env", PKG_DIR / ".env"):
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.strip() == "DEEPSEEK_API_KEY":
                val = val.split("#", 1)[0].strip().strip('"').strip("'")
                if val:
                    return val
    return fallback


# ---------------------------------------------------------------- LLM 默认参数


@dataclass
class Settings:
    """一次运行的全部可调参数。"""

    # --- 数据源 ---
    label_api_url: str = "https://api.fda.gov/drug/label.json"
    request_timeout: float = 30.0
    offline: bool = False          # True = 只读 fixtures/cache，绝不发网络请求
    use_cache: bool = True         # 命中 cache/ 就不重复请求 openFDA
    write_cache: bool = True       # 查到的新标签写回 cache/
    fixture_fallback: bool = False  # 网络失败时退回本地 fixtures

    # --- 模型 ---
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.0       # 医疗场景不要发挥，要稳定复现
    max_tokens: int = 4096
    llm_timeout: float = 120.0

    # --- Agent 循环 ---
    max_steps: int = 8             # 控制流上限，防止模型无限绕圈
    engine: str = "auto"           # auto | llm | template
    lang: str = "zh"               # zh = 完整中文报告；en = 英文版（main_sample.py 默认 en）
    compact: bool = False          # True = 只出一页简洁版报告（main_sample.py 默认 True）
    verbose: bool = True           # 打印每一步 [Observe]/[Tool]/[Gate] 轨迹
    trace: bool = True             # 把轨迹落盘到 traces/
    evidence_quote_limit: int = 12000  # 单节入证据库的截断长度（保证相互作用章节不被砍掉）

    _api_key: str = field(default="", repr=False)

    @property
    def api_key(self) -> str:
        if not self._api_key:
            self._api_key = read_deepseek_api_key()
        return self._api_key

    def apply_env_overrides(self) -> "Settings":
        """允许用环境变量覆盖模型与网关：DEEPSEEK_MODEL / DEEPSEEK_BASE_URL。"""
        model = (os.environ.get("DEEPSEEK_MODEL") or "").strip()
        if model:
            self.model = model
        base = (os.environ.get("DEEPSEEK_BASE_URL") or "").strip()
        if base:
            self.base_url = base
        return self

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    @property
    def cache_dir(self) -> Path:
        return CACHE_DIR

    @property
    def fixtures_dir(self) -> Path:
        return FIXTURES_DIR


def default_settings() -> Settings:
    return Settings()
