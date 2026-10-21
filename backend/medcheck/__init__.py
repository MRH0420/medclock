"""MedCheck —— 用药冲突辅助核查 Agent（只依据 FDA 官方说明书说话）。

对外只暴露最必要的几个名字。
"""

from .agent import AgentResult, MedCheckAgent
from .config import Settings, default_settings, read_deepseek_api_key

__version__ = "1.0.0"

__all__ = [
    "MedCheckAgent",
    "AgentResult",
    "Settings",
    "default_settings",
    "read_deepseek_api_key",
    "__version__",
]
