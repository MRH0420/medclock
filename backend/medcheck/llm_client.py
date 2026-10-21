"""DeepSeek 客户端（OpenAI 兼容协议）。

为什么不直接 `import openai`：赛前环境不一定装得上（本项目实测环境里 pip 被一个
死掉的代理挡住）。所以这里用标准库实现，装了 openai 包也能用，两条路任选。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .api_tool import build_opener
from .config import Settings


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    """模型有时会包一层 ```json，这里做一次宽容解析（只为了拿到 JSON）。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip().strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


class DeepSeekClient:
    """极简 chat 客户端：支持 tools（函数调用）与 json_object 输出。"""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.opener = build_opener()
        self._sdk = self._try_sdk()

    # ------------------------------------------------------------ 内部

    def _try_sdk(self):
        """如果环境里恰好有 openai 包，就用它（少踩协议细节）。"""
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return None
        try:
            return OpenAI(api_key=self.s.api_key, base_url=self.s.base_url)
        except Exception:
            return None

    def _endpoint(self) -> str:
        return f"{self.s.base_url.rstrip('/')}/chat/completions"

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        if not self.s.has_api_key:
            raise LLMError(
                "未配置 DeepSeek API Key（环境变量 DEEPSEEK_API_KEY / Agent_test/.env / "
                "config.py 里都为空）。"
            )
        if self._sdk is not None:
            return self._chat_sdk(messages, tools, json_mode, temperature)
        return self._chat_http(messages, tools, json_mode, temperature)

    def _chat_sdk(self, messages, tools, json_mode, temperature) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.s.model,
            "messages": messages,
            "temperature": self.s.temperature if temperature is None else temperature,
            "max_tokens": self.s.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._sdk.chat.completions.create(**kwargs)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - 统一包成 LLMError
            raise LLMError(f"DeepSeek 调用失败：{exc}") from exc
        msg = resp.choices[0].message
        return _normalize_message(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (msg.tool_calls or [])
                ],
            }
        )

    def _chat_http(self, messages, tools, json_mode, temperature) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.s.model,
            "messages": messages,
            "temperature": self.s.temperature if temperature is None else temperature,
            "max_tokens": self.s.max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            self._endpoint(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.s.api_key}",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with self.opener.open(req, timeout=self.s.llm_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            raise LLMError(f"DeepSeek HTTP {exc.code}：{detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"无法连接 {self.s.base_url}（{exc.reason}）") from exc
        except TimeoutError as exc:
            raise LLMError(f"DeepSeek 请求超时（>{self.s.llm_timeout}s）") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"DeepSeek 响应不是合法 JSON：{exc}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"DeepSeek 未返回 choices：{str(data)[:300]}")
        return _normalize_message(choices[0].get("message") or {})


def _normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
    """统一成 {role, content, tool_calls:[{id,name,arguments(JSON 字符串)}]}。"""
    out = {
        "role": "assistant",
        "content": msg.get("content") or "",
        "tool_calls": [],
    }
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        out["tool_calls"].append(
            {
                "id": tc.get("id") or f"call_{len(out['tool_calls'])}",
                "name": fn.get("name") or "",
                "arguments": fn.get("arguments") or "{}",
            }
        )
    return out


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    """模型的 arguments 是字符串，偶尔会是空串或坏 JSON —— 全部兜住。"""
    if not raw or not raw.strip():
        return {}
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return val if isinstance(val, dict) else {}


__all__ = ["DeepSeekClient", "LLMError", "_extract_json", "parse_tool_arguments"]
