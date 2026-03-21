"""
从模型回复文本中尽量稳健地解析根对象为 dict 的 JSON。

依次尝试多种策略；全部失败后再抛出 ValueError（附带最近若干条错误摘要）。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

try:
    import json5
except ImportError:
    json5 = None  # type: ignore


def _strip_leading_fence(s: str) -> str:
    """去掉开头的 ```json 围栏（若存在）。"""
    t = s.strip()
    m = re.match(r"^```(?:json)?\s*\n", t)
    if not m:
        return t
    rest = t[m.end() :]
    if rest.rstrip().endswith("```"):
        rest = rest.rstrip()[:-3].rstrip()
    return rest.strip()


def _fence_blocks_anywhere(s: str) -> List[str]:
    """提取文中所有 ``` ... ``` 代码块内容（含 json 标记）。"""
    out: List[str] = []
    for m in re.finditer(r"```(?:json)?\s*\n(.*?)```", s, flags=re.DOTALL):
        block = m.group(1).strip()
        if block:
            out.append(block)
    return out


def _extract_first_braced_object(s: str) -> Optional[str]:
    """
    从首个「{」起按括号深度截取到匹配的「}」，忽略字符串内的花括号。
    用于从夹杂说明文字的回复中抠出第一段完整 JSON 对象子串。
    """
    i = s.find("{")
    if i == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(s)):
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i : j + 1]
    return None


def _as_dict(obj: Any) -> Optional[Dict[str, Any]]:
    return obj if isinstance(obj, dict) else None


def _try_raw_decode_from_brace(s: str) -> Optional[Dict[str, Any]]:
    brace = s.find("{")
    if brace == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(s, brace)
        return _as_dict(obj)
    except json.JSONDecodeError:
        return None


def _try_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return _as_dict(json.loads(s))
    except json.JSONDecodeError:
        return None


def _try_json5_loads(s: str) -> Optional[Dict[str, Any]]:
    if json5 is None:
        return None
    try:
        return _as_dict(json5.loads(s))
    except Exception:
        return None


def _try_balanced_then_load(s: str) -> Optional[Dict[str, Any]]:
    sub = _extract_first_braced_object(s)
    if not sub:
        return None
    for fn in (_try_json_loads, _try_json5_loads, _try_raw_decode_from_brace):
        got = fn(sub)
        if got is not None:
            return got
    return None


def parse_model_json(text: str) -> Dict[str, Any]:
    """
    从模型回复中解析根 JSON 对象（dict）。

    策略顺序（对多份候选串各做一遍）：
    1. 全文 / 去围栏后的串：从首个「{」 raw_decode
    2. json.loads 整段
    3. json5.loads 整段（若已安装 json5，可容忍注释、尾逗号等）
    4. 括号平衡截取首段 `{...}` 后再 1～3
    候选串来源：去 BOM 后的全文、去行首围栏、文中所有 ``` 代码块。
    """
    if not isinstance(text, str):
        raise ValueError("模型返回类型须为字符串")
    t = text.strip()
    if not t:
        raise ValueError("模型返回空内容")

    t = t.lstrip("\ufeff").strip()

    candidates: List[str] = []
    candidates.append(t)
    candidates.append(_strip_leading_fence(t))
    candidates.extend(_fence_blocks_anywhere(t))

    seen: set[str] = set()
    uniq: List[str] = []
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)

    errors: List[str] = []

    for s in uniq:
        for name, fn in (
            ("raw_decode@{", _try_raw_decode_from_brace),
            ("json.loads", _try_json_loads),
            ("json5.loads", _try_json5_loads),
            ("balanced+load", _try_balanced_then_load),
        ):
            try:
                got = fn(s)
            except Exception as e:
                errors.append(f"{name}: {e!s}")
                continue
            if got is not None:
                return got

    tail = "; ".join(errors[-12:]) if errors else "无详细子错误"
    raise ValueError(
        "无法将模型回复解析为 JSON 对象（根须为 {...}）。"
        f"最近尝试: {tail}"
    )
