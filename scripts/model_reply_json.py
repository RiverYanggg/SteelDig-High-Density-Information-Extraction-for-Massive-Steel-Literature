"""
从模型回复文本中尽量稳健地解析根对象为 dict 的 JSON。

优先扫描全文内所有平衡 `{...}`，按与 paper_entity_schema.jsonc 一致的根级键命中数
选取最佳对象（缓解模型先输出长段思考、行内示例 `` `{}` `` 等导致的误解析）；
其余情况再回退到 raw_decode / json.loads / 围栏块等策略。全部失败则抛出 ValueError。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import json5
except ImportError:
    json5 = None  # type: ignore

# 与 paper_entity_schema.jsonc 根级键一致；用于从「夹杂思考过程」的回复中选出真正的抽取结果。
SCHEMA_ROOT_KEYS: Tuple[str, ...] = (
    "papers",
    "alloys",
    "processes",
    "samples",
    "processing_steps",
    "structures",
    "interfaces",
    "properties",
    "performance",
    "characterization_methods",
    "computational_details",
    "unmapped_findings",
)


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


def _extract_balanced_object_from(s: str, start: int) -> Optional[str]:
    """
    从 s[start] 必须为「{」的位置起，按括号深度截取到匹配的「}」，忽略字符串内的花括号。
    """
    if start < 0 or start >= len(s) or s[start] != "{":
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(s)):
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
                return s[start : j + 1]
    return None


def _extract_first_braced_object(s: str) -> Optional[str]:
    """
    从首个「{」起截取第一段完整 `{...}`（用于从夹杂说明文字的回复中抠 JSON）。
    """
    i = s.find("{")
    if i == -1:
        return None
    return _extract_balanced_object_from(s, i)


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


def _score_schema_dict(d: Dict[str, Any]) -> int:
    return sum(1 for k in SCHEMA_ROOT_KEYS if k in d)


def _parse_dict_from_substring(sub: str) -> Optional[Dict[str, Any]]:
    for fn in (_try_json_loads, _try_json5_loads):
        got = fn(sub)
        if got is not None:
            return got
    t = sub.strip()
    if t.startswith("{"):
        try:
            obj, _ = json.JSONDecoder().raw_decode(t)
            return _as_dict(obj)
        except json.JSONDecodeError:
            pass
    return None


def _best_dict_by_schema_scan(s: str) -> Optional[Dict[str, Any]]:
    """
    在全文内扫描每一处 `{`，尝试截取平衡括号并解析为对象；
    选取「命中 SCHEMA_ROOT_KEYS 数量最多、同分则更长」的 dict。
    用于规避：思考文字里出现 `` `{}` ``、示例片段 `[{"uuid":null}]` 等抢先被 `json.loads` 成 `{}` 或碎对象。
    """
    best: Optional[Tuple[int, int, Dict[str, Any]]] = None
    n = len(s)
    i = 0
    while i < n:
        if s[i] != "{":
            i += 1
            continue
        sub = _extract_balanced_object_from(s, i)
        if not sub or sub == "{}":
            i += 1
            continue
        d = _parse_dict_from_substring(sub)
        if d is None:
            i += 1
            continue
        sc = _score_schema_dict(d)
        L = len(sub)
        if best is None or sc > best[0] or (sc == best[0] and L > best[1]):
            best = (sc, L, d)
        i += 1
    if best is None or best[0] == 0:
        return None
    return best[2]


def parse_model_json(text: str) -> Dict[str, Any]:
    """
    从模型回复中解析根 JSON 对象（dict）。

    优先策略：在候选串中**扫描所有**平衡 `{...}`，按与 `paper_entity_schema.jsonc`
    根级键命中数量选取最佳对象（避免思考文字里抢先出现的 `` `{}` ``、示例片段等）。

    回退策略（对多份候选串各做一遍）：
    1. 从首个「{」 raw_decode
    2. json.loads 整段
    3. json5.loads 整段（若已安装 json5）
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

    global_best: Optional[Tuple[int, int, Dict[str, Any]]] = None
    for s in uniq:
        got = _best_dict_by_schema_scan(s)
        if got is None:
            continue
        sc = _score_schema_dict(got)
        L = len(json.dumps(got, ensure_ascii=False))
        if global_best is None or sc > global_best[0] or (
            sc == global_best[0] and L > global_best[1]
        ):
            global_best = (sc, L, got)

    if global_best is not None:
        return global_best[2]

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
