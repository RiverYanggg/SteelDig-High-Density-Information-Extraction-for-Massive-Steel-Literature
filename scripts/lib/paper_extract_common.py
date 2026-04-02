"""
文献实体抽取管线中与「OpenAI 兼容 API / Schema / Prompt / 用量统计 / 任务日志」相关的共用逻辑。

由 paper_entity_extract_text_once.py 与 paper_entity_extract_multi_once.py 导入；
二者在默认并发、system 消息中 Schema 前置说明、任务日志原子写入策略上仍有意保持差异。
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import json5
from openai import OpenAI

# ---------------------------------------------------------------------------
# 环境变量与默认值（与两脚本历史行为对齐）
# ---------------------------------------------------------------------------
ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_MODEL = "OPENAI_MODEL"
ENV_OPENAI_MAX_TOKENS = "OPENAI_MAX_TOKENS"
ENV_WORKERS = "STEELDIG_EXTRACT_WORKERS"

DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_OPENAI_API_KEY = "EMPTY"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 65536

# system 消息里 Schema JSON 与规则正文之间的衔接（纯文本脚本仅换行；多模态多一段说明）
SCHEMA_GAP_TEXT_ONLY = "\n\n 具体抽取规则见下方 Schema JSON。\n\n"
SCHEMA_PREAMBLE_MULTIMODAL = (
    "\n\n【JSON Schema（已由程序去除注释，键与嵌套结构必须与输出一致；"
    "可增键但不可擅自改名已有键）】\n"
)

stderr_exc_lock = threading.Lock()
task_log_write_lock = threading.Lock()


def try_load_dotenv(*, project_root: Path, cwd: Optional[Path] = None) -> None:
    """若已安装 python-dotenv，则从项目根与当前工作目录加载 .env。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(project_root / ".env")
    load_dotenv((cwd or Path.cwd()) / ".env")


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    durable: bool = False,
) -> None:
    """原子写文件：临时文件 + os.replace。durable=True 时 flush/fsync 并 best-effort 同步目录。"""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    if not durable:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
        return
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def default_workers(*, fallback: int) -> int:
    raw = (os.environ.get(ENV_WORKERS) or str(fallback)).strip()
    try:
        return max(1, int(raw, 10))
    except ValueError:
        return fallback


def effective_openai_base_url() -> str:
    return os.environ.get(ENV_OPENAI_BASE_URL, DEFAULT_OPENAI_BASE_URL).rstrip("/")


def openai_client() -> OpenAI:
    key = os.environ.get(ENV_OPENAI_API_KEY, DEFAULT_OPENAI_API_KEY)
    return OpenAI(api_key=key, base_url=effective_openai_base_url())


def resolve_chat_model_id(client: OpenAI, cli_model: Optional[str]) -> str:
    for candidate in ((cli_model or "").strip(), (os.environ.get(ENV_OPENAI_MODEL) or "").strip()):
        if candidate:
            return candidate
    try:
        listed = client.models.list()
    except Exception as e:
        raise SystemExit(
            "无法访问 vLLM 的 /v1/models（请检查 OPENAI_BASE_URL、服务是否已启动、"
            "OPENAI_API_KEY 是否与 --api-key 一致）。\n"
            f"详情: {e}"
        ) from e
    data = getattr(listed, "data", None) or []
    if not data:
        raise SystemExit(
            "/v1/models 未返回任何模型。请设置环境变量 OPENAI_MODEL 或使用 --model，"
            "名称须与 vLLM 注册的 id 完全一致（可对照启动日志或 curl /v1/models）。"
        )
    return str(data[0].id)


def get_obj_field(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_usage_from_completion(completion: Any) -> Optional[Dict[str, Any]]:
    usage = get_obj_field(completion, "usage")
    if usage is None:
        return None
    pt = get_obj_field(usage, "prompt_tokens")
    ct = get_obj_field(usage, "completion_tokens")
    tt = get_obj_field(usage, "total_tokens")
    details = get_obj_field(usage, "completion_tokens_details")
    reasoning_tok = None
    if details is not None:
        reasoning_tok = get_obj_field(details, "reasoning_tokens")

    out: Dict[str, Any] = {}
    if pt is not None:
        out["prompt_tokens"] = int(pt)
    if ct is not None:
        out["completion_tokens"] = int(ct)
    if tt is not None:
        out["total_tokens"] = int(tt)
    elif pt is not None and ct is not None:
        out["total_tokens"] = int(pt) + int(ct)
    if reasoning_tok is not None:
        out["completion_tokens_reasoning"] = int(reasoning_tok)
        if ct is not None:
            out["completion_tokens_response"] = max(0, int(ct) - int(reasoning_tok))

    return out if out else None


def extract_output_text_stats(message: Any) -> Dict[str, Any]:
    content = get_obj_field(message, "content")
    if not isinstance(content, str):
        content = str(content) if content is not None else ""
    reasoning = get_obj_field(message, "reasoning_content")
    if reasoning is None:
        reasoning = get_obj_field(message, "thinking")
    stats: Dict[str, Any] = {"response_chars": len(content)}
    if reasoning is not None and str(reasoning).strip():
        stats["reasoning_chars"] = len(str(reasoning))
    return stats


def usage_meta_from_completion(completion: Any, message: Any) -> Optional[Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    u = extract_usage_from_completion(completion)
    if u:
        meta["usage"] = u
    s = extract_output_text_stats(message)
    if s:
        meta["output_text_stats"] = s
    return meta if meta else None


def load_schema_object(schema_path: Path) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json5.load(f)


def load_system_prompt_from_markdown(md_path: Path, *, variant: str) -> str:
    if variant == "text":
        header = "## 系统提示词（纯文本）"
    elif variant == "multi":
        header = "## 系统提示词（多模态）"
    else:
        raise ValueError(f"未知 prompt variant: {variant!r}，应为 \"text\" 或 \"multi\"")
    raw = md_path.read_text(encoding="utf-8")
    m = re.search(
        re.escape(header) + r"\s*\n```(?:text)?\s*\n(.*?)```",
        raw,
        flags=re.DOTALL,
    )
    if not m:
        raise ValueError(
            f"无法在 {md_path} 中解析 {header} 下方的 ``` 系统提示词代码块"
        )
    return m.group(1).strip()


def build_system_content(
    schema_path: Path,
    prompt_md_path: Path,
    *,
    prompt_variant: str,
    schema_intro_before_json: str,
) -> str:
    """组装完整 system 消息：规则 Markdown 片段 + schema_intro + Schema JSON（紧凑单行）。"""
    schema_obj = load_schema_object(schema_path)
    schema_compact = json.dumps(schema_obj, ensure_ascii=False)
    system_rules = load_system_prompt_from_markdown(prompt_md_path, variant=prompt_variant)
    return system_rules + schema_intro_before_json + schema_compact


def aggregate_usage_for_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "completion_tokens_reasoning",
        "completion_tokens_response",
    )
    sums: Dict[str, int] = {k: 0 for k in keys}
    counts: Dict[str, int] = {k: 0 for k in keys}
    for it in items:
        u = it.get("usage")
        if not isinstance(u, dict):
            continue
        for k in keys:
            v = u.get(k)
            if v is not None:
                sums[k] += int(v)
                counts[k] += 1
    out: Dict[str, Any] = {}
    for k in keys:
        if counts[k] > 0:
            out[f"{k}_sum"] = sums[k]
    if out:
        out["items_with_usage_count"] = len(
            [it for it in items if isinstance(it.get("usage"), dict)]
        )
    return out


def write_task_log(
    path: Path,
    *,
    started_at: str,
    finished_at: Optional[str],
    wall_seconds: Optional[float],
    openai_base_url: str,
    model_id: str,
    workers: int,
    dry_run: bool,
    temperature: float,
    max_tokens: int,
    items: List[Dict[str, Any]],
    atomic_durable: bool = False,
    atomic_retries: int = 1,
) -> None:
    """写入批任务 JSON 日志；atomic_retries>1 时在 OSError 时短暂退避重试（多模态脚本历史行为）。"""
    summary: Dict[str, Any] = {"ok": 0, "skipped": 0, "failed": 0}
    processing_sum = 0.0
    for it in items:
        st = it.get("status")
        if st in summary:
            summary[st] += 1
        psec = it.get("processing_seconds")
        if psec is not None:
            processing_sum += float(psec)
    summary["processing_seconds_sum"] = round(processing_sum, 4)
    usage_agg = aggregate_usage_for_summary(items)
    if usage_agg:
        summary["usage_tokens"] = usage_agg
    wall_r = round(float(wall_seconds), 4) if wall_seconds is not None else None
    payload = {
        "started_at": started_at,
        "finished_at": finished_at,
        "batch_wall_seconds": wall_r,
        "wall_seconds": wall_r,
        "openai_base_url": openai_base_url,
        "model": model_id,
        "workers": workers,
        "dry_run": dry_run,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "summary": summary,
        "items": items,
    }
    path = path.resolve()
    blob = json.dumps(payload, ensure_ascii=False, indent=2)
    retries = max(1, int(atomic_retries))
    with task_log_write_lock:
        if retries <= 1:
            # 与历史 paper_entity_extract_text_once 一致：I/O 错误原样上抛，便于调用方处理
            atomic_write_text(path, blob, encoding="utf-8", durable=atomic_durable)
        else:
            last_err: Optional[OSError] = None
            for i in range(retries):
                try:
                    atomic_write_text(path, blob, encoding="utf-8", durable=atomic_durable)
                    last_err = None
                    break
                except OSError as e:
                    last_err = e
                    if i + 1 < retries:
                        time.sleep(0.2 * (i + 1))
            if last_err is not None:
                raise SystemExit(f"写入任务日志失败: {path} | 错误: {last_err}")
    if finished_at is not None:
        print(f"任务日志已写入: {path}", file=sys.stderr)
