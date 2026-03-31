#!/usr/bin/env python3
"""
一次性多模态文献实体抽取：读取 **build_multimodal_content** 已生成的
`*_multimodal_content.json`（OpenAI 兼容的 text + image_url 列表），
将其中本地文件路径在调用前转为 base64，再请求本地 vLLM（OpenAI 兼容 /v1），
按 paper_entity_schema.jsonc 输出 JSON。

多模态需 vLLM 侧部署**视觉语言模型**；若仅文本模型（如 Qwen3.5-9B 纯文本）
请改用 paper_entity_extract_text_once.py。

请先运行：
python scripts/clean_content_list.py
python scripts/build_multimodal_content.py

依赖：openai、json5、tqdm、python-dotenv（可选，用于 .env）

用法（在项目根目录）：
  # 先启动支持多模态的 vLLM；端口与 OPENAI_BASE_URL 一致（参见 digmodel/basemodel/start_vllm.sh）
  export OPENAI_BASE_URL=http://127.0.0.1:8001/v1
  export OPENAI_API_KEY=EMPTY
  export OPENAI_MODEL=<你的视觉模型 served-model-name>

  # 批量：扫描 multimodal_content 下所有 *_multimodal_content.json
  python scripts/paper_entity_extract_multi_once.py \\
    --input datasets/multimodal_content \\
    --output-dir datasets/output_multi

  # 单文件
  python scripts/paper_entity_extract_multi_once.py -i datasets/multimodal_content/0321_noted_multimodal_content.json

  # 并行、任务日志（每篇 token 与 summary.usage_tokens 汇总，与 paper_entity_extract_text_once.py 一致）
  python scripts/paper_entity_extract_multi_once.py \\
    --input datasets/multimodal_content \\
    --output-dir datasets/output_multi \\
    -j 4 \\
    --task-log datasets/output_multi/extract_multi_task_log.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

MULTIMODAL_INPUT_DIR = PROJECT_ROOT / "datasets" / "multimodal_content"
OUTPUT_MULTI_DIR = PROJECT_ROOT / "datasets" / "output_multi"

# =========================
# 可配置项（常修改参数集中，与 paper_entity_extract_text_once.py 对齐）
# =========================
ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_MODEL = "OPENAI_MODEL"
ENV_OPENAI_MAX_TOKENS = "OPENAI_MAX_TOKENS"
ENV_WORKERS = "STEELDIG_EXTRACT_WORKERS"

DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_OPENAI_API_KEY = "EMPTY"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 65536
DEFAULT_WORKERS = 3

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(Path.cwd() / ".env")
except ImportError:
    pass

from build_multimodal_content import encode_image  # noqa: E402
from model_reply_json import parse_model_json  # noqa: E402

try:
    import json5
except ImportError as e:
    raise SystemExit(
        "缺少依赖 json5，请执行: pip install json5\n"
        f"原始错误: {e}"
    ) from e

try:
    from tqdm.auto import tqdm
except ImportError as e:
    raise SystemExit(
        "缺少依赖 tqdm，请执行: pip install tqdm\n"
        f"原始错误: {e}"
    ) from e

from openai import OpenAI  # noqa: E402

_exc_lock = threading.Lock()


def _default_workers() -> int:
    raw = (os.environ.get(ENV_WORKERS) or str(DEFAULT_WORKERS)).strip()
    try:
        return max(1, int(raw, 10))
    except ValueError:
        return DEFAULT_WORKERS


def _effective_openai_base_url() -> str:
    return os.environ.get(ENV_OPENAI_BASE_URL, DEFAULT_OPENAI_BASE_URL).rstrip("/")


def _openai_client() -> OpenAI:
    """与 digmodel/basemodel/start_vllm.sh、test_vllm.py 一致：本地 OpenAI 兼容端点。"""
    key = os.environ.get(ENV_OPENAI_API_KEY, DEFAULT_OPENAI_API_KEY)
    return OpenAI(api_key=key, base_url=_effective_openai_base_url())


def _resolve_chat_model_id(client: OpenAI, cli_model: Optional[str]) -> str:
    """与 vLLM 返回的 model id 必须一致；未指定时取 /v1/models 第一个 id。"""
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


def _get_obj_field(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_usage_from_completion(completion: Any) -> Optional[Dict[str, Any]]:
    """从 completion.usage 提取 token；若存在 reasoning 细分则一并记录（与 text_once 一致）。"""
    usage = _get_obj_field(completion, "usage")
    if usage is None:
        return None
    pt = _get_obj_field(usage, "prompt_tokens")
    ct = _get_obj_field(usage, "completion_tokens")
    tt = _get_obj_field(usage, "total_tokens")
    details = _get_obj_field(usage, "completion_tokens_details")
    reasoning_tok = None
    if details is not None:
        reasoning_tok = _get_obj_field(details, "reasoning_tokens")

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
            out["completion_tokens_response"] = max(
                0, int(ct) - int(reasoning_tok)
            )

    return out if out else None


def _extract_output_text_stats(message: Any) -> Dict[str, Any]:
    """若 API 将 thinking 与最终回复分字段，则分别统计字符数。"""
    content = _get_obj_field(message, "content")
    if not isinstance(content, str):
        content = str(content) if content is not None else ""
    reasoning = _get_obj_field(message, "reasoning_content")
    if reasoning is None:
        reasoning = _get_obj_field(message, "thinking")
    stats: Dict[str, Any] = {"response_chars": len(content)}
    if reasoning is not None and str(reasoning).strip():
        stats["reasoning_chars"] = len(str(reasoning))
    return stats


def _usage_meta_from_completion(completion: Any, message: Any) -> Optional[Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    u = _extract_usage_from_completion(completion)
    if u:
        meta["usage"] = u
    s = _extract_output_text_stats(message)
    if s:
        meta["output_text_stats"] = s
    return meta if meta else None


def _guess_image_mime(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    return {
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }.get(ext, "image/jpeg")


def verify_multimodal_image_paths_exist(parts: List[Dict[str, Any]]) -> None:
    """检查 JSON 中非 data: 的 image_url 是否指向存在的本地文件。"""
    for p in parts:
        if p.get("type") != "image_url":
            continue
        inner = p.get("image_url") or {}
        url = inner.get("url") or ""
        if not isinstance(url, str):
            raise ValueError(f"无效的 image_url: {p!r}")
        if url.startswith("data:"):
            continue
        path = Path(url)
        if not path.is_file():
            raise FileNotFoundError(
                f"多模态 JSON 引用的图片不存在: {path}\n"
                "请确认已运行 build_multimodal_content 且图片仍在 paper_parsered 下。"
            )


def ensure_multimodal_payload_for_api(
    parts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    将 multimodal_content JSON 中的条目转为可直接发给 API 的 content。

    - text：原样保留
    - image_url.url 已为 data:...;base64,... 的，原样保留
    - image_url.url 为本地文件路径的，读入并转为 data URL（OpenAI 兼容多模态常用 base64）
    """
    out: List[Dict[str, Any]] = []
    for p in parts:
        if p.get("type") != "image_url":
            out.append(p)
            continue
        inner = p.get("image_url") or {}
        url = inner.get("url") or ""
        if not isinstance(url, str):
            raise ValueError(f"无效的 image_url: {p!r}")
        if url.startswith("data:"):
            out.append(p)
            continue
        path = Path(url)
        if not path.is_file():
            raise FileNotFoundError(
                f"多模态 JSON 引用的图片不存在: {path}\n"
                "请确认已运行 build_multimodal_content 且图片仍在 paper_parsered 下。"
            )
        mime = _guess_image_mime(path)
        b64 = encode_image(path)
        out.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )
    return out


def load_schema_object(schema_path: Path) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json5.load(f)


def load_system_prompt_from_markdown(md_path: Path, *, variant: str) -> str:
    """从 paper_entity_extraction_prompt.md 中提取对应模式的「系统提示词」代码块正文。

    variant: \"text\" -> ## 系统提示词（纯文本）；\"multi\" -> ## 系统提示词（多模态）
    """
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
    schema_path: Path, prompt_md_path: Path, *, prompt_variant: str
) -> str:
    """组装完整 system 消息（规则 + Schema JSON）。"""
    schema_obj = load_schema_object(schema_path)
    schema_compact = json.dumps(schema_obj, ensure_ascii=False)
    system_rules = load_system_prompt_from_markdown(
        prompt_md_path, variant=prompt_variant
    )
    return (
        system_rules
        + "\n\n【JSON Schema（已由程序去除注释，键与嵌套结构必须与输出一致；"
        "可增键但不可擅自改名已有键）】\n"
        + schema_compact
    )


def collect_multimodal_files(path: Path) -> List[Path]:
    """
    path 为文件：须为 *_multimodal_content.json。
    path 为目录：其下所有 *_multimodal_content.json（仅一层，按名排序）。
    """
    path = path.resolve()
    if path.is_file():
        if not path.name.endswith("_multimodal_content.json"):
            raise SystemExit(
                f"单文件须为 *_multimodal_content.json，当前: {path.name}"
            )
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*_multimodal_content.json"))
        if not files:
            raise SystemExit(
                f"目录中未找到 *_multimodal_content.json: {path}\n"
                "请先运行: python scripts/build_multimodal_content.py"
            )
        return list(files)
    raise SystemExit(f"路径不存在: {path}")


def derive_output_path(multimodal_path: Path, output_dir: Path) -> Path:
    """foo_multimodal_content.json -> output_dir/foo_entities.json"""
    name = multimodal_path.name
    if name.endswith("_multimodal_content.json"):
        out_name = name.replace("_multimodal_content.json", "_entities.json")
    else:
        out_name = f"{multimodal_path.stem}_entities.json"
    return output_dir.resolve() / out_name


def run_extraction(
    *,
    multimodal_json_path: Path,
    system_content: str,
    model: str,
    output_path: Optional[Path],
    dry_run: bool,
    temperature: float = 0.3,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    quiet: bool = False,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """返回 (解析后的 JSON 对象, token/输出分块元数据)。dry_run 时第二项为 None。

    quiet: 为 True 时不打印「已写入…」（多线程时避免输出交错）。
    """
    with open(multimodal_json_path, "r", encoding="utf-8") as f:
        raw_parts: List[Dict[str, Any]] = json.load(f)

    if not isinstance(raw_parts, list):
        raise ValueError(f"多模态 JSON 顶层须为数组: {multimodal_json_path}")

    if dry_run:
        verify_multimodal_image_paths_exist(raw_parts)
        multimodal_parts: List[Dict[str, Any]] = raw_parts
    else:
        multimodal_parts = ensure_multimodal_payload_for_api(raw_parts)

    user_tail = (
        "【任务】上文按阅读顺序给出了论文全文（含图注/表注/公式文本与对应插图）。"
        "请综合文本与图像信息，一次性抽取结构化实体。\n"
        "【输出】仅输出一个合法 JSON 对象，不要 Markdown 围栏、不要解释性文字；"
        "严格遵守系统提示中的类型约束（禁止 number/boolean 叶子类型）。"
    )

    user_content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": "【论文多模态内容 — 按顺序阅读】\n",
        },
        *multimodal_parts,
        {"type": "text", "text": user_tail},
    ]

    if dry_run:
        print(
            json.dumps(
                {
                    "multimodal_file": str(multimodal_json_path),
                    "system_len": len(system_content),
                    "user_parts": len(user_content),
                    "openai_base_url": _effective_openai_base_url(),
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "output": str(output_path) if output_path else None,
                    "usage_tokens": None,
                    "note": "dry_run 不调用 API，无 token 统计",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return {}, None

    client = _openai_client()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    msg = completion.choices[0].message
    raw_reply = msg.content
    if not raw_reply:
        raise RuntimeError("模型返回空内容")
    if not isinstance(raw_reply, str):
        raw_reply = str(raw_reply)

    usage_meta = _usage_meta_from_completion(completion, msg)

    try:
        result = parse_model_json(raw_reply)
    except ValueError as e:
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path = output_path.with_suffix(".raw.txt")
            raw_path.write_text(raw_reply, encoding="utf-8")
            if not quiet:
                print(f"JSON 解析失败，已保存原始回复: {raw_path}", file=sys.stderr)
            err = ValueError(f"{e}\n原始模型输出已写入: {raw_path}")
            if usage_meta:
                err.usage_meta = usage_meta  # type: ignore[attr-defined]
            raise err from e
        raise

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        if not quiet:
            print(f"已写入: {output_path}")

    return result, usage_meta if usage_meta else None


def _task_record(
    *,
    multimodal_json_path: Path,
    json_output_path: Path,
    status: str,
    processing_seconds: Optional[float],
    error: Optional[str],
    usage_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ps: Optional[float]
    if processing_seconds is None:
        ps = None
    else:
        ps = round(float(processing_seconds), 4)
    rec: Dict[str, Any] = {
        "multimodal_input": str(multimodal_json_path),
        "json_output": str(json_output_path),
        "status": status,
        "processing_seconds": ps,
        "error": error,
    }
    if usage_meta:
        if "usage" in usage_meta:
            rec["usage"] = usage_meta["usage"]
        if "output_text_stats" in usage_meta:
            rec["output_text_stats"] = usage_meta["output_text_stats"]
    return rec


def _run_one_extraction_job(
    *,
    multimodal_json_path: Path,
    json_output_path: Path,
    system_content: str,
    model_id: str,
    dry_run: bool,
    temperature: float,
    max_tokens: int,
    quiet: bool,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        _result, usage_meta = run_extraction(
            multimodal_json_path=multimodal_json_path,
            system_content=system_content,
            model=model_id,
            output_path=json_output_path,
            dry_run=dry_run,
            temperature=temperature,
            max_tokens=max_tokens,
            quiet=quiet,
        )
        elapsed = time.perf_counter() - t0
        return _task_record(
            multimodal_json_path=multimodal_json_path,
            json_output_path=json_output_path,
            status="ok",
            processing_seconds=elapsed,
            error=None,
            usage_meta=usage_meta,
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        usage_meta = getattr(e, "usage_meta", None)
        with _exc_lock:
            print(f"失败: {multimodal_json_path}", file=sys.stderr)
            traceback.print_exc()
        return _task_record(
            multimodal_json_path=multimodal_json_path,
            json_output_path=json_output_path,
            status="failed",
            processing_seconds=elapsed,
            error=f"{type(e).__name__}: {e}",
            usage_meta=usage_meta if isinstance(usage_meta, dict) else None,
        )


def _aggregate_usage_for_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def _write_task_log(
    path: Path,
    *,
    started_at: str,
    finished_at: str,
    wall_seconds: float,
    openai_base_url: str,
    model_id: str,
    workers: int,
    dry_run: bool,
    temperature: float,
    max_tokens: int,
    items: List[Dict[str, Any]],
) -> None:
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
    usage_agg = _aggregate_usage_for_summary(items)
    if usage_agg:
        summary["usage_tokens"] = usage_agg
    wall_r = round(wall_seconds, 4)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"任务日志已写入: {path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "多模态文献实体抽取（本地 vLLM OpenAI 兼容）：输入为 build_multimodal_content 生成的 "
            "*_multimodal_content.json，支持目录批量与跳过已抽取"
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=MULTIMODAL_INPUT_DIR,
        help="multimodal_content 目录或单个 *_multimodal_content.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_MULTI_DIR,
        help="结果 JSON 目录；每个 foo_multimodal_content.json 对应 foo_entities.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="仅单文件模式：指定完整输出路径时覆盖 --output-dir 的自动命名",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=PROJECT_ROOT / "prompts/paper_entity_schema.jsonc",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=PROJECT_ROOT / "prompts/paper_entity_extraction_prompt.md",
        help="含「系统提示词（多模态）」代码块的 Markdown（默认本仓库 prompts 文件）",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="vLLM 模型 id（须与 GET /v1/models 一致）；不设则用 OPENAI_MODEL；仍为空则自动取列表第一个",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="采样温度",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get(ENV_OPENAI_MAX_TOKENS, str(DEFAULT_MAX_TOKENS))),
        help="completion 最大生成 token（也可设 OPENAI_MAX_TOKENS）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查路径与消息规模，不调用 API；校验图片路径存在，不读盘编码 base64",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="默认跳过已存在的 *_entities.json；指定本项则强制重抽",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="批量时遇错立即退出；默认继续处理其余文件",
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=_default_workers(),
        help=(
            "并行线程数（仅对实际调用 API 的任务生效；默认 3，过大可能打满 vLLM）。"
            "也可用环境变量 STEELDIG_EXTRACT_WORKERS"
        ),
    )
    parser.add_argument(
        "--task-log",
        type=Path,
        default=None,
        help=(
            "任务日志 JSON：每篇 items[].usage（prompt/completion/total token 等）、"
            "items[].output_text_stats；summary.usage_tokens 为合计；另有 batch_wall_seconds 等"
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="不显示 tqdm 进度条（便于重定向日志）",
    )
    args = parser.parse_args()

    multimodal_arg = args.input.resolve()
    output_dir = args.output_dir.resolve()
    schema_path = args.schema.resolve()
    prompt_path = args.prompt.resolve()

    files = collect_multimodal_files(multimodal_arg)
    system_content = build_system_content(
        schema_path, prompt_path, prompt_variant="multi"
    )

    client = _openai_client()
    model_id = _resolve_chat_model_id(client, args.model)
    print(f"使用模型: {model_id}", file=sys.stderr)

    single_explicit_output: Optional[Path] = None
    if len(files) == 1 and args.output is not None:
        single_explicit_output = args.output.resolve()

    skip_existing = not args.no_skip and not args.dry_run

    workers = max(1, args.workers)
    if args.fail_fast and workers > 1:
        print(
            "提示: --fail-fast 与多线程并行互斥，已强制 --workers 1。",
            file=sys.stderr,
        )
        workers = 1
    if args.dry_run and workers > 1:
        print(
            "提示: --dry-run 使用单线程，已忽略 -j。",
            file=sys.stderr,
        )
        workers = 1

    log_items: List[Dict[str, Any]] = []
    to_process: List[Tuple[Path, Path]] = []

    for content_path in files:
        if single_explicit_output is not None:
            out_path = single_explicit_output
        else:
            out_path = derive_output_path(content_path, output_dir)

        if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
            print(f"跳过（已存在）: {out_path}")
            log_items.append(
                _task_record(
                    multimodal_json_path=content_path,
                    json_output_path=out_path,
                    status="skipped",
                    processing_seconds=None,
                    error=None,
                )
            )
            continue

        to_process.append((content_path, out_path))

    started_at = datetime.now().isoformat(timespec="seconds")
    t_wall0 = time.perf_counter()
    run_results: List[Dict[str, Any]] = []
    quiet = workers > 1

    if to_process:
        if workers == 1:
            bar = tqdm(
                to_process,
                desc="多模态提取",
                unit="篇",
                disable=args.no_progress,
            )
            for content_path, out_path in bar:
                rec = _run_one_extraction_job(
                    multimodal_json_path=content_path,
                    json_output_path=out_path,
                    system_content=system_content,
                    model_id=model_id,
                    dry_run=args.dry_run,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    quiet=quiet,
                )
                run_results.append(rec)
                ps = rec.get("processing_seconds")
                if ps is not None and hasattr(bar, "set_postfix_str"):
                    bar.set_postfix_str(f"本篇 {float(ps):.1f}s", refresh=False)
                if rec["status"] == "failed" and args.fail_fast:
                    break
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                future_map = {
                    ex.submit(
                        _run_one_extraction_job,
                        multimodal_json_path=pair[0],
                        json_output_path=pair[1],
                        system_content=system_content,
                        model_id=model_id,
                        dry_run=args.dry_run,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        quiet=quiet,
                    ): pair
                    for pair in to_process
                }
                done_iter = as_completed(future_map)
                if not args.no_progress:
                    done_iter = tqdm(
                        done_iter,
                        total=len(future_map),
                        desc="多模态提取",
                        unit="篇",
                    )
                for fut in done_iter:
                    rec = fut.result()
                    run_results.append(rec)
                    ps = rec.get("processing_seconds")
                    if ps is not None and hasattr(done_iter, "set_postfix_str"):
                        done_iter.set_postfix_str(f"刚完成 {float(ps):.1f}s", refresh=False)
                    if rec["status"] == "failed" and args.fail_fast:
                        ex.shutdown(wait=False, cancel_futures=True)
                        break

    wall_seconds = time.perf_counter() - t_wall0
    finished_at = datetime.now().isoformat(timespec="seconds")
    all_items = log_items + run_results
    all_items.sort(key=lambda x: x["multimodal_input"])

    summary = {"ok": 0, "skipped": 0, "failed": 0}
    for it in all_items:
        st = it.get("status")
        if st in summary:
            summary[st] += 1

    per_paper_sum = sum(
        float(it["processing_seconds"])
        for it in all_items
        if it.get("processing_seconds") is not None
    )
    uagg = _aggregate_usage_for_summary(all_items)
    usage_parts: List[str] = []
    if uagg:
        if "prompt_tokens_sum" in uagg:
            usage_parts.append(f"prompt_tokens 合计={uagg['prompt_tokens_sum']}")
        if "completion_tokens_sum" in uagg:
            usage_parts.append(f"completion_tokens 合计={uagg['completion_tokens_sum']}")
        if "total_tokens_sum" in uagg:
            usage_parts.append(f"total_tokens 合计={uagg['total_tokens_sum']}")
        if "completion_tokens_reasoning_sum" in uagg:
            usage_parts.append(
                f"completion 中 reasoning_tokens 合计={uagg['completion_tokens_reasoning_sum']}"
            )
        if "completion_tokens_response_sum" in uagg:
            usage_parts.append(
                f"completion 中应答 tokens 合计={uagg['completion_tokens_response_sum']}"
            )
    usage_line = f" | {' | '.join(usage_parts)}" if usage_parts else ""

    print(
        f"整批墙钟: {wall_seconds:.2f}s | 各篇处理时间合计: {per_paper_sum:.2f}s "
        f"（单篇见任务日志 items[].processing_seconds）| "
        f"成功 {summary['ok']} | 跳过 {summary['skipped']} | 失败 {summary['failed']}"
        f"{usage_line}",
        file=sys.stderr,
    )

    if args.task_log is not None:
        _write_task_log(
            args.task_log,
            started_at=started_at,
            finished_at=finished_at,
            wall_seconds=wall_seconds,
            openai_base_url=_effective_openai_base_url(),
            model_id=model_id,
            workers=workers,
            dry_run=args.dry_run,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            items=all_items,
        )

    if summary["failed"] > 0:
        print(
            f"\n共 {summary['failed']} 个文件处理失败（共 {len(files)} 个输入）。",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
