#!/usr/bin/env python3
"""
纯文本文献实体抽取（仅调用模型）：读取 **build_multimodal_content** 已生成的
`*_text_llm_input.json`（含 `text` 字段），组装 user 文本后请求本地 vLLM（OpenAI
兼容 /v1），将**完整模型原始输出**写入 `*_entities_text_only.raw.txt`。

JSON 解析请使用独立脚本（与模型调用解耦）：
  python scripts/paper_entity_parse_text_raw_to_json.py ...

请先运行：
  python scripts/clean_content_list.py
  python scripts/build_multimodal_content.py

依赖：openai、json5、tqdm、python-dotenv（可选，用于 .env）

用法（在项目根目录）：
  # 先启动 vLLM，见 digmodel/basemodel/start_vllm.sh（默认 PORT=8001 → base_url .../v1）
  export OPENAI_BASE_URL=http://127.0.0.1:8001/v1   # 可选，此为默认值
  export OPENAI_API_KEY=EMPTY                       # 可选；若 serve 时用了 --api-key 则填一致
  export OPENAI_MODEL=<与 GET /v1/models 中 id 一致>  # 可选；不设则自动取列表第一个模型

  # 批量：扫描 text_llm_input 下所有 *_text_llm_input.json → 写入 output_text 下 .raw.txt
  python scripts/paper_entity_extract_text_once.py \\
    --input datasets/text_llm_input \\
    --output-dir datasets/output_text

  # 单文件
  python scripts/paper_entity_extract_text_once.py \\
    -i datasets/text_llm_input/A_text_llm_input.json

  # 并行 4 线程、任务日志（含每篇 prompt/completion/total tokens，及可选 reasoning/应答拆分）
  python scripts/paper_entity_extract_text_once.py \\
    --input datasets/text_llm_input \\
    --output-dir datasets/output_text \\
    -j 4 \\
    --task-log datasets/output_text/extract_task_log.json

  # 第二步：解析原始输出为 JSON
  python scripts/paper_entity_parse_text_raw_to_json.py \\
    --input datasets/output_text \\
    --output-dir datasets/output_text
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

TEXT_LLM_INPUT_DIR = PROJECT_ROOT / "datasets" / "text_llm_input"
TEXT_RAW_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "output_text"

# =========================
# 可配置项（常修改参数集中）
# =========================
# 环境变量名
ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_MODEL = "OPENAI_MODEL"
ENV_OPENAI_MAX_TOKENS = "OPENAI_MAX_TOKENS"
ENV_WORKERS = "STEELDIG_EXTRACT_WORKERS"

# 默认值
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
    """从 completion.usage 提取 token；若存在 reasoning 细分（如 OpenAI o 系列、部分网关）则一并记录。"""
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
    """若 API 将 thinking 与最终回复分字段（如 reasoning_content / thinking），则分别统计字符数。"""
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


def load_schema_object(schema_path: Path) -> dict:
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
        system_rules + "\n\n" + schema_compact
    )


def load_paper_text_from_text_llm_input(path: Path) -> str:
    """读取 build_multimodal_content 写入的纯文本 LLM 中间 JSON，返回 text 字段。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"纯文本 LLM 输入 JSON 顶层须为对象: {path}")
    text = data.get("text")
    if not isinstance(text, str):
        raise ValueError(f"{path} 须包含字符串字段 \"text\"")
    return text


def collect_text_llm_input_files(path: Path) -> List[Path]:
    """
    path 为文件：须为 *_text_llm_input.json。
    path 为目录：其下所有 *_text_llm_input.json（仅一层，按名排序）。
    """
    path = path.resolve()
    if path.is_file():
        if not path.name.endswith("_text_llm_input.json"):
            raise SystemExit(
                f"单文件须为 *_text_llm_input.json，当前: {path.name}"
            )
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*_text_llm_input.json"))
        if not files:
            raise SystemExit(
                f"目录中未找到 *_text_llm_input.json: {path}\n"
                "请先运行: python scripts/build_multimodal_content.py"
            )
        return list(files)
    raise SystemExit(f"路径不存在: {path}")


def derive_raw_output_path(text_llm_input_path: Path, output_dir: Path) -> Path:
    """foo_text_llm_input.json -> output_dir/foo_entities_text_only.raw.txt"""
    name = text_llm_input_path.name
    if name.endswith("_text_llm_input.json"):
        stem = name[: -len("_text_llm_input.json")]
        out_name = f"{stem}_entities_text_only.raw.txt"
    else:
        out_name = f"{text_llm_input_path.stem}_entities_text_only.raw.txt"
    return output_dir.resolve() / out_name


def run_extraction(
    *,
    text_llm_input_path: Path,
    system_content: str,
    model: str,
    raw_output_path: Optional[Path],
    dry_run: bool,
    temperature: float = 0.3,
    max_tokens: int = 65536,
    quiet: bool = False,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """调用模型，返回 (原始文本, token/输出分块元数据)。

    元数据在成功调用 API 时通常含 ``usage``（prompt/completion/total token，及可选的
    reasoning/response 拆分）与 ``output_text_stats``（reasoning_content 与 content 的字符数）。
    dry_run 时第二项为 None。

    quiet: 为 True 时不打印「已写入…」（多线程时避免输出交错）。
    """
    paper_text = load_paper_text_from_text_llm_input(text_llm_input_path)

    user_tail = (
        "【任务】上文按阅读顺序给出了论文全文（含图注/表注/公式文本；未提供插图像素）。"
        "请仅依据文本与标注信息抽取结构化实体。\n"
        "【输出】仅输出一个合法 JSON 对象，不要 Markdown 围栏、不要解释性文字；"
        "严格遵守系统提示中的类型约束（禁止 number/boolean 叶子类型）。"
    )

    user_message = (
        "【论文纯文本内容 — 按顺序阅读】\n\n"
        + paper_text
        + "\n\n"
        + user_tail
    )

    if dry_run:
        print(
            json.dumps(
                {
                    "text_llm_input_file": str(text_llm_input_path),
                    "system_len": len(system_content),
                    "user_chars": len(user_message),
                    "openai_base_url": _effective_openai_base_url(),
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "raw_output": str(raw_output_path) if raw_output_path else None,
                    "usage_tokens": None,
                    "note": "dry_run 不调用 API，无 token 统计",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return "", None

    client = _openai_client()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
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

    meta: Dict[str, Any] = {}
    usage = _extract_usage_from_completion(completion)
    if usage:
        meta["usage"] = usage
    text_stats = _extract_output_text_stats(msg)
    if text_stats:
        meta["output_text_stats"] = text_stats

    if raw_output_path:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text(raw_reply, encoding="utf-8")
        if not quiet:
            print(f"已写入原始模型输出: {raw_output_path}")

    return raw_reply, meta if meta else None


def _task_record(
    *,
    text_llm_input_path: Path,
    raw_output_path: Path,
    status: str,
    processing_seconds: Optional[float],
    error: Optional[str],
    usage_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """processing_seconds: 单篇从读入输入到写完 raw 的耗时；skipped 为 null。

    usage_meta: run_extraction 返回的第二项，可含 usage（token）与 output_text_stats（分字段字符数）。
    """
    ps: Optional[float]
    if processing_seconds is None:
        ps = None
    else:
        ps = round(float(processing_seconds), 4)
    rec: Dict[str, Any] = {
        "text_llm_input": str(text_llm_input_path),
        "raw_output": str(raw_output_path),
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
    text_llm_input_path: Path,
    raw_output_path: Path,
    system_content: str,
    model_id: str,
    dry_run: bool,
    temperature: float,
    max_tokens: int,
    quiet: bool,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        _raw, usage_meta = run_extraction(
            text_llm_input_path=text_llm_input_path,
            system_content=system_content,
            model=model_id,
            raw_output_path=raw_output_path,
            dry_run=dry_run,
            temperature=temperature,
            max_tokens=max_tokens,
            quiet=quiet,
        )
        elapsed = time.perf_counter() - t0
        return _task_record(
            text_llm_input_path=text_llm_input_path,
            raw_output_path=raw_output_path,
            status="ok",
            processing_seconds=elapsed,
            error=None,
            usage_meta=usage_meta,
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        with _exc_lock:
            print(f"失败: {text_llm_input_path}", file=sys.stderr)
            traceback.print_exc()
        return _task_record(
            text_llm_input_path=text_llm_input_path,
            raw_output_path=raw_output_path,
            status="failed",
            processing_seconds=elapsed,
            error=f"{type(e).__name__}: {e}",
        )


def _aggregate_usage_for_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对 items[].usage 做求和，便于整批 token 汇总。"""
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
            "纯文本文献实体抽取（仅写原始模型输出 .raw.txt）：输入为 build_multimodal_content 生成的 "
            "*_text_llm_input.json；解析 JSON 请用 paper_entity_parse_text_raw_to_json.py"
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=TEXT_LLM_INPUT_DIR,
        help="text_llm_input 目录或单个 *_text_llm_input.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TEXT_RAW_OUTPUT_DIR,
        help="原始模型输出目录（默认 datasets/output_text，写入 *_entities_text_only.raw.txt）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="仅单文件模式：指定完整原始输出路径时覆盖 --output-dir 的自动命名",
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
        help="含「系统提示词（纯文本）」代码块的 Markdown（默认本仓库 prompts 文件）",
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
        help="采样温度（结构化抽取建议偏低）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get(ENV_OPENAI_MAX_TOKENS, str(DEFAULT_MAX_TOKENS))),
        help=(
            "单次 completion 最多生成的 token；须严格小于「模型上下文上限 − 本请求 prompt 占用」。"
            "若默认仍 400，请再调小或缩短输入。环境变量 OPENAI_MAX_TOKENS 可覆盖默认值。"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查路径与消息规模，不调用 API",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="默认跳过已存在的、非空的 *_entities_text_only.raw.txt；指定本项则强制重抽",
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
            "任务日志 JSON：每篇 items[].processing_seconds、items[].usage（prompt/completion/total token，"
            "及可选的 reasoning/response 拆分）、items[].output_text_stats；"
            "summary.usage_tokens 为各篇 token 合计；另有 batch_wall_seconds 等"
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="不显示 tqdm 进度条（便于重定向日志）",
    )
    args = parser.parse_args() # 解析命令行参数

    # 路径解析
    input_arg = args.input.resolve()
    output_dir = args.output_dir.resolve()
    schema_path = args.schema.resolve()
    prompt_path = args.prompt.resolve()

    files = collect_text_llm_input_files(input_arg) # 收集文本 LLM 输入文件
    # 构建系统提示词
    system_content = build_system_content(
        schema_path, prompt_path, prompt_variant="text"
    )

    client = _openai_client() # 创建 OpenAI 客户端
    model_id = _resolve_chat_model_id(client, args.model)
    print(f"使用模型: {model_id}", file=sys.stderr)

    # 单文件模式：指定完整原始输出路径时覆盖 --output-dir 的自动命名
    single_explicit_output: Optional[Path] = None
    if len(files) == 1 and args.output is not None:
        single_explicit_output = args.output.resolve()

    # 跳过已存在的、非空的 *_entities_text_only.raw.txt
    skip_existing = not args.no_skip and not args.dry_run

    # 并行线程数（仅对实际调用 API 的任务生效；默认 1，过大可能打满 vLLM）。
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

    log_items: List[Dict[str, Any]] = [] # 任务日志项列表
    to_process: List[Tuple[Path, Path]] = [] # 待处理任务列表

    for text_input_path in files: # 遍历文本 LLM 输入文件
        if single_explicit_output is not None:
            raw_path = single_explicit_output
        else:
            raw_path = derive_raw_output_path(text_input_path, output_dir) # 生成原始输出路径

        if (
            skip_existing
            and raw_path.is_file()
            and raw_path.stat().st_size > 0
        ):
            print(f"跳过（已存在非空原始输出）: {raw_path}")
            log_items.append(
                _task_record(
                    text_llm_input_path=text_input_path,
                    raw_output_path=raw_path,
                    status="skipped",
                    processing_seconds=None,
                    error=None,
                )
            )
            continue

        to_process.append((text_input_path, raw_path))

    started_at = datetime.now().isoformat(timespec="seconds") # 开始时间
    t_wall0 = time.perf_counter() # 开始时间
    run_results: List[Dict[str, Any]] = [] # 运行结果列表
    quiet = workers > 1 # 是否静默模式

    if to_process:
        if workers == 1: # 单线程模式
            bar = tqdm(
                to_process,
                desc="提取",
                unit="篇",
                disable=args.no_progress,
            )
            for text_input_path, raw_path in bar: # 遍历待处理任务
                rec = _run_one_extraction_job(
                    text_llm_input_path=text_input_path,
                    raw_output_path=raw_path,
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
                        text_llm_input_path=pair[0],
                        raw_output_path=pair[1],
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
                        desc="提取",
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
    all_items.sort(key=lambda x: x["text_llm_input"])

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
