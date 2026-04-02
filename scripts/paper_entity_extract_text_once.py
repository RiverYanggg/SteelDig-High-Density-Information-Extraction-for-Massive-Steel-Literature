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

  # 并行 4 线程；任务日志默认写入 output_dir 下 extract_text_task_log.<时间戳>.json，也可用 --task-log 指定路径
  python scripts/paper_entity_extract_text_once.py \\
    --input datasets/text_llm_input \\
    --output-dir datasets/output_text \\
    -j 4

  # 第二步：解析原始输出为 JSON
  python scripts/paper_entity_parse_text_raw_to_json.py \\
    --input datasets/output_text \\
    --output-dir datasets/output_text

共用逻辑见 scripts/lib/paper_extract_common.py（OpenAI、Schema、用量与任务日志等）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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

try:
    import json5  # noqa: F401 — 与 lib 共用依赖，此处先失败并给出中文提示
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

from lib.paper_extract_common import (  # noqa: E402
    ENV_OPENAI_MAX_TOKENS,
    SCHEMA_GAP_TEXT_ONLY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    aggregate_usage_for_summary,
    build_system_content,
    default_workers as _default_workers,
    effective_openai_base_url,
    extract_output_text_stats,
    extract_usage_from_completion,
    openai_client,
    resolve_chat_model_id,
    stderr_exc_lock,
    try_load_dotenv,
    write_task_log,
)

try_load_dotenv(project_root=PROJECT_ROOT)

DEFAULT_WORKERS = 2


def load_paper_text_from_text_llm_input(path: Path) -> str:
    """读取 build_multimodal_content 写入的纯文本 LLM 中间 JSON，返回 text 字段。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"纯文本 LLM 输入 JSON 顶层须为对象: {path}")
    text = data.get("text")
    if not isinstance(text, str):
        raise ValueError(f'{path} 须包含字符串字段 "text"')
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

    dry_run 时第二项为 None。quiet 为 True 时不打印「已写入…」。
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
                    "openai_base_url": effective_openai_base_url(),
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

    client = openai_client()
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
    usage = extract_usage_from_completion(completion)
    if usage:
        meta["usage"] = usage
    text_stats = extract_output_text_stats(msg)
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
        with stderr_exc_lock:
            print(f"失败: {text_llm_input_path}", file=sys.stderr)
            traceback.print_exc()
        return _task_record(
            text_llm_input_path=text_llm_input_path,
            raw_output_path=raw_output_path,
            status="failed",
            processing_seconds=elapsed,
            error=f"{type(e).__name__}: {e}",
        )


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
        default=_default_workers(fallback=DEFAULT_WORKERS),
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
            "任务日志 JSON 路径；未指定时默认写到 --output-dir 下 "
            "extract_text_task_log.<时间戳>.json。内容含每篇 processing_seconds、usage、output_text_stats 等"
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="不显示 tqdm 进度条（便于重定向日志）",
    )
    args = parser.parse_args()

    input_arg = args.input.resolve()
    output_dir = args.output_dir.resolve()
    schema_path = args.schema.resolve()
    prompt_path = args.prompt.resolve()

    files = collect_text_llm_input_files(input_arg)
    system_content = build_system_content(
        schema_path,
        prompt_path,
        prompt_variant="text",
        schema_intro_before_json=SCHEMA_GAP_TEXT_ONLY,
    )

    client = openai_client()
    model_id = resolve_chat_model_id(client, args.model)
    print(f"使用模型: {model_id}", file=sys.stderr)

    if args.task_log is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        task_log_path = output_dir / f"extract_text_task_log.{ts}.json"
    else:
        task_log_path = args.task_log.resolve()
    print(f"任务日志路径: {task_log_path}", file=sys.stderr)

    single_explicit_output: Optional[Path] = None
    if len(files) == 1 and args.output is not None:
        single_explicit_output = args.output.resolve()

    started_at = datetime.now().isoformat(timespec="seconds")
    t_wall0 = time.perf_counter()

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

    for text_input_path in files:
        if single_explicit_output is not None:
            raw_path = single_explicit_output
        else:
            raw_path = derive_raw_output_path(text_input_path, output_dir)

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
            if task_log_path is not None:
                write_task_log(
                    task_log_path,
                    started_at=started_at,
                    finished_at=None,
                    wall_seconds=None,
                    openai_base_url=effective_openai_base_url(),
                    model_id=model_id,
                    workers=workers,
                    dry_run=args.dry_run,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    items=sorted((log_items), key=lambda x: x["text_llm_input"]),
                    atomic_durable=False,
                    atomic_retries=1,
                )
            continue

        to_process.append((text_input_path, raw_path))
    run_results: List[Dict[str, Any]] = []
    quiet = workers > 1

    if to_process:
        if workers == 1:
            bar = tqdm(
                to_process,
                desc="提取",
                unit="篇",
                disable=args.no_progress,
            )
            for text_input_path, raw_path in bar:
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
                if task_log_path is not None:
                    snap_items = log_items + run_results
                    snap_items.sort(key=lambda x: x["text_llm_input"])
                    write_task_log(
                        task_log_path,
                        started_at=started_at,
                        finished_at=None,
                        wall_seconds=None,
                        openai_base_url=effective_openai_base_url(),
                        model_id=model_id,
                        workers=workers,
                        dry_run=args.dry_run,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        items=snap_items,
                        atomic_durable=False,
                        atomic_retries=1,
                    )
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
                    if task_log_path is not None:
                        snap_items = log_items + run_results
                        snap_items.sort(key=lambda x: x["text_llm_input"])
                        write_task_log(
                            task_log_path,
                            started_at=started_at,
                            finished_at=None,
                            wall_seconds=None,
                            openai_base_url=effective_openai_base_url(),
                            model_id=model_id,
                            workers=workers,
                            dry_run=args.dry_run,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            items=snap_items,
                            atomic_durable=False,
                            atomic_retries=1,
                        )
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
    uagg = aggregate_usage_for_summary(all_items)
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

    if task_log_path is not None:
        write_task_log(
            task_log_path,
            started_at=started_at,
            finished_at=finished_at,
            wall_seconds=wall_seconds,
            openai_base_url=effective_openai_base_url(),
            model_id=model_id,
            workers=workers,
            dry_run=args.dry_run,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            items=all_items,
            atomic_durable=False,
            atomic_retries=1,
        )

    if summary["failed"] > 0:
        print(
            f"\n共 {summary['failed']} 个文件处理失败（共 {len(files)} 个输入）。",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
