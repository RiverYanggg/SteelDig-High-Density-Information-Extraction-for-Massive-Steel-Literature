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

共用逻辑见 scripts/lib/paper_extract_common.py；多模态 payload 转换见 scripts/lib/multimodal_extract_payload.py。
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

MULTIMODAL_INPUT_DIR = PROJECT_ROOT / "datasets" / "multimodal_content"
OUTPUT_MULTI_DIR = PROJECT_ROOT / "datasets" / "output_multi"

try:
    import json5  # noqa: F401
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

from lib.multimodal_extract_payload import (  # noqa: E402
    ensure_multimodal_payload_for_api,
    verify_multimodal_image_paths_exist,
)
from lib.paper_extract_common import (  # noqa: E402
    ENV_OPENAI_MAX_TOKENS,
    SCHEMA_PREAMBLE_MULTIMODAL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    aggregate_usage_for_summary,
    build_system_content,
    default_workers as _default_workers,
    effective_openai_base_url,
    openai_client,
    resolve_chat_model_id,
    stderr_exc_lock,
    try_load_dotenv,
    usage_meta_from_completion,
    write_task_log,
)
from model_reply_json import parse_model_json  # noqa: E402

try_load_dotenv(project_root=PROJECT_ROOT)

DEFAULT_WORKERS = 3


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
    """返回 (解析后的 JSON 对象, token/输出分块元数据)。dry_run 时第二项为 None。"""
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
                    "openai_base_url": effective_openai_base_url(),
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

    client = openai_client()
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

    usage_meta = usage_meta_from_completion(completion, msg)

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
        with stderr_exc_lock:
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
        schema_path,
        prompt_path,
        prompt_variant="multi",
        schema_intro_before_json=SCHEMA_PREAMBLE_MULTIMODAL,
    )

    client = openai_client()
    model_id = resolve_chat_model_id(client, args.model)
    print(f"使用模型: {model_id}", file=sys.stderr)

    started_at = datetime.now().isoformat(timespec="seconds")
    t_wall0 = time.perf_counter()

    if args.task_log is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        task_log_path = output_dir / f"extract_multi_task_log.{ts}.json"
    else:
        task_log_path = args.task_log.resolve()
    print(f"任务日志路径: {task_log_path}", file=sys.stderr)

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
            if task_log_path is not None:
                snap = sorted(log_items, key=lambda x: x["multimodal_input"])
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
                    items=snap,
                    atomic_durable=True,
                    atomic_retries=3,
                )
            continue

        to_process.append((content_path, out_path))

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
                if task_log_path is not None:
                    snap_items = log_items + run_results
                    snap_items.sort(key=lambda x: x["multimodal_input"])
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
                        atomic_durable=True,
                        atomic_retries=3,
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
                    if task_log_path is not None:
                        snap_items = log_items + run_results
                        snap_items.sort(key=lambda x: x["multimodal_input"])
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
                            atomic_durable=True,
                            atomic_retries=3,
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
            atomic_durable=True,
            atomic_retries=3,
        )

    if summary["failed"] > 0:
        print(
            f"\n共 {summary['failed']} 个文件处理失败（共 {len(files)} 个输入）。",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
