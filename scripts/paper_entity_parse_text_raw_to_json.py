#!/usr/bin/env python3
"""
将 paper_entity_extract_text_once.py 写出的「原始模型输出」解析为 JSON 对象，
并保存为 *_entities_text_only.json（与多模态 datasets/output 中的命名语义对齐）。

不调用任何大模型，仅依赖 model_reply_json.parse_model_json（会按 schema 根级键命中数
优先选取 `{...}`，避免思考文字里抢先出现的 `` `{}` `` 等被误当作最终结果）。

用法（在项目根目录）：
  # 批量：目录下所有 *_entities_text_only.raw.txt（默认写入过程日志与失败清单）
  python scripts/paper_entity_parse_text_raw_to_json.py \\
    --input datasets/output_text \\
    --output-dir datasets/output_text

  # 单文件
  python scripts/paper_entity_parse_text_raw_to_json.py \\
    -i datasets/output_text/foo_entities_text_only.raw.txt

  # 显式指定 JSON 输出路径（仅当 --input 为单个文件时可用）
  python scripts/paper_entity_parse_text_raw_to_json.py \\
    -i datasets/output_text/foo_entities_text_only.raw.txt \\
    --output datasets/output_text/foo_entities_text_only.json

  # 指定日志与失败记录路径（便于固定路径做后续脚本处理）
  python scripts/paper_entity_parse_text_raw_to_json.py \\
    -i datasets/output_text \\
    --log-file /tmp/parse_run.log \\
    --failures-jsonl /tmp/parse_failures.jsonl

  # 也可直接编辑本脚本「用户配置区」中的路径，无参运行即使用该默认目录
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# -----------------------------------------------------------------------------
# 用户配置区（直接改这里；与 paper_entity_extract_text_once.py 的 output 目录对齐）
# -----------------------------------------------------------------------------
# 输入：含 *_entities_text_only.raw.txt 的目录，或单个 .raw.txt 文件的绝对/相对路径
#
#   Path("/home/caep-xuben/chenchengbing/yzj/datasets/llm_baseline_366/output_text")
USER_CONFIG_PARSE_INPUT: Path = Path(
    "/home/caep-xuben/chenchengbing/yzj/datasets/llm_baseline_366/output_text"
)
#
# JSON 输出目录（批量时与 raw 文件名组合生成 .json）
USER_CONFIG_PARSE_OUTPUT_DIR: Path = Path(
    "/home/caep-xuben/chenchengbing/yzj/datasets/llm_baseline_366/final_json"
)
#
# 过程日志 / 失败清单：设为 None 表示每次运行自动生成带时间戳的文件（在 output_dir/parse_text_raw_logs/）
# 设为具体路径则作为默认输出位置（命令行传 --log-file / --failures-jsonl 可覆盖）
USER_CONFIG_PARSE_LOG_FILE: Optional[Path] = None
USER_CONFIG_PARSE_FAILURES_JSONL: Optional[Path] = None
# -----------------------------------------------------------------------------

from model_reply_json import parse_model_json  # noqa: E402

_LOG = logging.getLogger(__name__)


def _default_run_paths(output_dir: Path) -> tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir.resolve() / "parse_text_raw_logs"
    return base / f"run_{ts}.log", base / f"failures_{ts}.jsonl"


def setup_logging(
    log_file: Optional[Path],
    *,
    verbose: bool,
) -> None:
    """控制台 + 可选文件；文件侧记录 DEBUG 级完整过程。"""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)


def collect_raw_files(path: Path) -> List[Path]:
    """
    path 为文件：须以 .raw.txt 结尾（推荐 *_entities_text_only.raw.txt）。
    path 为目录：其下所有 *_entities_text_only.raw.txt（仅一层，按名排序）。
    """
    path = path.resolve()
    if path.is_file():
        if not path.name.endswith(".raw.txt"):
            raise SystemExit(
                f"单文件须为 .raw.txt（模型原始输出），当前: {path.name}"
            )
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*_entities_text_only.raw.txt"))
        if not files:
            raise SystemExit(
                f"目录中未找到 *_entities_text_only.raw.txt: {path}\n"
                "请先运行: python scripts/paper_entity_extract_text_once.py"
            )
        return list(files)
    raise SystemExit(f"路径不存在: {path}")


def derive_json_output_path(raw_path: Path, output_dir: Path) -> Path:
    """foo_entities_text_only.raw.txt -> output_dir/foo_entities_text_only.json"""
    name = raw_path.name
    if name.endswith("_entities_text_only.raw.txt"):
        json_name = name.replace(
            "_entities_text_only.raw.txt", "_entities_text_only.json"
        )
    elif name.endswith(".raw.txt"):
        json_name = name[: -len(".raw.txt")] + ".json"
    else:
        json_name = f"{raw_path.stem}.json"
    return output_dir.resolve() / json_name


def _append_failure_record(
    failures_fp: Optional[TextIO],
    record: Dict[str, Any],
) -> None:
    if failures_fp is None:
        return
    failures_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    failures_fp.flush()


def parse_one(raw_path: Path, json_path: Path) -> None:
    _LOG.debug("读取原始文件: %s", raw_path)
    try:
        text = raw_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(
            f"无法用 UTF-8 解码（请检查文件编码）: {raw_path}"
        ) from e
    except OSError as e:
        raise OSError(f"读取失败: {raw_path}: {e}") from e

    _LOG.debug("原始文本长度: %d 字符", len(text))
    result = parse_model_json(text)
    if not isinstance(result, dict):
        raise TypeError(
            f"parse_model_json 返回非 dict（{type(result).__name__}），拒绝写入"
        )

    json_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except OSError as e:
        raise OSError(f"写入 JSON 失败: {json_path}: {e}") from e

    _LOG.info("已写入: %s", json_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 *_entities_text_only.raw.txt 解析 JSON，写入 *_entities_text_only.json"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=USER_CONFIG_PARSE_INPUT,
        help="含 .raw.txt 的目录，或单个 .raw.txt 文件（默认：脚本内 USER_CONFIG_PARSE_INPUT）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=USER_CONFIG_PARSE_OUTPUT_DIR,
        help="JSON 输出目录（批量模式下与默认命名组合；默认：USER_CONFIG_PARSE_OUTPUT_DIR）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="仅单文件模式：指定完整 JSON 输出路径时覆盖 --output-dir 的自动命名",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="默认跳过已存在且非空的对应 .json；指定本项则强制重写",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="批量时遇错立即退出；默认继续处理其余文件",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=USER_CONFIG_PARSE_LOG_FILE,
        help="过程日志；不设且 USER_CONFIG_PARSE_LOG_FILE 为 None 时写入 output_dir 下带时间戳的 .log",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="不写日志文件，仅控制台输出",
    )
    parser.add_argument(
        "--failures-jsonl",
        type=Path,
        default=USER_CONFIG_PARSE_FAILURES_JSONL,
        help="失败清单 JSONL；不设且 USER_CONFIG_PARSE_FAILURES_JSONL 为 None 时与本次 run 日志同目录",
    )
    parser.add_argument(
        "--no-failures-jsonl",
        action="store_true",
        help="不写失败清单文件",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="控制台也输出 DEBUG（默认仅 INFO）",
    )
    args = parser.parse_args()

    input_arg = args.input.resolve()
    output_dir = args.output_dir.resolve()

    raws = collect_raw_files(input_arg)
    single_explicit: Optional[Path] = None
    if len(raws) == 1 and args.output is not None:
        single_explicit = args.output.resolve()

    default_log, default_failures = _default_run_paths(output_dir)
    log_file: Optional[Path] = None
    if not args.no_log_file:
        log_file = default_log if args.log_file is None else args.log_file.resolve()

    failures_path: Optional[Path] = None
    if not args.no_failures_jsonl:
        if args.failures_jsonl is not None:
            failures_path = args.failures_jsonl.resolve()
        elif log_file is not None:
            # 与本次 run 日志同目录，时间戳与 run_<ts>.log 的 <ts> 对齐
            stem = log_file.stem
            ts_part = stem[4:] if stem.startswith("run_") else stem
            failures_path = log_file.parent / f"failures_{ts_part}.jsonl"
        else:
            failures_path = default_failures

    setup_logging(log_file, verbose=args.verbose)
    _LOG.info(
        "开始批量解析: 输入条目=%d, output_dir=%s, skip_existing=%s",
        len(raws),
        output_dir,
        not args.no_skip,
    )
    if log_file is not None:
        _LOG.info("过程日志文件: %s", log_file)
    if failures_path is not None:
        _LOG.info("失败清单（JSONL）: %s", failures_path)

    skip_existing = not args.no_skip
    failed: List[str] = []
    ok = 0
    skipped = 0

    failures_fp: Optional[TextIO] = None
    if failures_path is not None:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        failures_fp = open(failures_path, "w", encoding="utf-8")

    try:
        for idx, raw_path in enumerate(raws, start=1):
            if single_explicit is not None:
                json_path = single_explicit
            else:
                json_path = derive_json_output_path(raw_path, output_dir)

            _LOG.info(
                "[%d/%d] 处理: %s -> %s",
                idx,
                len(raws),
                raw_path,
                json_path,
            )

            if skip_existing and json_path.is_file() and json_path.stat().st_size > 0:
                _LOG.info("跳过（已存在非空 JSON）: %s", json_path)
                skipped += 1
                continue

            try:
                parse_one(raw_path, json_path)
                ok += 1
            except Exception as e:
                tb = traceback.format_exc()
                failed.append(f"{raw_path}: {e}")
                _LOG.error("解析失败: %s", raw_path, exc_info=True)
                _append_failure_record(
                    failures_fp,
                    {
                        "raw_path": str(raw_path.resolve()),
                        "json_path": str(json_path.resolve()),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": tb,
                    },
                )
                if args.fail_fast:
                    raise SystemExit(1) from e
    finally:
        if failures_fp is not None:
            failures_fp.close()

    _LOG.info(
        "结束: 成功=%d, 跳过=%d, 失败=%d, 总计=%d",
        ok,
        skipped,
        len(failed),
        len(raws),
    )
    if failed:
        _LOG.error("失败文件数: %d", len(failed))
        for line in failed:
            _LOG.error("  %s", line)
        print(
            f"\n共 {len(failed)} 个文件解析失败（共 {len(raws)} 个）。"
            f"详见日志与失败清单。",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
