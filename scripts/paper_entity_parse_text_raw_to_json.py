#!/usr/bin/env python3
"""
将 paper_entity_extract_text_once.py 写出的「原始模型输出」解析为 JSON 对象，
并保存为 *_entities_text_only.json（与多模态 datasets/output 中的命名语义对齐）。

不调用任何大模型，仅依赖 model_reply_json.parse_model_json（会按 schema 根级键命中数
优先选取 `{...}`，避免思考文字里抢先出现的 `` `{}` `` 等被误当作最终结果）。

用法（在项目根目录）：
  # 批量：目录下所有 *_entities_text_only.raw.txt
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
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "output_text"

from model_reply_json import parse_model_json  # noqa: E402


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


def parse_one(raw_path: Path, json_path: Path) -> None:
    text = raw_path.read_text(encoding="utf-8")
    result = parse_model_json(text)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"已写入: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 *_entities_text_only.raw.txt 解析 JSON，写入 *_entities_text_only.json"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="含 .raw.txt 的目录，或单个 .raw.txt 文件",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="JSON 输出目录（批量模式下与默认命名组合；默认 datasets/output_text）",
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
    args = parser.parse_args()

    input_arg = args.input.resolve()
    output_dir = args.output_dir.resolve()

    raws = collect_raw_files(input_arg)
    single_explicit: Optional[Path] = None
    if len(raws) == 1 and args.output is not None:
        single_explicit = args.output.resolve()

    skip_existing = not args.no_skip
    failed: List[str] = []

    for raw_path in raws:
        if single_explicit is not None:
            json_path = single_explicit
        else:
            json_path = derive_json_output_path(raw_path, output_dir)

        if skip_existing and json_path.is_file() and json_path.stat().st_size > 0:
            print(f"跳过（已存在非空 JSON）: {json_path}")
            continue

        try:
            parse_one(raw_path, json_path)
        except Exception as e:
            failed.append(f"{raw_path}: {e}")
            print(f"失败: {raw_path}", file=sys.stderr)
            traceback.print_exc()
            if args.fail_fast:
                raise SystemExit(1) from e

    if failed:
        print(
            f"\n共 {len(failed)} 个文件解析失败（共 {len(raws)} 个）。",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
