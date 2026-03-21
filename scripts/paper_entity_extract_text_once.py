#!/usr/bin/env python3
"""
纯文本文献实体抽取：读取 **build_multimodal_content** 已生成的
`*_text_llm_input.json`（含 `text` 字段，无图像），组装 user 文本后请求 Kimi，
按 paper_entity_schema.jsonc 输出 JSON。

请先运行：
  python scripts/clean_content_list.py
  python scripts/build_multimodal_content.py

依赖：openai、json5、python-dotenv（可选，用于 .env）

用法（在项目根目录）：
  export MOONSHOT_API_KEY=...   # 或 KIMI_API_KEY

  # 批量：扫描 text_llm_input 下所有 *_text_llm_input.json
  python scripts/paper_entity_extract_text_once.py \\
    --input datasets/text_llm_input \\
    --output-dir datasets/output_text

  # 单文件
  python scripts/paper_entity_extract_text_once.py \\
    -i datasets/text_llm_input/A_text_llm_input.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

TEXT_LLM_INPUT_DIR = PROJECT_ROOT / "datasets" / "text_llm_input"
# 与多模态抽取的 datasets/output 分离，避免同名 *_entities.json 互相覆盖
TEXT_ENTITIES_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "output_text"

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

from openai import OpenAI  # noqa: E402

from model_reply_json import parse_model_json  # noqa: E402


def load_schema_object(schema_path: Path) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json5.load(f)


def load_system_prompt_from_markdown(md_path: Path) -> str:
    """从 paper_entity_extraction_prompt.md 中提取「系统提示词」代码块正文。"""
    raw = md_path.read_text(encoding="utf-8")
    m = re.search(
        r"## 系统提示词.*?\n```(?:text)?\s*\n(.*?)```",
        raw,
        flags=re.DOTALL,
    )
    if not m:
        raise ValueError(f"无法在 {md_path} 中解析系统提示词代码块")
    return m.group(1).strip()


def build_system_content(schema_path: Path, prompt_md_path: Path) -> str:
    """组装完整 system 消息（规则 + Schema JSON）。"""
    schema_obj = load_schema_object(schema_path)
    schema_compact = json.dumps(schema_obj, ensure_ascii=False)
    system_rules = load_system_prompt_from_markdown(prompt_md_path)
    return (
        system_rules
        + "\n\n【JSON Schema（已由程序去除注释，键与嵌套结构必须与输出一致；"
        "可增键但不可擅自改名已有键）】\n"
        + schema_compact
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


def derive_output_path(text_llm_input_path: Path, output_dir: Path) -> Path:
    """foo_text_llm_input.json -> output_dir/foo_entities_text_only.json"""
    name = text_llm_input_path.name
    if name.endswith("_text_llm_input.json"):
        out_name = name.replace("_text_llm_input.json", "_entities_text_only.json")
    else:
        out_name = f"{text_llm_input_path.stem}_entities_text_only.json"
    return output_dir.resolve() / out_name


def run_extraction(
    *,
    text_llm_input_path: Path,
    system_content: str,
    model: str,
    output_path: Optional[Path],
    dry_run: bool,
    temperature: float = 1.0,
) -> Dict[str, Any]:
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

    raw_key = os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY")
    api_key = (raw_key or "").strip()
    if not api_key:
        raise SystemExit(
            "请设置环境变量 MOONSHOT_API_KEY 或 KIMI_API_KEY（Moonshot / Kimi OpenAPI 密钥）。"
        )

    if dry_run:
        print(
            json.dumps(
                {
                    "text_llm_input_file": str(text_llm_input_path),
                    "system_len": len(system_content),
                    "user_chars": len(user_message),
                    "model": model,
                    "temperature": temperature,
                    "output": str(output_path) if output_path else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return {}

    client = OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
    )
    raw_reply = completion.choices[0].message.content
    if not raw_reply:
        raise RuntimeError("模型返回空内容")

    result = parse_model_json(raw_reply)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"已写入: {output_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "纯文本文献实体抽取（Kimi）：输入为 build_multimodal_content 生成的 "
            "*_text_llm_input.json，支持目录批量与跳过已抽取"
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
        default=TEXT_ENTITIES_OUTPUT_DIR,
        help="纯文本抽取结果目录（默认 datasets/output_text，与多模态的 datasets/output 分开）",
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
    )
    parser.add_argument(
        "--model",
        default="kimi-k2.5",
        help="模型名（纯文本可选用与多模态相同或文档推荐文本模型）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="采样温度",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查路径与消息规模，不调用 API",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="默认跳过已存在的 *_entities_text_only.json；指定本项则强制重抽",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="批量时遇错立即退出；默认继续处理其余文件",
    )
    args = parser.parse_args()

    input_arg = args.input.resolve()
    output_dir = args.output_dir.resolve()
    schema_path = args.schema.resolve()
    prompt_path = args.prompt.resolve()

    files = collect_text_llm_input_files(input_arg)
    system_content = build_system_content(schema_path, prompt_path)

    single_explicit_output: Optional[Path] = None
    if len(files) == 1 and args.output is not None:
        single_explicit_output = args.output.resolve()

    skip_existing = not args.no_skip and not args.dry_run
    failed: List[str] = []

    for text_input_path in files:
        if single_explicit_output is not None:
            out_path = single_explicit_output
        else:
            out_path = derive_output_path(text_input_path, output_dir)

        if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
            print(f"跳过（已存在）: {out_path}")
            continue

        try:
            run_extraction(
                text_llm_input_path=text_input_path,
                system_content=system_content,
                model=args.model,
                output_path=out_path,
                dry_run=args.dry_run,
                temperature=args.temperature,
            )
        except Exception as e:
            failed.append(f"{text_input_path}: {e}")
            print(f"失败: {text_input_path}", file=sys.stderr)
            traceback.print_exc()
            if args.fail_fast:
                raise SystemExit(1) from e

    if failed:
        print(
            f"\n共 {len(failed)} 个文件处理失败（共 {len(files)} 个）。",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
