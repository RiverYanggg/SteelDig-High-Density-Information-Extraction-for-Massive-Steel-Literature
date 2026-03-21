#!/usr/bin/env python3
"""
一次性多模态文献实体抽取：读取 **build_multimodal_content** 已生成的
`*_multimodal_content.json`（OpenAI 兼容的 text + image_url 列表），
将其中本地文件路径在调用前转为 base64，再请求 Kimi，按 paper_entity_schema.jsonc 输出 JSON。

请先运行：
python scripts/clean_content_list.py
python scripts/build_multimodal_content.py

依赖：openai、json5、python-dotenv（可选，用于 .env）

用法（在项目根目录）：
  export MOONSHOT_API_KEY=...   # 或 KIMI_API_KEY

  # 批量：扫描 multimodal_content 下所有 *_multimodal_content.json
  python scripts/paper_entity_extract_once.py \\
    --input datasets/multimodal_content \\
    --output-dir datasets/output_multi

  # 单文件
  python scripts/paper_entity_extract_once.py -i datasets/multimodal_content/0321_noted_multimodal_content.json
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

from build_multimodal_content import encode_image  # noqa: E402
from model_reply_json import parse_model_json  # noqa: E402

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
    - image_url.url 为本地文件路径的，读入并转为 data URL（Kimi 需要 base64）
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
    temperature: float = 1.0,
) -> Dict[str, Any]:
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
                    "multimodal_file": str(multimodal_json_path),
                    "system_len": len(system_content),
                    "user_parts": len(user_content),
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
            {"role": "user", "content": user_content},
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
            "多模态文献实体抽取（Kimi）：输入为 build_multimodal_content 生成的 "
            "*_multimodal_content.json，支持目录批量与跳过已抽取"
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=PROJECT_ROOT / "datasets/multimodal_content",
        help="multimodal_content 目录或单个 *_multimodal_content.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets/output_multi",
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
    )
    parser.add_argument("--model", default="kimi-k2.5", help="多模态模型名")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="采样温度；kimi-k2.5 多模态目前仅允许 1.0（传其它值会 400）",
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
    args = parser.parse_args()

    multimodal_arg = args.input.resolve()
    output_dir = args.output_dir.resolve()
    schema_path = args.schema.resolve()
    prompt_path = args.prompt.resolve()

    files = collect_multimodal_files(multimodal_arg)
    system_content = build_system_content(schema_path, prompt_path)

    single_explicit_output: Optional[Path] = None
    if len(files) == 1 and args.output is not None:
        single_explicit_output = args.output.resolve()

    skip_existing = not args.no_skip and not args.dry_run
    failed: List[str] = []

    for content_path in files:
        if single_explicit_output is not None:
            out_path = single_explicit_output
        else:
            out_path = derive_output_path(content_path, output_dir)

        if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
            print(f"跳过（已存在）: {out_path}")
            continue

        try:
            run_extraction(
                multimodal_json_path=content_path,
                system_content=system_content,
                model=args.model,
                output_path=out_path,
                dry_run=args.dry_run,
                temperature=args.temperature,
            )
        except Exception as e:
            failed.append(f"{content_path}: {e}")
            print(f"失败: {content_path}", file=sys.stderr)
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
