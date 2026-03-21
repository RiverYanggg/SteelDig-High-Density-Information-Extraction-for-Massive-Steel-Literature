"""
多模态 Content 构建模块。

从 datasets/input_cleaned 读取清洗后的 *_content_list.json，
按多模态 API 的 content 格式（文本 + 图片/表格/公式）构建传入内容，
保存到 datasets/multimodal_content；同时生成纯文本 LLM 输入 JSON（仅 text 字段，无图像），
保存到 datasets/text_llm_input，供 paper_entity_extract_text_once 使用。
不调用大模型，仅构建并落盘。
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 路径与常量
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CLEANED_DIR = PROJECT_ROOT / "datasets" / "input_cleaned"
PAPER_PARSERED_DIR = PROJECT_ROOT / "datasets" / "paper_parsered"
MULTIMODAL_CONTENT_DIR = PROJECT_ROOT / "datasets" / "multimodal_content"
TEXT_LLM_INPUT_DIR = PROJECT_ROOT / "datasets" / "text_llm_input"

TEXT_LLM_INPUT_SUFFIX = "_text_llm_input.json"

TYPE_TEXT = "text"
TYPE_IMAGE = "image"
TYPE_TABLE = "table"
TYPE_EQUATION = "equation"


def _get_str(value: Any, default: str = "") -> str:
    """安全地获取字符串值，处理 None 和非字符串类型"""
    if value is None:
        return default
    return value.strip() if isinstance(value, str) else default


def _get_img_path(block: Dict[str, Any]) -> str:
    """从块中获取图片路径"""
    path = block.get("img_path") or block.get("image_path") or ""
    return _get_str(path)


def encode_image(image_path: Path) -> str:
    """将图片转换为 base64 编码
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        base64 编码的字符串
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def find_original_content_list_path(cleaned_filename: str) -> Optional[Path]:
    """
    根据清洗后的文件名，在 paper_parsered 下查找同名的原始 content_list 路径。
    用于解析图片等资源的相对路径。
    """
    for path in PAPER_PARSERED_DIR.rglob("*_content_list.json"):
        if path.name == cleaned_filename:
            return path
    return None


def resolve_image_path(img_path: str, base_dir: Optional[Path]) -> Optional[Path]:
    """
    将块中的 img_path（相对或绝对）解析为绝对路径。
    base_dir 为原始 content_list 所在目录（即 auto 目录）。
    """
    raw = _get_str(img_path)
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    if base_dir is not None:
        resolved = (base_dir / raw).resolve()
        if resolved.exists():
            return resolved
    return None


def _add_image_block(
    api_content: List[Dict[str, Any]],
    resolved_path: Optional[Path],
    base_dir: Optional[Path],
    include_base64: bool,
    text_blocks: List[str],
) -> None:
    """
    添加图片块到 api_content。
    
    Args:
        api_content: 目标内容列表
        resolved_path: 解析后的图片路径
        base_dir: 基准目录，用于计算相对路径
        include_base64: 是否转换为 base64
        text_blocks: 需要添加的文本块列表（如 caption、table_caption 等）
    """
    if include_base64 and resolved_path and resolved_path.exists():
        b64 = encode_image(resolved_path)
        api_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    elif resolved_path:
        api_content.append({
            "type": "image_url",
            "image_url": {"url": str(resolved_path)},
        })
    
    for text in text_blocks:
        if text:
            api_content.append({"type": "text", "text": text})


def build_content_for_api(
    content_list: List[Dict[str, Any]],
    base_dir: Optional[Path] = None,
    *,
    include_base64: bool = False,
) -> List[Dict[str, Any]]:
    """
    将清洗后的 content 列表转换为多模态 API 所需的 content 格式。

    支持的块类型：
    - text: {"type": "text", "text": "..."}
    - image: 图片块（含可选 caption）
    - table: 表格块（含 table_caption / table_footnote）
    - equation: 公式块（含公式 text）

    Args:
        content_list: 清洗后的内容列表
        base_dir: 原始 content_list 所在目录（用于解析相对路径）
        include_base64: 是否将图片转为 base64 编码（默认 False，仅保存路径）

    Returns:
        API 所需的 content 格式列表
        
    Note:
        当 include_base64=False 时，图片类块保存为：
        {"type": "image_url", "image_url": {"url": "<绝对路径>"}}
        便于后续调用 API 时再编码为 base64，避免 JSON 过大。
    """
    api_content: List[Dict[str, Any]] = []

    for item in content_list:
        block_type = item.get("type")

        if block_type == TYPE_TEXT:
            text = _get_str(item.get("text"))
            if text:
                api_content.append({"type": "text", "text": text})
            continue

        if block_type == TYPE_IMAGE:
            img_path = _get_img_path(item)
            resolved = resolve_image_path(img_path, base_dir) if img_path else None
            caption = _get_str(item.get("caption"))
            text_blocks = [f"[caption] {caption}"] if caption else []
            _add_image_block(api_content, resolved, base_dir, include_base64, text_blocks)
            continue

        if block_type == TYPE_TABLE:
            img_path = _get_img_path(item)
            resolved = resolve_image_path(img_path, base_dir) if img_path else None
            cap_raw = item.get("table_caption") or []
            fn_raw = item.get("table_footnote") or []
            cap_text = " ".join(cap_raw).strip() if isinstance(cap_raw, list) else _get_str(cap_raw)
            fn_text = " ".join(fn_raw).strip() if isinstance(fn_raw, list) else _get_str(fn_raw)
            text_blocks = []
            if cap_text:
                text_blocks.append(f"[caption] {cap_text}")
            if fn_text:
                text_blocks.append(f"[footnote] {fn_text}")
            _add_image_block(api_content, resolved, base_dir, include_base64, text_blocks)
            continue

        if block_type == TYPE_EQUATION:
            img_path = _get_img_path(item)
            resolved = resolve_image_path(img_path, base_dir) if img_path else None
            eq_text = _get_str(item.get("text"))
            text_blocks = [f"[equation] {eq_text}"] if eq_text else []
            _add_image_block(api_content, resolved, base_dir, include_base64, text_blocks)
            continue

        text = _get_str(item.get("text"))
        if text:
            api_content.append({"type": "text", "text": text})

    return api_content


def iter_text_segments_from_content_list(
    content_list: List[Dict[str, Any]],
) -> List[str]:
    """
    按阅读顺序从清洗后的 content 列表产出纯文本片段（不含图片）。

    与 build_content_for_api 对齐：正文保留；图/表/公式块仅保留 [caption] / [footnote] /
    [equation] 等标注与脚注文本，不发送图像。
    """
    segments: List[str] = []

    for item in content_list:
        block_type = item.get("type")

        if block_type == TYPE_TEXT:
            text = _get_str(item.get("text"))
            if text:
                segments.append(text)
            continue

        if block_type == TYPE_IMAGE:
            caption = _get_str(item.get("caption"))
            if caption:
                segments.append(f"[caption] {caption}")
            continue

        if block_type == TYPE_TABLE:
            cap_raw = item.get("table_caption") or []
            fn_raw = item.get("table_footnote") or []
            cap_text = (
                " ".join(cap_raw).strip() if isinstance(cap_raw, list) else _get_str(cap_raw)
            )
            fn_text = (
                " ".join(fn_raw).strip() if isinstance(fn_raw, list) else _get_str(fn_raw)
            )
            if cap_text:
                segments.append(f"[caption] {cap_text}")
            if fn_text:
                segments.append(f"[footnote] {fn_text}")
            continue

        if block_type == TYPE_EQUATION:
            eq_text = _get_str(item.get("text"))
            if eq_text:
                segments.append(f"[equation] {eq_text}")
            continue

        text = _get_str(item.get("text"))
        if text:
            segments.append(text)

    return segments


def build_plain_text_from_content_list(content_list: List[Dict[str, Any]]) -> str:
    """将 content_list 转为单段可送入纯文本模型的论文全文字符串。"""
    parts = iter_text_segments_from_content_list(content_list)
    return "\n\n".join(parts)


def build_one_file(
    input_path: Path,
    output_dir: Path = MULTIMODAL_CONTENT_DIR,
    *,
    include_base64: bool = False,
    text_output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    对单个清洗后的 content_list JSON 构建多模态 content 并写入 output_dir；
    同时写入纯文本 LLM 输入 JSON 到 text_output_dir（默认 datasets/text_llm_input）。

    Args:
        input_path: 清洗后的 JSON 路径（通常在 input_cleaned 下）
        output_dir: 多模态 content 输出目录
        include_base64: 是否将图片转为 base64 写入（默认 False，仅写绝对路径，便于小文件）
        text_output_dir: 纯文本 LLM 中间 JSON 目录；为 None 时使用 TEXT_LLM_INPUT_DIR

    Returns:
        多模态输出文件路径；失败返回 None
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("读取失败 %s: %s", input_path, e)
        return None

    if not isinstance(data, list):
        logger.warning("跳过非列表格式: %s", input_path)
        return None

    base_dir: Optional[Path] = None
    original = find_original_content_list_path(input_path.name)
    if original is not None:
        base_dir = original.parent
    else:
        logger.warning("未找到原始 content_list 路径，图片将使用相对路径: %s", input_path.name)

    content = build_content_for_api(data, base_dir, include_base64=include_base64)

    output_dir.mkdir(parents=True, exist_ok=True)
    # 输出文件名与清洗文件一致，或可改为 _multimodal_content.json 等
    output_path = output_dir / input_path.name.replace("_content_list.json", "_multimodal_content.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

    logger.info("已构建: %s -> %s", input_path.name, output_path)

    text_dir = text_output_dir if text_output_dir is not None else TEXT_LLM_INPUT_DIR
    text_dir.mkdir(parents=True, exist_ok=True)
    text_body = build_plain_text_from_content_list(data)
    text_path = text_dir / input_path.name.replace(
        "_content_list.json", TEXT_LLM_INPUT_SUFFIX
    )
    text_payload: Dict[str, Any] = {
        "source_content_list": input_path.name,
        "text": text_body,
    }
    with open(text_path, "w", encoding="utf-8") as f:
        json.dump(text_payload, f, ensure_ascii=False, indent=2)
    logger.info("已写入纯文本 LLM 输入: %s", text_path)

    return output_path


def build_all(
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    *,
    include_base64: bool = False,
    text_output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    批量构建：遍历 input_dir 下所有 *_content_list.json，生成多模态 content 写入 output_dir，
    并生成纯文本 LLM 输入 JSON 写入 text_output_dir（默认 datasets/text_llm_input）。

    Args:
        input_dir: 清洗结果目录，默认 datasets/input_cleaned
        output_dir: 多模态 content 输出目录，默认 datasets/multimodal_content
        include_base64: 是否在 JSON 中内联图片 base64（默认 False，使用相对路径）
        text_output_dir: 纯文本 LLM 中间 JSON 目录，默认 TEXT_LLM_INPUT_DIR

    Returns:
        成功写入的多模态输出文件路径列表
    """
    input_dir = input_dir or INPUT_CLEANED_DIR
    output_dir = output_dir or MULTIMODAL_CONTENT_DIR

    if not input_dir.exists():
        logger.warning("目录不存在: %s", input_dir)
        return []

    files = sorted(input_dir.glob("*_content_list.json"))
    if not files:
        logger.warning("未找到任何 *_content_list.json: %s", input_dir)
        return []

    results: List[Path] = []
    for path in files:
        try:
            out = build_one_file(
                path,
                output_dir,
                include_base64=include_base64,
                text_output_dir=text_output_dir,
            )
            if out is not None:
                results.append(out)
        except Exception as e:
            logger.exception("构建失败 %s: %s", path, e)

    logger.info("共处理 %d 个文件，成功 %d 个", len(files), len(results))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_all()
