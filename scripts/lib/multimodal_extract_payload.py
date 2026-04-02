"""将 build_multimodal_content 产出的多模态 JSON 转为可直接调用 Chat Completions 的 content 列表。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from build_multimodal_content import encode_image


def guess_image_mime(image_path: Path) -> str:
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
        mime = guess_image_mime(path)
        b64 = encode_image(path)
        out.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )
    return out
