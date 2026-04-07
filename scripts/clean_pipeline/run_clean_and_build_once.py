"""
一键串联执行：
1) 清洗输入目录下的 *_content_list.json 到 datasets/input_cleaned
2) 构建 datasets/multimodal_content 与 datasets/text_llm_input

该脚本只执行一次，不做循环与守护。
"""

from __future__ import annotations

import logging
from pathlib import Path

try:
    from .build_multimodal_content import (
        INPUT_CLEANED_DIR,
        MULTIMODAL_CONTENT_DIR,
        TEXT_LLM_INPUT_DIR,
        build_all,
    )
    from .json_cleaner import PAPER_PARSERED_DIR, clean_all_papers
except ImportError:
    # 兼容直接运行：python scripts/clean_pipeline/run_clean_and_build_once.py
    from build_multimodal_content import (  # type: ignore
        INPUT_CLEANED_DIR,
        MULTIMODAL_CONTENT_DIR,
        TEXT_LLM_INPUT_DIR,
        build_all,
    )
    from json_cleaner import PAPER_PARSERED_DIR, clean_all_papers  # type: ignore

# -----------------------------------------------------------------------------
# 用户配置区（直接改这里）
# -----------------------------------------------------------------------------
# 输入目录：这里应放原始的 *_content_list.json 文件（会递归查找）
# 例如：
#   /path/to/your_data/paper_parsered
#   /path/to/another_dir
USER_INPUT_DIR: Path = Path("/home/caep-xuben/chenchengbing/yzj/experiments/exp_0405/data/mineru_data")
#
# 你只需要修改上面这一行，即可切换输入数据路径。
#
# 输出目录（可按需修改）
# - 清洗后的 *_content_list.json 输出到这里
USER_CLEANED_OUTPUT_DIR: Path = Path("/home/caep-xuben/chenchengbing/yzj/experiments/exp_0405/data/mineru_data_cleaned")
# - 多模态 content 输出到这里
USER_MULTIMODAL_OUTPUT_DIR: Path = Path("/home/caep-xuben/chenchengbing/yzj/experiments/exp_0405/data/mineru_data_multimodal_input")
# - 纯文本 LLM 输入 JSON 输出到这里
USER_TEXT_OUTPUT_DIR: Path = Path("/home/caep-xuben/chenchengbing/yzj/experiments/exp_0405/data/mineru_data_text_llm_input")


def run_once() -> int:
    """执行一次全流程，成功返回 0，失败返回非 0。"""
    logger = logging.getLogger(__name__)

    logger.info("步骤 1/2: 开始清洗 JSON -> %s", USER_CLEANED_OUTPUT_DIR)
    cleaned_files = clean_all_papers(
        input_dir=USER_INPUT_DIR,
        output_dir=USER_CLEANED_OUTPUT_DIR,
    )
    if not cleaned_files:
        logger.error("没有可用的清洗结果，流程终止。")
        return 1

    logger.info("步骤 2/2: 开始构建多模态与纯文本输入")
    built_files = build_all(
        input_dir=USER_CLEANED_OUTPUT_DIR,
        output_dir=USER_MULTIMODAL_OUTPUT_DIR,
        include_base64=False,
        text_output_dir=USER_TEXT_OUTPUT_DIR,
        original_search_root=USER_INPUT_DIR,
    )
    if not built_files:
        logger.error("构建阶段未生成输出，流程终止。")
        return 2

    logger.info(
        "流程完成：清洗 %d 个文件，构建 %d 个文件。输出目录：%s / %s",
        len(cleaned_files),
        len(built_files),
        USER_MULTIMODAL_OUTPUT_DIR,
        USER_TEXT_OUTPUT_DIR,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(run_once())

