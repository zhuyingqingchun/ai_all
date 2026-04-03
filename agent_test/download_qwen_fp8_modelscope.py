#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from pathlib import Path

from modelscope import snapshot_download

MODEL_REPOS = {
    # 推荐先下这个：更适合你现在的 agent 工程化路线
    "next80b_fp8": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",

    # 如果你就是要更大的官方 FP8 instruct
    "qwen235b_fp8": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 ModelScope SDK 下载 Qwen FP8 模型"
    )
    parser.add_argument(
        "--model",
        choices=MODEL_REPOS.keys(),
        default="next80b_fp8",
        help="要下载的模型别名",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="下载到哪个目录，例如 /mnt/PRO6000_disk/models/Qwen",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="可选，指定模型 revision；默认下载最新版本",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="可选，ModelScope 缓存目录；不填则使用默认缓存",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_id = MODEL_REPOS[args.model]
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 下载后的模型目录名，直接取 repo_id 最后一段
    model_dir_name = repo_id.split("/")[-1]
    local_dir = output_root / model_dir_name

    print("=" * 80)
    print("开始下载")
    print(f"repo_id   : {repo_id}")
    print(f"local_dir : {local_dir}")
    if args.cache_dir:
        print(f"cache_dir : {args.cache_dir}")
    if args.revision:
        print(f"revision  : {args.revision}")
    print("=" * 80)

    kwargs = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
    }

    if args.revision:
        kwargs["revision"] = args.revision

    if args.cache_dir:
        cache_dir = Path(args.cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs["cache_dir"] = str(cache_dir)

    try:
        downloaded_dir = snapshot_download(**kwargs)
    except KeyboardInterrupt:
        print("\n用户中断下载。")
        sys.exit(130)
    except Exception as e:
        print("\n下载失败：")
        print(str(e))
        sys.exit(1)

    print("\n下载完成。")
    print(f"模型目录: {downloaded_dir}")

    # 简单列一下关键文件是否存在
    expected_files = [
        "config.json",
        "tokenizer_config.json",
        "generation_config.json",
    ]

    print("\n关键文件检查：")
    for name in expected_files:
        p = Path(downloaded_dir) / name
        print(f"  {name:<24} {'OK' if p.exists() else 'MISSING'}")

    print("\n可用模型别名：")
    for k, v in MODEL_REPOS.items():
        print(f"  {k:<14} -> {v}")


if __name__ == "__main__":
    main()