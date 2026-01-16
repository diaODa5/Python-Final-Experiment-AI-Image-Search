#!/usr/bin/env python3
"""
Download images from a huge CSV until SUCCESS count reaches target (default 12000).

Fixes:
- No generator re-entrancy: URL iteration happens ONLY in the main thread.
- Uses a bounded in-flight queue of futures; continues reading CSV until success target reached.
- Supports resume: counts existing *.jpg in save_dir and continues numbering.
"""

import os
import re
import time
import argparse
from typing import Iterator, Optional, Tuple

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm


# ----------------------------
# Defaults
# ----------------------------
CSV_PATH = "data.csv"
SAVE_DIR = "gallery_images"
TARGET_COL = "image_url"
DEFAULT_TARGET = 12000
DEFAULT_CHUNKSIZE = 200000
DEFAULT_WORKERS = 12
DEFAULT_TIMEOUT = 25
DEFAULT_RETRIES = 2
DEFAULT_MAX_ATTEMPTS = 200000  # safety cap (number of URL attempts)


_NUM_JPG = re.compile(r"^(\d+)\.jpg$", re.I)


def _scan_existing(save_dir: str) -> Tuple[int, int]:
    """
    Returns (existing_count, next_index).
    Looks for files like 000001.jpg; next_index = max_index + 1.
    """
    if not os.path.exists(save_dir):
        return 0, 0
    mx = -1
    cnt = 0
    for fn in os.listdir(save_dir):
        m = _NUM_JPG.match(fn)
        if m:
            cnt += 1
            mx = max(mx, int(m.group(1)))
    return cnt, mx + 1 if mx >= 0 else 0


def iter_urls_stream(csv_path: str, col: str, chunksize: int, encoding: Optional[str]) -> Iterator[str]:
    """
    Stream URLs from a huge CSV, reading only one column.
    """
    reader = pd.read_csv(
        csv_path,
        usecols=[col],
        chunksize=chunksize,
        encoding=encoding if encoding else None,
    )
    for chunk in reader:
        s = chunk[col].dropna()
        for v in s.values:
            u = str(v).strip()
            if len(u) >= 5:
                yield u


def _download_one(
    idx_url: Tuple[int, str],
    save_dir: str,
    timeout: int,
    retries: int,
    check_content_type: bool,
    backoff_base: float,
) -> bool:
    """
    Download one image. Returns True if saved or already exists.
    """
    idx, url = idx_url
    save_path = os.path.join(save_dir, f"{idx:06d}.jpg")

    # resume-safe
    if os.path.exists(save_path):
        return True

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }

    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            if check_content_type:
                ctype = resp.headers.get("Content-Type", "")
                if "image" not in ctype.lower():
                    raise RuntimeError(f"Not image content-type: {ctype}")

            content = resp.content
            if not content or len(content) < 200:  # avoid tiny error pages
                raise RuntimeError("Empty/too small content")

            with open(save_path, "wb") as f:
                f.write(content)
            return True
        except Exception:
            # backoff before retry
            if attempt < retries:
                time.sleep(backoff_base * (2 ** attempt))
            continue
    return False


def main():
    ap = argparse.ArgumentParser(description="Download images until SUCCESS count reaches target.")
    ap.add_argument("--csv", default=CSV_PATH, help="CSV path (default: data.csv)")
    ap.add_argument("--save_dir", default=SAVE_DIR, help="Output folder (default: gallery_images)")
    ap.add_argument("--col", default=TARGET_COL, help="URL column name (default: image_url)")
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET, help="Target SUCCESS count (default: 12000)")
    ap.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE, help="CSV streaming chunk size")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Download threads (default: 12)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout seconds (default: 25)")
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries per URL (default: 2)")
    ap.add_argument("--max_attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                    help="Safety cap on URL attempts (default: 200000)")
    ap.add_argument("--encoding", default="", help="CSV encoding (default: try utf-8 then gbk)")
    ap.add_argument("--check_content_type", action="store_true",
                    help="Only save when Content-Type is image/* (recommended)")
    ap.add_argument("--backoff", type=float, default=0.6, help="Retry backoff base seconds (default: 0.6)")
    ap.add_argument("--inflight", type=int, default=0,
                    help="Max in-flight futures (default: workers*4)")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"错误: 找不到 {args.csv}")
        return

    os.makedirs(args.save_dir, exist_ok=True)

    existing, next_idx = _scan_existing(args.save_dir)
    if existing >= args.target:
        print(f"已存在 {existing} 张 >= target {args.target}，无需下载。目录: {os.path.abspath(args.save_dir)}")
        return

    inflight = args.inflight if args.inflight > 0 else args.workers * 4

    # Choose encoding: user-specified first, else try utf-8 then gbk
    encodings = [args.encoding] if args.encoding else ["utf-8", "gbk"]
    url_iter = None
    last_err = None
    for enc in encodings:
        try:
            url_iter = iter_urls_stream(args.csv, args.col, args.chunksize, enc if enc else None)
            # smoke test one element without consuming it permanently: just create generator (no next call)
            break
        except Exception as e:
            last_err = e
            url_iter = None

    if url_iter is None:
        print(f"读取 CSV 失败: {last_err}")
        print(f"请确认列名是否为: {args.col}，或用 --encoding 指定编码。")
        return

    success = existing
    attempts = 0
    pending = {}

    print(f"当前已有: {existing} 张，将继续下载到: {args.target} 张")
    print(f"并发: {args.workers}，in-flight: {inflight}，retries: {args.retries}，timeout: {args.timeout}s")
    if args.check_content_type:
        print("启用 Content-Type 校验：只保存 image/* 响应")

    pbar = tqdm(total=args.target, initial=success, desc="成功下载", unit="img")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        # Main loop: keep filling tasks and collecting results until target reached or attempts exhausted
        while success < args.target and attempts < args.max_attempts:
            # Fill up in-flight queue
            while len(pending) < inflight and success < args.target and attempts < args.max_attempts:
                try:
                    url = next(url_iter)
                except StopIteration:
                    # No more URLs in CSV
                    break

                idx = next_idx
                next_idx += 1
                attempts += 1

                fut = ex.submit(
                    _download_one,
                    (idx, url),
                    args.save_dir,
                    args.timeout,
                    args.retries,
                    args.check_content_type,
                    args.backoff,
                )
                pending[fut] = idx

            if not pending:
                # Nothing in flight and cannot add more (StopIteration or attempts cap)
                break

            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                pending.pop(fut, None)
                ok = False
                try:
                    ok = bool(fut.result())
                except Exception:
                    ok = False
                if ok:
                    success += 1
                    pbar.update(1)

        pbar.close()

    print("-" * 30)
    print("下载结束")
    print(f"成功: {success} 张（新增 {success - existing}）")
    print(f"尝试: {attempts} 条 URL")
    if success < args.target:
        print(f"未达到目标 {args.target}，可能是可用 URL 不足 / 反爬限制较强。可尝试：降低并发 --workers 8，开启 --check_content_type，增加 --max_attempts。")
    print(f"图片保存在: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()
