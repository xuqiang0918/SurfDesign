#!/usr/bin/env python
"""把 surface lmdb 的 .partN 分片合并成最终 lmdb（分块提交，可重入）。

为什么单独写一个
----------------
`build_surface_lmdb.py::merge_parts` 对每个分片用**单个**写事务（~1400 条 × ~375 KB
≈ 525 MB 脏页），LMDB 的 `MDB_TXN_FULL`（脏页上限）有被触发的风险。本脚本按
`--commit-every` 分块提交，并且**可重入**：目标已存在时先删再重建，失败可反复重试。

用法
----
    python merge_surface_lmdb.py --out <surface_pymol_bb.lmdb> [--glob '<out>.part*'] \
        [--commit-every 200] [--keep-parts]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import lmdb

MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB，与 cath.py 保持一致


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="最终 lmdb 路径")
    ap.add_argument("--glob", default=None, help="分片 glob，默认 '<out>.part*'（会剔除 *-lock）")
    ap.add_argument("--commit-every", type=int, default=200)
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args()

    pat = args.glob or (args.out + ".part*")
    parts = sorted(p for p in glob.glob(pat) if not p.endswith("-lock"))
    if not parts:
        sys.exit(f"没有匹配到分片: {pat}")
    print(f"[merge] {len(parts)} 个分片 -> {args.out}")

    for p in (args.out, args.out + "-lock"):
        if os.path.exists(p):
            os.remove(p)

    dst = lmdb.open(args.out, map_size=MAP_SIZE, create=True, subdir=False, readonly=False)
    total = 0
    t0 = time.time()
    txn = dst.begin(write=True)
    try:
        for p in parts:
            src = lmdb.open(p, map_size=MAP_SIZE, subdir=False, readonly=True, lock=False)
            n_part = 0
            with src.begin() as stxn:
                for k, v in stxn.cursor():
                    txn.put(k, v)
                    n_part += 1
                    total += 1
                    if total % args.commit_every == 0:
                        txn.commit()
                        txn = dst.begin(write=True)
            src.close()
            print(f"[merge]   {os.path.basename(p)}: {n_part} 条（累计 {total}）", flush=True)
    finally:
        txn.commit()
    dst.close()

    size = os.path.getsize(args.out)
    print(f"[merge] 写入 {total} 条 -> {args.out}  {size / 1024 ** 3:.2f} GiB  "
          f"用时 {(time.time() - t0) / 60:.1f} min")

    if not args.keep_parts:
        for p in parts:
            os.remove(p)
        print(f"[merge] 已删除 {len(parts)} 个分片")


if __name__ == "__main__":
    main()
