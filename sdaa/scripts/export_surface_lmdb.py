#!/usr/bin/env python
"""把 surface lmdb 导出为跨平台可移植的 numpy 数据流（在源机上运行）。

为什么需要这一步
----------------
1. **页大小固化**：LMDB 数据文件在创建时把页大小写进 meta page（x86_64 = 4096 B，
   loongarch64 = 16384 B）。跨页大小的 lmdb 文件无法直接 mmap 打开，因此不能把
   `surface_pymol_bb.lmdb` 原样拷到另一架构的机器上。
2. **跨 torch 版本**：lmdb value 是 `pickle.dumps([pts, norms, hp, hbond])`，其中四个
   元素都是 **torch 张量**。直接跨 torch 版本（2.4 → 2.12）反序列化有风险。

所以统一转成 **numpy 数组**（`torch.Tensor.numpy()` 共享底层 buffer，dtype 逐位保留），
落成单个 pickle 流文件；目标机再用 `torch.from_numpy` 还原，并把 value 重新
`pickle.dumps` 写进本机原生 lmdb —— 这样两端喂给模型的表面数据是逐位一致的，
跨平台的精度/性能对比才是"纯平台差异"。

输出
----
- `<out>`          **长度前缀分帧**的 pickle 记录流：每条 = `struct.pack('<Q', n)` + `pickle.dumps((name, arrs))`。
  不能用"同一个 Pickler 连续 dump"的做法 —— Pickler 的 memo 会跨次累积，而读取端每次
  `pickle.load` 都重置 memo，导致 `UnpicklingError: Memo value not found at index N`。
- `<out>.manifest` 逐条目数据指纹（md5），供目标机导入后逐字节核对。

用法
----
    python export_surface_lmdb.py --lmdb <path> --out <path.pkl>
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import struct
import sys
import time

import numpy as np
import torch

import lmdb

MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB，与 cath.py 保持一致


def write_record(f, obj) -> int:
    """写一条长度前缀分帧的 pickle 记录，返回写入字节数。"""
    payload = pickle.dumps(obj, protocol=4)
    f.write(struct.pack("<Q", len(payload)))
    f.write(payload)
    return 8 + len(payload)


def arr_fingerprint(arrs) -> str:
    """对一组数组做与导出/导入两侧无关的稳定指纹（dtype + shape + 原始字节）。"""
    h = hashlib.md5()
    for a in arrs:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def to_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj
    return np.asarray(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True, help="源 lmdb 文件路径")
    ap.add_argument("--out", required=True, help="输出 pickle 流路径")
    ap.add_argument("--report-every", type=int, default=1000)
    args = ap.parse_args()

    if not os.path.exists(args.lmdb):
        sys.exit(f"lmdb 不存在: {args.lmdb}")

    env = lmdb.open(args.lmdb, map_size=MAP_SIZE, subdir=False,
                    readonly=True, lock=False, meminit=False, readahead=False)

    manifest = open(args.out + ".manifest", "w", encoding="utf-8")
    n = 0
    dtypes = {}
    t0 = time.time()

    with env.begin() as txn:
        total = txn.stat()["entries"]
        print(f"[export] 源 lmdb: {args.lmdb}")
        print(f"[export] 条目数 : {total}")
        print(f"[export] 输出   : {args.out}")

        with open(args.out, "wb") as f:
            for key, raw in txn.cursor():
                name = key.decode("utf-8")
                value = pickle.loads(raw)
                arrs = [to_numpy(v) for v in value]
                for a in arrs:
                    dtypes[str(a.dtype)] = dtypes.get(str(a.dtype), 0) + 1
                write_record(f, (name, arrs))
                manifest.write(f"{name}\t{arr_fingerprint(arrs)}\n")
                n += 1
                if n % args.report_every == 0 or n == total:
                    el = time.time() - t0
                    print(f"[export] {n}/{total}  {el:.0f}s  {n / max(el, 1e-9):.1f} 条/s  "
                          f"out={os.path.getsize(args.out) / 1e9:.2f}GB")

    manifest.close()
    env.close()

    if n != total:
        sys.exit(f"!! 只导出 {n} 条，源有 {total} 条")
    print(f"[export] 完成 {n} 条，用时 {time.time() - t0:.0f}s")
    print(f"[export] 出现过的 dtype 分布: {dtypes}")


if __name__ == "__main__":
    main()
