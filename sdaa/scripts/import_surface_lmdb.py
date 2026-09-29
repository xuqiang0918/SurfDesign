#!/usr/bin/env python
"""在目标机上把导出的 numpy 数据流重建为本机原生 lmdb（跨架构迁移用）。

配套 `export_surface_lmdb.py`。LMDB 页大小随架构固化（x86=4096 / loongarch64=16384），
文件不可直接拷贝，所以在目标机重建一个原生 lmdb，写回上游期望的
`pickle.dumps([pts, norms, hp, hbond])`（`cath.py::Featurizer.get_surface` 的读取格式）。

--verify 会读回 lmdb 并对每个条目重算数据指纹，与导出侧 `<pkl>.manifest` 逐条比对，
用于证明"两端喂给模型的表面数据逐位一致"（否则跨平台精度对比没有意义）。

输入格式见 `export_surface_lmdb.py`：长度前缀分帧的 pickle 记录流。
默认每 100 条提交一次事务（单条 ~0.5MB × 16K 页 ≈ 33 页，避免 LMDB `MDB_TXN_FULL`）。

用法
----
    python import_surface_lmdb.py --pkl <path.pkl> --lmdb <path.lmdb> \
        [--manifest <path.pkl>.manifest] [--verify] [--commit-every 100]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import struct
import sys
import time

import lmdb
import numpy as np
import torch

MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB，与 cath.py 保持一致


def iter_records(f):
    """读长度前缀分帧的 pickle 记录流（见 export_surface_lmdb.py 的格式说明）。"""
    while True:
        hdr = f.read(8)
        if len(hdr) == 0:
            return
        if len(hdr) != 8:
            raise EOFError("记录头被截断")
        (n,) = struct.unpack("<Q", hdr)
        payload = f.read(n)
        if len(payload) != n:
            raise EOFError(f"记录体被截断（期望 {n}，实得 {len(payload)}）")
        yield pickle.loads(payload)


def arr_fingerprint(arrs) -> str:
    h = hashlib.md5()
    for a in arrs:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def tensors_from(arrs):
    return [torch.from_numpy(np.ascontiguousarray(a)) for a in arrs]


def do_import(args):
    env = lmdb.open(args.lmdb, map_size=MAP_SIZE, subdir=False, create=True, readonly=False)
    n = 0
    t0 = time.time()
    with open(args.pkl, "rb") as f:
        txn = env.begin(write=True)
        try:
            for name, arrs in iter_records(f):
                txn.put(name.encode("utf-8"), pickle.dumps(tensors_from(arrs)))
                n += 1
                if n % args.commit_every == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                if n % 2000 == 0:
                    el = time.time() - t0
                    print(f"[import] {n} 条  {el:.0f}s  {n / max(el, 1e-9):.1f} 条/s  "
                          f"lmdb={os.path.getsize(args.lmdb) / 1e9:.2f}GB")
        finally:
            txn.commit()
    env.close()
    print(f"[import] 完成 {n} 条，用时 {time.time() - t0:.0f}s -> {args.lmdb}")
    return n


def do_verify(args):
    if not args.manifest or not os.path.exists(args.manifest):
        sys.exit(f"--verify 需要 manifest 文件: {args.manifest}")
    want = {}
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            name, fp = line.rstrip("\n").split("\t")
            want[name] = fp

    env = lmdb.open(args.lmdb, map_size=MAP_SIZE, subdir=False,
                    create=False, readonly=True, lock=False, readahead=False, meminit=False)
    bad, checked = [], 0
    with env.begin() as txn:
        got = txn.stat()["entries"]
        for key, raw in txn.cursor():
            name = key.decode("utf-8")
            value = pickle.loads(raw)
            arrs = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                    for v in value]
            fp = arr_fingerprint(arrs)
            if want.get(name) != fp:
                bad.append(name)
            checked += 1
    env.close()

    print(f"[verify] lmdb 条目 {got} / manifest 条目 {len(want)} / 实际核对 {checked}")
    if bad:
        print(f"[verify] !! 指纹不一致 {len(bad)} 条，前 5: {bad[:5]}")
    if got != len(want) or checked == 0 or checked != got or bad:
        print("[verify] 结论: 不一致")
        return 1
    print("[verify] 结论: 与导出侧逐条指纹完全一致（数据逐位相同）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--verify", action="store_true", help="导入完成后读回 lmdb 逐条核对指纹")
    ap.add_argument("--verify-only", action="store_true", help="跳过导入，只核对已有 lmdb")
    ap.add_argument("--commit-every", type=int, default=100)
    args = ap.parse_args()

    if args.manifest is None:
        args.manifest = args.pkl + ".manifest"

    if not args.verify_only:
        if os.path.exists(args.lmdb):
            sys.exit(f"目标已存在，拒绝覆盖（先手动删除）: {args.lmdb}")
        do_import(args)

    if args.verify or args.verify_only:
        return do_verify(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
