"""按 split 生成 SurfDesign 的 surface 缓存（surface_<mode>[_bb].lmdb）。

背景
----
上游 ``src/datamodules/datasets/cath.py::CATH()`` 在 lmdb 缺失时会对
``chain_set.jsonl`` 里的**全量**条目现场生成 surface 缓存（PyMOL surface
-> obj -> 逐点化学属性 -> lmdb）。实测单条约 1.2 s、obj 约 4.6 MB，全量
22508 条外推约 7.6 h、obj 缓存约 103 GB。

本脚本复用 ``cath.py`` 里同一套函数（``cath_to_pdb`` /
``load_point_cloud_by_file_extension`` / ``computer_property``），仅做两件事：

1. 把生成范围收敛到指定 split（推理只需 test），避免无谓开销；
2. 采用「边生成边删 obj」，把峰值磁盘占用从 ~103 GB 压到 GB 级
   （obj 只是中间缓存，``CATH()`` 里本来就有 ``os.path.exists`` 复用逻辑）。

生成结果与上游 CATH() 完全同构：lmdb 的 key 取 ``entry['name']``，
value 是 ``pickle.dumps([pts, norms, hp, hbond])``；同时写出与上游格式一致的
``structure.pkl``（``[SequenceWrapper(dataset), alphabet_set]``），
使官方 ``CATH()`` 后续运行直接命中缓存。

用法
----
    python sdaa/scripts/build_surface_lmdb.py --split test
    python sdaa/scripts/build_surface_lmdb.py --split all --workers 8
    python sdaa/scripts/build_surface_lmdb.py --split test --limit 5   # 冒烟
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Sequence

import numpy as np

# --- 让脚本可以按 <repo>/sdaa/scripts/xxx.py 的方式直接运行 ------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB，与 cath.py 保持一致
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

_pymol_ready = False


# --------------------------------------------------------------------------- #
# dataset 构造：与 cath.py::CATH() 中的逻辑逐行一致（只把范围收敛到 split）
# --------------------------------------------------------------------------- #
def build_dataset(jsonl_path: str, max_length: int = 500) -> tuple[List[Dict], set]:
    alphabet_set = set(AA_ALPHABET)
    discard_count = {"bad_chars": 0, "too_long": 0}
    dataset: List[Dict] = []
    is_cath43 = "cath_4.3" in jsonl_path

    with open(jsonl_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        entry = json.loads(r"{}".format(line)) if is_cath43 else json.loads(line)
        seq = entry["seq"]
        bad_chars = set(seq).difference(alphabet_set)
        if len(bad_chars) == 0:
            if len(entry["seq"]) <= max_length:
                mask = ~np.isnan(entry["coords"]["CA"])[:, 0]
                entry["seq"] = "".join(np.array(list(entry["seq"]))[mask])
                for key, val in entry["coords"].items():
                    entry["coords"][key] = np.asarray(val, dtype=np.float32)[mask]
                dataset.append(entry)
            else:
                discard_count["too_long"] += 1
        else:
            discard_count["bad_chars"] += 1

    print(
        f"[build] jsonl {len(lines)} 条 -> 保留 {len(dataset)} 条 "
        f"(剔除: {discard_count})",
        flush=True,
    )
    return dataset, alphabet_set


def load_structure_cache(root: str, cache_path: str, jsonl_path: str, max_length: int):
    """有 structure.pkl 就读它，否则现场构造并落盘（格式与上游一致）。"""
    if os.path.exists(cache_path):
        print(f"[build] 命中 structure 缓存 {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            dataset, alphabet_set = pickle.load(f)
        return dataset, alphabet_set

    dataset, alphabet_set = build_dataset(jsonl_path, max_length=max_length)

    # 与上游 CATH() 落盘格式保持一致
    from torch.utils.data.datapipes.map import SequenceWrapper

    with open(cache_path, "wb") as f:
        pickle.dump([SequenceWrapper(dataset), alphabet_set], f)
    print(f"[build] 已写 structure 缓存 {cache_path}", flush=True)
    return dataset, alphabet_set


# --------------------------------------------------------------------------- #
# worker：一个进程一个 PyMOL 实例，写自己的分片 lmdb
# --------------------------------------------------------------------------- #
def _init_pymol():
    global _pymol_ready
    from pymol import cmd  # noqa: F401
    import pymol2
    import pymol.invocation

    pymol.invocation.parse_args(["pymol", "-q"])
    pymol2.SingletonPyMOL().start()
    _pymol_ready = True


def _surface_one(entry: Dict, root: str, surf_mode: str, bb_surface: bool):
    """生成单条的 (pts, norms, hp, hbond)，返回 pickled bytes。"""
    from src.datamodules.datasets.cath import (
        cath_to_pdb,
        computer_property,
        load_point_cloud_by_file_extension,
    )

    suffix = "_bb" if bb_surface else ""
    pdb_path = os.path.join(root, f"pdb{suffix}/{entry['name']}.pdb")
    if not os.path.exists(pdb_path):
        cath_to_pdb(entry, pdb_path)

    surf_dir = os.path.join(root, f"surface{suffix}")
    obj_path = os.path.join(surf_dir, f"{entry['name']}.obj")

    if not os.path.exists(obj_path) or os.path.getsize(obj_path) == 0:
        if surf_mode == "pymol":
            from pymol import cmd

            cmd.reinitialize()
            cmd.load(pdb_path)
            cmd.show_as("surface")
            # 调整顶点坐标系，见 pymolwiki: Model_Space_and_Camera_Space
            cmd.set_view((1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 300, 1))
            cmd.save(obj_path)
        else:
            raise ValueError("本脚本目前只实现 surf_mode='pymol'（与 cath_4.3 配置一致）")

    pts, norms = load_point_cloud_by_file_extension(obj_path)
    hp, hbond = computer_property(pdb_path, pts, surf_mode)
    return pickle.dumps([pts, norms, hp, hbond]), obj_path


def worker(payload):
    # 注意：PyMOL 实例由 Pool(initializer=_init_pymol) 在每个子进程内启动一次，
    # 这里**不能**再调一次 —— SingletonPyMOL().start() 调第二次会直接抛
    # RuntimeError('can only start SingletonPyMOL once')。
    rank, entries, root, surf_mode, bb_surface, part_path, keep_obj, resume = payload
    import lmdb

    env = lmdb.open(part_path, map_size=MAP_SIZE, create=True, subdir=False, readonly=False)
    n_done = n_skip = 0
    t0 = time.time()

    for entry in entries:
        name = entry["name"]
        try:
            if resume:
                with env.begin() as txn:
                    if txn.get(name.encode()) is not None:
                        n_skip += 1
                        continue
            blob, obj_path = _surface_one(entry, root, surf_mode, bb_surface)
            with env.begin(write=True, buffers=True) as txn:
                txn.put(name.encode(), blob)
            n_done += 1
            if not keep_obj:
                try:
                    os.remove(obj_path)
                except OSError:
                    pass
        except Exception as exc:  # 单条失败不拖垮整体
            print(f"[w{rank}] FAIL {name}: {type(exc).__name__}: {exc}", flush=True)

        if n_done % 50 == 0 and n_done:
            rate = (time.time() - t0) / n_done
            print(
                f"[w{rank}] {n_done}/{len(entries)} skip={n_skip} "
                f"{rate:.2f}s/条",
                flush=True,
            )

    env.close()
    print(f"[w{rank}] 完成 done={n_done} skip={n_skip} 用时 {time.time()-t0:.1f}s", flush=True)
    return rank, n_done, n_skip


def merge_parts(part_paths: Sequence[str], out_path: str, commit_every: int = 200):
    """把分片合并成最终 lmdb。

    必须**分块提交**：每个分片约 1400 条、单条 ~375 KB，一个事务就是 500 MB 级脏页，
    会撞上 LMDB 的脏页上限（MDB_TXN_FULL）。上游 `cath.py` 是「每条一个事务」，
    这里取折中：每 `commit_every` 条提交一次，兼顾安全与写入速度。
    """
    import lmdb

    print(f"[merge] 合并 {len(part_paths)} 个分片 -> {out_path}", flush=True)
    dst = lmdb.open(out_path, map_size=MAP_SIZE, create=True, subdir=False, readonly=False)
    total = 0
    txn = dst.begin(write=True)
    try:
        for p in part_paths:
            src = lmdb.open(p, map_size=MAP_SIZE, subdir=False, readonly=True, lock=False)
            with src.begin() as stxn:
                for k, v in stxn.cursor():
                    txn.put(k, v)
                    total += 1
                    if total % commit_every == 0:
                        txn.commit()
                        txn = dst.begin(write=True)
            src.close()
    finally:
        txn.commit()
    dst.close()
    print(f"[merge] 写入 {total} 条", flush=True)
    return total


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="生成 SurfDesign surface lmdb 缓存")
    ap.add_argument("--root", default=None, help="数据目录，默认 <repo>/data/cath_4.3")
    ap.add_argument("--split", default="test", help="test | validation | train | all")
    ap.add_argument("--splits-json", default="splits.json")
    ap.add_argument("--jsonl", default="chain_set.jsonl")
    ap.add_argument("--surf-mode", default="pymol")
    ap.add_argument("--bb-surface", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=500)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（冒烟用）")
    ap.add_argument("--keep-obj", action="store_true", help="保留 obj（默认用完即删）")
    ap.add_argument("--no-resume", action="store_true", help="不跳过已存在的条目")
    args = ap.parse_args()

    root = args.root or os.path.join(_REPO_ROOT, "data", "cath_4.3")
    root = os.path.abspath(root)
    suffix = "_bb" if args.bb_surface else ""
    out_path = os.path.join(root, f"surface_{args.surf_mode}{suffix}.lmdb")
    structure_cache = os.path.join(root, "structure.pkl")

    os.makedirs(os.path.join(root, f"pdb{suffix}"), exist_ok=True)
    os.makedirs(os.path.join(root, f"surface{suffix}"), exist_ok=True)

    dataset, _ = load_structure_cache(
        root, structure_cache, os.path.join(root, args.jsonl), args.max_length
    )

    splits_path = os.path.join(root, args.splits_json)
    if args.split == "all":
        wanted = {e["name"] for e in dataset}
    else:
        with open(splits_path) as f:
            splits = json.load(f)
        key = "validation" if args.split in ("valid", "validation") else args.split
        wanted = set(splits[key])
        print(f"[build] split={args.split} 名单 {len(wanted)} 条", flush=True)

    entries = [e for e in dataset if e["name"] in wanted]
    if args.limit:
        entries = entries[: args.limit]
    print(f"[build] 待生成 {len(entries)} 条 -> {out_path}", flush=True)

    if not entries:
        print("[build] 没有待处理条目，退出")
        return

    nw = max(1, min(args.workers, len(entries)))
    resume = not args.no_resume
    t0 = time.time()

    if nw == 1:
        # 单进程不走 Pool，需要自己起 PyMOL
        _init_pymol()
        part = out_path + ".part0"
        worker((0, entries, root, args.surf_mode, bool(args.bb_surface), part,
                args.keep_obj, resume and os.path.exists(part)))
        if os.path.exists(part):
            os.replace(part, out_path)
    else:
        chunks = [entries[i::nw] for i in range(nw)]
        parts = [out_path + f".part{i}" for i in range(nw)]
        payloads = [
            (i, chunks[i], root, args.surf_mode, bool(args.bb_surface), parts[i],
             args.keep_obj, resume and os.path.exists(parts[i]))
            for i in range(nw)
        ]
        with Pool(nw, initializer=_init_pymol) as pool:
            pool.map(worker, payloads)
        merge_parts(parts, out_path)
        for p in parts:
            os.remove(p)

    size = os.path.getsize(out_path)
    print(
        f"[done] {out_path}  {size/1024**3:.2f} GiB  用时 {(time.time()-t0)/60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
