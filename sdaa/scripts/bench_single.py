"""SurfDesign 单样本推理性能 + 峰值显存（走官方 test.py 的同一套构件与推理入口）。

为什么单独测
------------
``test.py`` 一次跑完整个 test split（cath_4.3 为 1864 条），墙钟时间由数据规模决定，看不出
"一条样本本身多慢"。这里复用与 ``test.py`` 完全相同的构件
（``utils.resolve_experiment_config`` 解析出的 datamodule + ``CMLM``），只喂 1 条样本，
测单条延迟与峰值显存。两侧同一脚本，只换设备。

与 test.py 的一致性
-------------------
- 配置：同样的 ``configs/test.yaml`` + ``experiment=fixedbb/surf_design_esm2_650m``
- 构件：``utils.instantiate_from_config`` 建 datamodule / task（common_pipeline 同路径）
- 取数：``datamodule.setup('test')`` 后直接从 ``test_dataset`` 取<b>中位长度</b>的 3 条样本，
  用 ``datamodule.collate_batch``（即 ``Featurizer``）出 batch —— 与 DataLoader 里
  ``collate_fn`` 走的是同一个对象，只是绕开 MaxTokensBatchSampler 以便控制样本长度。
- 推理：直接调 ``pl_module.predict_step``，即 test.py 里被 ``on_prediction_mode``
  换到 ``test_step`` 位置的那个函数（含生成器全部 max_iter 轮迭代）。

用法
----
    # 6.21 (CUDA)
    python sdaa/scripts/bench_single.py --max-iter 5 --n 3 --repeat 5
    # 115 (SDAA)
    /home/py312/bin/python3 sdaa/scripts/bench_single.py --max-iter 5 --n 3 --repeat 5
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_EXP = "logs/fixedbb/cath_4.3/surf_design_esm2_650m"


# --------------------------------------------------------------------------- #
# 设备工具：SDAA 与 CUDA 的显存/同步接口同名，取到哪个用哪个
# --------------------------------------------------------------------------- #
def _accel_mod(dev: str):
    if dev == "sdaa":
        return getattr(torch, "sdaa", None)
    if dev == "cuda":
        return torch.cuda
    return None


def _api(dev: str, name: str):
    mod = _accel_mod(dev)
    return getattr(mod, name, None) if mod is not None else None


def _sync(dev: str) -> None:
    fn = _api(dev, "synchronize")
    if fn is not None:
        fn()


def _peak_mem(dev: str) -> int:
    fn = _api(dev, "max_memory_allocated")
    return fn() if fn is not None else 0


def _reset_peak(dev: str) -> None:
    fn = _api(dev, "reset_peak_memory_stats")
    if fn is not None:
        fn()


def mem_apis(dev: str) -> list:
    mod = _accel_mod(dev)
    if mod is None:
        return []
    return sorted(a for a in dir(mod) if "memory" in a.lower() or a in ("synchronize", "empty_cache"))


def to_device(obj, dev: str):
    """把 batch 搬到设备。没有 Trainer 就用不了 PL 的 transfer_batch_to_device，
    而 utils.recursive_to 先试 ``.cuda()``（SDAA 上抛的不是 RuntimeError，捕不到）——自己写。"""
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, dict):
        return {k: to_device(v, dev) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, dev) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(v, dev) for v in obj)
    if hasattr(obj, "to") and hasattr(obj, "keys"):  # PyG Batch / HeteroData
        return obj.to(dev)
    return obj


def pick_samples(dataset, n: int):
    """取长度接近中位数的 n 条样本下标（官方 sampler 是按长度排序的，前几条都是最短的）。"""
    lengths = [len(dataset[i]["seq"]) for i in range(len(dataset))]
    order = sorted(range(len(dataset)), key=lambda i: lengths[i])
    mid = len(order) // 2
    picks = []
    for d in range(len(order)):
        for j in (mid + d, mid - d - 1):
            if 0 <= j < len(order) and order[j] not in picks:
                picks.append(order[j])
            if len(picks) >= n:
                return picks[:n]
    return picks


def main():
    ap = argparse.ArgumentParser(description="SurfDesign 单样本推理性能 + 峰值显存")
    ap.add_argument("--experiment", default="fixedbb/surf_design_esm2_650m")
    ap.add_argument("--experiment-path", default=None, help=f"默认 <repo>/{_DEFAULT_EXP}")
    ap.add_argument("--ckpt", default="best.remap.ckpt", help="key 对齐后的 ckpt（见 remap_ckpt_keys.py）")
    ap.add_argument("--device", default=None, help="默认自动探测（sdaa > cuda > cpu）")
    ap.add_argument("--max-iter", type=int, default=None, help="不传则用实验配置里的值（5）")
    ap.add_argument("--n", type=int, default=3, help="测几条样本")
    ap.add_argument("--repeat", type=int, default=5, help="每条样本重复次数（报中位数）")
    ap.add_argument("--warmup", type=int, default=1, help="每条样本预热次数")
    args = ap.parse_args()

    repo = _REPO
    os.environ.setdefault("PROJECT_ROOT", repo)
    sys.path.insert(0, repo)
    os.chdir(repo)

    exp_dir = os.path.abspath(args.experiment_path or os.path.join(repo, _DEFAULT_EXP))
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(exp_dir, "checkpoints", args.ckpt)
    if not os.path.exists(ckpt):
        sys.exit(f"找不到 checkpoint: {ckpt}")

    # utils.parse_cli_override_args() 内部读 sys.argv，本脚本自己的参数得先摘掉
    sys.argv = [sys.argv[0]]

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from src import utils

    with initialize_config_dir(config_dir=os.path.join(repo, "configs"), version_base=None):
        cfg = compose(config_name="test.yaml",
                      overrides=[f"experiment={args.experiment}", f"experiment_path={exp_dir}", "ckpt_path=best.ckpt"])
    cfg = utils.resolve_experiment_config(cfg)  # 内部会 os.chdir(experiment_path)
    cfg = utils.extras(cfg)

    if args.max_iter is not None:
        OmegaConf.update(cfg, "task.generator.max_iter", args.max_iter)
    OmegaConf.update(cfg, "datamodule.num_workers", 0)  # 本脚本自己取 batch，不走 DataLoader worker
    max_iter = cfg.task.generator.max_iter

    dev = args.device or utils.accelerator.device_type()

    print("=" * 72)
    print(f"experiment_dir : {exp_dir}")
    print(f"ckpt           : {ckpt}")
    print(f"torch          : {torch.__version__}")
    print(f"device         : {dev}")
    # generator 段只有 max_iter / strategy（没有 temperature），逐键打印避免 KeyError
    _gen = cfg.task.generator
    _gen_s = ", ".join(f"{k}={v}" for k, v in _gen.items()) if hasattr(_gen, "items") else str(_gen)
    print(f"generator      : {_gen_s}")
    print(f"batch          : batch_size=1（单样本口径，直接调 predict_step）")
    print(f"surf_lmdb      : {cfg.datamodule.alphabet.featurizer_cfg.surf_path}")
    print(f"device mem api : {mem_apis(dev)}")
    print("=" * 72)

    t0 = time.time()
    datamodule = utils.instantiate_from_config(cfg=cfg.datamodule, group="datamodule")
    pl_module = utils.instantiate_from_config(cfg=cfg.task, group="task", model=cfg.model)
    n_params = sum(p.numel() for p in pl_module.parameters())
    print(f"[1] 构件 datamodule + CMLM     : {time.time() - t0:.1f}s   参数量 {n_params / 1e6:.1f}M")

    t0 = time.time()
    pl_module.load_from_ckpt(ckpt)
    # 顺序有讲究：build_torchmetric() 建的 MeanMetric 是 nn.Module，必须夹在 .to(dev) **之前**，
    # 否则指标留在 CPU、predict_step 里 self.acc.update(..., cuda 张量) 直接 device mismatch。
    # test.py 里由 Trainer 保证这个顺序（setup 之后才把模型搬设备）。
    pl_module.build_criterion()
    pl_module.build_torchmetric()
    pl_module = pl_module.to(dev).eval()
    pl_module._stage = "test"
    print(f"[2] 载 ckpt + 搬 {dev}          : {time.time() - t0:.1f}s   param device = {next(pl_module.parameters()).device}")

    t0 = time.time()
    datamodule.setup("test")
    ds = datamodule.test_dataset
    idx = pick_samples(ds, args.n)
    print(f"[3] datamodule.setup + 定位样本 : {time.time() - t0:.1f}s   dataset={len(ds)} 条，选中 {len(idx)} 条")

    if not idx:
        sys.exit("test_dataset 为空")

    _reset_peak(dev)
    rows = []
    # log_metrics 必须为 True（即默认值）：CMLM.predict_step 里 recovery_acc_per_sample 只在
    # log_metrics 为真时才算，但紧接着无条件塞进 results -> 传 False 会 UnboundLocalError。
    # 指标本身很轻（一次比较 + 一次 update），对计时没有可观测影响。
    for i in idx:
        # 每条样本单独成一个 batch（真正的单样本口径）
        batch = to_device(datamodule.collate_batch([ds[i]]), dev)
        # forward() 内部会 pop('tokens')，重复计时时得补回去
        tokens = batch["tokens"]
        name = batch["names"][0]
        length = int(batch["coord_mask"][0].sum())
        for _ in range(args.warmup):
            batch["tokens"] = tokens
            with torch.inference_mode():
                pl_module.predict_step(batch, 0, log_metrics=True)
        _sync(dev)
        ts = []
        for _ in range(args.repeat):
            batch["tokens"] = tokens
            _sync(dev)
            t1 = time.perf_counter()
            with torch.inference_mode():
                pl_module.predict_step(batch, 0, log_metrics=True)
            _sync(dev)
            ts.append(time.perf_counter() - t1)
        med = statistics.median(ts)
        rows.append((name, length, med, min(ts), max(ts)))
        print(f"    {name}  L={length}  中位 {med * 1e3:.1f} ms  (min {min(ts) * 1e3:.1f} / max {max(ts) * 1e3:.1f}, n={args.repeat})")

    peak = _peak_mem(dev)
    print("=" * 72)
    print(f"单样本延迟中位数 : {statistics.median([r[2] for r in rows]) * 1e3:.1f} ms  (n={len(rows)} 条样本)")
    print(f"峰值显存（{dev}） : {peak / 1024 ** 3:.2f} GiB")
    print("=" * 72)
    print("| 样本 | 长度 L | 中位延迟 ms | min ms | max ms |")
    print("|---|---|---|---|---|")
    for name, length, med, lo, hi in rows:
        print(f"| {name} | {length} | {med * 1e3:.1f} | {lo * 1e3:.1f} | {hi * 1e3:.1f} |")


if __name__ == "__main__":
    main()
