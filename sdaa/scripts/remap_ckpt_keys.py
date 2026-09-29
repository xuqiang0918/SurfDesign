"""把 ckpt 里 PyG ``HeteroConv`` 的子模块名对齐到当前 PyG 的命名。

背景（实测依据）
----------------
PyG 的 ``HeteroConv`` 用 ``torch_geometric.nn.ModuleDict`` 存各 edge type 的子模块，
key 由 ``ModuleDict.to_internal_key()`` 生成::

    torch_geometric/nn/module_dict.py:31
        key = f"<{'___'.join(key)}>"

即 edge type ``('surface', 'inter', 'protein')`` 落成 ``<surface___inter___protein>``。
更早的 PyG 走 ``edge_type_to_str`` = ``'__'.join`` -> ``surface__inter__protein``。

作者的 ckpt 是旧命名训练的。把新 PyG 装出来的模型直接
``load_state_dict(..., strict=False)`` 会出现 7 个 missing + 7 个 unexpected
（实测 PyG 2.6.1 与 2.8.0 表现一致）::

    missing    model.interact.convs.0.convs.<surface___inter___protein>.att
    unexpected model.interact.convs.0.convs.surface__inter__protein.att

``strict=False`` 下这批权重被**静默丢弃**，表面-主链交互卷积保持随机初始化，
精度指标会因此失真（而日志里只有一行 print，很容易被忽略）。
本脚本只改 key 名，其余字段一字不动。

改名映射由**模型自己的 state_dict key 集合**推导（去掉 ``<>`` / ``___`` 归一化后比对），
逐个校验 shape 一致，写完复跑一次 ``load_state_dict`` 要求 missing/unexpected 都为 0。

用法
----
    python sdaa/scripts/remap_ckpt_keys.py --ckpt-in <原始.ckpt> --ckpt-out <对齐后.ckpt>
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_EXP = "logs/fixedbb/cath_4.3/surf_design_esm2_650m"


def _norm(key: str) -> str:
    """把新旧两种 HeteroConv 命名归一化到同一形式。

    ``<surface___inter___protein>`` 与 ``surface__inter__protein`` -> ``surface__inter__protein``
    """
    return key.replace("<", "").replace(">", "").replace("___", "__")


def build_model(repo: str, exp_dir: str, experiment: str):
    # configs/paths/default.yaml 用 ${oc.env:PROJECT_ROOT}，compose 阶段就会求值
    os.environ.setdefault("PROJECT_ROOT", repo)
    sys.path.insert(0, repo)
    os.chdir(repo)
    sys.argv = [sys.argv[0]]  # utils.parse_cli_override_args() 会读 sys.argv，先摘掉本脚本的参数
    from hydra import compose, initialize_config_dir

    from src import utils

    with initialize_config_dir(config_dir=os.path.join(repo, "configs"), version_base=None):
        cfg = compose(config_name="test.yaml",
                      overrides=[f"experiment={experiment}", f"experiment_path={exp_dir}", "ckpt_path=best.ckpt"])
    cfg = utils.resolve_experiment_config(cfg)
    cfg = utils.extras(cfg)
    return utils.instantiate_from_config(cfg=cfg.task, group="task", model=cfg.model)


def main():
    ap = argparse.ArgumentParser(description="对齐 ckpt 里的 PyG HeteroConv 子模块命名")
    ap.add_argument("--ckpt-in", required=True)
    ap.add_argument("--ckpt-out", required=True)
    ap.add_argument("--repo", default=_REPO)
    ap.add_argument("--experiment-path", default=None, help=f"默认 <repo>/{_DEFAULT_EXP}")
    ap.add_argument("--experiment", default="fixedbb/surf_design_esm2_650m")
    ap.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    args = ap.parse_args()

    ckpt_in = os.path.abspath(args.ckpt_in)
    ckpt_out = os.path.abspath(args.ckpt_out)
    if ckpt_in == ckpt_out:
        sys.exit("--ckpt-in 与 --ckpt-out 不能相同（原始 ckpt 保持不动）")
    if os.path.exists(ckpt_out) and not args.overwrite:
        sys.exit(f"输出已存在，加 --overwrite 才覆盖: {ckpt_out}")

    print("=" * 72)
    print(f"torch      : {torch.__version__}")
    print(f"torch_geometric: {__import__('torch_geometric').__version__}")
    print(f"ckpt in    : {ckpt_in}")
    print(f"ckpt out   : {ckpt_out}")
    print("=" * 72)

    exp_dir = os.path.abspath(args.experiment_path or os.path.join(args.repo, _DEFAULT_EXP))
    model = build_model(args.repo, exp_dir, args.experiment)
    model_keys = list(model.state_dict().keys())

    by_norm: dict = {}
    for k in model_keys:
        n = _norm(k)
        if n in by_norm and by_norm[n] != k:
            sys.exit(f"归一化冲突：{by_norm[n]} 与 {k} 归一化后相同，拒绝继续")
        by_norm[n] = k

    ck = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    model_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}

    new_sd, renamed, skipped = {}, [], []
    for k, v in sd.items():
        if k in model_shapes:
            new_sd[k] = v
            continue
        cand = by_norm.get(_norm(k))
        if cand is not None and tuple(v.shape) == model_shapes[cand]:
            new_sd[cand] = v
            renamed.append((k, cand))
        else:
            new_sd[k] = v
            skipped.append(k)

    print(f"\nstate_dict 条目 : {len(sd)} -> {len(new_sd)}")
    print(f"改名            : {len(renamed)} 条")
    for a, b in renamed:
        print(f"   {a}\n-> {b}")
    if skipped:
        print(f"未匹配（保持原名，会在下面报 unexpected）: {len(skipped)} 条")
        for k in skipped:
            print(f"   {k}")

    ck["state_dict"] = new_sd
    torch.save(ck, ckpt_out)
    print(f"\n[写盘] {ckpt_out}  {os.path.getsize(ckpt_out) / 1024 ** 3:.2f} GiB")

    # 复核：用同一份模型再加载一次，要求零 missing / 零 unexpected
    sd2 = torch.load(ckpt_out, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(sd2, strict=False)
    print(f"[复核] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    for k in missing:
        print("   missing   ", k)
    for k in unexpected:
        print("   unexpected", k)
    if missing or unexpected:
        sys.exit("对齐后仍有键不匹配，输出不可用")
    print("[复核] 全部 %d 个键匹配，shape 逐一相同。" % len(sd2))


if __name__ == "__main__":
    main()
