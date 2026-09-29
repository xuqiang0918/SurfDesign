"""从 configs/ 组合出官方推理入口所需的实验目录（含 .hydra/config.yaml）。

为什么需要它
------------
官方 ``test.py``（``utils.resolve_experiment_config``）与 ``Designer``
（``utils.load_from_experiment``）都不直接吃 configs/*.yaml，而是要求
``experiment_path`` 指向一个 **训练时留下的实验目录**：里面必须有
``.hydra/config.yaml``（hydra 在训练时自动落盘的完整配置），
checkpoint 则放在 ``checkpoints/`` 下。

我们手上只有单个 ``acc_median_92.21.ckpt``（来自作者训练目录
``.../logs/fixedbb/cath_4.3/surf_design_esm2_650m/checkpoints/``），
没有完整的实验目录。本脚本用 hydra 的 compose API 按与训练完全一致的
方式组合出该配置并落盘，从而让官方入口可以原样运行。

用法
----
    python sdaa/scripts/make_exp_dir.py \
        --experiment fixedbb/surf_design_esm2_650m \
        --datamodule cath_4.3
"""

from __future__ import annotations

import argparse
import os

from omegaconf import OmegaConf

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description="生成官方 test.py / Designer 需要的实验目录")
    ap.add_argument("--experiment", default="fixedbb/surf_design_esm2_650m")
    ap.add_argument("--datamodule", default="cath_4.3", help="留空则不覆盖 experiment 里的默认值")
    ap.add_argument("--name", default=None, help="实验名，默认 fixedbb/<datamodule>/<exp名>")
    ap.add_argument("--out", default=None, help="实验目录，默认 <repo>/logs/<name>")
    ap.add_argument("--project-root", default=None, help="默认 <repo>")
    args = ap.parse_args()

    project_root = os.path.abspath(args.project_root or _REPO_ROOT)
    os.environ["PROJECT_ROOT"] = project_root

    exp_short = args.experiment.split("/")[-1]
    name = args.name or f"fixedbb/{args.datamodule}/{exp_short}"
    out_dir = os.path.abspath(args.out or os.path.join(project_root, "logs", name))

    # 注意：不要传 "logger=null" —— hydra 会把它解析成 NoneType 并抛
    # ValueError: Config group override must be a string or a list.
    # configs/config.yaml 的 defaults 里 logger 已经是 null，无需覆盖。
    overrides = [
        f"experiment={args.experiment}",
        f"name={name}",
    ]
    if args.datamodule:
        overrides.append(f"datamodule={args.datamodule}")

    # hydra 需要 cwd 在能解析 config_path 的地方；用 initialize_config_dir 显式指定
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.join(project_root, "configs"), version_base=None):
        cfg = compose(config_name="config.yaml", overrides=overrides)

    OmegaConf.resolve(cfg)

    hydra_dir = os.path.join(out_dir, ".hydra")
    os.makedirs(hydra_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    cfg_path = os.path.join(hydra_dir, "config.yaml")
    OmegaConf.save(cfg, cfg_path)

    print(f"[exp] 实验目录   : {out_dir}")
    print(f"[exp] 配置已写入 : {cfg_path}")
    print(f"[exp] checkpoint : {ckpt_dir}/")
    print(f"[exp] name       : {cfg.name}")
    print(f"[exp] data_dir   : {cfg.datamodule.data_dir}")
    print(f"[exp] surf_path  : {cfg.datamodule.alphabet.featurizer_cfg.surf_path}")
    print(f"[exp] task       : {cfg.task._target_}")
    print(f"[exp] model      : {cfg.model._target_}")
    print(f"[exp] monitor    : {cfg.train.monitor} ({cfg.train.mode})")


if __name__ == "__main__":
    main()
