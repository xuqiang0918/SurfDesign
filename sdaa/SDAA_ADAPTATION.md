# SurfDesign SDAA 适配说明

- 适配对象：SurfDesign（KDD'25，`esm2_650m` 固定骨架表面序列设计）
- base commit：`9ce30392`
- 分支：`xuqiang0918/SurfDesign` → `adapt/sdaa`
- SDAA：`10.71.15.115` / 容器 `sdaa`（镜像 `3.2.1-torch_sdaa3.2.1`）/ loongarch64 / 32 卡
- 对照：`10.10.6.21` / A100-PCIE-40GB

## 1. 结论

- 官方推理入口 `test.py --mode=predict` 在 SDAA 上可直接运行，权重加载 `0 missing / 0 unexpected`
- 精度与 CUDA 一致：`acc_median` 差 0.014 / 0.008，`acc_worst` 完全相同（§6）
- 单样本延迟 20.1×（133.2 ms → 2683.5 ms），峰值显存相当（2.56 / 2.58 GiB）（§5）
- 改动 9 个跟踪文件（+110 / −30）+ 新增 `src/utils/accelerator.py`（150 行），拆两笔提交
- 未做任何计算组件替换；未验证：`eval_sc=True`（需 ESMFold）、`train.py` 路径（本次只验证推理）

## 2. 环境（SDAA 侧）

| 项 | 值 |
|---|---|
| 主机 / 容器 | `10.71.15.115`，`sdaa` |
| 架构 / 卡 | loongarch64 / SDAA × 32 |
| Python | `/home/py312/bin/python3`（3.12.13） |
| PyTorch / PyG / PL | 2.12.0（Torch-SDAA 3.2.1）/ 2.8.0.post1 / 2.6.5 |
| lmdb | 2.3.0 |

依赖（已就位）：`torch_cluster` `torch_scatter` `torch_sparse` `tmtools` `trimesh` `pymol2`
`sklearn` `hydra` `omegaconf` `rootutils` `torchmetrics` `colorlog` `einops`

**平台注意**：`source setvars.sh` 要在 `set -u` 关闭状态下执行 —— §4 的封装脚本已内置。

ESM-2 权重走 `torch.hub.load_state_dict_from_url` 下**权重文件**到 `~/.cache/torch/hub/checkpoints/`
（不是拉 GitHub 仓库），该目录就位即**完全离线可用**。

## 3. 改动清单

### 3.1 `[sdaa-adapt]` 平台适配

| 文件 | 改动 |
|---|---|
| `src/utils/accelerator.py` | 新增（150 行）。设备判断收口到一处：`sdaa_available()` / `device_type()` / `SDAAAccelerator` / `accelerator_kwargs()` |
| `src/utils/__init__.py` | 转发 |
| `test.py` / `train.py` | `Trainer` 改用 `**utils.accelerator_kwargs()` |

`train.py` 是关键：Teco 的 torch 里 `torch.cuda` 存在但 `is_available()` 恒为 False，不改会静默落 CPU。
该函数 SDAA 上返回 `{"accelerator": SDAAAccelerator(), "devices": 1}`，CUDA 上返回 `{}`（`**` 展开等于没传，
上游行为不变）。必须传**实例**：PL `_choose_strategy` 的白名单只认 `CUDAAccelerator`/`MPSAccelerator`
或字符串 `"cuda"/"gpu"/"mps"`，传字符串 `"sdaa"` 会被静默改成 CPU。

### 3.2 `[upstream]` 通用兼容（与平台无关）

判据是**依赖版本**、不是显卡：上游代码锁在旧版本上（作者那年的 torch/PyG/PL/Python），
新版把这些 API 改名、删除或改了默认行为，换到 NVIDIA 新环境一样会撞，因此与平台适配分开归档。

| 文件 | 现象（版本门槛） | 修法 |
|---|---|---|
| `test.py` / `src/tasks/cmlm.py` | `torch.load` 拒载 ckpt（torch ≥ 2.6 默认按 `weights_only=True` 处理，实测 2.12 拒载） | `weights_only=False` |
| `src/tasks/__init__.py` | `on_test_epoch_end() missing 'outputs'`（PL ≥ 2.0 钩子不再带 outputs，实测签名 `(self) -> None`） | 重写 `on_prediction_mode()`：按实际存在的钩子选择性交换，自攒 `results` 注入 |
| `src/datamodules/datasets/cath.py` | `cannot import name 'one_to_three'`（Biopython 新版已移除；实测 SDAA 侧 1.88 无此名，CUDA 侧 1.79 仍有） | `try/except` 兜底，用 `index_to_three(one_to_index(x))` 等价实现 |
| `src/datamodules/datasets/cath.py` | `'NodeStorage' object has no attribute 'edge_index'`（PyG ≥ 2.6 transform 先 copy） | 接住 `self.kg(...)` 返回值 |
| `src/models/surf_design/egnn.py` | `'Inspector' object has no attribute 'distribute'`（PyG ≥ 2.6 改名 `collect_param_data`，实测两侧均 `distribute=False`） | `getattr` 兼容新旧名字 |
| `src/models/lm_design/esm2_adapter.py` | dataclasses 可变默认值报错（Python ≥ 3.11） | `field(default_factory=...)` |
| `scripts/design_pdb.py` | `ModuleNotFoundError`（与版本无关，上游自身入口 bug） | 修正 import 路径 |

两侧实测对照（2026-09-29）：PL 钩子无 outputs、PyG `Inspector.distribute` 已移除 —— 这两条在
A100 侧同样成立；`weights_only` 拒载、`one_to_three` 缺失、Python ≥ 3.11 只在 SDAA 侧成立
（那边 torch / Biopython / Python 更新）。同一份改动在 A100 侧也一并应用，因此归为通用兼容。

## 4. 运行

### 前置 1：surface 缓存（必需）

`data/cath_4.3/surface_pymol_bb.lmdb`，22401 条 / 7.94 GiB。缺失或 < 1 GB 时上游会自动重建
（单条实测约 1.2 s，`chain_set.jsonl` 全量 22508 条外推约 7.6 h）：

```bash
python sdaa/scripts/build_surface_lmdb.py --split all --workers 64
```

lmdb 页大小随架构固化（x86=4096 / loongarch64=16384），**不能直接拷**；已有源机缓存时可迁移：
`export_surface_lmdb.py`（源机导出）→ `import_surface_lmdb.py --verify`（目标机重建）。

### 前置 2：权重 remap（必需）

作者 ckpt 有 7 个键与当前 PyG 的命名规则不符，`strict=False` 下**静默丢弃**
（surface↔protein 交互卷积保持随机初始化、精度失真但不报错）：

```
原始 ckpt：  model.interact.convs.0.convs.surface__inter__protein.att
当前 PyG：   model.interact.convs.0.convs.<surface___inter___protein>.att
```

```bash
python sdaa/scripts/remap_ckpt_keys.py \
  --ckpt-in  logs/fixedbb/cath_4.3/surf_design_esm2_650m/checkpoints/best.ckpt \
  --ckpt-out logs/fixedbb/cath_4.3/surf_design_esm2_650m/checkpoints/best.remap.ckpt
```

（输出已存在时加 `--overwrite`。）remap 后 `0 missing / 0 unexpected`。
**CUDA 侧（PyG 2.6.1）同样要 remap**，本文件全部结果都跑在 `best.remap.ckpt` 上。

### 跑推理

```bash
TAG=sdaa CARD=0 MAXIT=5 bash sdaa/scripts/run_test_sdaa.sh
```

全量 test 1864 条约 85 min；加 `LIMIT=N` 只跑前 N 个 batch 冒烟。
脚本已内置必需参数（`experiment` / `experiment_path` / `ckpt_path=best.remap.ckpt` / `mode=predict`，
缺任一会报错），并处理容器、选卡与环境变量。

### 单样本性能测量

```bash
CARD=0 N=3 REPEAT=5 MAXIT=5 bash sdaa/scripts/run_bench_sdaa.sh    # SDAA
python sdaa/scripts/bench_single.py --max-iter 5 --n 3 --repeat 5   # CUDA
```

## 5. 性能

| 平台 | 输入 | 单样本延迟 | 峰值显存 | 参数量 |
|---|---|---|---|---|
| A100-PCIE-40GB × 1 | L=130 / bs=1 / max_iter=5 | 133.2 ms | 2.56 GiB | 656.3M |
| SDAA × 1 | 同上 | **2683.5 ms** | 2.58 GiB | 656.3M |

**20.1×**。三条样本 2679.3 / 2685.6 / 2683.5 ms。

## 6. 精度

口径：官方 `test.py --mode=predict`，`acc_median` = 逐样本复现率中位数。

| split | n | 平台 | `test/acc` | `acc_median` | `acc_worst` | 全量耗时 |
|---|---|---|---|---|---|---|
| test | 1864 | CUDA | 90.1646 | 90.3846 | 64.1791 | 180 s |
| test | 1864 | **SDAA** | 90.1672 | 90.3704 | **64.1791** | 5110 s |
| validation | 1516 | CUDA | 91.9577 | 92.2078 | 78.5714 | 145 s |
| validation | 1516 | **SDAA** | 91.9544 | **92.2156** | **78.5714** | 4413 s |

两侧 `acc_worst` 完全相同，`acc_median` 差 0.014 / 0.008。

ckpt 名里的 `92.21` 对应 **validation** split（配置里 `train.monitor: "val/acc_median"`），
不是 README 用的 test split —— 两个口径都对，只是分母不同。

## 7. 交付物

| 路径 | 内容 |
|---|---|
| `sdaa/SDAA_ADAPTATION.md` | 本文件 |
| `sdaa/patches/0001-upstream-compat-SurfDesign.patch` | `[upstream]`，6 文件 +105 / −27 |
| `sdaa/patches/0002-sdaa-adapt-SurfDesign.patch` | `[sdaa-adapt]`，4 文件 +155 / −3 |
| `sdaa/scripts/remap_ckpt_keys.py` | ckpt 键重映射（§4 前置 2） |
| `sdaa/scripts/build_surface_lmdb.py` / `merge_surface_lmdb.py` | surface 缓存生成与分片合并 |
| `sdaa/scripts/export_surface_lmdb.py` / `import_surface_lmdb.py` | 缓存跨架构迁移（`--verify` 逐条比对指纹） |
| `sdaa/scripts/make_exp_dir.py` | 由 configs 组合出官方入口要求的实验目录 |
| `sdaa/scripts/run_test_cuda.sh` / `run_test_sdaa.sh` | 两侧官方推理入口封装 |
| `sdaa/scripts/bench_single.py` / `run_bench_sdaa.sh` | 单样本延迟 + 峰值显存 |
| `sdaa/scripts/split_diff.py` | 把整仓 diff 拆成上面两个 patch（`--verify` 反向校验） |
| `sdaa/scripts/probe_upstream.py` | 探依赖的 API 状态，用于判定改动归 `[upstream]` 还是 `[sdaa-adapt]`（§3.2） |
