"""探测当前环境里，[upstream] 那几条兼容改动各自的触发条件是否成立。

用途：区分「换平台的必要改动」与「库版本升级导致的改动」，给文档里的分类提供实测依据。
"""
import inspect
import sys

print("python   ", sys.version.split()[0])
import torch

print("torch    ", torch.__version__)
import pytorch_lightning as pl

print("lightning", pl.__version__)
import torch_geometric as pyg

print("PyG      ", pyg.__version__)
import Bio

print("biopython", Bio.__version__)
print()

params = inspect.signature(torch.load).parameters
print("[1] torch.load weights_only 默认 =", params["weights_only"].default if "weights_only" in params else "无此参数")

print("[2] LightningModule.on_test_epoch_end 签名 =", inspect.signature(pl.LightningModule.on_test_epoch_end))

try:
    from Bio.SeqUtils import one_to_three

    print("[3] Bio.SeqUtils.one_to_three 可用 = True")
except Exception as exc:
    print("[3] Bio.SeqUtils.one_to_three 可用 = False (%s)" % type(exc).__name__)

try:
    from torch_geometric.inspector import Inspector as PygInspector

    print(
        "[4] PyG Inspector.distribute =",
        hasattr(PygInspector, "distribute"),
        "| collect_param_data =",
        hasattr(PygInspector, "collect_param_data"),
    )
except Exception as exc:
    print("[4] 无法导入 torch_geometric.inspector (%s)" % type(exc).__name__)

print("[5] Python >= 3.11（dataclasses 禁止可变默认值）= ", sys.version_info >= (3, 11))

# [6] 实测默认行为：拿一个含非张量对象的 ckpt 试读，验证 weights_only 的真实默认语义
_probe = "/tmp/_probe_upstream_ckpt.pth"
try:
    torch.save({"t": torch.zeros(2), "obj": object()}, _probe)
    try:
        torch.load(_probe)
        print("[6] 默认 torch.load 读非纯张量 ckpt = 可以（不会拒载）")
    except Exception as exc:
        print("[6] 默认 torch.load 读非纯张量 ckpt = 拒载 (%s)" % type(exc).__name__)
        print("     ->", str(exc).splitlines()[0][:110])
    finally:
        import os

        os.path.exists(_probe) and os.remove(_probe)
except Exception as exc:
    print("[6] 探测失败:", type(exc).__name__, exc)
