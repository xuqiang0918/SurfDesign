#!/usr/bin/env python
"""按关注点把整仓 diff 拆成两个 patch（不碰 git index）。

背景
----
SurfDesign 的适配改动混着两类关注点：

- ``[upstream]``：上游代码锁在旧依赖版本上导致的通用兼容性修复
  （torch>=2.6 的 ``weights_only``、PL>=2.0 的钩子重命名、PyG>=2.6 的
  transform 返回值/Inspector 改名、Python>=3.11 的 dataclass、Biopython>=1.80）
  —— **与平台无关**，在 CUDA 与 SDAA 两侧都会触发，可以单独归档成 upstream patch。
- ``[sdaa-adapt]``：真正的平台适配（新增 ``src/utils/accelerator.py``、
  把 ``torch.cuda.*`` 换成抽象后端、Trainer 走 ``accelerator_kwargs()``）。

难点是 ``test.py`` **同时含两类改动**（两处相隔很远的 hunk），
所以不能按文件切，得**按 hunk 切**：hunk 里出现 ``accelerator_kwargs``
或 ``[sdaa-adapt]`` 就归 sdaa，否则归 upstream。

为什么不直接 ``git format-patch``
--------------------------------
format-patch 要求先有 commit，而按关注点拆 commit 需要 ``git add -p`` 交互式
挑 hunk。本脚本先把两个 patch 生成好，随后的两次 commit 就能直接
``git apply`` 对应 patch 后提交，等价于逐 hunk 挑选但完全可复现。

用法
----
    python sdaa/scripts/split_diff.py --repo <repo>
    python sdaa/scripts/split_diff.py --repo <repo> --verify   # 额外校验两个 patch 能反向应用

输出
----
    <repo>/sdaa/patches/0001-upstream-compat-SurfDesign.patch
    <repo>/sdaa/patches/0002-sdaa-adapt-SurfDesign.patch
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

# --- 按文件归类；'mixed' 表示该文件要按 hunk 逐个判断 -------------------------
GROUP = {
    "src/utils/accelerator.py": "sdaa",      # 新增文件
    "src/utils/__init__.py": "sdaa",
    "train.py": "sdaa",
    "test.py": "mixed",
}
DEFAULT_GROUP = "upstream"
SDAA_HUNK_HINTS = ("accelerator_kwargs", "[sdaa-adapt]", "accelerator.py")

P1_NAME = "0001-upstream-compat-SurfDesign.patch"
P2_NAME = "0002-sdaa-adapt-SurfDesign.patch"


def run(cwd: str, *args: str) -> str:
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"命令失败: {' '.join(args)}\n{r.stderr}")
    return r.stdout


def parse_diff(diff_text: str):
    """把 unified diff 切成 [(header_lines, file_path, [hunks])]。"""
    out, header, path, hunks, cur = [], [], None, [], None
    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if path is not None:
                if cur:
                    hunks.append(cur)
                out.append((header, path, hunks))
            m = re.match(r"diff --git a/(.+?) b/(.+)\n?$", line)
            header, path, hunks, cur = [line], (m.group(2) if m else None), [], None
        elif line.startswith("@@"):
            if cur:
                hunks.append(cur)
            cur = [line]
        elif cur is not None:
            cur.append(line)
        else:
            header.append(line)
    if path is not None:
        if cur:
            hunks.append(cur)
        out.append((header, path, hunks))
    return out


def new_file_diff(repo: str, rel: str) -> tuple:
    """为一个未跟踪文件造出标准的「new file」unified diff。"""
    abs_path = os.path.join(repo, rel)
    with open(abs_path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    header = [
        f"diff --git a/{rel} b/{rel}\n",
        "new file mode 100644\n",
        "index 0000000..0000000\n",
        "--- /dev/null\n",
        f"+++ b/{rel}\n",
    ]
    hunk = [f"@@ -0,0 +1,{len(lines)} @@\n"] + [f"+{l}\n" for l in lines]
    return header, rel, [hunk]


def count(hunks) -> tuple:
    add = sum(1 for h in hunks for l in h if l.startswith("+") and not l.startswith("+++"))
    rem = sum(1 for h in hunks for l in h if l.startswith("-") and not l.startswith("---"))
    return add, rem


def render(chunks) -> str:
    buf = []
    for header, _path, hunks in chunks:
        buf.extend(header)
        for h in hunks:
            buf.extend(h)
    return "".join(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--outdir", default=None, help="默认 <repo>/sdaa/patches")
    ap.add_argument("--new-files", default="src/utils/accelerator.py",
                    help="逗号分隔的未跟踪但需纳入 patch 的文件")
    ap.add_argument("--verify", action="store_true",
                    help="生成后用 git apply --reverse --check 校验（工作区已打上这些改动时应当通过）")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    outdir = args.outdir or os.path.join(repo, "sdaa", "patches")
    os.makedirs(outdir, exist_ok=True)

    files = parse_diff(run(repo, "git", "diff"))

    # 补充未跟踪的新文件（git diff 默认不含）
    tracked = {p for _h, p, _k in files}
    for rel in [s.strip() for s in args.new_files.split(",") if s.strip()]:
        if rel in tracked:
            continue
        if not os.path.exists(os.path.join(repo, rel)):
            print(f"[warn] 跳过不存在的文件: {rel}")
            continue
        files.append(new_file_diff(repo, rel))

    sdaa_chunks, up_chunks = [], []
    detail = []
    for header, path, hunks in files:
        grp = GROUP.get(path, DEFAULT_GROUP)
        if grp == "mixed":
            for h in hunks:
                blob = "".join(h)
                dst = "sdaa" if any(k in blob for k in SDAA_HUNK_HINTS) else "upstream"
                (sdaa_chunks if dst == "sdaa" else up_chunks).append((header, path, [h]))
                a, r = count([h])
                detail.append((f"{path}  [hunk] -> {dst}", a, r))
        else:
            (sdaa_chunks if grp == "sdaa" else up_chunks).append((header, path, hunks))
            a, r = count(hunks)
            detail.append((f"{path}  [{grp}]", a, r))

    p_up = os.path.join(outdir, P1_NAME)
    p_sdaa = os.path.join(outdir, P2_NAME)
    with open(p_up, "w", encoding="utf-8", newline="") as fh:
        fh.write(render(up_chunks))
    with open(p_sdaa, "w", encoding="utf-8", newline="") as fh:
        fh.write(render(sdaa_chunks))

    print("=== 归类明细 ===")
    for name, a, r in detail:
        print(f"  {name:<52} +{a} -{r}")
    for tag, path in (("upstream", p_up), ("sdaa-adapt", p_sdaa)):
        a, r = count([h for _hd, _p, hk in (up_chunks if tag == "upstream" else sdaa_chunks) for h in hk])
        print(f"=== {tag}: {os.path.relpath(path, repo)}  +{a} -{r} ===")

    if args.verify:
        ok = True
        for tag, path in (("upstream", p_up), ("sdaa-adapt", p_sdaa)):
            r = subprocess.run(["git", "apply", "--reverse", "--check", path],
                               cwd=repo, capture_output=True, text=True)
            status = "OK（可反向应用 = 工作区已含这些改动）" if r.returncode == 0 else f"FAIL: {r.stderr.strip()}"
            print(f"[verify] {tag}: {status}")
            ok = ok and r.returncode == 0
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
