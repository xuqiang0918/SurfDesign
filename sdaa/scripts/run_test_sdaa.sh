#!/bin/bash
# 115（SDAA）跑官方推理入口 test.py，hydra 命令与 run_test_cuda.sh 逐字一致，只换平台环境。
#
# 平台注意（SDAA 容器固定事实）：
# - 必须先 source /opt/tecoai/setvars.sh，否则 torch_sdaa 后端加载失败
# - Python 固定 /home/py312/bin/python3（3.12.13）；容器默认 /usr/bin/python3 是 3.11 且没有 torch
# - 可见卡变量是 SDAA_VISIBLE_DEVICES（CUDA_VISIBLE_DEVICES 在 SDAA 上完全无效）
# - HOME 必须指到 /data/application/xuqiang：容器里 HOME=/，否则 torch hub 找不到
#   ESM-2 权重、matplotlib 也会往只读的 /.config 写
#
# 环境变量：TAG / MAXIT / CKPT / LIMIT / EXTRA / CARD（见 run_test_cuda.sh）
set -u
REPO=/data/application/xuqiang/SurfDesign
EXPNAME=logs/fixedbb/cath_4.3/surf_design_esm2_650m
TAG="${TAG:-sdaa}"
MAXIT="${MAXIT:-5}"
CKPT="${CKPT:-best.remap.ckpt}"
LIMIT="${LIMIT:-}"
SPLIT="${SPLIT:-test}"   # test | valid（README: data_split 支持 valid/test）
EXTRA="${EXTRA:-}"
CARD="${CARD:-0}"
LOG=/tmp/test_sdaa_${TAG}_$(date +%Y%m%d_%H%M%S).log

LIMIT_OPT=""
# limit_test_batches 不在 configs/trainer/default.yaml 里，struct 模式下必须用 + 追加
if [ -n "$LIMIT" ]; then LIMIT_OPT="+trainer.limit_test_batches=$LIMIT"; fi

echo "[run] ckpt=$CKPT max_iter=$MAXIT limit=${LIMIT:-all} card=$CARD extra='${EXTRA}'"
echo "[run] log -> $LOG"
T0=$(date +%s)
docker exec -u 1007:1007 -e USER=xuqiang -e HOME=/data/application/xuqiang \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/ti_xq -e SDAA_VISIBLE_DEVICES="$CARD" sdaa bash -c "
set -u
# 注意：/opt/tecoai/setvars.sh 第 6 行会引用 \$LD_LIBRARY_PATH，而容器里该变量默认未定义，
# 在 set -u 下直接致命退出（配合 >/dev/null 2>&1 就变成「零输出 + rc=127 静默秒退」）。
# 所以 source 前后必须把 nounset 关掉再打开。
set +u
source /opt/tecoai/setvars.sh >/dev/null 2>&1
set -u
cd $REPO
export PROJECT_ROOT=\$PWD
export PYTHONPATH=\$PWD
export HOME=/data/application/xuqiang
echo \"[env] \$(date '+%F %T') teco-smi 卡数=\$(/opt/tecoai/bin/teco-smi 2>/dev/null | grep -c '^Index ') card=\$SDAA_VISIBLE_DEVICES\"
# 选卡前先看真实可用显存：这台机器上 torch.sdaa 报的单卡 total 只有 ~14.94 GiB（不是硬件标称 64 GB），
# 且机器共享，别人占了的卡照样显示 OK 但只剩几 GiB，直接跑会 OOM。
/home/py312/bin/python3 -c \"import torch; f,t=torch.sdaa.mem_get_info(); print('[env] SDAA free=%.2f GiB / total=%.2f GiB' % (f/1024**3, t/1024**3))\" 2>/dev/null
/home/py312/bin/python3 test.py \
  experiment=fixedbb/surf_design_esm2_650m \
  experiment_path=$EXPNAME \
  data_split="${SPLIT}" \
  ckpt_path=$CKPT \
  task.generator.max_iter=$MAXIT \
  mode=predict \
  $LIMIT_OPT $EXTRA
RC_PY=\$?
echo \"[env] python exit=\$RC_PY\"
exit \$RC_PY
" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
T1=$(date +%s)
echo "[run] exit=$RC elapsed=$((T1 - T0))s"

# 逐样本结果落在 open(f'./test_tau{tau}.fasta')，即**仓库根**（相对 CWD），不是实验目录。
# 优先从日志里抓 "Saving predictions to <abs path>"。
FASTA=$(grep -aoE 'Saving predictions to [^ ]+\.fasta' "$LOG" | tail -1 | sed 's/Saving predictions to //')
[ -z "$FASTA" ] && FASTA="$REPO/test_tau0.fasta"
echo "=== 落盘文件 ==="
ls -l "$REPO"/test_tau*.fasta "$REPO"/native.fasta 2>&1
if [ -f "$FASTA" ]; then
  echo "=== $FASTA 逐样本 AAR 统计 ==="
  grep -oE 'AAR=[0-9.]+' "$FASTA" | sed 's/AAR=//' | sort -n | awk '
    {a[NR]=$1; s+=$1}
    END {if (NR==0) {print "(空)"; exit}
         m = (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2
         printf "n=%d  mean=%.2f  median=%.2f  min=%.2f  max=%.2f\n", NR, s/NR, m, a[1], a[NR]}'
else
  echo "!! 没找到 $FASTA"
fi
echo "=== 日志里的指标行 ==="
grep -aoE 'test/(acc|acc_median|acc_worst|ppl|nll_loss|loss)[^0-9]{0,14}[0-9]+\.[0-9]+' "$LOG" | sort -u | tail -20

# 逐样本 fasta 的路径由 experiment_path 决定（不含 TAG），并行跑多个 split 时会互相覆盖。
# 跑完立刻按 TAG 落一份副本，保证并行也能各自留档。
if [ -f "$FASTA" ]; then
  cp -f "$FASTA" "/tmp/fasta_${TAG}.fa" && echo "[copy] /tmp/fasta_${TAG}.fa  ($(wc -l < "/tmp/fasta_${TAG}.fa") 行)"
fi
