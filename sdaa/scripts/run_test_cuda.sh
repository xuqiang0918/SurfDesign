#!/bin/bash
# 6.21（CUDA 基线）跑官方推理入口 test.py。与 run_test_sdaa.sh 共用同一条 hydra 命令，只换平台环境。
#
# README 的命令无法直接用，有三处必须补齐（都不改代码，只在命令行补）：
#   1) configs/config.yaml 默认 experiment 指向不存在的文件 -> 必须传 experiment=
#   2) test.py 不像 train.py 那样调 rootutils.setup_root -> 必须手动 export PROJECT_ROOT
#      （configs/paths/default.yaml 用 ${oc.env:PROJECT_ROOT}，未设置直接报错）
#   3) ckpt 必须用 key 对齐过的 best.remap.ckpt —— 作者的 ckpt 是旧 PyG 命名，
#      直接加载会有 7 个键静默丢失（见 sdaa/scripts/remap_ckpt_keys.py）
#
# 环境变量（都有默认值）：
#   TAG=名字  MAXIT=生成迭代数(默认5，与实验配置一致)  CKPT=checkpoints 下的文件名
#   LIMIT=只跑前 N 个 batch（冒烟用，留空=全量）  EXTRA=追加的 hydra override
set -u
REPO=/data/application/xuqiang/SurfDesign
EXPNAME=logs/fixedbb/cath_4.3/surf_design_esm2_650m
PY=/data/application/xuqiang/anaconda3/envs/surf/bin/python

TAG="${TAG:-cuda}"
MAXIT="${MAXIT:-5}"
CKPT="${CKPT:-best.remap.ckpt}"
LIMIT="${LIMIT:-}"
SPLIT="${SPLIT:-test}"   # test | valid（README: data_split 支持 valid/test）
EXTRA="${EXTRA:-}"

cd "$REPO" || exit 1
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

LIMIT_OPT=""
# limit_test_batches 不在 configs/trainer/default.yaml 里，struct 模式下必须用 + 追加
if [ -n "$LIMIT" ]; then LIMIT_OPT="+trainer.limit_test_batches=$LIMIT"; fi
LOG=/tmp/test_cuda_${TAG}_$(date +%Y%m%d_%H%M%S).log

echo "[env] $(date '+%F %T') GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | head -1)"
echo "[run] ckpt=$CKPT max_iter=$MAXIT limit=${LIMIT:-all} extra='${EXTRA}'"
echo "[run] log -> $LOG"
T0=$(date +%s)
"$PY" test.py \
  experiment=fixedbb/surf_design_esm2_650m \
  experiment_path="$EXPNAME" \
  data_split="${SPLIT}" \
  ckpt_path="$CKPT" \
  task.generator.max_iter="$MAXIT" \
  mode=predict \
  $LIMIT_OPT $EXTRA 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
T1=$(date +%s)
echo "[run] exit=$RC elapsed=$((T1 - T0))s"

# 官方 on_predict_epoch_end 把逐样本结果写成 <temperature>.fasta，表头带每条的 AAR；
# 直接统计它即可复现 test/acc_median（temperature 默认 0）。
# 注意路径：CMLM.save_prediction 里是 open(f'./test_tau{tau}.fasta','w')，**相对 CWD**
# （即仓库根），不是实验目录。所以先从日志里抓 "Saving predictions to <abs path>"，
# 抓不到再退回仓库根。
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
