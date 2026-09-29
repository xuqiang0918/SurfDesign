#!/bin/bash
# 115（SDAA）单样本推理性能 + 峰值显存。与 run_test_sdaa.sh 同一套平台环境。
#
# 为什么单独一个脚本：bench_single.py 走「官方构件 + 直接调 predict_step」，
# 需要一个干净的平台环境（source setvars / SDAA_VISIBLE_DEVICES / USER / HOME），
# 这些从命令行传进去引号太深，封在这里更省事。
#
# 环境变量：CARD（默认 0）  N（样本数，默认 3）  REPEAT（默认 5）  MAXIT（默认 5）
set -u
REPO=/data/application/xuqiang/SurfDesign
CARD="${CARD:-0}"
N="${N:-3}"
REPEAT="${REPEAT:-5}"
MAXIT="${MAXIT:-5}"

echo "[bench] card=$CARD n=$N repeat=$REPEAT max_iter=$MAXIT"
docker exec -u 1007:1007 -e USER=xuqiang -e HOME=/data/application/xuqiang \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/ti_xq -e SDAA_VISIBLE_DEVICES="$CARD" sdaa bash -c "
set -u
# setvars.sh 不是 set -u 安全的（会引用未定义的 \$LD_LIBRARY_PATH），source 前后要关/开 nounset
set +u
source /opt/tecoai/setvars.sh >/dev/null 2>&1
set -u
cd $REPO
export PROJECT_ROOT=\$PWD
export PYTHONPATH=\$PWD
export HOME=/data/application/xuqiang
/home/py312/bin/python3 sdaa/scripts/bench_single.py --max-iter $MAXIT --n $N --repeat $REPEAT
RC=\$?
echo \"[bench] python exit=\$RC\"
exit \$RC
"
echo "[bench] docker exit=$?"
