#!/bin/bash
source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=3000
export LR=0.0005
export MINIBS=1
export TP=8
export PP=1
export CP=1
export FP8_ACT=1

# Частый validation
export VAL_CHECK_INTERVAL=50
export LIMIT_VAL_BATCHES=1.0

# Увеличенное время
export WALLTIME_RUNANDTIME=80  # 80 минут на тренировку
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=1
export DGXNGPU=8
