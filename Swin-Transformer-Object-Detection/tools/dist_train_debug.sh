#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

#TORCH_DISTRIBUTED_DEBUG="DETAIL" PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#/opt/conda/bin/python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
/opt/conda/bin/python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}