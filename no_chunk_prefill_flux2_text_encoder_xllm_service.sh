#!/bin/bash
set -e

rm -rf core.*

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
#export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_IF_BASE_PORT=43432  # HCCL 通信基础端口

# # 4. 算子库日志
# export ASDOPS_LOG_TO_STDOUT=1
# #export ASDOPS_LOG_LEVEL=ERROR
# export ASDOPS_LOG_LEVEL=INFO
# export ASDOPS_LOG_TO_FILE=1

# # 5. 加速库日志
# export ATB_LOG_LEVEL=DEBUG
# export ATB_LOG_TO_FILE=1
# export ATB_LOG_TO_STDOUT=1

#export SPDLOG_LEVEL=debug
export ASCEND_MODULE_LOG_LEVEL=ATB=0
export ASDOPS_LOG_TO_FILE=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export MINDIE_LOG_LEVEL=DEBUG

#MODEL_PATH="/export/home/models/Mistral-Small-3.1-24B-Base-2503"               # 模型路径
MODEL_PATH="/export/home/models/flux2/text_encoder/"
MASTER_NODE_ADDR="127.0.0.1:9748"                  # Master 节点地址（需全局一致）
START_PORT=18001                                   # 服务起始端口
START_DEVICE=8                                    # 起始逻辑设备号
LOG_DIR="log"                                      # 日志目录
NNODES=2                                           # 节点数（当前脚本启动 1 个进程）

mkdir -p $LOG_DIR

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/mistral_node_$i.log"
  ./build/xllm/core/server/xllm \
    --model $MODEL_PATH \
    --devices="npu:$DEVICE" \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --max_memory_utilization=0.86 \
    --block_size=128 \
    --tp_size=2 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=true \
    --enable_return_mm_full_embeddings=true \
    --enable_mistral_prompt_to_message=true \
    --task="embed" \
    --enable_shm=true \
    --node_rank=$i \ > $LOG_FILE 2>&1 &
done

