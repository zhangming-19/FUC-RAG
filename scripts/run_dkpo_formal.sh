#!/bin/bash

set -euo pipefail


if [ "$#" -ne 2 ]; then
    echo "Usage:"
    echo "  bash run_dkpo_formal.sh ETA EXP_NAME"
    echo
    echo "Example:"
    echo "  bash run_dkpo_formal.sh 0.1 dkpo_eta010"
    exit 1
fi


ETA="$1"
EXP_NAME="$2"


ROOT="/home/feiyang/xle/gemini"

PROJECT="$ROOT/code/ParamMute"

MODEL_PATH="$ROOT/models/Meta-Llama-3-8B-Instruct"

TRAIN_FILE="$ROOT/data/parammute/ParamMute-Training-Data/pip_kag_train_dkpo_v31.jsonl"


OUT_DIR="$PROJECT/outputs/dkpo/$EXP_NAME"

LOG_DIR="$PROJECT/logs/dkpo/$EXP_NAME"


cd "$PROJECT"


if [ -f "env_offline.sh" ]; then
    source env_offline.sh
fi


export CUDA_VISIBLE_DEVICES=0,1

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export TOKENIZERS_PARALLELISM=false


mkdir -p \
"$OUT_DIR" \
"$LOG_DIR"


echo "============================================================"
echo "D-KPO FORMAL TRAINING"
echo "============================================================"

echo "EXP_NAME=$EXP_NAME"
echo "ETA=$ETA"

echo
echo "MODEL_PATH=$MODEL_PATH"
echo "TRAIN_FILE=$TRAIN_FILE"

echo
echo "max_steps=2100"
echo "max_len=1024"

echo
echo "per_device_batch=1"
echo "gradient_accumulation=4"
echo "world_size=2"
echo "effective_batch=8"

echo
echo "lambda_train=0"

echo
echo "gamma_g=1"
echo "gamma_c=1"

echo
echo "alpha=0.5"
echo "beta=0.5"
echo "eta=$ETA"

echo "============================================================"


torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/2_tuning/train.py \
  --model_name_or_path "$MODEL_PATH" \
  --train_file "$TRAIN_FILE" \
  --max_len 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_steps 2100 \
  --save_steps 300 \
  --logging_steps 10 \
  --lr_scheduler_type cosine \
  --gradient_checkpointing False \
  --bf16 True \
  --warmup_ratio 0.1 \
  --weight_decay 0.00001 \
  --learning_rate 0.0001 \
  --output_dir "$OUT_DIR" \
  --overwrite_output_dir True \
  --train_mode dkpo_contrastive \
  --use_lora True \
  --model_type LlamaForInputContrastivew_act_inhibit \
  --report_to none \
  --logging_dir "$LOG_DIR" \
  --initial_margin 1 \
  --final_margin 1 \
  --alpha 0.5 \
  --beta 0.5 \
  --cd_margin 1 \
  --dkpo_eta "$ETA" \
  --ddp_find_unused_parameters False \
  --inhibit_strength 0.0 \
  --inhibit_layer_list \
    25 27 26 28 24 23 29 21

