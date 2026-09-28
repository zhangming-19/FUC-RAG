#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/feiyang/xle/gemini/code/ParamMute"
ENV_DIR="/home/feiyang/xle/gemini/envs/parammute"

PY="$ENV_DIR/bin/python"

MODEL="$ROOT/outputs/dkpo/dkpo_rgdu_native19to26_bs32_cd000_rgdu010_gate1"
EVALUATOR="$ROOT/src/3_evaluate/eval_CoConflictQA_dkpo_seeded.py"
DATA_DIR="$ROOT/data/CoConflictQA/test"

EXP_NAME="rgdu_infer_ratio_sweep_seed42_strict_20260905"

OUT_ROOT="$ROOT/outputs/dkpo/unified_eval/generation/$EXP_NAME"
LOG_ROOT="$ROOT/logs/dkpo/unified_eval/$EXP_NAME"

SEED=42

LAYERS=(19 20 21 22 23 24 25 26)

RATIOS=(
  0.0
  0.1
  0.2
  0.25
  0.3
  0.4
  0.5
  0.6
  0.7
  0.8
  0.9
  1.0
)

DATASETS=(
  "NaturalQuestionsShort:$DATA_DIR/NaturalQuestionsShort_kc.jsonl"
  "NewsQA:$DATA_DIR/NewsQA_kc.jsonl"
  "SQuAD:$DATA_DIR/SQuAD_kc.jsonl"
  "SearchQA:$DATA_DIR/SearchQA_kc.jsonl"
  "TriviaQA-web:$DATA_DIR/TriviaQA-web_kc.jsonl"
  "hotpotq:$DATA_DIR/hotpotq_kc.jsonl"
)

cd "$ROOT"

export PYTHONPATH="$ROOT/src/transformers/src:$ROOT/src/2_tuning${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

# ------------------------------------------------------------
# Preflight
# ------------------------------------------------------------

if [ ! -x "$PY" ]; then
    echo "[ERROR] Python not found: $PY"
    exit 1
fi

if [ ! -s "$MODEL/adapter_model.safetensors" ]; then
    echo "[ERROR] RGDU adapter not found:"
    echo "$MODEL/adapter_model.safetensors"
    exit 1
fi

if [ ! -s "$MODEL/adapter_config.json" ]; then
    echo "[ERROR] RGDU adapter_config.json not found"
    exit 1
fi

if [ ! -s "$EVALUATOR" ]; then
    echo "[ERROR] evaluator not found: $EVALUATOR"
    exit 1
fi

if ! grep -Fq -- '--seed' "$EVALUATOR"; then
    echo "[ERROR] evaluator does not expose --seed"
    exit 1
fi

if ! grep -Fq 'SEEDED GENERATION' "$EVALUATOR"; then
    echo "[ERROR] evaluator does not contain seeded-generation marker"
    exit 1
fi

for item in "${DATASETS[@]}"; do
    IFS=: read -r name data <<< "$item"
    if [ ! -s "$data" ]; then
        echo "[ERROR] missing dataset: $data"
        exit 1
    fi
done

# ------------------------------------------------------------
# Experiment configuration record
# ------------------------------------------------------------

cat > "$OUT_ROOT/experiment_config.txt" <<EOF
experiment=$EXP_NAME
model=$MODEL
model_type=RGDU-DKPO_PEFT
base_model=Meta-Llama-3-8B-Instruct

evaluator=$EVALUATOR

seed=$SEED

schema=base
use_chat_template=True
max_new_tokens=32

do_sample=True
temperature=0.6
top_p=0.9

act_inhibit_layers=19,20,21,22,23,24,25,26

ratios=0.0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0

datasets=NaturalQuestionsShort,NewsQA,SQuAD,SearchQA,TriviaQA-web,hotpotq

selection_metric=MR_min
total_tasks=72
EOF

echo "============================================================"
echo "RGDU-DKPO INFERENCE RATIO SWEEP"
echo "============================================================"
echo "MODEL=$MODEL"
echo "SEED=$SEED"
echo "OUT_ROOT=$OUT_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "RATIOS=${RATIOS[*]}"
echo "LAYERS=${LAYERS[*]}"
echo "TOTAL_TASKS=72"
echo "============================================================"

ratio_tag() {
    local ratio="$1"
    echo "${ratio//./p}"
}

run_one() {

    local gpu="$1"
    local name="$2"
    local data="$3"
    local ratio="$4"

    local tag
    tag="$(ratio_tag "$ratio")"

    local out_dir="$OUT_ROOT/ratio_$tag"
    local log_dir="$LOG_ROOT/ratio_$tag"

    mkdir -p "$out_dir" "$log_dir"

    local result="$out_dir/${name}_res.json"
    local eval_log="$out_dir/${name}.log"
    local stdout_log="$log_dir/${name}.stdout.log"

    # Safe resume:
    # only skips completed jobs from THIS fresh experiment namespace.
    if [ -s "$result" ] &&
       [ -s "$eval_log" ] &&
       grep -q "final evaluation" "$eval_log"; then

        echo "[SKIP COMPLETE] GPU=$gpu ratio=$ratio dataset=$name"
        return 0
    fi

    echo
    echo "------------------------------------------------------------"
    echo "[START]"
    echo "GPU=$gpu"
    echo "ratio=$ratio"
    echo "dataset=$name"
    echo "data=$data"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES="$gpu" \
    "$PY" -u "$EVALUATOR" \
        --model_name "$MODEL" \
        --data_path "$data" \
        --schema base \
        --output_path "$result" \
        --log_path "$eval_log" \
        --use_chat_template True \
        --max_new_tokens 32 \
        --seed "$SEED" \
        --act_inhibit_ratio "$ratio" \
        --act_inhibit_layer_list "${LAYERS[@]}" \
        > "$stdout_log" 2>&1

    if [ ! -s "$result" ]; then
        echo "[ERROR] result missing: $result"
        return 1
    fi

    if [ ! -s "$eval_log" ]; then
        echo "[ERROR] eval log missing: $eval_log"
        return 1
    fi

    if ! grep -q "final evaluation" "$eval_log"; then
        echo "[ERROR] final evaluation marker missing:"
        echo "$eval_log"
        return 1
    fi

    if ! grep -Fq \
       "SEEDED GENERATION: seed=42, do_sample=True, temperature=0.6, top_p=0.9" \
       "$stdout_log"; then

        echo "[ERROR] seeded-generation marker missing:"
        echo "$stdout_log"
        return 1
    fi

    if ! grep -Fq "PEFT adapter loaded successfully" "$stdout_log"; then
        echo "[ERROR] PEFT adapter load marker missing:"
        echo "$stdout_log"
        return 1
    fi

    echo "[DONE] GPU=$gpu ratio=$ratio dataset=$name"

    grep -E \
      'Step: [0-9]+: pc [-+0-9.eE]+, po [-+0-9.eE]+, mr [-+0-9.eE]+, em [-+0-9.eE]+\.' \
      "$eval_log" \
      | tail -1 || true
}

run_worker() {

    local gpu="$1"
    local parity="$2"
    local ratio="$3"

    local i
    local name
    local data

    for i in "${!DATASETS[@]}"; do

        if (( i % 2 != parity )); then
            continue
        fi

        IFS=: read -r name data <<< "${DATASETS[$i]}"

        run_one \
            "$gpu" \
            "$name" \
            "$data" \
            "$ratio"
    done
}

for ratio in "${RATIOS[@]}"; do

    echo
    echo "============================================================"
    echo "START RATIO=$ratio"
    echo "============================================================"

    run_worker 0 0 "$ratio" &
    PID0=$!

    run_worker 1 1 "$ratio" &
    PID1=$!

    STATUS0=0
    STATUS1=0

    wait "$PID0" || STATUS0=$?
    wait "$PID1" || STATUS1=$?

    if [ "$STATUS0" -ne 0 ] || [ "$STATUS1" -ne 0 ]; then

        echo
        echo "[ERROR] RATIO=$ratio worker failed"
        echo "GPU0_STATUS=$STATUS0"
        echo "GPU1_STATUS=$STATUS1"
        exit 1
    fi

    COMPLETED_RESULTS="$(
        find "$OUT_ROOT" \
            -type f \
            -name '*_res.json' \
            | wc -l
    )"

    echo
    echo "============================================================"
    echo "DONE RATIO=$ratio"
    echo "COMPLETED_RESULTS=$COMPLETED_RESULTS / 72"
    echo "============================================================"
done

RESULT_COUNT="$(
    find "$OUT_ROOT" \
        -type f \
        -name '*_res.json' \
        | wc -l
)"

FINAL_LOG_COUNT="$(
    grep -Rl \
        "final evaluation" \
        "$OUT_ROOT" \
        --include='*.log' \
        2>/dev/null \
        | wc -l
)"

SEED_MARKER_COUNT="$(
    grep -Rl \
        "SEEDED GENERATION: seed=42, do_sample=True, temperature=0.6, top_p=0.9" \
        "$LOG_ROOT" \
        --include='*.stdout.log' \
        2>/dev/null \
        | wc -l
)"

ADAPTER_LOAD_COUNT="$(
    grep -Rl \
        "PEFT adapter loaded successfully" \
        "$LOG_ROOT" \
        --include='*.stdout.log' \
        2>/dev/null \
        | wc -l
)"

echo
echo "============================================================"
echo "FINAL AUDIT"
echo "============================================================"
echo "RESULT_JSON=$RESULT_COUNT / 72"
echo "FINAL_EVAL_LOG=$FINAL_LOG_COUNT / 72"
echo "SEEDED_GENERATION=$SEED_MARKER_COUNT / 72"
echo "PEFT_ADAPTER_LOAD=$ADAPTER_LOAD_COUNT / 72"
echo "============================================================"

if [ "$RESULT_COUNT" -ne 72 ] ||
   [ "$FINAL_LOG_COUNT" -ne 72 ] ||
   [ "$SEED_MARKER_COUNT" -ne 72 ] ||
   [ "$ADAPTER_LOAD_COUNT" -ne 72 ]; then

    echo "RGDU INFERENCE RATIO SWEEP: INCOMPLETE"
    exit 1
fi

echo
echo "============================================================"
echo "RGDU INFERENCE RATIO SWEEP: PASS"
echo "ALL 72 TASKS FINISHED"
echo "============================================================"
