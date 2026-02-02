#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models

# --- Check for minimum number of arguments ---
if [ "$#" -lt 5 ]; then
    echo "错误: 至少需要 5 个基本参数。"
    echo "用法: $0 [MODE] [MODEL] [DATASET] [GPU_DEVICE] [SAVE_ID] [TRAINING_ARGS...] [EXTRA_ARGS...]"
    exit 1
fi

# --- Read common arguments for all modes ---
MODE=$1
MODEL=$2
DATASET=$3
GPU_DEVICE=$4
SAVE_ID=$5

# --- Construct paths ---
FULL_DATA_PATH=$DATA_PATH/$DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"


# --- Main Logic based on MODE ---

if [ $MODE == "train" ]
then
    echo "Start Training......"

    # --- Check for minimum training arguments ---
    if [ "$#" -lt 15 ]; then
        echo "错误: 训练模式至少需要 15 个参数。"
        echo "用法: $0 train [MODEL] ... [PHASE_WEIGHT] [EXTRA_ARGS...]"
        exit 1
    fi

    # --- Read all known training arguments ---
    BATCH_SIZE=$6
    NEGATIVE_SAMPLE_SIZE=$7
    HIDDEN_DIM=$8
    GAMMA=$9
    ALPHA=${10}
    LEARNING_RATE=${11}
    MAX_STEPS=${12}
    TEST_BATCH_SIZE=${13}
    MODULUS_WEIGHT=${14}
    PHASE_WEIGHT=${15}

    # --- Isolate extra arguments (the core of the change) ---
    shift 15
    EXTRA_ARGS="$@"

    # --- Define model-specific arguments to avoid code duplication ---
    MODEL_SPECIFIC_ARGS=""
    if [ $MODEL == "HAKE" ]
    then
        MODEL_SPECIFIC_ARGS="-mw $MODULUS_WEIGHT -pw $PHASE_WEIGHT"
    elif [ $MODEL == "ModE" ]
    then
        # ModE does not use modulus_weight or phase_weight in the original script
        # So, this variable remains empty, which is correct.
        : # The colon is a no-op, just a placeholder.
    fi

    # --- Construct and run the single, unified training command ---
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/runs.py --do_train \
        --do_valid \
        --do_test \
        --data_path $FULL_DATA_PATH \
        --model $MODEL \
        -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
        -g $GAMMA -a $ALPHA \
        -lr $LEARNING_RATE --max_steps $MAX_STEPS \
        -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
        $MODEL_SPECIFIC_ARGS \
        $EXTRA_ARGS

elif [ $MODE == "valid" ]
then
    echo "Start Evaluation on Valid Data Set......"
    # Evaluation modes don't need extra arguments
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/runs.py --do_valid -init $SAVE

elif [ $MODE == "test" ]
then
    echo "Start Evaluation on Test Data Set......"
    # Evaluation modes don't need extra arguments
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/runs.py --do_test -init $SAVE

else
   echo "Unknown MODE: $MODE"
   exit 1
fi