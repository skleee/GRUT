#!/bin/bash
# Usage: bash test_grut_test.sh -d Beauty -e 30 -g 0 -l 0.3 -c /path/to/log/Beauty/run_dir
while getopts d:e:g:l:c: opt; do
    case $opt in
        d) DATASET=$OPTARG ;;
        e) EPOCH_NUM=$OPTARG ;;
        g) GPU_NUM=$OPTARG ;;
        l) LAMBDA=$OPTARG ;;
        c) CHECKPOINT_DIR=$OPTARG ;;
    esac
done

# Per-dataset hyperparams — same as tune_valid
ITEM_ID_TYPE=split
if [ "${DATASET}" = "Beauty" ]; then
    NUM_CLUSTER=128; ID_LEN=7
    EPSILON=0.01; TOP_K=1; LAST_N=5; TAU=128; C_MIN=0.8; DAYS=7
    ITEM_ID=hierarchy_v1_c${NUM_CLUSTER}_l${ID_LEN}_len32768_split
elif [ "${DATASET}" = "Toys" ]; then
    NUM_CLUSTER=32; ID_LEN=5
    EPSILON=0.01; TOP_K=1; LAST_N=2; TAU=1024; C_MIN=0.8; DAYS=30
    ITEM_ID=hierarchy_v1_c${NUM_CLUSTER}_l${ID_LEN}_len32768_split
elif [ "${DATASET}" = "Sports" ]; then
    NUM_CLUSTER=32; ID_LEN=7
    EPSILON=0.001; TOP_K=1; LAST_N=3; TAU=128; C_MIN=0.9; DAYS=30
    ITEM_ID=hierarchy_v1_c${NUM_CLUSTER}_l${ID_LEN}_len32768_split
elif [ "${DATASET}" = "Yelp" ]; then
    NUM_CLUSTER=32; ID_LEN=9
    EPSILON=0.01; TOP_K=1; LAST_N=3; TAU=256; C_MIN=0.8; DAYS=30
    ITEM_ID=hierarchy_v1_c${NUM_CLUSTER}_l${ID_LEN}_len128_split
else
    echo "Unknown dataset: ${DATASET}"; exit 1
fi

mkdir -p ../logs
LOG_FILE=../logs/test_grut_test_${DATASET}_e${EPOCH_NUM}_l${LAMBDA}.log

echo ">>>>>>>>>>>>>>>>>>>>> ${DATASET} TEST e${EPOCH_NUM} LAMBDA=${LAMBDA}" | tee -a ${LOG_FILE}
CUDA_VISIBLE_DEVICES=${GPU_NUM} python ../src/main_generative_grut.py --datasets ${DATASET} \
  --distributed 0 --gpu 0 --beam_size 50 \
  --train 0 --test_by_valid 0 \
  --item_prompt_max_len 128 --max_his 20 \
  --rec_batch_size 32 --gradient_accumulation_steps 2 \
  --rec_lr 1e-3 --scheduler linear_warmup --eval_batch_size 1 --warmup_prop 0.05 \
  --rec_epochs 30 --test_epoch_rec 10 --valid_by_test 0 \
  --save_rec_epochs 10 --save_predictions 1 \
  --debug_test_small_set 0 --debug_train_100 0 --debug_test_100 0 \
  --verbose_input_output 1 \
  --item_id_type ${ITEM_ID_TYPE} --hierarchical_id_type ${ITEM_ID} \
  --temporal_feature target_rel_abs \
  --global_context all_weight \
  --global_context_last_N ${LAST_N} --global_context_top_K ${TOP_K} \
  --global_context_epsilon ${EPSILON} \
  --global_context_decay_tau ${TAU} --global_context_decay_clip ${C_MIN} \
  --prompt_sentence_for_fid_input 1 --use_fid_position_embedding 1 \
  --trend_reranking_days ${DAYS} --trend_reranking_lambda ${LAMBDA} \
  --trend_reranking_use_valid_on_test 1 \
  --metrics 'hit@1,hit@3,hit@5,hit@10,hit@20,hit@50,ndcg@1,ndcg@3,ndcg@5,ndcg@10,ndcg@20,ndcg@50' \
  --rec_model_path ${CHECKPOINT_DIR}/model_rec_phase_1_epoch_${EPOCH_NUM}.pt \
  --item_prompt all_text --cf_model sasrec --id_linking 1 \
  --top_k_similar_item 10 --lexical_id_type_user idgenrec \
  2>&1 | tee -a ${LOG_FILE}
