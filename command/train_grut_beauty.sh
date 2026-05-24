# GRUT beauty (paper table 8)

SEED=2023

ITEM_ID_TYPE=split
ID_LEN=7
NUM_CLUSTER=128

ITEM_ID=hierarchy_v1_c${NUM_CLUSTER}_l${ID_LEN}_len32768_split

EPSILON=0.01
TOP_K=1
LAST_N=5
TAU=128
C_MIN=0.8

echo ">>>>>>>>>>>>>>>>>>>>> Beauty SEED: ${SEED} ITEM_ID: ${ITEM_ID}"

CUDA_VISIBLE_DEVICES=0,1 python ../src/main_generative_grut.py --datasets Beauty \
  --distributed 1 \
  --master_port 2443 \
  --gpu 0,1 \
  --seed ${SEED} \
  --train 1 \
  --item_prompt_max_len 128 \
  --max_his 20 \
  --rec_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --rec_lr 1e-3 \
  --scheduler linear_warmup \
  --eval_batch_size 1 \
  --warmup_prop 0.05 \
  --rec_epochs 30 \
  --test_epoch_rec 10 \
  --valid_by_test 0 \
  --save_rec_epochs 5 \
  --save_predictions 1 \
  --debug_train_100 0 \
  --debug_test_100 0 \
  --debug_test_small_set 0 \
  --verbose_input_output 1 \
  --item_id_type ${ITEM_ID_TYPE} \
  --hierarchical_id_type ${ITEM_ID} \
  --temporal_feature target_rel_abs \
  --global_context all_weight \
  --global_context_last_N ${LAST_N} \
  --global_context_top_K ${TOP_K} \
  --global_context_epsilon ${EPSILON} \
  --global_context_decay_tau ${TAU} \
  --global_context_decay_clip ${C_MIN} \
  --prompt_sentence_for_fid_input 1 \
  --use_fid_position_embedding 1 \
  --item_prompt all_text \
  --cf_model sasrec \
  --id_linking 1 \
  --top_k_similar_item 10 \
  --lexical_id_type_user idgenrec
