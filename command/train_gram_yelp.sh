# GRAM yelp (base config)

SEED=2023

ITEM_ID_TYPE=split
ID_LEN=9
NUM_CF=5
NUM_CLUSTER=32

ITEM_ID=hierarchy_v1_c${NUM_CLUSTER}_l${ID_LEN}_len128_split # hierarchy_v1_c32_l9_len128_split
echo ">>>>>>>>>>>>>>>>>>>>> Yelp SEED: ${SEED} ITEM_ID: ${ITEM_ID}" 

CUDA_VISIBLE_DEVICES=0,1 python ../src/main_generative_gram.py --datasets Yelp \
  --distributed 1 \
  --master_port 1143 \
  --gpu 0,1 \
  --seed ${SEED} \
  --train 1 \
  --item_prompt_max_len 128 \
  --item_prompt all_text \
  --cf_model sasrec \
  --id_linking 1 \
  --max_his 20 \
  --rec_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --rec_lr 1e-3 \
  --rec_epochs 30 \
  --test_epoch_rec 5 \
  --save_rec_epochs 5 \
  --save_predictions 1 \
  --top_k_similar_item ${NUM_CF} \
  --item_id_type ${ITEM_ID_TYPE} \
  --hierarchical_id_type ${ITEM_ID} \
  --lexical_id_type_user idgenrec

