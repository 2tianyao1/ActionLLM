cd "/home/ty/code/LLMAction_after/LLMAction_01_4"
splits=(1 2 3 4)

for spl in "${splits[@]}"
do
  for i in {0..19}
  do
    CUDA_VISIBLE_DEVICES=1 python eval.py \
    --checkpoint_file "./result/bf/sp${spl}_weights/checkpoint-${i}.pth" \
    --pred_file "./result/bf/excel/bf_sp${spl}.xlsx" \
    --dataset breakfast \
    --max_seq_len 1627 \
    --ck_num ${i} \
    --split ${spl} \
    --sample_rate 6 \
    --adapter_dim 4 \
    --longest_past_feature_length 812 \
    --action_class 48 \
    --n_query 8 \
    --multi_hidden_proj 128
  done
done
