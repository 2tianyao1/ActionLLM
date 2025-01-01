cd "/home/tianyao/code/ty/LLMAction_after/LLMAction_01_4"
splits=(1 2 3 4 5)

for spl in "${splits[@]}"
do
  for i in {0..24}
  do
    CUDA_VISIBLE_DEVICES=0 python eval.py \
    --checkpoint_file "./result/per_sp/50s/sp${spl}_weights/checkpoint-${i}.pth" \
    --pred_file "./result/per_sp/50s/excel/bf_sp${spl}.xlsx" \
    --dataset 50_salads \
    --max_seq_len 2270 \
    --ck_num ${i} \
    --split $spl \
    --sample_rate 8 \
    --action_class 19 \
    --n_query 20 \
    --adapter_dim 4 \
    --multi_hidden_proj 128
  done
done

