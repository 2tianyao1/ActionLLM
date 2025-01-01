cd "/path/to/ActionLLM"
splits=(1 2 3 4 5)

for sp in "${splits[@]}"
do
  CUDA_VISIBLE_DEVICES=0 python train.py \
      --output_dir "./result/per_sp/50s/sp${sp}_weights" \
      --log_dir "./result/per_sp/50s/sp${sp}_logs" \
      --dataset 50_salads \
      --max_seq_len 2270 \
      --epochs 25 \
      --split ${sp} \
      --sample_rate 8 \
      --action_class 19 \
      --batch_size 1 \
      --adapter_dim 4 \
      --n_query 20 \
      --lr 0.001 \
      --multi_hidden_proj 128
done
