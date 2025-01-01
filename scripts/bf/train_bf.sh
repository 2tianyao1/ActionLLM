cd "/path/to/ActionLLM"
splits=(1 2 3 4)

for bf in "${splits[@]}"
do
  CUDA_VISIBLE_DEVICES=1 python train.py \
      --output_dir "./result/bf/sp${sp}_weights" \
      --log_dir "./result/bf/sp${sp}_logs" \
      --dataset breakfast \
      --batch_size 1 \
      --max_seq_len 1627 \
      --epochs 20 \
      --split $sp \
      --sample_rate 6 \
      --adapter_dim 4 \
      --action_class 48 \
      --n_query 8 \
      --multi_hidden_proj 128 \
      --lr 0.0001
done