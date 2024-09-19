#!/bin/env bash
export OMP_NUM_THREADS=8
# export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
# export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets
export TORCH_CUDA_ARCH_LIST="7.5"

export CUDA_VISIBLE_DEVICES=0
NUM_GPU=1

ARGS="
--n_gpu $NUM_GPU
--strategy deepspeed_stage_2
--output_dir checkpoints/test
--run_name orca2-lb-9b
--seed 42
--train_set_path DKYoon/metamath-200k
--output_exists True
--enc_name_or_path DKYoon/mt5-small-lm-adapt
--lm_name_or_path facebook/opt-125m
--alignments linear 
--enc_hidden_size 512
--lm_hidden_size 768
--max_length 256
--max_length_enc 128
--freeze_language_model True
--freeze_encoder True
--learning_rate_alignment 6e-4
--learning_rate_enc 2e-5
--w_decay_alignment 0.0
--w_decay_enc 0.1
--warmup_steps 0
--per_device_train_batch_size 4
--per_device_eval_batch_size 16
--gradient_accumulation_steps 8
--logging_steps 10
--num_train_epochs 1
--dataloader_num_workers 16
--bf16 True
"

echo $ARGS
if [ $NUM_GPU == 1 ]; then
    echo "running on a single GPU"
    python train_langbridge.py $ARGS
else
    echo "running on multiple GPUs"
    torchrun --nproc_per_node $NUM_GPU train_langbridge.py $ARGS
fi