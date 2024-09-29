#!/bin/env bash
export OMP_NUM_THREADS=8
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export HUGGING_FACE_HUB_TOKEN=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
# export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
# export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets

export CUDA_VISIBLE_DEVICES=0
NUM_GPU=1

ARGS="
--n_gpu $NUM_GPU
--strategy deepspeed_stage_2
--output_dir checkpoints/Llama-3.2-1B-Instruct-rr
--run_name Llama-3.2-1B-Instruct-test-rr
--seed 42
--train_set_path DKYoon/metamath-200k
--output_exists True
--enc_name_or_path DKYoon/mt5-small-lm-adapt
--lm_name_or_path meta-llama/Llama-3.2-1B-Instruct
--alignments rr
--enc_hidden_size 512
--lm_hidden_size 1536
--max_length 128
--max_length_enc 1024
--freeze_language_model True
--freeze_encoder True
--learning_rate_alignment 6e-4
--learning_rate_enc 2e-5
--w_decay_alignment 0.0
--w_decay_enc 0.1
--warmup_steps 0
--per_device_train_batch_size 28
--per_device_eval_batch_size 16
--gradient_accumulation_steps 8
--logging_steps 10
--num_train_epochs 1
--dataloader_num_workers 12
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