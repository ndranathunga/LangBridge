# export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
# export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets
# export CUDA_VISIBLE_DEVICES=0
# export CC=/usr/bin/gcc-10
# export CXX=/usr/bin/g++-10
export HUGGING_FACE_HUB_TOKEN=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

python eval_langbridge.py \
  --checkpoint_path checkpoints/MetaMath-7B-V1.0-rr/epoch=1-step=1562 \
  --enc_tokenizer DKYoon/mt5-xl-lm-adapt \
  --lm_tokenizer meta-math/MetaMath-7B-V1.0 \
  --tasks mgsm_es\
  --instruction_template metamath \
  --batch_size 16 \
  --output_path eval_outputs/mgsm/MetaMath-7B-rr-768 \
  --device cuda:0 \
  --no_cache


