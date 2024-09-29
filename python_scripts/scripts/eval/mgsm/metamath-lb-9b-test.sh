# export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
# export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets
# export CUDA_VISIBLE_DEVICES=0
# export CC=/usr/bin/gcc-10
# export CXX=/usr/bin/g++-10
export HUGGING_FACE_HUB_TOKEN=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

python eval_langbridge.py \
  --checkpoint_path checkpoints/metamath-lb-9b-rr/epoch=1-step=892 \
  --enc_tokenizer DKYoon/mt5-small-lm-adapt \
  --lm_tokenizer facebook/opt-125m \
  --tasks mgsm_es\
  --instruction_template metamath \
  --batch_size 1 \
  --output_path eval_outputs/mgsm/metamath-langbridge_9b \
  --device cuda:0 \
  --no_cache


