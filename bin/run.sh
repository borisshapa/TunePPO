NUM_GPUS=$(nvidia-smi -L | wc -l)
NUM_GPUS_FOR_TRAINING=$((NUM_GPUS - 1))

CUDA_VISIBLE_DEVICES=$NUM_GPUS_FOR_TRAINING vllm serve /data/models/judge --enable-reasoning --reasoning-parser deepseek_r1 --max-model-len 12000 & 

sleep 150

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS_FOR_TRAINING - 1))) tune run --nproc_per_node $NUM_GPUS_FOR_TRAINING receipts/ppo/ppo.py --config receipts/ppo/configs/tldr_mistral_7b_ppo.yaml