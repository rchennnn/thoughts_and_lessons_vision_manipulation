#!/bin/bash
#SBATCH --job-name=generate_embeddings_1
#SBATCH --output=slurm/%j-%x.out
#SBATCH --partition=job
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

python -m run_metaworld \
    --env_name button-press-v2 \
    --num_epochs 101 \
    --base_encoder vip \
    --pos_contrib 0 \
    --num_train 2000 \
    --num_eval 50 \
    --img_size 224 \
    --silent False \
    --seed 22315
