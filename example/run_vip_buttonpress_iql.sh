#!/bin/bash
#SBATCH --job-name=generate_embeddings_1
#SBATCH --output=slurm/%j-%x.out
#SBATCH --partition=job
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

python -m run_metaworld_iql \
     --base_encoder vip \
     --iql_seed 42 \
     --env_name button-press-v2 \
     --traj_len 100 \
     --num_traj 2000 \
     --train_steps 10000 \
     --pos_embed 1
