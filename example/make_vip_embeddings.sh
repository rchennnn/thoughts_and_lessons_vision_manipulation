#!/bin/bash
#SBATCH --job-name=generate_embeddings_1
#SBATCH --output=slurm/%j-%x.out
#SBATCH --partition=job
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

# make the embeddings by running the expert policy forward on a fixed seed to generate the expert trajectories
python -u -m make_embeddings_metaworld \
        --mode make \
        --env_name button-press-v2 \
        --num_trajs 2500 \
        --seed 42 \
        --embedding_file "/home/ubuntu/metaworld/button_press_data" \
        --pickle_file "/home/ubuntu/metaworld/button_press"

# process the expert trajectories to get vision embeddings and save them to a pickle file
python -u -m make_embeddings_metaworld \
            --mode process \
            --env_name button-press-v2 \
            --embedding_file "/home/ubuntu/metaworld/button_press_data" \
            --pickle_file "/home/ubuntu/metaworld/button_press" \
            --img_size 224 \
            --encoder vip \
            --embedding_name "vip_embeddings"

