#!/bin/bash
#SBATCH --job-name=generate_embeddings_1
#SBATCH --output=slurm/%j-%x.out
#SBATCH --partition=job
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1


export MUJOCO_GL=egl
cd ~/thoughts_and_lessons_vision_manipulation


envs=(
  'push-v2'
  'assembly-v2'
  'push-wall-v2'
  'faucet-close-v2'
  'bin-picking-v2'
  'button-press-v2'
  'hammer-v2'
  'coffee-pull-v2'
  'dial-turn-v2'
  'hammer-v2'
  'pick-place-v2'
  'plate-slide-v2'
  'soccer-v2'
  'stick-push-v2'
  'sweep-v2'
  'window-open-v2'
  'window-close-v2'
  'door-close-v2'
  'stick-pull-v2'
  'push-back-v2'
  'reach-v2'
  'reach-wall-v2'
  'shelf-place-v2'
  'sweep-into-v2'
  'sweep-v2'
  'window-open-v2'
  'window-close-v2'
)

encoders=("sam" "clip" "mvp" "mae" "dino" "r3m" "vc1" "ibot" "obj_rn" "vip" "moco")

set -x

for env in "${envs[@]}"; do
    # Extract the environment name without the '-v2' suffix
    env_name="${env%-v2}"
    # Replace hyphens with underscores in the environment name
    env_name_underscore="${env_name//-/_}"
    
    python -u -m make_embeddings_metaworld \
        --mode make \
        --env_name "$env" \
        --num_trajs 2500 \
        --seed 42 \
        --embedding_file "/home/ubuntu/metaworld/${env_name_underscore}_data" \
        --pickle_file "/home/ubuntu/metaworld/${env_name_underscore}"

    for encoder in "${encoders[@]}"; do
        echo "processing $env with $encoder"
        python -u -m make_embeddings_metaworld \
            --mode process \
            --env_name "$env" \
            --embedding_file "/home/ubuntu/metaworld/${env_name_underscore}_data" \
            --pickle_file "/home/ubuntu/metaworld/${env_name_underscore}" \
            --img_size 224 \
            --encoder "$encoder" \
            --embedding_name "${encoder}_embeddings$(if [ "$encoder" = "sam" ]; then echo "224"; fi)"
    done
done


# Example usage:
# python -u -m make_embeddings_metaworld \
#             --mode process \
#             --env_name button-press-v2 \
#             --embedding_file "/home/ubuntu/metaworld/button_press_data" \
#             --pickle_file "/home/ubuntu/metaworld/button_press" \
#             --img_size 224 \
#             --encoder ibot \
#             --embedding_name "ibot_embeddings"

