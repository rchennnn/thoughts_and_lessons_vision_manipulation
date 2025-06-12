# Thoughts and Lessons on Using Visual Foundation Models for Manipulation

## Installation 

Create a Python 3.11 environment with conda

```
conda create -n vfm python==3.11
```

All necessary dependencies are in `dependencies.sh`.

## Encoders

Some weights are provided through installing the respective repo libraries.

Encoder | Architecture | Weights 
--------|--------------|---------
SAM | ViT | [SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
CLIP| ViT | [CLIP](https://github.com/mlfoundations/open_clip)
DINO| ViT | [DINOv2](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_linear_head.pth)
MAE| ViT | [MAE](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)
MVP| ViT | [MVP](https://github.com/ir413/mvp)
R3M| ResNet50 | [R3M](https://github.com/facebookresearch/r3m)
VIP| ResNet50 | [VIP](https://github.com/facebookresearch/vip)
VC1| ViT | [VC1](https://github.com/facebookresearch/eai-vc)
MOCO| ResNet50 | [MOCO](https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar)
IBOT| ViT | [iBOT](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth)
OBJRN|  ResNet50 | [Object365](https://drive.google.com/file/d/1FLPcRcAKaYBZrJQ7uYz0ST0WPrgacwm6/view?usp=sharing)


## Trajectory Collection  

Trajectories are collected via `make_embeddings_metaworld.sh`, where the `env` variables and `encoders` are cycled through the script, collecting all trajectories. Please note this procedure will produce trajectory data sets that total up to 3 TB in size.

## Training Behavior Cloning

Once the datasets have been created, we can train the behavior cloning policies with the following scripts. The following hyperparameters are used for the experiments.

```
python -m run_metaworld_bc --env_name push --num_epochs 101 --base_encoder sam --pos_contrib 0 --num_train 2500 --num_eval 50 --img_size 224 --silent False --seed 22315 # set the seed to anything
```

This script will run the policy forward.

## Training IQL

We can run IQL with the following script. The script was adapted from [CORL](https://github.com/tinkoff-ai/CORL) which provides single file implementations of offline algorithms. 

```
# python -m run_metaworld_iql --base_encoder sam --iql_seed 42 --env_name push-v2 --traj_len 100 --num_traj 2000 --train_steps 10000
```

## Reconstruction

We can train a reconstructor by the following, which will produce embeddings and reconstructions for a given trajectory and encoder.

```
python -m train_recon --embedding_name sam --env_name assembly
```

## Note on Gymnasium Environments

The environments were trained with HER, and uses a different data collection mechanism. The code is currently being prepared.