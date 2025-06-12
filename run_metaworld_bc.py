from utils.utils import parse_args, enablePrint, blockPrint
from utils.encoders import SAMPolicy, MVPPolicy, CLIPPolicy, MAEPolicy, DinoV2Policy, R3MPolicy, VIPPolicy, VC1Policy, MoCoV3Policy, IBOTPolicy, ObjRNPolicy
from utils.fetch_utils import get_data, train_policy, play_metaworld_policy

import gymnasium_robotics
import numpy as np
import torch
import importlib


'''
python -m run_metaworld --env_name push --num_epochs 101 --base_encoder moco --pos_contrib 0 --num_train 2000 --num_eval 50 --img_size 224 --silent False --seed 22315
'''


EMBEDDINGS ={
    'sam' : 'sam_embeddings224', 
    'mvp' : 'mvp_embeddings',
    'mae' : 'mae_embeddings',
    'dino' : 'dino_embeddings',
    'clip' : 'clip_embeddings',
    'cnn' : 'images',
    'r3m' : 'r3m_embeddings',
    'vip' : 'vip_embeddings',
    'vc1' : 'vc1_embeddings',
    'moco' : 'moco_embeddings',
    'ibot' : 'ibot_embeddings',
    'obj_rn' : 'obj_rn_embeddings'
}

MODELS = {
    'sam' : SAMPolicy,
    'mvp' : MVPPolicy,
    'mae' : MAEPolicy,
    'dino' : DinoV2Policy,
    'clip' : CLIPPolicy,
    'r3m' : R3MPolicy,
    'vip' : VIPPolicy,
    'vc1' : VC1Policy,
    'moco' : MoCoV3Policy,
    'ibot' : IBOTPolicy,
    'obj_rn' : ObjRNPolicy,
}

def get_data_dir(env_name):
    env_name = env_name.replace('-', '_')
    data_dir = f'/home/ubuntu/metaworld/{env_name}_data.hdf5'
    print('data from:', data_dir)
    return data_dir
    

if __name__ == "__main__":
    
    args = parse_args()
    num_epochs = args.num_epochs
    env_name = args.env_name
    print('num_epochs:', num_epochs)
    print('env_name:', env_name)
    
    try:
        module_name = env_name
        from_part = 'metaworld.policies.sawyer_' + env_name.replace('-', '_') + '_v2_policy' 
        mod_name_env = env_name.replace('-', ' ')
        mod_name_env = ''.join([i.capitalize() for i in mod_name_env.split()])
        import_part = 'Sawyer' + mod_name_env + 'V2Policy'
        module = importlib.import_module(from_part)
        p = getattr(module, import_part)
        print('expert:', module, 'module:', import_part)
    except:
        print('Failed to import expert policy module')
        print(from_part, import_part)
        raise ModuleNotFoundError
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_sam, eval_sam = get_data(
        data_dir=get_data_dir(env_name),
        num_train=args.num_train, 
        num_test=args.num_eval,
        traj_length=100,
        embedding_name=EMBEDDINGS[args.base_encoder],
        seed = args.seed,
        device=device)
    
    init_embeds = (train_sam['init_embed'], eval_sam['init_embed'])
    goal_embeds = (train_sam['goal_embed'], eval_sam['goal_embed'])
    init_state_space = (train_sam['init_state'], eval_sam['init_state'])
    goal_state_space = (train_sam['goal_state'], eval_sam['goal_state'])
    actions = (train_sam['actions'], eval_sam['actions'])
    
    print(init_embeds[0].shape)
    print(init_state_space[0].shape[1])
    if args.silent:
        blockPrint()
    model = MODELS[args.base_encoder](device=device, 
                                      state_size=init_state_space[0].shape[1], 
                                      action_size=actions[0].shape[1],
                                      layers=[128, 128, 64], 
                                      pos_contrib=args.pos_contrib)    
    
    
    policy = train_policy(init_embeds=init_embeds, goal_embeds=goal_embeds, 
                          init_states=init_state_space, goal_states=goal_state_space,
                          actions=actions,
                          epochs=num_epochs, #1001 or 401
                          mb_size=256, # 256
                          model=model)
    
    heads = policy.get_heads()
    # torch.save(heads.state_dict(), 'trained_modules/frozen_heads/push_heads_dino.pth')
    
    successes = play_metaworld_policy(the_policy=policy, env_name = env_name + '-v2', 
                          seed=1337,
                          trajectories=21,
                          size=args.img_size,
                          expert=p)
    if args.silent:
        enablePrint()
    
    output = [args.env_name, args.num_train, args.num_epochs, args.base_encoder, str(np.mean(successes))]
    output = [str(i) for i in output]
    print(','.join(output))