# Usage

# python -m make_embeddings_metaworld \
#   --mode make \
#   --env_name push-v2 \
#   --num_trajs 50 \
#   --seed 42 \
#   --pickle_file push \
#   --embedding_file push_data


# python -m make_embeddings_metaworld \
#   --mode process \
#   --env_name push-v2 \
#   --num_trajs 2300 \
#   --pickle_file push \
#   --embedding_file push_data \
#   --img_size 96 \
#   --encoder sam \
#   --embedding_name sam_embeddings

# given hdf5 file with observations, make embeddings by running environment,
# store each trajectory images in temp, and get embeddings thereof
# ONLY for use with fetch envs

from typing import Tuple
from utils.policy_net import PolicyNetwork
import utils.encoders as encoders
import imageio

import torch
import torchvision.transforms as T
import h5py
import numpy as np
import gymnasium as gym
import pickle
import importlib

from argparse import ArgumentParser
from namespace import Namespace

_ENCODERS = {
    "sam": encoders.SAMPolicy,
    "mvp": encoders.MVPPolicy,
    "clip": encoders.CLIPPolicy,
    "mae": encoders.MAEPolicy,
    "dino": encoders.DinoV2Policy,
    "r3m": encoders.R3MPolicy,
    "vip": encoders.VIPPolicy,
    "vc1": encoders.VC1Policy,
    "moco": encoders.MoCoV3Policy,
    "ibot": encoders.IBOTPolicy,
    "obj_rn": encoders.ObjRNPolicy
}


def get_encoder(encoder_name: str) -> Tuple[PolicyNetwork, T.Compose]:
    policy_object: PolicyNetwork = _ENCODERS[encoder_name](device="cuda")
    return policy_object.encoder, policy_object.preprocess


def parse_arguments() -> Namespace:
    parse = ArgumentParser()
    parse.add_argument("--embedding_name", type=str)
    parse.add_argument("--embedding_file", type=str)
    parse.add_argument("--mode", type=str, default="None")

    parse.add_argument("--pickle_file", type=str)
    parse.add_argument("--num_trajs", type=int)
    parse.add_argument("--seed", type=int)
    parse.add_argument("--img_size", type=int, default=224)
    parse.add_argument("--env_name", type=str)
    parse.add_argument("--encoder", type=str, default="sam")
    parse.add_argument("--save_action", type=int, default=0)

    return parse.parse_args()


def hdf_augment(output_dir, is_new, dset_name, dset_data) -> None:
    # Create hdf5 data set. If data set exists, augment it larger
    if is_new:
        h = h5py.File(output_dir, "a")
        h.create_dataset(
            dset_name,
            data=dset_data,
            chunks=dset_data.shape,
            maxshape=tuple([None] * len(dset_data.shape)),
        )
        h.close()
    else:
        h = h5py.File(output_dir, "a")
        current_shape = h[dset_name].shape
        new_shape = dset_data.shape
        final_shape = (new_shape[0] + current_shape[0], *current_shape[1:])
        h[dset_name].resize(final_shape)
        h[dset_name][current_shape[0] : final_shape[0], :] = dset_data
        h.close()


def get_trajectories(args: Namespace) -> None:
    from metaworld import MT1

    try:
        module_name = args.env_name
        from_part = (
            "metaworld.policies.sawyer_" + args.env_name.replace("-", "_") + "_policy"
        )
        mod_name_env = args.env_name.replace("-", " ")
        mod_name_env = "".join([i.capitalize() for i in mod_name_env.split()])
        import_part = "Sawyer" + mod_name_env + "Policy"
        module = importlib.import_module(from_part)
        p = getattr(module, import_part)
        print("expert:", module, "module:", import_part)
    except:
        print("Failed to import expert policy module")
        print(from_part, import_part)
        raise ModuleNotFoundError
    print("Imported expert policy module", p)

    mt1 = MT1(args.env_name, seed=args.seed, num_goals=args.num_trajs)

    env = mt1.train_classes[args.env_name]()
    policy = p()
    log_successes = 0
    succeeded_tasks = []
    for traj, task in enumerate(mt1.train_tasks):
        env.set_task(task)
        obs, info = env.reset()
        actions = np.zeros((1, 150, env.action_space.shape[0]))
        state_size = env.get_arm_state().shape
        state_space = np.zeros((1, 150, state_size[0]))
        for step in range(150):
            a = policy.get_action(obs)
            actions[0, step] = a
            state_space[0, step] = env.get_arm_state()
            obs, _, _, _, info = env.step(a)
            done = int(info["success"]) == 1

        if done:
            succeeded_tasks.append(task)
            hdf_augment(
                output_dir=args.embedding_file + ".hdf5",
                is_new=log_successes == 0,
                dset_name="actions",
                dset_data=actions,
            )
            hdf_augment(
                output_dir=args.embedding_file + ".hdf5",
                is_new=log_successes == 0,
                dset_name="state",
                dset_data=state_space,
            )
            log_successes += 1

        if (traj + 1) % 100 == 0:
            print(
                "Completed trajectory",
                traj + 1,
                ". Success rate:",
                log_successes / (traj + 1),
            )

    with open(args.pickle_file + ".pkl", "wb") as f:
        pickle.dump(succeeded_tasks, f)

    name = f"{'/'.join(args.pickle_file.split('/')[:-1])}/success_rate.log"

    with open(name, "a+") as f:
        f.write(f"{args.env_name},")
        f.write(f"{log_successes / args.num_trajs},")
        f.write(f"{len(succeeded_tasks)}\n")
    print(
        "Finished making trajectories. Success rate:",
        log_successes / args.num_trajs,
        "Total trajectories:",
        len(succeeded_tasks),
    )


def process_data(args: Namespace) -> None:
    from metaworld import MT1

    if not int(args.save_action):
        print("Not saving actions...")

    img_size = args.img_size
    env_name = args.env_name
    device = "cuda"
    model, preprocess = get_encoder(args.encoder)
    model.eval()

    mt1 = MT1(env_name, seed=args.seed, num_goals=1)
    size = args.img_size
    env = mt1.train_classes[env_name](
        render_mode="rgb_array",
    ) 
    env.mujoco_renderer.default_cam_config = {
        "distance": 1.75,
        "azimuth": 135.0,
        "elevation": -45.0,
    }

    with open(args.pickle_file + ".pkl", "rb") as f:
        train_tasks = pickle.load(f)

    with h5py.File(args.embedding_file + ".hdf5", "r") as f:
        actions = f["actions"][:]

    traj_steps = actions.shape[1]

    log_successes = 0
    for traj, task in enumerate(train_tasks):
        env.set_task(task)
        obs, info = env.reset()
        traj_images = np.zeros((traj_steps, 3, img_size, img_size))
        for step in range(traj_steps):
            old_obs = obs
            a = actions[traj, step]
            img = env.render()
            actions[0, step] = a
            traj_images[step] = img.transpose((2, 0, 1))
            obs, _, _, _, info = env.step(a)
            done = int(info["success"]) == 1
        if done:
            log_successes += 1

        with torch.no_grad():
            traj_images = torch.from_numpy(traj_images).to(
                device=device, dtype=torch.float32
            )
            traj_images = preprocess(traj_images)
            embeds = model(traj_images)
            embeds = embeds.unsqueeze(0).detach().cpu().numpy()
        hdf_augment(
            output_dir=args.embedding_file + ".hdf5",
            is_new=traj == 0,
            dset_name=args.embedding_name,
            dset_data=embeds,
        )
        if int(args.save_action):
            hdf_augment(
                output_dir=args.embedding_file + ".hdf5",
                is_new=traj == 0,
                dset_name="actions",
                dset_data=actions,
            )
        if (traj + 1) % 10 == 0:
            print(
                f"finished trajectory {traj + 1}.",
                f"Success rate: {log_successes / (traj + 1)}",
            )

    print(
        "Finished processing trajectories. Success rate:",
        log_successes / len(train_tasks),
        "Total trajectories:",
        len(train_tasks),
    )
    
def get_goals_and_objects_pos(args: Namespace) -> None:
    from metaworld import MT1
    
    with open(args.pickle_file + ".pkl", "rb") as f:
        train_tasks = pickle.load(f)
    env_name = args.env_name
    
    with h5py.File(args.embedding_file + ".hdf5", "r") as f:
        actions = f["actions"]
        num_steps = actions.shape[1]
    
    try:
        module_name = args.env_name
        from_part = (
            "metaworld.policies.sawyer_" + args.env_name.replace("-", "_") + "_policy"
        )
        mod_name_env = args.env_name.replace("-", " ")
        mod_name_env = "".join([i.capitalize() for i in mod_name_env.split()])
        import_part = "Sawyer" + mod_name_env + "Policy"
        module = importlib.import_module(from_part)
        p = getattr(module, import_part)
        print("expert:", module, "module:", import_part)
    except:
        print("Failed to import expert policy module")
        print(from_part, import_part)
        raise ModuleNotFoundError
    print("Imported expert policy module", p)

    mt1 = MT1(env_name, seed=args.seed, num_goals=args.num_trajs)

    env = mt1.train_classes[env_name]()
    
    env.set_task(train_tasks[0])
    env.reset()
    goal_array = env._get_pos_goal()
    object_array = env._get_pos_objects()
    goal_size = goal_array.shape[0]
    object_size = object_array.shape[0]
    
    policy = p()
    log_successes = 0
    succeeded_tasks = []
    for traj, task in enumerate(train_tasks):
        env.set_task(task)
        obs, info = env.reset()
        goals = env._get_pos_goal()
        objects_space = np.zeros((1, num_steps, object_size))
        goal_space = np.zeros((1, num_steps, goal_size))
        for step in range(num_steps):
            a = policy.get_action(obs)
            objects_space[0, step] = env._get_pos_objects()
            goal_space[0, step] = env._get_pos_goal()
            obs, _, _, _, info = env.step(a)
            done = int(info["success"]) == 1

        if done:
            succeeded_tasks.append(task)
            hdf_augment(
                output_dir=args.embedding_file + ".hdf5",
                is_new=log_successes == 0,
                dset_name="object_pos",
                dset_data=objects_space,
            )
            hdf_augment(
                output_dir=args.embedding_file + ".hdf5",
                is_new=log_successes == 0,
                dset_name="goal_pos",
                dset_data=goal_space,
            )
            log_successes += 1

        if (traj + 1) % 100 == 0:
            print(
                "Completed trajectory",
                traj + 1,
                ". Success rate:",
                log_successes / (traj + 1),
            )


def concatenate(args: Namespace) -> None:
    h = h5py.File(args.data_file, "a")
    g = h5py.File(args.embedding_file, "r")
    h.create_dataset(
        args.embedding_name,
        data=g["embeddings"][:],
        maxshape=(None, *g["embeddings"][:].shape[1:]),
    )
    g.close()
    h.close()


args = parse_arguments()

if args.mode == "make":
    get_trajectories(
        args=args
    )  # trajectory tasks saved as pickle given by args.pickle_file
elif args.mode == "process":
    process_data(
        args=args
    )  # generates embeddings from trajectories of given encoder in file specified by args.embedding_file
elif args.mode == "concat":
    concatenate(
        args=args
    )  # concatenates embeddings to data file given by args.data_file
elif args.mode == "get_goals_and_objects_pos":
    get_goals_and_objects_pos(
        args=args
    )  # generates goal and object positions from trajectories of given encoder in file specified by args.embedding_file
else:
    print("Missing --mode make or process or concat")
