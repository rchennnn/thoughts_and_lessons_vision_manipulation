import argparse
import h5py
import numpy as np
from metaworld import MT1
from metaworld.envs.mujoco.sawyer_xyz import SawyerXYZEnv
import importlib
import torchvision.utils as vutils
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.decoder import Decoder, LinearDecoder, PerceptualLoss, VGGPerceptualLoss

import tqdm

# conda activate mlenv
# export MUJOCO_GL=egl
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

def get_model_size(model):
    # Count the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Estimate memory usage in bytes (assuming float32)
    size_in_bytes = total_params * 4  # 4 bytes for float32
    size_in_mb = size_in_bytes / (1024 ** 3)  # Convert to GB
    
    return total_params, size_in_mb

def get_embedding_name(embedder):
    mapper = {
        'clip': 'clip_embeddings',
        'sam': 'sam_embeddings224',
        'r3m': 'r3m_embeddings',
        'dino': 'dino_embeddings',
        'mae': 'mae_embeddings',
        'mvp': 'mvp_embeddings',
        'vip': 'vip_embeddings',
        'vc1': 'vc1_embeddings',
        'moco': 'moco_embeddings',
        'ibot': 'ibot_embeddings'
    }
    return mapper[embedder]

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_name', type=str, default='clip')
    parser.add_argument('--env_name', type=str, default='assembly')
    parser.add_argument('--num_epochs', type=int, default=130)
    parser.add_argument('--linear', type=int, default=0)
    parser.add_argument('--future', type=int, default=5)
    return parser.parse_args()

vgg = VGGPerceptualLoss(resize=False)
def reg_loss(inputs, outputs):
    floss = vgg(inputs, outputs)
    mloss = nn.MSELoss()
    mloss = mloss(inputs, outputs)
    return mloss +  0.1* floss

def get_images(env_name, indices, tasks, traj_length):
    images = np.zeros((len(indices), traj_length, 3, 224, 224))

    mt1 = MT1(env_name, seed=42, num_goals=3000)
    env : SawyerXYZEnv = mt1.train_classes[env_name](render_mode='rgb_array')
    env.mujoco_renderer.default_cam_config = {"distance": 1.75, "azimuth": 135.0, "elevation": -45.0}
    try:
        from_part = (
            "metaworld.policies.sawyer_" + env_name.replace("-", "_") + "_policy"
        )
        mod_name_env = env_name.replace("-", " ")
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
    expert = p()

    # Generate images for training data
    for traj_num, i in tqdm.tqdm(enumerate(indices)):
        env.set_task(tasks[i])
        obs, _ = env.reset()
        for k in range(traj_length):
            action = expert.get_action(obs)
            obs, _, _, _, _ = env.step(action)
            img = env.render() # outputs (224, 224, 3)
            images[traj_num, k] = np.transpose(img, (2, 0, 1))  

    return images

def load_and_split_hdf5_data(file_path, dataset_name, pkl_path, test_size=0.2, num_trajectories=100):
    '''
    """
    Load and split HDF5 data into training and testing sets.

    Parameters:
    file_path (str): The path to the HDF5 file containing the dataset.
    dataset_name (str): The name of the dataset within the HDF5 file.
    pkl_path (str): The path to the pickle file containing tasks.
    test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.

    Returns:
    tuple: A tuple containing:
        - train_data (List[Tuple[np.array, np.array]]): A list of tuples where each tuple contains a training data point and its corresponding index.
        - test_data (List[Tuple[np.array, np.array]]): A list of tuples where each tuple contains a testing data point and its corresponding index.
        - task_indices (dict): A dictionary mapping indices to tasks.
        - env (str): The environment name extracted from the file path.
        
        E.g train_data[0][0].shape = (traj_length, embedding size) and train_data[0][1] = index of pickled task
    """
    '''
    # Extract the environment name from the file path
    print(file_path)
    env_name = file_path.split('/')[-1].split('_')[:-1]
    env_name = '_'.join(env_name)
    env_name = env_name.replace('_', '-') + '-v2'
    print(env_name)
    
    # Load tasks from the pickle file
    import pickle
    with open(pkl_path, 'rb') as pkl_file:
        tasks = pickle.load(pkl_file)
    
    # load data set and produce splits
    with h5py.File(file_path, 'r') as file:
        dataset = file[dataset_name][:num_trajectories]
        dataset_size = dataset.shape[0]
        traj_length = dataset.shape[1]
        split_index = int(dataset_size * (1 - test_size))
        
        train_data = dataset[:split_index]
        test_data = dataset[split_index:]
        
        task_indices = {i: tasks[i] for i in range(len(tasks))}
        
        train_indices = np.arange(split_index)
        test_indices = np.arange(split_index, dataset_size)
    
    temp_env_name = env_name[:-3]
    data_dir = f'/home/ubuntu/metaworld/imgs' 
    # create images for training and testing by running env forward with expert
    
    if not os.path.exists(os.path.join(data_dir, f'{env_name}_images_compressed.npz')):
        print('Generating images at ', os.path.join(data_dir, f'{env_name}_images_compressed.npz'))
        train_images = get_images(env_name, train_indices, task_indices, traj_length)
        test_images = get_images(env_name, test_indices, task_indices, traj_length)
        images = np.vstack((train_images, test_images))
        assert images.shape[0] == train_images.shape[0] + test_images.shape[0]
        assert images.shape[1] == train_images.shape[1] == test_images.shape[1]
        assert images.shape[2] == train_images.shape[2] == test_images.shape[2]
        assert images.shape[3] == train_images.shape[3] == test_images.shape[3]
        np.savez_compressed(os.path.join(data_dir, f'{env_name}_images_compressed.npz'), images)
    else:
        images = np.load(os.path.join(data_dir, f'{env_name}_images_compressed.npz'))
        images = images['arr_0']
        train_images = images[:split_index]
        test_images = images[split_index:]
        assert train_images.shape[0] + test_images.shape[0] == images.shape[0]
        assert train_images.shape[1] == test_images.shape[1]
        assert train_images.shape[2] == test_images.shape[2]
        assert train_images.shape[3] == test_images.shape[3]
    
    return train_data, test_data, task_indices, env_name, train_images, test_images

def test_load_and_split_hdf5_data(): # only used for testing purposes
    file_path = "/home/ubuntu/metaworld/assembly_data.hdf5"
    pkl_path = "/home/ubuntu/metaworld/assembly.pkl"
    dataset_name = "vip_embeddings"
    test_size = 0.2
    
    train_data, test_data, task_indices, env = load_and_split_hdf5_data(file_path, dataset_name, pkl_path, test_size)
    print("Training Data:", train_data[0][0].shape, train_data[0][1])
    print("Testing Data:", test_data[0][0].shape, test_data[0][1])
    print("Task Indices:", task_indices.keys())
    print("Environment:", env)

def train_decoder(args, decoder,
                  train_data, future_train_images,
                  test_data, future_test_images,
                  optimizer, loss_function, num_epochs=10, optim_scheduler=None, 
                  traj_length=100, embedding_name=None, env_name=None):
    # train_data is a list of tuples (embeddings, indices)
    decoder.train()  # Set the decoder to training mode
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(10):
            mb_indices = np.random.randint(0, train_data.shape[0], size=32)
            inputs = train_data[mb_indices]  # Get the n-th tensor of shape (100, 512)
            images = future_train_images[mb_indices]
            inputs = torch.tensor(inputs).to('cuda')  # Convert to tensor
            
            noise = torch.randn_like(inputs) * inputs.std(dim=0, keepdim=True)  # Generate noise proportional to the std of the embeddings
            inputs += noise  # Add noise to the inputs
            images = torch.tensor(images).to('cuda')
            
            inputs = torch.flatten(inputs, 1)  
            images = images.float() / 255.0  
            
            optimizer.zero_grad() 
            outputs = decoder(inputs) 
            loss = loss_function(outputs, images) 
            loss.backward() 
            optimizer.step()  
            total_loss += loss.item() 
        # scheduler.step(total_loss / len(train_data))
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_data)}')
    
    # Optionally, evaluate on test data
    decoder.eval()  # Set the decoder to evaluation mode
    with torch.no_grad():
        test_loss = 0
        mb_indices = np.arange(traj_length)
        inputs = test_data[mb_indices]
        images = future_test_images[mb_indices]
        inputs = torch.tensor(inputs).to('cuda')
        images = torch.tensor(images).to('cuda')
        images = images.float() / 255.0  
        inputs = torch.flatten(inputs, 1)  
        outputs = decoder(inputs) 
        loss = loss_function(outputs, images)  # Compute loss
        test_loss += loss.item()  # Accumulate test loss
            
        print(f'Test Loss: {test_loss / len(test_data)}')

        # Create directory for saving images if it doesn't exist
        save_dir = f'/shares/bcs516/ryan/metaworld_data_lambda/recons_future/{env_name}/{embedding_name}/'
        if args.future == 20:
            save_dir = f'/home/ubuntu/metaworld/recons_future20/{env_name}/{embedding_name}/'
        if args.future == 5:
            save_dir = f'/home/ubuntu/metaworld/recons_future5/{env_name}/{embedding_name}/'
        os.makedirs(save_dir, exist_ok=True)

        test_data = test_data.reshape(-1, traj_length, *test_data.shape[1:])
        test_images = torch.tensor(future_test_images).to('cuda')
        images = test_images.view(-1, traj_length, 3, 224, 224)
        images_to_eval = images[:,[15, traj_length-5]].squeeze()
        inputs_to_eval = test_data[:,[15, traj_length-5]]
        inputs_to_eval = inputs_to_eval.reshape(-1, *inputs_to_eval.shape[2:])
        inputs_to_eval = torch.tensor(inputs_to_eval).to('cuda')
        inputs_to_eval = torch.flatten(inputs_to_eval, 1)

        outputs_to_eval = decoder(inputs_to_eval)
        outputs_to_eval = outputs_to_eval.view(-1, 2, *outputs_to_eval.shape[1:])
        print(outputs_to_eval.shape)
        print(images_to_eval.shape)
        print('SAVING IMAGES TO', save_dir)
        for test_idx in range(outputs_to_eval.shape[0]):
            recons_path = os.path.join(save_dir, f'{test_idx}_{5}_recons.png')
            vutils.save_image(outputs_to_eval[test_idx, 0], recons_path, normalize=False)
            recons_path = os.path.join(save_dir, f'{test_idx}_{traj_length-5}_recons.png')
            vutils.save_image(outputs_to_eval[test_idx, 1], recons_path, normalize=False)

            # Save original images with suffix _original_{i}.png
            recons_path = os.path.join(save_dir, f'{test_idx}_{5}_original.png')
            vutils.save_image(images_to_eval[test_idx, 0] / 255.0, recons_path, normalize=False)
            recons_path = os.path.join(save_dir, f'{test_idx}_{traj_length-5}_original.png')
            vutils.save_image(images_to_eval[test_idx, 1] / 255.0, recons_path, normalize=False)



if __name__ == "__main__":
    
    args = argparser()
    embedding_name = get_embedding_name(args.embedding_name)
    num_steps = args.future
    
    mod_env_name = args.env_name.replace('-', '_')

    file_path = f"/home/ubuntu/metaworld/{mod_env_name}_data.hdf5"
    pkl_path = f"/home/ubuntu/metaworld/{mod_env_name}.pkl"
    test_size = 0.8
    num_trajectories = 50 
    
    train_data, test_data, tasks, env_name, train_images, test_images = load_and_split_hdf5_data(file_path, 
                                                                                                 embedding_name, 
                                                                                                 pkl_path, 
                                                                                                 test_size, 
                                                                                                 num_trajectories)

    'train_data has shape (9, 100, embedding_size)'
    
    future_train_data = train_data[:, num_steps:]
    future_test_data = test_data[:, num_steps:]
    train_data = train_data[:, :-num_steps]
    test_data = test_data[:, :-num_steps]
    
    future_train_images = train_images[:, num_steps:]
    future_test_images = test_images[:, num_steps:]
    train_images = train_images[:, :-num_steps]
    test_images = test_images[:, :-num_steps]
    
    traj_length = train_data.shape[1] # number of steps in a trajectory
    train_data = train_data.reshape(-1, *train_data.shape[2:])  # Collapse the first two dimensions
    test_data = test_data.reshape(-1, *test_data.shape[2:])      # Collapse the first two dimensions
    future_train_data = future_train_data.reshape(-1, *future_train_data.shape[2:])
    future_test_data = future_test_data.reshape(-1, *future_test_data.shape[2:])
    
    
    train_images = train_images.reshape(-1, *train_images.shape[2:])  # Collapse the first two dimensions
    test_images = test_images.reshape(-1, *test_images.shape[2:])    # Collapse the first two dimensions
    future_train_images = future_train_images.reshape(-1, *future_train_images.shape[2:])
    future_test_images = future_test_images.reshape(-1, *future_test_images.shape[2:])
    
    print('test data shape', test_data.shape)
    
    embedding_size = int(np.prod(train_data.shape[1:]))  # Get the product of dimensions from position 1 and higher
    print(embedding_size)
    loss = PerceptualLoss(args.embedding_name, device='cuda')
    loss = VGGPerceptualLoss(resize=False)
    
    # Initialize the Decoder
    if args.linear:
        decoder = LinearDecoder(input_size=embedding_size)
    else:
        decoder = Decoder(input_size=embedding_size)  # Assuming embedding size is the second dimension
    decoder.to('cuda')
    optimizer = torch.optim.NAdam(decoder.parameters(), lr=0.0001)
    loss_function = reg_loss
    total_params, size_in_mb = get_model_size(decoder)
    print(f'Model size: {size_in_mb} GB')
    
    train_decoder(args, decoder, train_data, future_train_images, test_data, future_test_images, optimizer, loss_function, num_epochs=args.num_epochs, 
                  traj_length=traj_length, embedding_name=args.embedding_name, optim_scheduler=None, env_name=args.env_name)
    