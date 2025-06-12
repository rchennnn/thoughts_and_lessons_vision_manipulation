from vqvae.auto_encoder import VQ_CVAE
import torch 
import torchvision.transforms as T
import h5py
import numpy as np


resize = T.Resize((96, 96))

model = VQ_CVAE(d=256).to('cuda')

optimizer = torch.optim.NAdam(params=model.parameters(), lr=0.0001)


h = h5py.File('/projects/p31777/ryan/fphab/fphab.hdf5')
init = resize(torch.from_numpy(h['init_frames'][0:3200].transpose((0,3,1,2)))).to('cuda') / 255
goal = resize(torch.from_numpy(h['goal_frames'][0:3200].transpose((0,3,1,2)))).to('cuda') / 255
distances = torch.from_numpy(h['distances'][0:3200]).to('cuda', dtype=torch.float32)

eval_init = resize(torch.from_numpy(h['init_frames'][3200:3500].transpose((0,3,1,2)))).to('cuda') / 255
eval_goal = resize(torch.from_numpy(h['goal_frames'][3200:3500].transpose((0,3,1,2)))).to('cuda') / 255
eval_distances = torch.from_numpy(h['distances'][3200:3500]).to('cuda', dtype=torch.float32)

for epoch in range(40):
    train_epoch_losses = {'mse': 0, 'vq': 0, 'commitment': 0}
    test_epoch_losses = {'mse': 0, 'vq': 0, 'commitment': 0}
    for _ in range(100):
        train_idx = np.random.randint(1000, size=32)
        # goal_train = np.random.randint(1000, size=32)
                
        recon, ze, embedding, _ = model(init[train_idx].to(dtype=torch.float32))
        loss = model.loss_function(init[train_idx].to(dtype=torch.float32), recon, ze, embedding, _)
        latest_losses = model.latest_losses()
        train_epoch_losses = {k: train_epoch_losses[k] + v for k, v in latest_losses.items()}
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    for _ in range(100):
        with torch.no_grad():
            eval_idx = np.random.randint(200, size=32)
                    
            recon, ze, embedding, _ = model(eval_init[eval_idx].to(dtype=torch.float32))
            loss = model.loss_function(eval_init[eval_idx].to(dtype=torch.float32), recon, ze, embedding, _)
            latest_losses = model.latest_losses()
            test_epoch_losses = {k: test_epoch_losses[k] + v for k, v in latest_losses.items()}

    print('Train: Epoch', epoch, {k: v.item()/100 for k, v in train_epoch_losses.items()})
    print('\t\t Eval: Epoch', epoch, {k: v.item()/100 for k, v in test_epoch_losses.items()})
    
torch.save(model.encoder.state_dict(), './experiments/utils/cnn_extractor/cnn_encoder.pth')