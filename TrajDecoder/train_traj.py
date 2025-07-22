import os 
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json

import options.option_traj as option_traj 
import utils.utils_model as utils_model 
from dataset.traj_dataset import *
from model.traj_model import *
import warnings
from torch.utils.data import DataLoader
import torch.nn.functional as F
from vis import render_sample
from tqdm import tqdm
import torch.nn as nn
import codecs as cs
import re

warnings.filterwarnings('ignore')


##### ---- Exp dirs ---- #####
args = option_traj.get_args_parser()
torch.manual_seed(args.seed)

device = args.device
args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.ckpt_dir, exist_ok = True)
if args.fkout_dir is not None:
    os.makedirs(args.fkout_dir, exist_ok = True)


##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Dataloader ---- #####
split_file_pth = f"../data/dancernum_split/split_files/split_dancerNum_{args.required_dancer_num}.txt"
split_filenames = []
with cs.open(split_file_pth, 'r') as f: 
    for line in f.readlines():
        split_filenames.append(line.strip())

required_dancer_num = args.required_dancer_num


train_tensor_dataset_path = os.path.join(
    args.processed_data_dir, f"train_tensor_dataset.pkl"
)
test_tensor_dataset_path = os.path.join(
    args.processed_data_dir, f"test_tensor_dataset.pkl"
)

# If caching is enabled and cached dataset files exist,
# load the preprocessed datasets directly from disk to save time.
if (
    not args.no_cache
    and os.path.isfile(train_tensor_dataset_path) 
    and os.path.isfile(test_tensor_dataset_path)
):
    train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
    test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))

else: 
    train_dataset = TrajDataset(
                    data_path=args.data_path,
                    backup_path=args.processed_data_dir,
                    train=True,
                    force_reload=args.force_reload,
                    required_dancer_num = args.required_dancer_num, 
                    split_file = split_filenames,
                )
    test_dataset = TrajDataset(
                    data_path=args.data_path,
                    backup_path=args.processed_data_dir,
                    train=False,
                    normalizer=train_dataset.normalizer,
                    force_reload=args.force_reload,
                    required_dancer_num = args.required_dancer_num, 
                    split_file = split_filenames,
                )
    pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
    pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))


train_data_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(int(4 * 0.75), 32),
            pin_memory=True,
            drop_last=True,
        )
train_loader_iter = cycle(train_data_loader)
test_data_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)
test_loader_iter = cycle(test_data_loader)

##### ---- Network ---- #####
net = TrajDecoder(nfeats = args.nfeats, 
                  trans_layer = args.trans_layer, 
                  window_size = args.window_size,

                  ) 

ckpt_iter = 0 
if args.checkpoint is not None:
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True) 
    print('loading checkpoint from {}'.format(args.checkpoint))

    # load iteration
    filename = os.path.basename(args.checkpoint)
    match = re.search(r'epoch-(\d+)\.pth', filename)
    if match:
        ckpt_iter = int(match.group(1))
    else:
        ckpt_iter = 0  

    print(f'Resumed ckpt_iter: {ckpt_iter}')

net.train()
net.to(device)

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, net, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_func = F.mse_loss
nb_iter, avg_loss, avg_recon_loss, avg_vloss, avg_dis_loss = 0, 0., 0., 0., 0.

##### ---- Training ---- #####
while nb_iter <= args.total_iter:
    # Load a batch of data
    batch = next(train_loader_iter)
    x, cond, filename, wavnames = batch  # Shape: (batch_size, dancer_num, 150, 3)

    # Design for autoregressive prediction:
    # Slide forward by 100 frames with a window size of 25.
    window_size = args.window_size
    step = args.step

    # Randomly choose a start frame for the prediction segment
    rand_start = torch.randint(low = 0,high = 150 - window_size-step,size = (1,))[0] 
    rand_end = rand_start + window_size

    # Music has double the FPS of motion, so adjust music indices accordingly
    rand_music_start = rand_start * 2 
    rand_music_end = rand_music_start + window_size*2 + step*2

    # Define reconstruction range (target to predict)
    recon_start = rand_start + step
    recon_end = rand_end + step
    
    # Select only x and y coordinates for motion and move data to device
    x = x[:,:,:,[0,1]].to(device).float()
    cond = cond.to(device).float()

    # Input segment for conditioning
    x_cond = x[:,:,rand_start:rand_end]

    # Target segment for reconstruction
    x_target = x[:,:,recon_start:recon_end]

    # Model prediction for future trajectory (step frames ahead)
    pre_traj = net(x_cond, cond[:, rand_music_start:rand_music_end]) 

    # Reconstruction loss between predicted and target motion
    # Sliding window approach: generate previous results + next moment's motion
    loss_recon = loss_func(pre_traj, x_target, reduction="none")

    # Compute distance loss between trajectories
    target_dis = x_target[:,1:] - x_target[:,:-1]
    recon_dis = pre_traj[:,1:] - pre_traj[:,:-1]
    dis_loss = loss_func(target_dis, recon_dis, reduction="none")

    # Compute velocity loss
    target_v = x_target[:, : ,1:] - x_target[:, :, :-1]
    model_out_v = pre_traj[:, : ,1:] - pre_traj[:, :, :-1]
    v_loss = loss_func(target_v, model_out_v, reduction="none")

    # Combine losses with weighting
    loss = loss_recon.mean() + 2 * dis_loss.mean() + 2 * v_loss.mean()

    ## global loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # logs
    nb_iter += 1
    avg_loss += loss.item() 
    avg_recon_loss += loss_recon.mean().item() 
    avg_dis_loss += dis_loss.mean().item()
    avg_vloss += v_loss.mean().item()

    if nb_iter % args.print_iter ==  0 :
        avg_loss = avg_loss / args.print_iter
        avg_recon_loss = avg_recon_loss / args.print_iter
        avg_dis_loss = avg_dis_loss / args.print_iter
        avg_vloss = avg_vloss / args.print_iter

        writer.add_scalar('./Loss/train', avg_loss, nb_iter)
        msg = f"Train. Iter {nb_iter} : Loss. {avg_loss:.5f} recon. {avg_recon_loss:.5f} dis_loss. {avg_dis_loss:.5f} v_loss. {avg_vloss:.5f}"
        logger.info(msg)
        avg_loss = 0.
        avg_recon_loss = 0.
        avg_dis_loss = 0.
        avg_vloss = 0.

        # === Evaluation and Rendering ===

        # Fetch the next test batch
        batch = next(test_loader_iter)
        x, cond, filename, wavname = batch

        # Number of samples to render
        render_count = 2

        # Select only x and y coordinates for motion (drop z)
        x = x[:,:,:,[0,1]]

        # Predict the future trajectory using the trained network
        # Input: last 'window_size' frames before the prediction step
        pre_traj = net(x[:,:,-(step + window_size):-step].to(device).float(), cond[:, -(step + window_size)*2:].to(device).float()) 

        # Concatenate predicted trajectory with earlier frames
        samples_xz = torch.cat([x[:,:,:-window_size].to(pre_traj), pre_traj], dim = 2) 
        
        # Optional: process trajectory with smoothing or constraints
        samples_xz = utils_model.kalman_smooth_batch(samples_xz.cpu().detach().numpy())
        samples_xz = torch.from_numpy(samples_xz).to(dtype=pre_traj.dtype, device=pre_traj.device)

        # Prepare trajectory for rendering
        b,dn,seq,c = samples_xz.shape

        # Initialize zero-padded array for full 3D coordinates (x, y, z)
        samples = torch.zeros(b,dn,seq,3).to(samples_xz)
        samples[:,:,:,[0,1]] = samples_xz

        # Reshape to (batch_size, frames, 3) for normalization
        samples = samples.reshape(b, -1, 3) 

        # Unnormalize the sample back to original scale
        samples = test_dataset.normalizer.unnormalize(samples) 


        # Render the motion sequence
        render_sample(
            samples[:render_count].reshape(render_count, dn, seq, 3) .permute(0,2,1,3), # [*, 150, 3, 3]
            nb_iter + ckpt_iter,
            args.render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="normal",
            constraint=None,
            start_point=None,
            render=True,
            fk_out = args.fkout_dir,  # Optional: output FK data if directory specified
            required_dancer_num = args.required_dancer_num,
            x_0 = None,
        )
        print('Done Rrendering~')
        

        # Save the ckpt
        torch.save({'net' : net.state_dict(),},
                   os.path.join(args.ckpt_dir, f'epoch-{nb_iter + ckpt_iter}.pth'))
        print(os.path.join(args.ckpt_dir, f'epoch-{nb_iter + ckpt_iter}.pth'))
        print('saved..')

    if nb_iter == args.total_iter: 
        break            