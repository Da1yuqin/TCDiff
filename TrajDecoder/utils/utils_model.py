import numpy as np 
import torch
import torch.optim as optim
import logging
import os 
import sys 
from scipy.signal import savgol_filter
from filterpy.kalman import KalmanFilter

def kalman_smooth_batch(xy_batch, dt=1.0, process_noise_std=1e-2, measurement_noise_std=1e-1):
    """
    Smooth a batch of XY trajectories using a Kalman filter.
    
    Args:
        xy_batch: np.ndarray of shape (batch_size, dancer_num, seq_len, 2)
        dt: time step
        process_noise_std: std of process noise
        measurement_noise_std: std of measurement noise

    Returns:
        xy_smooth_batch: np.ndarray of same shape as input
    """

    batch_size, dancer_num, seq_len, _ = xy_batch.shape
    xy_smooth_batch = np.zeros_like(xy_batch)

    for b in range(batch_size):
        for d in range(dancer_num):
            xy = xy_batch[b, d]  # shape (seq_len, 2)
            
            # Create Kalman filter
            kf = KalmanFilter(dim_x=4, dim_z=2)
            
            # State transition matrix
            kf.F = np.array([[1, 0, dt, 0],
                             [0, 1, 0, dt],
                             [0, 0, 1,  0],
                             [0, 0, 0,  1]])
            
            # Measurement function
            kf.H = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]])
            
            # Initial state covariance
            kf.P *= 10.0
            
            # Measurement noise covariance
            kf.R = np.eye(2) * measurement_noise_std ** 2
            
            # Process noise covariance
            q = process_noise_std
            kf.Q = np.array([[q, 0, 0, 0],
                             [0, q, 0, 0],
                             [0, 0, q, 0],
                             [0, 0, 0, q]])
            
            # Initial state: (x, y, vx, vy)
            x0, y0 = xy[0]
            kf.x = np.array([[x0],
                             [y0],
                             [0.0],
                             [0.0]])
            
            xy_smooth = []

            for pos in xy:
                kf.predict()
                kf.update(pos)
                xy_smooth.append(kf.x[:2, 0])

            xy_smooth = np.array(xy_smooth)
            xy_smooth_batch[b, d] = xy_smooth

    return xy_smooth_batch



##### ---- Funcs ---- #####
def smooth_data(input_data, window_length=21, polyorder=3):
    input_data = input_data.detach().numpy()
    smoothed_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        for j in range(input_data.shape[1]):
            for k in range(input_data.shape[3]):
                smoothed_data[i, j, :, k] = savgol_filter(input_data[i, j, :, k], window_length, polyorder)
    return torch.from_numpy(smoothed_data)

def offset2xyz(offset, start_xyz):
    '''
    input: 
        offset: (bs, dancer_num(3), sq, 3)
    output:
        xyz: (bs, dn, sq, 3)
    '''
    b,dn,s,c = offset.shape
    xyz = start_xyz 
    for i in range(1, s): 
        xyz = torch.cat(
            [xyz, start_xyz + torch.sum(offset[:, :, 0:i, :], dim=2, keepdim=True)], 
            dim=2)

    return xyz 

def Process_traj(traj, v_max = 0.01, var_frame = 15): 
     # Save the initial position of the trajectory
    start_xyz = traj[:, :, 0:1] 
    # xyz -> v: # Compute velocity offsets from position differences
    offset = traj[:,:,1:] - traj[:,:,:-1]

    # Clip velocities to stay within the maximum speed limit
    offset = torch.clamp(offset, -v_max, v_max)

    # Limit how often the velocity can change by holding it constant
    # over intervals of var_frame frames
    for i in range(0, offset.shape[2], var_frame):
        offset[:,:,i:i+var_frame] = offset[:,:,i:i+1]

    # v -> xyz: # Integrate velocity offsets back to absolute positions
    xyz = offset2xyz(offset, start_xyz)
    return xyz

    
def getCi(accLog):

    mean = np.mean(accLog)
    std = np.std(accLog)
    ci95 = 1.96*std/np.sqrt(len(accLog))

    return mean, ci95

def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger

## Optimizer
def initial_optim(decay_option, lr, weight_decay, net, optimizer) : 
    
    if optimizer == 'adamw' : 
        optimizer_adam_family = optim.AdamW
    elif optimizer == 'adam' : 
        optimizer_adam_family = optim.Adam
    if decay_option == 'all':
        #optimizer = optimizer_adam_family(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        optimizer = optimizer_adam_family(net.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=weight_decay)
        
    elif decay_option == 'noVQ':
        all_params = set(net.parameters())
        no_decay = set([net.vq_layer])
        
        decay = all_params - no_decay
        optimizer = optimizer_adam_family([
                    {'params': list(no_decay), 'weight_decay': 0}, 
                    {'params': list(decay), 'weight_decay' : weight_decay}], lr=lr)
        
    return optimizer


def get_motion_with_trans(motion, velocity) : 
    '''
    motion : torch.tensor, shape (batch_size, T, 72), with the global translation = 0
    velocity : torch.tensor, shape (batch_size, T, 3), contain the information of velocity = 0
    
    '''
    trans = torch.cumsum(velocity, dim=1)
    trans = trans - trans[:, :1] ## the first root is initialized at 0 (just for visualization)
    trans = trans.repeat((1, 1, 21))
    motion_with_trans = motion + trans
    return motion_with_trans
    