import copy
import os
import pickle
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from p_tqdm import p_map
from pytorch3d.transforms import (axis_angle_to_quaternion,
                                  quaternion_to_axis_angle)
from tqdm import tqdm

from dataset.quaternion import ax_from_6v, quat_slerp
from vis import skeleton_render

from .utils import extract, make_beta_schedule
import copy


def identity(t, *args, **kwargs):
    return t

def offset2xy(offset):
    '''
    input: 
        offset: (bs, sq, dancer_num(3), 3)
    output:
        xy: (bs, sq*dn, 2+1)
    '''
    b,s,dn,c = offset.shape
    xy = offset[:, :1, :, :2] 
    for i in range(1, s): 
        xy = torch.cat(
            [xy, offset[:, :1, :, :2] + torch.sum(offset[:, 1:i, :, :2], dim=1, keepdim=True)], 
            dim=1)
    xyz = torch.cat([xy, offset[:, :, :, 2:]], dim=3)
    return xyz.reshape(b,-1,3)


def offset2xyz(offset):
    '''
    input: 
        offset: (bs, sq, dancer_num(3), 3)
    output:
        xyz: (bs, sq*dn, 3)
    '''
    b,s,dn,c = offset.shape
    xyz = offset[:, :1, :, :] 
    for i in range(1, s): 
        xyz = torch.cat(
            [xyz, offset[:, :1, :, :] + torch.sum(offset[:, 1:i, :, :], dim=1, keepdim=True)], 
            dim=1)

    return xyz.reshape(b,-1,3)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        repr_dim,
        smpl,
        n_timestep=1000,
        schedule="linear",
        loss_type="l1",
        clip_denoised=True,
        predict_epsilon=True,
        guidance_weight=3,
        use_p2=False,
        cond_drop_prob=0.2,
        seq_len = 150,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model
        self.ema = EMA(0.9999)
        self.master_model = copy.deepcopy(self.model)
        self.seq_len = seq_len

        self.cond_drop_prob = cond_drop_prob

        # make a SMPL instance for FK module
        self.smpl = smpl

        betas = torch.Tensor(
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timestep = int(n_timestep)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.guidance_weight = guidance_weight

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # p2 weighting
        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 0.5 if use_p2 else 0
        self.register_buffer(
            "p2_loss_weight",
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -self.p2_loss_weight_gamma,
        )

        ## get loss coefficients and initialize objective
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def model_predictions(self, x, cond, t, weight=None, clip_x_start = False):
        weight = weight if weight is not None else self.guidance_weight
        model_output = self.model.guided_forward(x, cond, t, weight) # torch.Size([2, 150, 453])
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        x_start = model_output
        x_start = maybe_clip(x_start) # torch.Size([2, 150, 453])
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start # torch.Size([2, 150, 453])

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        # guidance clipping
        if t[0] > 1.0 * self.n_timestep:
            weight = min(self.guidance_weight, 0)
        elif t[0] < 0.1 * self.n_timestep:
            weight = min(self.guidance_weight, 1)
        else:
            weight = self.guidance_weight

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model.guided_forward(x, cond, t, weight)
        )

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, cond=cond, t=t
        )
        noise = torch.randn_like(model_mean)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(noise.shape) - 1))
        )
        x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_out, x_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        # default to diffusion over whole timescale
        start_point = self.n_timestep if start_point is None else start_point
        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)

        if return_diffusion:
            diffusion = [x]

        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, _ = self.p_sample(x, cond, timesteps)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x
    
    @torch.no_grad() 
    def ddim_sample_Footwork(self, shape, cond, x_0 = None, **kwargs):
        # We implemented code to facilitate experiments focusing on the control of lower-limb movements.
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        if x_0 is not None:
            x_0 = x_0.to(device)
            _, seq, c = x_0.shape
            x_0 = x_0.reshape(-1, 150, seq//150, c)
            x = x.reshape(-1, 150, seq//150, c)
            x[:,:,:,[4,4+1]] = x_0[:,:,:,[0,1]]

            # Replace the displacement data for footsteps 1, 2, 3, 4, 5, 7, 8, 10, and 11
            for i in [1,2,3,4,5,7,8,10,11]:
                x[:,75:120,:,4+3+(i-1)*6:4+3+i*6] = x_0[:,75:120,:,4+3+(i-1)*6:4+3+i*6] 
            x = x.reshape(-1, seq, c)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            if x_0 is not None: 
                # Replace the x_past segment with the ground truth
                _, seq, dn, c = x_0.shape
                x = x.reshape(-1, seq, dn, c)

                # Replace the displacement components with those from x_0
                x[:,:,:,[4,4+1]] = x_0[:,:,:,[0,1]]

                # Replace the displacement data for specific footstep segments
                for i in [1,2,3,4,5,7,8,10,11]:
                    x[:,75:120,:,4+3+(i-1)*6:4+3+i*6] = x_0[:,75:120,:,4+3+(i-1)*6:4+3+i*6] 
            
                x = x.reshape(-1, seq*dn, c)
        

        if x_0 is not None: # At t == 0, step is skipped, so an extra update is necessary.
            _, seq, dn, c = x_0.shape
            x = x.reshape(-1, seq, dn, c)

            # Replace the initial displacement components (e.g. global XY translation)
            x[:,:,:,[4,4+1]] = x_0[:,:,:,[0,1]]
            
            # --- Linear interpolation fusion for specific footstep segments ---

            # The following code performs partial replacement and fusion of displacement data
            # for specific footstep segments. Instead of a hard overwrite, it uses linear
            # interpolation at the sequence boundaries to ensure smooth transitions.

            # Define the width of the interpolation window (number of frames)
            width = 10
            
            # Generate weights for linear interpolation, from 0 to 1 across the defined window.
            # Shape: (1, width, 1, 1) to broadcast across batch and feature dimensions.
            weights = torch.from_numpy(np.linspace(0, 1, width)).to(x)
            weights = weights[None, :, None, None]

            # Loop over the selected footsteps indices (each corresponds to a segment of rotation data)
            for i in [1,2,3,4,5,7,8,10,11]:
                # Interpolate the start segment (smooth transition from original x to x_0)
                x[:, 75:75 + width, :,4+3+(i-1)*6:4+3+i*6] = weights * x_0[:,75:75 + width, :,4+3+(i-1)*6:4+3+i*6] + (1-weights) * x[:, 75:75 + width, :,4+3+(i-1)*6:4+3+i*6] 

                # Fully replace the middle segment with values from x_0
                x[:, 75 + width : -width, :,4+3+(i-1)*6:4+3+i*6] = x_0[:,75 + width : -width,:,4+3+(i-1)*6:4+3+i*6] 

                # Interpolate the end segment (smooth transition back to original x)
                x[:, 120-width:120,:,4+3+(i-1)*6:4+3+i*6] = (1-weights) * x_0[:,120-width:120,:,4+3+(i-1)*6:4+3+i*6] + weights * x[:, 120-width:120,:,4+3+(i-1)*6:4+3+i*6] 

            x = x.reshape(-1, seq*dn, c)

        return x # torch.Size([bs, seq*dn, 151])

    @torch.no_grad() 
    def ddim_sample(self, shape, cond, x_0 = None, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        if x_0 is not None:
            x_0 = x_0.to(device)
            _, seq, _ = x_0.shape 
            x_0 = x_0.reshape(-1, 150, seq//150, 3)
            x = x.reshape(-1, 150, seq//150, 151)

            x[:,:,:,[4,4+1]] = x_0[:,:,:,[0,1]]
            x = x.reshape(-1, seq, 151)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            if x_0 is not None: # replace x- and y-axis trajectory
                _, seq, dn, _ = x_0.shape
                x = x.reshape(-1, seq, dn, 151)
                x[:,:,:,[4,4+1]] = x_0[:,:,:,[0,1]]
                x = x.reshape(-1, seq*dn, 151)
        

        if x_0 is not None: # At t == 0, step is skipped, so an extra update is necessary.
            _, seq, dn, _ = x_0.shape
            x = x.reshape(-1, seq, dn, 151)
            x[:,:,:,[4,4+1]] = x_0[:,:,:,[0,1]] # replace x- and y-axis trajectory
 
            
            x = x.reshape(-1, seq*dn, 151)

        return x # torch.Size([2, seq*dn, 151])


    @torch.no_grad() 
    def long_ddim_sample(self, shape, cond, x_0, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1
        
        if batch == 1:
            return self.ddim_sample(shape, cond)

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        weights = np.clip(np.linspace(0, self.guidance_weight * 2, sampling_timesteps), None, self.guidance_weight)
        time_pairs = list(zip(times[:-1], times[1:], weights)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)
        
        if x_0 is not None:
            x_0 = x_0.to(device)
            b,seq,dn,_ = x_0.shape 
            x_0 = x_0.reshape(-1, seq, dn, 3)
            x = x.reshape(-1, seq, dn, 151)
            x[:,:,:,[4,4+1]] = x_0[:,:,:,[0,1]]
            x = x.reshape(-1, seq*dn, 151)

        assert batch > 1
        assert self.seq_len%2 ==0
        half = self.seq_len // 2

        x_start = None

        for time, time_next, weight in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, weight=weight, clip_x_start = self.clip_denoised) 

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            # Compute x_{t-1} from x_0 using the DDPM update rule
            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            if x_0 is not None: 
                # Replace the x_past portion with the ground truth values
                _, seq, dn, _ = x_0.shape
                x = x.reshape(-1, seq, dn, 151)
                x[:,:,:,[4,4+1]] = x_0[:,:,:,[0,1]]
                x = x.reshape(-1, seq*dn, 151)

            if time > 0: 
                # the first half of each sequence is the second half of the previous one
                x = x.reshape(shape[0], self.seq_len, shape[1]//self.seq_len, shape[2])
                x[1:, :half] = x[:-1, half:]
                x = x.reshape(shape[0],-1,shape[2])
        
        if x_0 is not None: 
            # Replace the x_past portion with the ground truth values again after all updates
            _, seq, dn, _ = x_0.shape
            x = x.reshape(-1, seq, dn, 151)
            x[:,:,:,[4,4+1]] = x_0[:,:,:,[0,1]]
            x = x.reshape(-1, seq*dn, 151)
                
        return x # (slice_num(18), sq(150), 453)


    @torch.no_grad()
    def inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
            # value_ = self.q_sample(value, timesteps - 1) if (i > 0) else value # This may cause abrupt positional changes, but it can be useful for verifying whether the input motion is reasonable.
            x = value_ * mask + (1.0 - mask) * x

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def long_inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        assert x.shape[1] % 2 == 0
        if batch_size == 1:
            # there's no continuation to do, just do normal
            return self.p_sample_loop(
                shape,
                cond,
                noise=noise,
                constraint=constraint,
                return_diffusion=return_diffusion,
                start_point=start_point,
            )
        assert batch_size > 1
        half = x.shape[1] // 2

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            if i > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:] 

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def conditional_sample(
        self, shape, cond, constraint=None, *args, horizon=None, **kwargs
    ):
        """
            conditions : [ (time, state), ... ]
        """
        device = self.betas.device
        horizon = horizon or self.horizon

        return self.p_sample_loop(shape, cond, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = ( # Forward equation of DPM, used to derive x_t at any timestep from x_start. The loss_simple objective is applied here.
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, trj_dist = None): 

        # (bs, dancer_num, 150, 151) -> (bs, 150, dancer_num, 151)
        bs, dancer_num, sq, c = x_start.shape
        x_start = x_start.permute(0,2,1,3) # (bs, dancer_num, seq_len, 151) - > (bs, seq_len, dancer_num, 151)

        # # Transition from full-noise to partial-noise scheduling, where noise is added only to the latter half of the data sequence.
        # This aims to enhance the model’s ability to retain temporal information.
        noise = torch.randn_like(x_start) # b, sq//2, dn, c
        x_noisy= self.q_sample(x_start=x_start, t=t, noise=noise)

        # In this variant (outputting x_0), no noise is added to the trajectory part;
        # it is restored via direct value assignment.
        # The trajectory data is also fed into the modulation module.
        x_noisy[:,:,:,[4,4+1]] = x_start[:,:,:,[4,4+1]]
        x_noisy = x_noisy.reshape(bs,sq*dancer_num, c)

        # reconstruct
        x_recon = self.model(x_noisy, cond, t, cond_drop_prob=self.cond_drop_prob, trj_dist = trj_dist) # (bs, 150, dancer_num, 151) | model.decoder

        model_out = x_recon
        if self.predict_epsilon: # origin DPM
            target = noise
        else: # this one
            target = x_start


        # split off contact from the rest
        model_out = model_out.reshape(bs, sq, dancer_num, c) # (bs, sq(150), dancer_num(3), c(151))
        target = target.reshape(bs, sq, dancer_num, c)

        # full reconstruction loss
        loss = self.loss_fn(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        model_contact, model_out = torch.split(
            model_out, (4, model_out.shape[3] - 4), dim=3 # (bs, sq(150), dancer_num(3), 3 + 22*6 == 147)
        )
        target_contact, target = torch.split(target, (4, target.shape[3] - 4), dim=3) # (bs, sq(150), dancer_num(3), 4)

        # velocity loss 
        target_v = target[:, 1:] - target[:, :-1]
        model_out_v = model_out[:, 1:] - model_out[:, :-1]
        v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

        # FK loss
        b, s, dancer_num, c = model_out.shape 

        # unnormalize
        # model_out = self.normalizer.unnormalize(model_out)
        # target = self.normalizer.unnormalize(target)

        # X, Q
        model_x = model_out[:, :, :,:3] # root | (bs, sq, dancer_num(3), 3) -> (bs, sq * dancer_num(3), 3)
        model_q = ax_from_6v(model_out[:, :, :, 3:].reshape(b, s*dancer_num, -1, 6)) # body | (bs, sq, dancer_num(3), c-3) -> (bs, sq*dancer_num(3), (c-3)) -> (bs, sq*dancer_num(3), 24, 6)
        target_x = target[:, :, :,:3] # gt root | (bs, sq, dancer_num(3), 3)
        target_q = ax_from_6v(target[:, :, :, 3:].reshape(b, s*dancer_num, -1, 6)) # gt body | (bs, sq, dancer_num(3), c-3) -> (bs, sq*dancer_num(3), (c-3)) -> (bs, sq*dancer_num(3), 24, 6)

        # offset -> xyz | or not
        # model_x = offset2xyz(model_x) # [2, 453, 3]
        # target_x = offset2xyz(target_x)
        # model_x = offset2xy(model_x) # [2, 453, 3]
        # target_x = offset2xy(target_x)

        model_x = model_x.reshape(b, s*dancer_num, -1)
        target_x = target_x.reshape(b, s*dancer_num, -1)

        # perform FK
        model_xp = self.smpl.forward(model_q, model_x) # (bs, sq*dancer_num(3), 24, 3)
        target_xp = self.smpl.forward(target_q, target_x)

        # compute relative distence
        model_relative = model_xp[:,:,1:] - model_xp[:,:,0:1]
        target_relative = target_xp[:,:,1:] - target_xp[:,:,0:1]

        # relative fk loss
        fk_loss = self.loss_fn(model_relative, target_relative, reduction="none") 
        fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")
        fk_loss = fk_loss * extract(self.p2_loss_weight, t, fk_loss.shape)

        # foot skate loss
        foot_idx = [7, 8, 10, 11]

        static_idx = model_contact > 0.95  # N x S x 4 -> N x S x dancer_num x 4
        model_xp = model_xp.reshape(b, s, dancer_num, 24, 3) # (bs, sq, dancer_num(3), 24, 3)
        model_feet = model_xp[:, :, :,foot_idx]  # foot positions (bs, sq, dancer_num(3), 4, 3)
        model_foot_v = torch.zeros_like(model_feet)
        model_foot_v[:, :-1] = ( # v (bs, sq, dancer_num(3), 4, 3)
            model_feet[:, 1:, :, :, :] - model_feet[:, :-1, :, :, :]
        )  # (N, S-1, dancer_num(3), 4, 3)
        model_foot_v[~static_idx] = 0
        foot_loss = self.loss_fn(
            model_foot_v, torch.zeros_like(model_foot_v), reduction="none"
        )
        foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")

        losses = (
            0.636 * loss.mean(),
            2.964 * v_loss.mean(),
            0.646 * fk_loss.mean(),
            10.942 * foot_loss.mean(),  
        )
        return sum(losses), losses


    def loss(self, x, cond, t_override=None, trj_dist = None): 
        batch_size = len(x)
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x.device).long()
        else:
            t = torch.full((batch_size,), t_override, device=x.device).long()
        return self.p_losses(x, cond, t, trj_dist = trj_dist)

    def forward(self, x, cond, t_override=None, trj_dist = None): 
        return self.loss(x, cond, t_override, trj_dist = trj_dist)
    

    def partial_denoise(self, x, cond, t):
        x_noisy = self.noise_to_t(x, t)
        return self.p_sample_loop(x.shape, cond, noise=x_noisy, start_point=t)

    def noise_to_t(self, x, timestep):
        batch_size = len(x)
        t = torch.full((batch_size,), timestep, device=x.device).long()
        return self.q_sample(x, t) if timestep > 0 else x

    def render_sample( 
        self,
        shape,
        cond,
        normalizer,
        epoch,
        render_out,
        fk_out=None,
        name=None,
        sound=True,
        mode="normal",
        noise=None,
        constraint=None,
        sound_folder="ood_sliced",
        start_point=None,
        render=True,
        required_dancer_num = 4,
        x_0 = None,
        render_len = 512,
    ):
        if isinstance(shape, tuple):
            if mode == "inpaint": # Inpainting mode: fills in missing frames within an existing sequence
                func_class = self.inpaint_loop
            elif mode == "normal":# Standard generation mode for regular-length motion sequences
                func_class = self.ddim_sample
            elif mode == "long": # Long generation mode for producing extended motion sequences
                func_class = self.long_ddim_sample
            elif mode == "ctrl": # Control mode for motion retargeting based on given trajectories and partial footstep constraints
                func_class = self.ddim_sample_Footwork
            else:
                assert False, "Unrecognized inference mode"
            samples = (
                func_class(
                    shape,
                    cond,
                    noise=noise,
                    constraint=constraint,
                    start_point=start_point,
                    x_0 = x_0,
                )
                .detach()
                .cpu()
            )
        else:
            samples = shape

        # (b, s*dancer_num, c)
        b,s,_ = samples.shape # [*, 450, 151]
        samples = normalizer.unnormalize(samples) # torch.Size([2, 150*dancer_num, 151])

        # (b, s*dancer_num, c) -> (b, s, dancer_num, c)
        samples = samples.reshape(b, 150, s//150, 151) # [*, 150, 3, 151]
        b, s, ds, c = samples.shape # In long_DDPM_sample, b refers to the number of audio segments rather than the batch size.

        if len(samples.shape) == 4 and samples.shape[3] == 151: # (b, s, dancer_num, c)
            sample_contact, samples = torch.split(
                samples, (4, samples.shape[3] - 4), dim=3
            )
        else:
            sample_contact = None

        # do the FK all at once
            # pos offset -> xyz
        # We tried transforming the offset representation into absolute 3D or 2D positions,
            # but this approach did not show significant benefits:
        # pos = offset2xyz(samples[:, :, :, :3]).to(cond.device)  # (b, s*dn, 3)
        # pos = offset2xy(samples[:, :, :, :3]).to(cond.device)   # (b, s*dn, 3)

        samples = samples.reshape(b,-1,c-4) # (b, s, dancer_num, c-4(147)) -> (b, s*dancer_num, c-4(147))
        b, s, c = samples.shape # (b, s*dancer_num, c-4(147))
        pos = samples[:, :, :3].to(cond.device)  # (b, s*dancer_num, 3) | np.zeros((sample.shape[0], 3))
        q = samples[:, :, 3:].reshape(b, s, 24, 6) # (b, s*dancer_num, 24, 6)
        # go 6d to ax
        q = ax_from_6v(q).to(cond.device) # (b, s*dancer_num, 24, 3)


        if mode == "long": #For testing, outputs concatenated motion data (pos, q). When a full audio clip is input, b denotes the number of audio slices.
            b, _, c1, c2 = q.shape
            pos = pos.reshape(b, 150, required_dancer_num, -1) # (b, s*dancer_num, 3) -> (b, s, dancer_num, 3)
            q = q.reshape(b, 150, required_dancer_num, c1, c2) # (b, s*dancer_num, 24, 3) -> (b, s, dancer_num, 24, 3)
            b, s, dn, c1, c2 = q.shape
            assert s % 2 == 0
            half = s // 2

            if b >= 1: # If there is more than one slice
                # Concatenate each dancer's motion separately
                pos_all = torch.tensor([]).to(pos.device)
                q_all = torch.tensor([]).to(pos.device)
                for dancer_i in range(dn):
                    cur_pos = pos[:, :, dancer_i, :].reshape(b, 150, -1)    
                    cur_q = q[:, :, dancer_i, :, :].reshape(b, 150, c1, c2)  
                    # if long mode, stitch position using linear interp
                    fade_out = torch.ones((1, s, 1)).to(cur_pos.device)
                    fade_in = torch.ones((1, s, 1)).to(cur_pos.device)
                    fade_out[:, half:, :] = torch.linspace(1, 0, half)[None, :, None].to(
                        cur_pos.device
                    )
                    fade_in[:, :half, :] = torch.linspace(0, 1, half)[None, :, None].to(
                        cur_pos.device
                    )

                    cur_pos[:-1] *= fade_out
                    cur_pos[1:] *= fade_in

                    full_pos = torch.zeros((s + half * (b - 1), 3)).to(cur_pos.device)
                    idx = 0
                    for pos_slice in cur_pos:
                        full_pos[idx : idx + s] += pos_slice
                        idx += half

                    # stitch joint angles with slerp
                    slerp_weight = torch.linspace(0, 1, half)[None, :, None].to(cur_pos.device)

                    left, right = cur_q[:-1, half:], cur_q[1:, :half]
                    # convert to quat
                    left, right = (
                        axis_angle_to_quaternion(left),
                        axis_angle_to_quaternion(right),
                    )
                    merged = quat_slerp(left, right, slerp_weight)  # (b-1) x half x ...
                    # convert back
                    merged = quaternion_to_axis_angle(merged)

                    full_q = torch.zeros((s + half * (b - 1), c1, c2)).to(cur_pos.device)
                    full_q[:half] += cur_q[0, :half]
                    idx = half
                    for q_slice in merged:
                        full_q[idx : idx + half] += q_slice
                        idx += half
                    full_q[idx : idx + half] += cur_q[-1, half:]

                    pos_all = torch.concat([pos_all,full_pos.reshape(-1, 1, 3)],dim = 1) # (b*s, dancer_num, 3)
                    q_all = torch.concat([q_all,full_q.reshape(-1, 1, 24, 3)],dim = 1) # (b*s, dancer_num, 24, 3)

                # reshape for fk 
                bs, dn, _ = pos_all.shape
                full_pos = pos_all.reshape(1, bs*dn, 3) # (b*s, dancer_num, 3) -> (1, b*s, dancer_num, 3)
                full_q = q_all.reshape(1, bs*dn, 24, 3) # (1, b*s, dancer_num, 24, 3)

            else:
                full_pos = pos
                full_q = q

            full_q = full_q.reshape(1,-1,24,3)
            full_pos = full_pos.reshape(1,-1,3)
            full_pose = (
                self.smpl.forward(full_q, full_pos).detach().cpu().numpy()
            )  # b, s*dancer_num, 24, 3

            # reshape (1, b*s*dancer_num, 24, 3) -> (1, b*s, dancer_num, 24, 3) -> (1, dancer_num, b*s, 24, 3)
            
            full_pose = full_pose.reshape(1, bs, dn, 24, 3)
            full_pose = np.transpose(full_pose, (0,2,1,3,4)) # (b, s, dancer_num, 24, 3) -> (b, dancer_num, s, 24, 3)

            # squeeze the batch dimension away and render
            skeleton_render(
                full_pose[0,:,:render_len], # (b, dancer_num, s, 24, 3) -> (dancer_num, s, 24, 3)
                epoch=f"{epoch}",
                out=render_out,
                name=name,
                sound=sound,
                stitch=True,
                sound_folder=sound_folder,
                render=render
            )
            if fk_out is not None: # If None, this branch needs to be executed to save the output FK SMPL data.
                outname = f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.pkl'
                Path(fk_out).mkdir(parents=True, exist_ok=True)
                pickle.dump(
                    {
                        "smpl_poses": full_q.squeeze(0).reshape((-1, 72)).cpu().numpy(),
                        "smpl_trans": full_pos.squeeze(0).cpu().numpy(),
                        "full_pose": full_pose[0],
                    },
                    open(os.path.join(fk_out, outname), "wb"),
                )
            return
        poses = self.smpl.forward(q, pos).detach().cpu().numpy() # [2, 450, 24, 3] [2, 3, 3] | (b, s*dancer_num, 24, 3), key points
        # permute contact (b, seq, dancer_num, 4) -> (b, dancer_num, seq, 4)
        sample_contact = np.transpose(sample_contact,(0,2,1,3))

        sample_contact = (
            sample_contact.detach().cpu().numpy() 
            if sample_contact is not None
            else None
        )

        b = poses.shape[0]
        poses = poses.reshape(b,-1,required_dancer_num,24,3)
        poses = np.transpose(poses,(0,2,1,3,4)) # (b, s, dancer_num, 24, 3) -> (b, dancer_num, s, 24, 3)

        def inner(xx):
            num, pose = xx
            filename = name[num] if name is not None else None
            contact = sample_contact[num] if sample_contact is not None else None
            skeleton_render(
                pose,
                epoch=f"e{epoch}_b{num}",
                out=render_out,
                name=filename,
                sound=sound,
                contact=contact,
            )

        p_map(inner, enumerate(poses)) # perform render here

        if fk_out is not None and mode != "long": 
            Path(fk_out).mkdir(parents=True, exist_ok=True)
            for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
                path = os.path.normpath(filename)
                pathparts = path.split(os.sep)
                pathparts[-1] = pathparts[-1].replace("npy", "wav")
                # path is like "data/train/features/name"
                pathparts[2] = "wav_sliced"
                audioname = os.path.join(*pathparts)
                outname = f"{epoch}_{num}_{pathparts[-1][:-4]}.pkl"
                pickle.dump(
                    {
                        "smpl_poses": qq.reshape((-1, 72)).cpu().numpy(),
                        "smpl_trans": pos_.cpu().numpy(),
                        "full_pose": pose,
                    },
                    open(f"{fk_out}/{outname}", "wb"),
                )
    