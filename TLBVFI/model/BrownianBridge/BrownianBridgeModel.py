import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np

from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel

# Helper for extracting schedule values
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class BrownianBridgeModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        model_params = model_config.BB.params
        self.num_timesteps = model_params.num_timesteps
        self.mt_type = model_params.mt_type
        self.max_var = model_params.max_var if model_params.__contains__("max_var") else 1
        self.eta = model_params.eta if model_params.__contains__("eta") else 1
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.steps = None
        self.register_schedule()

        self.objective = model_params.objective
        self.image_size = model_params.UNetParams.image_size
        self.channels = model_params.UNetParams.in_channels
        self.condition_key = model_params.UNetParams.condition_key

        self.denoise_fn = UNetModel(**vars(model_params.UNetParams))

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        
        m_tminus = np.append(0, m_t[:-1]) 
        variance_t = 2. * (m_t - m_t ** 2) * self.max_var 
        variance_tminus = np.append(0., variance_t[:-1]) 
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2 
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t 
        
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t.clamp(min=0))
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t).clamp(min=1e-12)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        elif self.objective == 'BB':
            x0_recon = -objective_recon + x_t 
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            objective_recon = self.denoise_fn(x_t, timesteps=t, cond = None, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape).float()
            m_nt = extract(self.m_t, n_t, x_t.shape).float()
            var_t = extract(self.variance_t, t, x_t.shape).float()
            var_nt = extract(self.variance_t, n_t, x_t.shape).float()
            
            eps = 1e-12
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt).clamp(min=eps) ** 2) * var_nt / var_t.clamp(min=eps)
            sigma_t = torch.sqrt(sigma2_t.clamp(min=0)) * self.eta

            noise = torch.randn_like(x_t).float()
            x_t_f = x_t.float()
            x0_recon_f = x0_recon.float()
            y_f = y.float()

            x_tminus_mean = (1. - m_nt) * x0_recon_f + m_nt * y_f + torch.sqrt(((var_nt - sigma2_t) / var_t.clamp(min=eps)).clamp(min=0)) * \
                            (x_t_f - (1. - m_t) * x0_recon_f - m_t * y_f)

            return (x_tminus_mean + sigma_t * noise).type(x_t.dtype), x0_recon

    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        img = y
        for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
            img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
        return img

    @torch.no_grad()
    def sample(self, y, z, context_y=None, context_z=None, clip_denoised=True):
        return self.p_sample_loop(y, z, context_y, context_z, clip_denoised)
