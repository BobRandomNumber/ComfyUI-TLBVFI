# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange,repeat

from model.utils import instantiate_from_config
from model.BrownianBridge.base.modules.attention import LinearAttention, SpatialCrossAttentionWithPosEmb
from model.BrownianBridge.base.modules.maxvit import SpatialCrossAttentionWithMax, MaxAttentionBlock

from VFI.archs.VFIformer_arch import VFIformer,FlowRefineNet_Multis,FlowRefineNet_Multis_our
from VFI.archs.warplayer import warp
import torch.nn.functional as F

from model.BrownianBridge.base.modules.diffusionmodules.util import GroupNorm32

def Normalize(in_channels, num_groups=4):
    return GroupNorm32(num_groups, in_channels, eps=1e-6, affine=True)

def Rearrange(x,frames = 3, back = False):
        if back:
            x = x.permute(0,2,1,3,4) ## B C F H W --> B F C H W
            x = rearrange(x,'b f c h w -> (b f) c h w') ## BF C H W
        else:
            x = torch.chunk(x,frames) ## F    B C H W
            x = torch.stack(x,2) ## B C F H W
        return x

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class IdentityWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x,ctx = None):
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)

        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)


        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
  
        if self.use_conv_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1)
        else:
            self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                out_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)    

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        h = h + x
        return h

class ResnetBlock_Dec(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        
        self.norm1 = Normalize(in_channels)

        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=(1,1,1),
                                     padding=(1,1,1))

        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=(1,1,1),
                                     padding=(0,1,1))

        if self.use_conv_shortcut:
            self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=(3,3,3),
                                                    stride=1,
                                                    padding=(0,1,1))
        else:
            self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=(3,1,1),
                                                    stride=(1,1,1),
                                                    padding=(0,0,0))

    def forward(self, x, temb):
        x = Rearrange(x)
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = Rearrange(h,back = True)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)

        h = Rearrange(h)
        h = self.conv2(h)

        if self.use_conv_shortcut:
            x = self.conv_shortcut(x)
        else:
            x = self.nin_shortcut(x)
        
        h = h + x
        h = h.squeeze(2)
        return h

class ResnetBlock_fusion(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        
        self.norm1 = Normalize(in_channels)

        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=(1,1,1),
                                     padding=(1,1,1))

        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=(1,1,1),
                                     padding=(1,1,1))

        if self.use_conv_shortcut:
            self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=(3,3,3),
                                                    stride=1,
                                                    padding=(1,1,1))
        else:
            self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        x = Rearrange(x)
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = Rearrange(h,back = True)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)

        h = Rearrange(h)
        h = self.conv2(h)

        if self.use_conv_shortcut:
            x = self.conv_shortcut(x)
        else:
            x = self.nin_shortcut(x)

        h = h + x
        h = Rearrange(h,back = True)
        return h

class LinAttnBlock(LinearAttention):
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, 1)
        self.k = torch.nn.Conv2d(in_channels, in_channels, 1)
        self.v = torch.nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x): 
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w).permute(0,2,1)
        k = k.reshape(b,c,h*w)
        w_ = torch.bmm(q.float(), k.float()) * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2).type(x.dtype)

        v = v.reshape(b,c,h*w)
        h_ = torch.bmm(v, w_.permute(0,2,1)).reshape(b,c,h,w)
        h_ = self.proj_out(h_)
        return x+h_

class STAttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(in_channels, in_channels, 1)
        self.k = torch.nn.Conv3d(in_channels, in_channels, 1)
        self.v = torch.nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = torch.nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x): 
        h_ = torch.chunk(x,3)
        h_ = torch.stack(h_,2)
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b,c,f,h,w = q.shape
        q = rearrange(q,'b c f h w -> b (f h w) c')
        k = rearrange(k,'b c f h w -> b c (f h w)')
        w_ = torch.bmm(q.float(), k.float()) * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2).type(x.dtype)

        v = rearrange(v,'b c f h w -> b c (f h w)')
        h_ = torch.bmm(v, w_.permute(0,2,1))
        h_ = rearrange(h_,'b c (f h w) -> b c f h w',f=f,h=h,w=w)
        h_ = self.proj_out(h_)
        h_ = h_.permute(0,2,1,3,4)
        h_ = rearrange(h_,'b f c h w -> (b f) c h w')
        return x+h_

def make_attn(in_channels, attn_type="vanilla"):
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    elif attn_type == 'max':
        return MaxAttentionBlock(in_channels, heads=1, dim_head=in_channels)
    else:
        return LinAttnBlock(in_channels)

def make_st_attn(in_channels, attn_type="vanilla"):
    return STAttnBlock(in_channels)

class WaveletTransform3D(torch.nn.Module):
    def __init__(self):
        super(WaveletTransform3D, self).__init__()
        low_filter = torch.tensor([1/2, 1/2], dtype=torch.float32).view(1, 1, -1)/torch.sqrt(torch.tensor(2.0))
        high_filter = torch.tensor([-1/2, 1/2], dtype=torch.float32).view(1, 1, -1)/torch.sqrt(torch.tensor(2.0))
        self.register_buffer('low_filter', low_filter, persistent=False)
        self.register_buffer('high_filter', high_filter, persistent=False)

    def conv1d_flat(self, x, filter, dim):
        if dim == 0:  # frames
            b, c, f, h, w = x.shape
            x_flat = x.permute(0, 1, 3, 4, 2).reshape(b * c * h * w, f)
            x_conv = F.conv1d(x_flat.unsqueeze(1).float(), filter.float(), padding=0, stride=1).squeeze(1).type(x.dtype)
            x = x_conv.view(b, c, h, w, -1).permute(0, 1, 4, 2, 3)
        elif dim == 1:  # height
            b, c, f, h, w = x.shape
            x_flat = x.permute(0, 1, 2, 4, 3).reshape(b * c * f * w, h)
            x_padded = F.pad(x_flat.unsqueeze(1).float(), (1, 0), mode='replicate')
            x_conv = F.conv1d(x_padded, filter.float(), padding=0, stride=1).squeeze(1).type(x.dtype)
            x = x_conv.view(b, c, f, w, -1).permute(0, 1, 2, 4, 3)
        elif dim == 2:  # width
            b, c, f, h, w = x.shape
            x_flat = x.permute(0, 1, 2, 3, 4).reshape(b * c * f * h, w)
            x_padded = F.pad(x_flat.unsqueeze(1).float(), (1, 0), mode='replicate')
            x_conv = F.conv1d(x_padded, filter.float(), padding=0, stride=1).squeeze(1).type(x.dtype)
            x = x_conv.view(b, c, f, h, -1)
        return x

    def forward(self, x):
        L = self.conv1d_flat(x, self.low_filter, 0)
        H = self.conv1d_flat(x, self.high_filter, 0)
        LL = self.conv1d_flat(L, self.low_filter, 1)
        LH = self.conv1d_flat(L, self.high_filter, 1)
        HL = self.conv1d_flat(H, self.low_filter, 1)
        HH = self.conv1d_flat(H, self.high_filter, 1)
        LLL = self.conv1d_flat(LL, self.low_filter, 2)
        LLH = self.conv1d_flat(LL, self.high_filter, 2)
        LHL = self.conv1d_flat(LH, self.low_filter, 2)
        LHH = self.conv1d_flat(LH, self.high_filter, 2)
        HLL = self.conv1d_flat(HL, self.low_filter, 2)
        HLH = self.conv1d_flat(HL, self.high_filter, 2)
        HHL = self.conv1d_flat(HH, self.low_filter, 2)
        HHH = self.conv1d_flat(HH, self.high_filter, 2)
        return LLL, torch.cat((LLH, LHL, LHH, HLL, HLH, HHL, HHH),dim = 2).squeeze(1)

class Frequency_block(nn.Module):
    def __init__(self,in_channels = 1, out_channels = 256):
        super().__init__()
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 5, 2, 2)
        self.norm2 = Normalize(out_channels)
        self.non_lin = nn.ReLU()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 5, 2, 2)
        self.shortcut = torch.nn.Conv2d(in_channels, out_channels, 5, 4, 1)
    def forward(self, x):
        h = self.norm1(x)
        h = self.non_lin(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.non_lin(h)
        h = self.conv2(h)
        x = self.shortcut(x)
        return h + x

class Frequency_extractor(nn.Module):
    def __init__(self, out_channels = 256,num_blocks = 5):
        super().__init__()
        self.num_blocks = (num_blocks-1)//2
        self.conv_in = torch.nn.Conv2d(21, 64, 3, 2, 1)
        Blocks = []
        for i in range(self.num_blocks-1):
            Blocks.append(Frequency_block(64,64))
        Blocks.append(Frequency_block(64,out_channels))
        self.Blocks = torch.nn.Sequential(*Blocks)
    def forward(self, x, temb):
        x = self.conv_in(x)
        x = self.Blocks(x)
        return x

class FIEncoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.wavelet_transform = WaveletTransform3D()
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, 3, 1, 1)
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        curr_res = resolution
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = int(ch*in_ch_mult[i_level])
            block_out = int(ch*ch_mult[i_level])
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=0, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            down.downsample = Downsample(block_in, resamp_with_conv)
            curr_res = curr_res // 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock_fusion(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout)
        self.mid.attn_1 = make_st_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock_fusion(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout)
        self.frequency_extractor = Frequency_extractor(block_in,self.num_resolutions)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in, 2*z_channels if double_z else z_channels, 3, 1, 1)

    def forward(self, x, ret_feature=False):
        if x.min() < 0: x = x/2 + 0.5
        vid = Rearrange(x).clone().detach()
        grayscale_coeffs = torch.tensor([0.2989, 0.5870, 0.1140], dtype=x.dtype, device=x.device).view(1,3,1,1,1)
        vid = (vid * grayscale_coeffs).sum(dim = 1, keepdim = True)
        low_freq, high_freq_1 = self.wavelet_transform(vid) 
        low_freq, high_freq_2 = self.wavelet_transform(low_freq)
        high_freq = torch.cat([high_freq_1,high_freq_2],dim = 1)
        high_freq_fea = self.frequency_extractor(high_freq,None)
        phi_list = []
        hs = [self.conv_in(x)]
        reshaped = torch.chunk(hs[-1],3)
        phi_list.append(torch.cat([reshaped[0],reshaped[-1]]))
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], None)
                if len(self.down[i_level].attn) > 0: h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            hs.append(self.down[i_level].downsample(hs[-1]))
            reshaped = torch.chunk(hs[-1],3)
            phi_list.append(torch.cat([reshaped[0],reshaped[-1]]))
        h = hs[-1]
        h = Rearrange(h)
        h = h  + torch.sigmoid(high_freq_fea.unsqueeze(2))*h
        h = Rearrange(h,back = True)
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = Rearrange(h)
        h = self.conv_out(h)
        h = Rearrange(h,back = True)
        return h, phi_list

class FlowEncoder(FIEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class FlowDecoderWithResidual(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", num_head_channels=32, num_heads=1, cond_type=None,load_VFI = None,
                 **ignorekwargs):
        super().__init__()
        def OutputHead(c_in):
            return torch.nn.Sequential(
                    torch.nn.Conv2d(c_in, 64, 3, 1, 1), Normalize(64, 16), torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 32, 3, 1, 1), Normalize(32, 8), torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 4, 3, 1, 1))
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end
        vfi = VFIformer()
        self.flownet = vfi.flownet
        self.refinenet = FlowRefineNet_Multis_our(c = self.ch)
        for p in self.flownet.parameters(): p.requires_grad = False
        block_in = int(ch*ch_mult[self.num_resolutions-1])
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.conv_in = torch.nn.Conv3d(z_channels, block_in, 3, 1, 1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock_fusion(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout)
        self.mid.attn_1 = make_st_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock_Dec(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block, fusion, attn = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
            block_out = int(ch*ch_mult[i_level])
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=0, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions: attn.append(make_attn(block_in, attn_type=attn_type))
            if num_head_channels == -1: dim_head = block_in // num_heads
            else: num_heads, dim_head = block_in // num_head_channels, num_head_channels
            if cond_type == 'cross_attn':
                cross_attn = SpatialCrossAttentionWithPosEmb(in_channels=block_in, heads=num_heads, dim_head=dim_head)
            elif cond_type == 'max_cross_attn':
                ctx_dim = block_in*2 if i_level > 2 else block_in*4
                cross_attn = SpatialCrossAttentionWithMax(in_channels=block_in, heads=num_heads, dim_head=dim_head, ctx_dim=ctx_dim)
            else: cross_attn = IdentityWrapper()
            up = nn.Module()
            up.block, up.attn, up.cross_attn, up.fusion = block, attn, cross_attn, fusion
            up.upsample = Upsample(block_in, resamp_with_conv)
            curr_res *= 2
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, block_in, 3, 1, 1)
        self.moduleout = OutputHead(c_in=block_in)

    def forward(self, z, cond_dict,flow = None):
        phi_list, frame_prev, frame_next = cond_dict['phi_list'], cond_dict['frame_prev'], cond_dict['frame_next']
        back = False
        if frame_prev.min() < 0:
            back, frame_prev, frame_next = True, frame_prev/2 + 0.5, frame_next/2 + 0.5
        B, _, H, W = frame_prev.size()
        if flow is not None: _, c0, c1 = self.refinenet.get_context(phi_list[:-2], flow)
        else:
            flow, _ = self.flownet(torch.cat((frame_prev, frame_next), 1))
            flow, c0, c1 = self.refinenet(phi_list[:-2], flow)
        phi_list, c0, c1 = phi_list[1:], c0[1:], c1[1:]
        warped_img0, warped_img1 = warp(frame_prev, flow[:, :2]), warp(frame_next, flow[:, 2:])
        z = Rearrange(z)
        h = self.conv_in(z)
        h = Rearrange(h,back = True)
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)
        for i_level in reversed(range(self.num_resolutions)):
            ctx = None
            if phi_list[i_level] is not None:
                if i_level > 2: ctx = torch.cat([phi_list[i_level][:B],phi_list[i_level][B:]],dim =1)
                else: ctx = torch.cat([phi_list[i_level][:B],phi_list[i_level][B:],c0[i_level],c1[i_level]], dim=1)
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h, None)
                if len(self.up[i_level].attn) > 0: h = self.up[i_level].attn[i_block](h)
            h = self.up[i_level].cross_attn(h, ctx)
            h = self.up[i_level].upsample(h)
        if self.give_pre_end: return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        out = self.moduleout(h)
        mask1 = torch.sigmoid(out[:,3:4])
        mask2, res = 1 - mask1, torch.sigmoid(out[:,:3])*2 - 1
        out = (warped_img0.float()*mask1.float() + warped_img1.float()*mask2.float() + res.float()).type(warped_img0.dtype)
        if back: out = (out.clamp(0, 1)*2 - 1)
        else: out = out.clamp(-1, 1)
        return out

    def get_flow(self, img0, img1,feats):
        flow, _ = self.flownet(torch.cat((img0, img1), 1))
        flow, _, _ = self.refinenet(feats,flow)
        return flow
