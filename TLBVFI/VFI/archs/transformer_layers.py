import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from model.utils import DropPath, to_2tuple, trunc_normal_
sys.path.append('../..')
from VFI.archs.warplayer import warp

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn.float()).type(attn.dtype)
        else:
            attn = self.softmax(attn.float()).type(attn.dtype)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WindowCrossAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table_x = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        self.relative_position_bias_table_y = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.merge1 = nn.Linear(dim*2, dim)
        self.merge2 = nn.Linear(dim, dim)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table_x, std=.02)
        trunc_normal_(self.relative_position_bias_table_y, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask_x=None, mask_y=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())

        relative_position_bias = self.relative_position_bias_table_x[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask_x is not None:
            nW = mask_x.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask_x.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn.float()).type(attn.dtype)
        else:
            attn = self.softmax(attn.float()).type(attn.dtype)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()

        B_, N, C = y.shape
        kv = self.kv(y).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1).contiguous())
        relative_position_bias = self.relative_position_bias_table_y[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask_y is not None:
            nW = mask_y.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask_y.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn.float()).type(attn.dtype)
        else:
            attn = self.softmax(attn.float()).type(attn.dtype)

        attn = self.attn_drop(attn)
        y = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        x = self.merge2(self.act(self.merge1(torch.cat([x, y], dim=-1)))) + x
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TFL(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_crossattn=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_crossattn = use_crossattn

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        if not use_crossattn:
            self.attn = WindowAttention(
                dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = WindowCrossAttention(
                dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            if not use_crossattn:
                attn_mask = self.calculate_mask(self.input_resolution)
                self.register_buffer("attn_mask", attn_mask)
            else:
                attn_mask_x = self.calculate_mask(self.input_resolution)
                attn_mask_y = self.calculate_mask2(self.input_resolution)
                self.register_buffer("attn_mask_x", attn_mask_x)
                self.register_buffer("attn_mask_y", attn_mask_y)
        else:
            if not use_crossattn:
                attn_mask = None
                self.register_buffer("attn_mask", attn_mask)
            else:
                attn_mask_x = None
                attn_mask_y = None
                self.register_buffer("attn_mask_x", attn_mask_x)
                self.register_buffer("attn_mask_y", attn_mask_y)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def calculate_mask2(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        img_mask_down = F.interpolate(img_mask.permute(0, 3, 1, 2).contiguous(), scale_factor=0.5, mode="bilinear", align_corners=False)
        img_mask_down = F.pad(img_mask_down, (self.window_size//4, self.window_size//4, self.window_size//4, self.window_size//4), mode='reflect')
        mask_windows_down = F.unfold(img_mask_down, kernel_size=self.window_size, dilation=1, padding=0, stride=self.window_size//2)
        mask_windows_down = mask_windows_down.view(self.window_size*self.window_size, -1).permute(1, 0).contiguous()

        attn_mask = mask_windows_down.unsqueeze(1) - mask_windows_down.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if not self.use_crossattn:
            if self.input_resolution == x_size:
                attn_windows = self.attn(x_windows, mask=self.attn_mask)
            else:
                attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))
        else:
            shifted_x_down = F.interpolate(shifted_x.permute(0, 3, 1, 2).contiguous(), scale_factor=0.5, mode="bilinear", align_corners=False)
            shifted_x_down = F.pad(shifted_x_down, (self.window_size//4, self.window_size//4, self.window_size//4, self.window_size//4), mode='reflect')
            x_windows_down = F.unfold(shifted_x_down, kernel_size=self.window_size, dilation=1, padding=0, stride=self.window_size//2)
            x_windows_down = x_windows_down.view(B, C, self.window_size*self.window_size, -1)
            x_windows_down = x_windows_down.permute(0, 3, 2, 1).contiguous().view(-1, self.window_size*self.window_size, C)

            if self.input_resolution == x_size:
                attn_windows = self.attn(x_windows, x_windows_down, mask_x=self.attn_mask_x, mask_y=self.attn_mask_y)
            else:
                attn_windows = self.attn(x_windows, x_windows_down, mask_x=self.calculate_mask(x_size).to(x.device), mask_y=self.calculate_mask2(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, use_crossattn=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        if use_crossattn is None:
            use_crossattn = [False for i in range(depth)]

        self.blocks = nn.ModuleList([
            TFL(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 use_crossattn=use_crossattn[i])
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class RTFL(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv', use_crossattn=None):
        super(RTFL, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.use_crossattn = use_crossattn

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint,
                                         use_crossattn=use_crossattn)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, self.embed_dim, x_size[0], x_size[1])
        return x

class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported.')
        super(Upsample, self).__init__(*m)

class TFModel(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3, out_chans=3, fuse_c=16,
                 embed_dim=96, depths=[[1,1], [3,3], [3,3], [3,3]], num_heads=[[6,6], [6,6], [6,6], [6,6]],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, img_range=1., resi_connection='1conv', use_crossattn=None,
                 **kwargs):
        super(TFModel, self).__init__()
        self.img_range = img_range
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr0 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[0]))]
        dpr1 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[1]))]
        dpr2 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[2]))]
        dpr3 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[3]))]

        self.layers0 = nn.ModuleList()
        for i_layer in range(len(depths[0])):
            self.layers0.append(RTFL(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                         depth=depths[0][i_layer], num_heads=num_heads[0][i_layer], window_size=window_size,
                         mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr0[sum(depths[0][:i_layer]):sum(depths[0][:i_layer + 1])], norm_layer=norm_layer,
                         downsample=None, use_checkpoint=use_checkpoint, img_size=(img_size[0]//8, img_size[1]//8),
                         patch_size=patch_size, resi_connection=resi_connection, use_crossattn=use_crossattn[0][i_layer]))

        self.layers1 = nn.ModuleList()
        for i_layer in range(len(depths[1])):
            self.layers1.append(RTFL(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                         depth=depths[1][i_layer], num_heads=num_heads[1][i_layer], window_size=window_size,
                         mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr1[sum(depths[1][:i_layer]):sum(depths[1][:i_layer + 1])], norm_layer=norm_layer,
                         downsample=None, use_checkpoint=use_checkpoint, img_size=(img_size[0]//4, img_size[1]//4),
                         patch_size=patch_size, resi_connection=resi_connection, use_crossattn=use_crossattn[1][i_layer]))

        self.layers2 = nn.ModuleList()
        for i_layer in range(len(depths[2])):
            self.layers2.append(RTFL(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                         depth=depths[2][i_layer], num_heads=num_heads[2][i_layer], window_size=window_size,
                         mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr2[sum(depths[2][:i_layer]):sum(depths[2][:i_layer + 1])], norm_layer=norm_layer,
                         downsample=None, use_checkpoint=use_checkpoint, img_size=(img_size[0]//2, img_size[1]//2),
                         patch_size=patch_size, resi_connection=resi_connection, use_crossattn=use_crossattn[2][i_layer]))

        self.layers3 = nn.ModuleList()
        for i_layer in range(len(depths[3])):
            self.layers3.append(RTFL(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                         depth=depths[3][i_layer], num_heads=num_heads[3][i_layer], window_size=window_size,
                         mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr3[sum(depths[3][:i_layer]):sum(depths[3][:i_layer + 1])], norm_layer=norm_layer,
                         downsample=None, use_checkpoint=use_checkpoint, img_size=(img_size[0], img_size[1]),
                         patch_size=patch_size, resi_connection=resi_connection, use_crossattn=use_crossattn[3][i_layer]))

        self.conv_after_body0 = nn.Sequential(nn.Conv2d(embed_dim+fuse_c*2, embed_dim, 3, 2, 1), nn.LeakyReLU(0.2, True),
                                              nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(0.2, True))
        self.conv_after_body1 = nn.Sequential(nn.Conv2d(embed_dim+fuse_c*4, embed_dim, 3, 2, 1), nn.LeakyReLU(0.2, True),
                                              nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(0.2, True))
        self.conv_after_body2 = nn.Sequential(nn.Conv2d(embed_dim+fuse_c*8, embed_dim, 3, 2, 1), nn.LeakyReLU(0.2, True),
                                              nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(0.2, True))

        self.conv_up0 = nn.Sequential(nn.ConvTranspose2d(embed_dim+fuse_c*16, embed_dim, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.conv_up1 = nn.Sequential(nn.ConvTranspose2d(2*embed_dim, embed_dim, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.conv_up2 = nn.Sequential(nn.ConvTranspose2d(2*embed_dim, embed_dim, 4, 2, 1), nn.LeakyReLU(0.2, True))

        self.conv_last1 = nn.Sequential(nn.Conv2d(embed_dim*2, embed_dim, 3, 1, 1), nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(embed_dim, embed_dim, 3, 1, 1))
        self.conv_last2 = nn.Sequential(nn.Conv2d(embed_dim, embed_dim//2, 3, 1, 1), nn.LeakyReLU(0.2, True),
                                        nn.Conv2d(embed_dim//2, out_chans, 3, 1, 1))

        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, layers):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape: x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in layers: x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x, c0, c1):
        s0 = self.conv_first(x.contiguous())
        fea0 = self.forward_features(s0, self.layers0)
        s1 = self.conv_after_body0(torch.cat([fea0, c0[0], c1[0]], dim=1))
        fea1 = self.forward_features(s1, self.layers1)
        s2 = self.conv_after_body1(torch.cat([fea1, c0[1], c1[1]], dim=1))
        fea2 = self.forward_features(s2, self.layers2)
        s3 = self.conv_after_body2(torch.cat([fea2, c0[2], c1[2]], dim=1))
        fea3 = self.forward_features(s3, self.layers3)
        fea3 = self.conv_up0(torch.cat([fea3, c0[3], c1[3]], dim=1))
        fea2 = self.conv_up1(torch.cat([fea3, fea2], dim=1))
        fea1 = self.conv_up2(torch.cat([fea2, fea1], dim=1))
        out = self.conv_last1(torch.cat([fea1, fea0], dim=1)) + s0
        out = self.conv_last2(out)
        return out
