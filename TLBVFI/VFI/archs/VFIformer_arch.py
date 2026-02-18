import os
import sys
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from model.utils import trunc_normal_
sys.path.append('../..')
from VFI.archs.warplayer import warp
from VFI.archs.transformer_layers import TFModel

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super().__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class IFBlock(nn.Module):
    def __init__(self, in_planes, scale=1, c=64):
        super().__init__()
        self.scale = scale
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c), conv(c, c), conv(c, c), conv(c, c),
            conv(c, c), conv(c, c), conv(c, c), conv(c, c),
        )
        self.conv1 = nn.ConvTranspose2d(c, 4, 4, 2, 1)

    def forward(self, x):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor= 1. / self.scale, mode="bilinear", align_corners=False)
        x = self.conv0(x)
        x = self.convblock(x) + x
        x = self.conv1(x)
        flow = x
        if self.scale != 1:
            flow = F.interpolate(flow, scale_factor= self.scale, mode="bilinear", align_corners=False)
        return flow

class IFNet(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.block0 = IFBlock(6, scale=4, c=240)
        self.block1 = IFBlock(10, scale=2, c=150)
        self.block2 = IFBlock(10, scale=1, c=90)

    def forward(self, x):
        flow0 = self.block0(x)
        F1 = flow0
        F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_img0 = warp(x[:, :3], F1_large[:, :2])
        warped_img1 = warp(x[:, 3:], F1_large[:, 2:4])
        flow1 = self.block1(torch.cat((warped_img0, warped_img1, F1_large), 1))
        F2 = (flow0 + flow1)
        F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_img0 = warp(x[:, :3], F2_large[:, :2])
        warped_img1 = warp(x[:, 3:], F2_large[:, 2:4])
        flow2 = self.block2(torch.cat((warped_img0, warped_img1, F2_large), 1))
        F3 = (flow0 + flow1 + flow2)
        return F3, [F1, F2, F3]

class FlowRefineNetA(nn.Module):
    def __init__(self, context_dim, c=16, r=1, n_iters=4):
        super().__init__()
        corr_dim, flow_dim, motion_dim, hidden_dim = c, c, c, c
        self.n_iters, self.r = n_iters, r
        self.n_pts = (r * 2 + 1) ** 2
        self.occl_convs = nn.Sequential(nn.Conv2d(2 * context_dim, hidden_dim, 1), nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, hidden_dim, 1), nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, 1, 1), nn.Sigmoid())
        self.corr_convs = nn.Sequential(nn.Conv2d(self.n_pts, hidden_dim, 1), nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, corr_dim, 1), nn.PReLU(corr_dim))
        self.flow_convs = nn.Sequential(nn.Conv2d(2, hidden_dim, 3, 1, 1), nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, flow_dim, 3, 1, 1), nn.PReLU(flow_dim))
        self.motion_convs = nn.Sequential(nn.Conv2d(corr_dim + flow_dim, motion_dim, 3, 1, 1), nn.PReLU(motion_dim))
        self.gru = nn.Sequential(nn.Conv2d(motion_dim + context_dim * 2 + 2, hidden_dim, 3, 1, 1), nn.PReLU(hidden_dim),
                                 nn.Conv2d(hidden_dim, flow_dim, 3, 1, 1), nn.PReLU(flow_dim))
        self.flow_head = nn.Sequential(nn.Conv2d(flow_dim, hidden_dim, 3, 1, 1), nn.PReLU(hidden_dim),
                                       nn.Conv2d(hidden_dim, 2, 3, 1, 1))

    def L2normalize(self, x, dim=1):
        eps = 1e-5
        x_f = x.float()
        norm = x_f.pow(2).sum(dim=dim, keepdim=True).add(eps).sqrt()
        return (x_f / norm).type(x.dtype)

    def forward_once(self, x0, x1, flow0, flow1):
        B, C, H, W = x0.size()
        x0_unfold = F.unfold(x0, kernel_size=(self.r * 2 + 1), padding=1).view(B, C * self.n_pts, H, W)
        x1_unfold = F.unfold(x1, kernel_size=(self.r * 2 + 1), padding=1).view(B, C * self.n_pts, H, W)
        contents0, contents1 = warp(x0_unfold, flow0).view(B, C, self.n_pts, H, W), warp(x1_unfold, flow1).view(B, C, self.n_pts, H, W)
        fea0, fea1 = contents0[:, :, self.n_pts // 2, :, :], contents1[:, :, self.n_pts // 2, :, :]
        occl = self.occl_convs(torch.cat([fea0, fea1], dim=1))
        fea = fea0 * occl + fea1 * (1 - occl)
        fea_view = fea.permute(0, 2, 3, 1).contiguous().view(B * H * W, 1, C)
        contents0 = contents0.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, self.n_pts, C)
        contents1 = contents1.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, self.n_pts, C)
        fea_view, contents0, contents1 = self.L2normalize(fea_view, -1), self.L2normalize(contents0, -1), self.L2normalize(contents1, -1)
        corr0, corr1 = torch.einsum('bic,bjc->bij', fea_view, contents0), torch.einsum('bic,bjc->bij', fea_view, contents1)
        corr0 = self.corr_convs(corr0.view(B, H, W, self.n_pts).permute(0, 3, 1, 2).contiguous())
        corr1 = self.corr_convs(corr1.view(B, H, W, self.n_pts).permute(0, 3, 1, 2).contiguous())
        flow0_fea, flow1_fea = self.flow_convs(flow0), self.flow_convs(flow1)
        motion0, motion1 = self.motion_convs(torch.cat([corr0, flow0_fea], 1)), self.motion_convs(torch.cat([corr1, flow1_fea], 1))
        flow0 = flow0 + self.flow_head(self.gru(torch.cat([fea, fea0, motion0, flow0], 1)))
        flow1 = flow1 + self.flow_head(self.gru(torch.cat([fea, fea1, motion1, flow1], 1)))
        return flow0, flow1

    def forward(self, x0, x1, flow0, flow1):
        for i in range(self.n_iters):
            flow0, flow1 = self.forward_once(x0, x1, flow0, flow1)
        return torch.cat([flow0, flow1], dim=1)

class FlowRefineNet_Multis_our(nn.Module):
    def __init__(self, c=24, n_iters=1):
        super().__init__()
        self.rf_block1 = FlowRefineNetA(context_dim= c, c= c, r=1, n_iters=n_iters)
        self.rf_block2 = FlowRefineNetA(context_dim= c, c= c, r=1, n_iters=n_iters)
        self.rf_block3 = FlowRefineNetA(context_dim=2 * c, c=2 * c, r=1, n_iters=n_iters)
        self.rf_block4 = FlowRefineNetA(context_dim=2 * c, c=2 * c, r=1, n_iters=n_iters)

    def forward(self, feats, flow):
        s_1,s_2,s_3,s_4 = feats
        bs = s_1.size(0)//2
        flow = F.interpolate(flow, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
        flow = self.rf_block4(s_4[:bs], s_4[bs:], flow[:, :2], flow[:, 2:4])
        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.
        flow = self.rf_block3(s_3[:bs], s_3[bs:], flow[:, :2], flow[:, 2:4])
        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.
        flow = self.rf_block2(s_2[:bs], s_2[bs:], flow[:, :2], flow[:, 2:4])
        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.
        flow = self.rf_block1(s_1[:bs], s_1[bs:], flow[:, :2], flow[:, 2:4])
        c0, c1 = [s_1[:bs], s_2[:bs], s_3[:bs], s_4[:bs]], [s_1[bs:], s_2[bs:], s_3[bs:], s_4[bs:]]
        return flow, self.warp_fea(c0, flow[:, :2]), self.warp_fea(c1, flow[:, 2:4])

    def warp_fea(self, feas, flow): 
        outs = []
        for fea in feas:
            outs.append(warp(fea, flow))
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        return outs

    def get_context(self, feats, flow):
        s_1,s_2,s_3,s_4 = feats
        bs = s_1.size(0)//2
        c0, c1 = [s_1[:bs], s_2[:bs], s_3[:bs], s_4[:bs]], [s_1[bs:], s_2[bs:], s_3[bs:], s_4[bs:]]
        return flow, self.warp_fea(c0, flow[:, :2]), self.warp_fea(c1, flow[:, 2:4])

class FlowRefineNet_Multis(nn.Module):
    def __init__(self, c=24, n_iters=1):
        super().__init__()
        self.conv1, self.conv2 = Conv2(3, c, 1), Conv2(c, 2 * c)
        self.conv3, self.conv4 = Conv2(2 * c, 4 * c), Conv2(4 * c, 8 * c)
        self.rf_block1 = FlowRefineNetA(context_dim=c, c=c, r=1, n_iters=n_iters)
        self.rf_block2 = FlowRefineNetA(context_dim=2 * c, c=2 * c, r=1, n_iters=n_iters)
        self.rf_block3 = FlowRefineNetA(context_dim=4 * c, c=4 * c, r=1, n_iters=n_iters)
        self.rf_block4 = FlowRefineNetA(context_dim=8 * c, c=8 * c, r=1, n_iters=n_iters)

    def get_context(self, x0, x1, flow):
        bs = x0.size(0)
        inp = torch.cat([x0, x1], 0)
        s_1, s_2, s_3, s_4 = self.conv1(inp), self.conv2(self.conv1(inp)), self.conv3(self.conv2(self.conv1(inp))), self.conv4(self.conv3(self.conv2(self.conv1(inp))))
        c0, c1 = [s_1[:bs], s_2[:bs], s_3[:bs], s_4[:bs]], [s_1[bs:], s_2[bs:], s_3[bs:], s_4[bs:]]
        return flow, self.warp_fea(c0, flow[:, :2]), self.warp_fea(c1, flow[:, 2:4])

    def forward(self, x0, x1, flow):
        bs = x0.size(0)
        inp = torch.cat([x0, x1], 0)
        s_1 = self.conv1(inp)
        s_2 = self.conv2(s_1)
        s_3 = self.conv3(s_2)
        s_4 = self.conv4(s_3)
        flow = F.interpolate(flow, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
        flow = self.rf_block4(s_4[:bs], s_4[bs:], flow[:, :2], flow[:, 2:4])
        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.
        flow = self.rf_block3(s_3[:bs], s_3[bs:], flow[:, :2], flow[:, 2:4])
        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.
        flow = self.rf_block2(s_2[:bs], s_2[bs:], flow[:, :2], flow[:, 2:4])
        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.
        flow = self.rf_block1(s_1[:bs], s_1[bs:], flow[:, :2], flow[:, 2:4])
        c0, c1 = [s_1[:bs], s_2[:bs], s_3[:bs], s_4[:bs]], [s_1[bs:], s_2[bs:], s_3[bs:], s_4[bs:]]
        return flow, self.warp_fea(c0, flow[:, :2]), self.warp_fea(c1, flow[:, 2:4])

    def warp_fea(self, feas, flow):
        outs = []
        for fea in feas:
            outs.append(warp(fea, flow))
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        return outs

class VFIformer(nn.Module):
    def __init__(self):
        super().__init__()
        c, height, width, window_size, embed_dim = 24, 192, 192, 8, 160
        self.flownet = IFNet()
        self.refinenet = FlowRefineNet_Multis(c=c, n_iters=1)
        self.fuse_block = nn.Sequential(nn.Conv2d(12, 2*c, 3, 1, 1), nn.LeakyReLU(0.2, True),
                                         nn.Conv2d(2*c, 2*c, 3, 1, 1), nn.LeakyReLU(0.2, True))
        self.transformer = TFModel(img_size=(height, width), in_chans=2*c, out_chans=4, fuse_c=c,
                                          window_size=window_size, img_range=1.,
                                          depths=[[3, 3], [3, 3], [3, 3], [1, 1]],
                                          embed_dim=embed_dim, num_heads=[[2, 2], [2, 2], [2, 2], [2, 2]], mlp_ratio=2,
                                          resi_connection='1conv',
                                          use_crossattn=[[[False, False, False, False], [True, True, True, True]],
                                                      [[False, False, False, False], [True, True, True, True]],
                                                      [[False, False, False, False], [True, True, True, True]],
                                                      [[False, False, False, False], [False, False, False, False]]])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_flow(self, img0, img1):
        imgs = torch.cat((img0, img1), 1)
        flow, _ = self.flownet(imgs)
        flow, _, _ = self.refinenet(img0, img1, flow)
        return flow

    def forward(self, img0, img1, flow_pre=None):
        imgs = torch.cat((img0, img1), 1)
        if flow_pre is not None:
            flow = flow_pre
            _, c0, c1 = self.refinenet.get_context(img0, img1, flow)
        else:
            flow, _ = self.flownet(imgs)
            flow, c0, c1 = self.refinenet(img0, img1, flow)
        warped_img0, warped_img1 = warp(img0, flow[:, :2]), warp(img1, flow[:, 2:])
        x = self.fuse_block(torch.cat([img0, img1, warped_img0, warped_img1], 1))
        refine_output = self.transformer(x, c0, c1)
        res, mask = torch.sigmoid(refine_output[:, :3]) * 2 - 1, torch.sigmoid(refine_output[:, 3:4])
        return torch.clamp(warped_img0 * mask + warped_img1 * (1 - mask) + res, 0, 1), flow
