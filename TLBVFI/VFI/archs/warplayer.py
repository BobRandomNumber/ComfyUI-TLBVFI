import torch
import torch.nn as nn

backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()), str(tenInput.dtype))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device, dtype=tenInput.dtype).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device, dtype=tenInput.dtype).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k].float() + tenFlow.float()).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput.float(), grid=g, mode='bilinear', padding_mode='border', align_corners=True).type(tenInput.dtype)


def flow_reversal(flow):
    # flow: (B, 2, H, W)
    B, _, H, W = flow.size()
    flow_r = warp(flow, flow.clone())
    flow_r = -1 * flow_r
    return flow_r