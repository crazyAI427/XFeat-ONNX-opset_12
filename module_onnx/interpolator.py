import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions. """
    def __init__(self, mode = 'bicubic', align_corners = False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """ Normalize coords to [-1,1]. """
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        B, C, Hx, Wx = x.shape
        N = pos.shape[1]

        # Normalize to [0, H-1] / [0, W-1]
        pos_x = pos[..., 0] * (Wx - 1) / W
        pos_y = pos[..., 1] * (Hx - 1) / H

        x0 = pos_x.floor().long().clamp(0, Wx - 1)
        x1 = (x0 + 1).clamp(0, Wx - 1)
        y0 = pos_y.floor().long().clamp(0, Hx - 1)
        y1 = (y0 + 1).clamp(0, Hx - 1)

        wa = (x1.float() - pos_x) * (y1.float() - pos_y)
        wb = (x1.float() - pos_x) * (pos_y - y0.float())
        wc = (pos_x - x0.float()) * (y1.float() - pos_y)
        wd = (pos_x - x0.float()) * (pos_y - y0.float())

        def gather_nd(img, x_idx, y_idx):
            B, C, H, W = img.shape
            N = x_idx.shape[1]
            idx = y_idx * W + x_idx  # [B, N]
            img_flat = img.view(B, C, -1)  # [B, C, H*W]
            idx_exp = idx.unsqueeze(1).expand(-1, C, -1)  # [B, C, N]
            gathered = torch.gather(img_flat, 2, idx_exp)  # [B, C, N]
            return gathered.permute(0, 2, 1)  # [B, N, C]

        Ia = gather_nd(x, x0, y0)
        Ib = gather_nd(x, x0, y1)
        Ic = gather_nd(x, x1, y0)
        Id = gather_nd(x, x1, y1)

        out = wa.unsqueeze(-1) * Ia + wb.unsqueeze(-1) * Ib + wc.unsqueeze(-1) * Ic + wd.unsqueeze(-1) * Id
        return out  # [B, N, C]