import os
import numpy as np
import torch

from module_onnx.model import *
from module_onnx.interpolator import InterpolateSparse2d
import torch.nn.functional as F

class XFeat(nn.Module):
    """
		Implements the inference module for XFeat.
		It supports inference for both sparse and semi-dense feature extraction & matching.
	"""

    def __init__(self, weights=None, top_k=4096, multiscale=False):
        super().__init__()
        # self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dev = torch.device('cpu')
        self.net = XFeatModel().to(self.dev)
        self.top_k = top_k
        self.multiscale = multiscale

        if weights is not None:
            if isinstance(weights, str):
                print('loading weights from: ' + weights)
                self.net.load_state_dict(torch.load(weights, map_location=self.dev), strict=False)
            else:
                self.net.load_state_dict(weights)

        self.interpolator = InterpolateSparse2d('bicubic')

    # ========================== XFeat.detectAndCompute ==========================
    @torch.inference_mode()
    def detectAndCompute(self, x):
        # ── pre‑process ────────────────────────────────────────────────────────────
        x, rh, rw       = self.preprocess_tensor(x)        # H, W divisible by 32
        B, _, Hc, Wc    = x.shape                          # coarse resolution

        # ── network forward ───────────────────────────────────────────────────────
        M, K, Hconf     = self.net(x)                      # M: feats, K: logits
        M               = F.normalize(M, dim=1)            # channel‑wise ℓ2

        # ── score map & raw key‑point grid ────────────────────────────────────────
        heat            = self.get_kpts_heatmap(K)         # (B,1,H*8,W*8)
        kpts            = self.NMS(heat, 0.05, 5)          # (B, Hc*Wc, 2)

        # ── reliability scores (nearest × bilinear) ───────────────────────────────
        near  = InterpolateSparse2d('nearest')(heat, kpts, Hc, Wc)[..., 0]   # (B, N)
        bilin = InterpolateSparse2d('bilinear')(Hconf, kpts, Hc, Wc)[..., 0] # (B, N)

        valid           = torch.any(kpts != 0, dim=-1).float()       # (B, N)
        scores          = near * bilin + (1. - valid) * -1e6         # invalid → −1e6

        # ── keep the best self.top_k per image, already sorted ↓ ─────────────────
        Kkeep                   = min(self.top_k, scores.shape[-1])
        scores_top, idx         = torch.topk(scores, Kkeep, dim=-1, largest=True,
                                            sorted=True)            # (B, K)

        gather1d                = lambda src: torch.gather(src, -1, idx)
        k_sel                   = torch.stack([gather1d(kpts[..., 0]),
                                            gather1d(kpts[..., 1])], dim=-1)  # (B,K,2)
        f_sel                   = self.interpolator(M, k_sel, Hc, Wc)            # (B,K,64)

        # ── rescale to the original image resolution ─────────────────────────────
        scale                    = torch.tensor([rw, rh], device=k_sel.device).view(1, 1, 2)
        k_sel                    = k_sel * scale

        # ── flatten the batch so the public API matches the original ─────────────
        keypoints   = k_sel.reshape(-1, 2)
        descriptors = F.normalize(f_sel, dim=-1).reshape(-1, f_sel.shape[-1])
        scores      = scores_top.reshape(-1)

        return {
            'keypoints'  : keypoints,
            'descriptors': descriptors,
            'scores'     : scores,
        }



    @torch.inference_mode()
    def detectAndComputeDense(self, x):
        """
            Compute dense *and coarse* descriptors. Supports batched mode.

            input:
                x -> torch.Tensor(B, C, H, W): grayscale or rgb image
                top_k -> int: keep best k features
            return: features sorted by their reliability score -- from most to least
                List[Dict]:
                    'keypoints'    ->   torch.Tensor(top_k, 2): coarse keypoints
                    'scales'       ->   torch.Tensor(top_k,): extraction scale
                    'descriptors'  ->   torch.Tensor(top_k, 64): coarse local features
        """
        if self.multiscale:
            print("TODO: Multiscale export")
            exit(-1)
            mkpts, sc, feats = self.extract_dualscale(x, self.top_k)
        else:
            mkpts, feats = self.extractDense(x)
            sc = torch.ones(mkpts.shape[:1], device=mkpts.device)

        return {'keypoints': mkpts,
                'descriptors': feats,
                'scales': sc}

    def create_xy(self, h, w, dev):
        y, x = torch.meshgrid(torch.arange(h, device=dev),
                              torch.arange(w, device=dev), indexing='ij')
        xy = torch.cat([x[..., None], y[..., None]], -1).reshape(-1, 2).float()
        return xy

    def extractDense(self, x):

        x, rh1, rw1 = self.preprocess_tensor(x)
        # x, self.net.scales[0], self.net.scales[1] = self.preprocess_tensor(x)
        M1, K1, H1 = self.net(x)
        _, C, _H1, _W1 = M1.shape
        xy1 = (self.create_xy(_H1, _W1, M1.device) * 8)

        M1 = M1[0].permute(1, 2, 0).flatten(0, 1)  # 1, H*W, C
        H1 = H1[0].permute(1, 2, 0).flatten(0)  # 1, H*W

        # _, top_k = torch.topk(H1, k = min(H1.shape[1], top_k), dim=-1)
        k = min(H1.shape[0], self.top_k)
        values, indices = torch.topk(H1, k, dim=0)
        feats = M1.index_select(0, indices)
        mkpts = xy1.index_select(0, indices)

        # Avoid warning of torch.tensor being treated as a constant when exporting to ONNX
        # mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, -1)
        mkpts[..., 0] = mkpts[..., 0] * rw1
        mkpts[..., 1] = mkpts[..., 1] * rh1

        return mkpts, feats

    @torch.inference_mode()
    def match_xfeat(self, img1, img2):
        """
            Simple extractor and MNN matcher.
            For simplicity, it does not support batched mode due to possibly different number of kpts.
            input:
                img1 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
                img2 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
                top_k -> int: keep best k features
            returns:
                mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
        """
        # img1 = self.parse_input(img1)
        # img2 = self.parse_input(img2)

        out1 = self.detectAndCompute(img1)
        out2 = self.detectAndCompute(img2)

        idxs0, idxs1 = self.match(out1['descriptors'], out2['descriptors'])
        return out1['keypoints'][idxs0], out2['keypoints'][idxs1]

    @torch.inference_mode()
    def match_xfeat_star(self, img1, img2):
        """
            Simple extractor and MNN matcher.
            For simplicity, it does not support batched mode due to possibly different number of kpts.
            input:
                img1 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
                img2 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
                top_k -> int: keep best k features
            returns:
                mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
        """
        out1 = self.detectAndComputeDense(img1)
        out2 = self.detectAndComputeDense(img2)
        idxs0, idxs1 = self.match(out1['descriptors'], out2['descriptors'])

        return self.refine_matches( out1["keypoints"], out1["descriptors"], idxs0,
                                    out2["keypoints"], out2["descriptors"], idxs1,
                                    out1["scales"])


    @torch.inference_mode()
    def match(self, feats1, feats2, min_cossim=-1):

        cossim = feats1 @ feats2.t()
        cossim_t = feats2 @ feats1.t()

        _, match12 = cossim.max(dim=1)
        _, match21 = cossim_t.max(dim=1)

        idx0 = torch.arange(match12.shape[0], device=match12.device)
        mutual = match21[match12] == idx0

        if min_cossim > 0:
            cossim, _ = cossim.max(dim=1)
            good = cossim > min_cossim
            idx0 = idx0[mutual & good]
            idx1 = match12[mutual & good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]

        return idx0, idx1

    @torch.inference_mode()
    def match_onnx(self, mkpts0, feats0, mkpts1, feats1):
        """
        Mutual‑nearest‑neighbour matcher — TensorRT‑friendly version.

        Returns
        -------
        mkpts0_pad, mkpts1_pad :  (top_k, 2)
            • The coordinates are **exactly** the same as the original
            implementation for every true match.
            • Rows where mkpts0_pad[i] == (0,0) contain no match and
            should be dropped *after* TensorRT inference.
        """

        # ------------------------------------------------------------------
        # 1. cosine‑similarity matrix
        cossim  = feats0 @ feats1.t()                # (N0 (=top_k), N1)

        # 2. best neighbours (row‑wise and column‑wise)
        _, best1 = torch.max(cossim, dim=1)          # (N0,)  j index
        _, best0 = torch.max(cossim, dim=0)          # (N1,)  i index

        # 3. mutual check
        idx0     = torch.arange(best1.size(0), device=best1.device)
        back     = best0.gather(0, best1)            # (N0,)
        mutual   = back.eq(idx0).float().unsqueeze(-1)   # (N0,1)  mask 1/0

        # 4. keep coordinates, zero‑out non‑mutual rows
        mk0_pad = mkpts0 * mutual
        mk1_pad = mkpts1.index_select(0, best1) * mutual

        # ------------------------------------------------------------------
        # Shapes are constant (top_k, 2)  →  no NonZero, no dynamic Slice
        return mk0_pad, mk1_pad




    @torch.inference_mode()
    def match_star_onnx(self, mkpts0, feats0, mkpts1, feats1, sc0):
        idx0, idx1 = self.match(feats0, feats1, min_cossim=0.82)
        # Refine coarse matches
        return self.refine_matches(mkpts0, feats0, idx0, mkpts1, feats1, idx1, sc0, fine_conf=0.25)

    def subpix_softmax2d(self, heatmaps, temp=3):
        N, H, W = heatmaps.shape
        heatmaps = torch.softmax(temp * heatmaps.view(-1, H * W), -1).view(-1, H, W)
        x, y = torch.meshgrid(torch.arange(H, device=heatmaps.device), torch.arange(W, device=heatmaps.device),
                              indexing='ij')
        x = x - (W // 2)
        y = y - (H // 2)

        coords_x = (x[None, ...] * heatmaps)
        coords_y = (y[None, ...] * heatmaps)
        coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H * W, 2)
        coords = coords.sum(1)

        return coords

    def refine_matches(self, mk0, d0, i0, mk1, d1, i1, sc0, fine_conf=0.25):
        f1  = d0.index_select(0, i0)
        f2  = d1.index_select(0, i1)
        p0  = mk0.index_select(0, i0)
        p1  = mk1.index_select(0, i1)
        sc0 = sc0.index_select(0, i0)

        off   = self.net.fine_matcher(torch.cat([f1, f2], -1))
        conf  = F.softmax(off*3, -1).max(-1)[0]
        off   = self.subpix_softmax2d(off.view(-1,8,8))
        p0   += off * sc0[:, None]
        p1   += off * sc0[:, None]

        # keep only confident matches without boolean indexing
        conf_mask = (conf > fine_conf).float()
        k_good    = int(conf_mask.sum().item())
        if k_good == 0:
            return p0[:0], p1[:0]
        _, top    = torch.topk(conf_mask, k_good)
        return p0.index_select(0, top), p1.index_select(0, top)



    def preprocess_tensor(self, x):
        """ Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
        x = x.to(self.dev).float()

        H, W = x.shape[-2:]
        _H, _W = (H // 32) * 32, (W // 32) * 32
        rh, rw = H / _H, W / _W

        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        return x, rh, rw

    def get_kpts_heatmap(self, kpts, softmax_temp=1.0):
        scores = F.softmax(kpts * softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
        return heatmap

    def NMS(self, x, threshold=0.05, kernel_size=5):
        # x: (1, 1, H, W)
        B, _, H, W = x.shape
        device = x.device

        local_max = F.max_pool2d(
            x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
        keep = (x == local_max) & (x > threshold)  # (B, 1, H, W)

        # Create a full (x, y) coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        coords = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).float()  # (H*W, 2)

        keep_flat = keep.view(B, -1).float()  # (B, H*W)
        coords = coords.unsqueeze(0).repeat(B, 1, 1)  # (B, H*W, 2)

        mkpts = coords * keep_flat.unsqueeze(-1)  # Zero out non-max entries

        # Remove zeroed-out keypoints (i.e. where keep == 0) later during inference
        return mkpts  # (B, H*W, 2), filter non-zero later





