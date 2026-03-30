"""TimesFM + N-HiTS 계층 디코더 (외인변수 블록 포함)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from timesfm.timesfm_2p5.timesfm_2p5_torch import (
    TimesFM_2p5_200M_torch,
    TimesFM_2p5_200M_torch_module,
)
from timesfm.torch.util import update_running_stats, revin


# ---------------------------------------------------------------------------
# 블록 정의
# ---------------------------------------------------------------------------

class TargetBlock(nn.Module):
    """시간 계층 블록: 특정 해상도에서 예측 계수를 출력하고 보간."""

    def __init__(self, num_patches, embed_dim, pool_kernel, n_coeffs, horizon):
        super().__init__()
        self.pool_kernel = pool_kernel
        self.horizon = horizon
        self.n_coeffs = n_coeffs

        if pool_kernel > 1:
            self.pool = nn.MaxPool1d(kernel_size=pool_kernel)
        else:
            self.pool = None

        # Flatten 대신 평균 풀링 → embed_dim만 FC 입력으로 사용
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_coeffs),
        )

    def forward(self, embeddings):
        # embeddings: [B, num_patches, embed_dim]
        if self.pool is not None:
            x = embeddings.permute(0, 2, 1)          # [B, embed_dim, num_patches]
            x = self.pool(x)                          # [B, embed_dim, pooled_patches]
            x = x.permute(0, 2, 1)                    # [B, pooled_patches, embed_dim]
        else:
            x = embeddings

        x = x.mean(dim=1)                             # [B, embed_dim]  ← 평균 풀링
        coeffs = self.fc(x)                            # [B, n_coeffs]

        if self.n_coeffs == self.horizon:
            return coeffs
        out = F.interpolate(
            coeffs.unsqueeze(1), size=self.horizon, mode="linear", align_corners=False
        ).squeeze(1)                                   # [B, horizon]
        return out


class CovariateBlock(nn.Module):
    """외인변수 블록: 하나의 covariate가 예측에 기여하는 부분을 출력."""

    def __init__(self, embed_dim, cov_len, n_coeffs, horizon):
        super().__init__()
        self.horizon = horizon
        self.n_coeffs = n_coeffs

        self.cov_encoder = nn.Sequential(
            nn.Linear(cov_len, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_coeffs),
        )

    def forward(self, embedding, covariate):
        # embedding:  [B, embed_dim]  (마지막 패치의 임베딩)
        # covariate:  [B, cov_len]    (context + horizon 전체)
        # 인스턴스별 정규화: 데이터셋 간 스케일 차이 해소
        cov_mean = covariate.mean(dim=-1, keepdim=True)
        cov_std = covariate.std(dim=-1, keepdim=True).clamp(min=1e-6)
        covariate = (covariate - cov_mean) / cov_std
        cov_feat = self.cov_encoder(covariate)                # [B, 128]
        combined = torch.cat([embedding, cov_feat], dim=-1)   # [B, embed_dim+128]
        coeffs = self.fc(combined)                            # [B, n_coeffs]

        if self.n_coeffs == self.horizon:
            return coeffs
        out = F.interpolate(
            coeffs.unsqueeze(1), size=self.horizon, mode="linear", align_corners=False
        ).squeeze(1)
        return out


class NHiTSHead(nn.Module):
    """N-HiTS 스타일 계층 디코더: Target 블록 + Covariate 블록."""

    def __init__(self, num_patches, embed_dim, horizon, covariate_cols, cov_len):
        super().__init__()
        self.covariate_cols = covariate_cols

        # Target 블록 (시간 계층) — 원본 N-HiTS식 공격적 분리
        n_trend = max(2, horizon // 48)     # h=96→2, h=192→4, h=336→7, h=720→15
        n_seasonal = max(4, horizon // 8)   # h=96→12, h=192→24, h=336→42, h=720→90
        self.trend = TargetBlock(num_patches, embed_dim, pool_kernel=16, n_coeffs=n_trend, horizon=horizon)
        self.seasonal = TargetBlock(num_patches, embed_dim, pool_kernel=4, n_coeffs=n_seasonal, horizon=horizon)
        self.detail = TargetBlock(num_patches, embed_dim, pool_kernel=1, n_coeffs=horizon, horizon=horizon)

        # Covariate 블록 (외인변수별)
        self.cov_blocks = nn.ModuleDict({
            col: CovariateBlock(embed_dim, cov_len, n_coeffs=horizon // 2, horizon=horizon)
            for col in covariate_cols
        })

    def forward(self, output_embeddings, covariates=None, return_decomposition=False):
        # output_embeddings: [B, num_patches, embed_dim]
        # covariates: {col_name: [B, cov_len]} or None

        trend_out = self.trend(output_embeddings)
        seasonal_out = self.seasonal(output_embeddings)
        detail_out = self.detail(output_embeddings)

        pred = trend_out + seasonal_out + detail_out

        cov_outs = {}
        if covariates is not None:
            last_emb = output_embeddings[:, -1, :]    # [B, embed_dim]
            for col in self.covariate_cols:
                if col in covariates:
                    cov_out = self.cov_blocks[col](last_emb, covariates[col])
                    cov_outs[col] = cov_out
                    pred = pred + cov_out

        if return_decomposition:
            return pred, {
                "trend": trend_out,
                "seasonal": seasonal_out,
                "detail": detail_out,
                **cov_outs,
            }
        return pred


# ---------------------------------------------------------------------------
# 전체 모델
# ---------------------------------------------------------------------------

class TimesFMWithNHiTSHead(nn.Module):
    """TimesFM backbone + N-HiTS 커스텀 헤드."""

    def __init__(self, horizon, context_len, covariate_cols=None, unfreeze_last_n=3):
        super().__init__()

        # 백본 로드
        pretrained = TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch", torch_compile=False,
        )
        self.backbone = pretrained.model
        self.patch_len = self.backbone.p       # 32
        self.embed_dim = self.backbone.md      # 1280
        self.num_layers = self.backbone.x      # 20
        self.context_len = context_len
        self.horizon = horizon

        num_patches = context_len // self.patch_len
        cov_len = context_len + horizon
        covariate_cols = covariate_cols or []

        # 커스텀 헤드
        self.head = NHiTSHead(
            num_patches=num_patches,
            embed_dim=self.embed_dim,
            horizon=horizon,
            covariate_cols=covariate_cols,
            cov_len=cov_len,
        )

        # Freeze 전략: 전체 freeze → 마지막 N개 레이어만 unfreeze
        for param in self.backbone.parameters():
            param.requires_grad = False

        for i in range(self.num_layers - unfreeze_last_n, self.num_layers):
            for param in self.backbone.stacked_xf[i].parameters():
                param.requires_grad = True

    def forward(self, context, masks, covariates=None, return_decomposition=False):
        # context: [B, context_len]
        # masks:   [B, context_len]  (True = 패딩)
        # covariates: {col: [B, cov_len]} or None

        B = context.shape[0]
        device = context.device
        num_patches = self.context_len // self.patch_len

        # 패치화
        patched_inputs = context.reshape(B, -1, self.patch_len)   # [B, num_patches, 32]
        patched_masks = masks.reshape(B, -1, self.patch_len)      # [B, num_patches, 32]

        # 원본과 동일한 running stats 정규화
        n = torch.zeros(B, device=device)
        mu = torch.zeros(B, device=device)
        sigma = torch.zeros(B, device=device)
        patch_mu = []
        patch_sigma = []
        for i in range(num_patches):
            (n, mu, sigma), _ = update_running_stats(
                n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)
        context_mu = torch.stack(patch_mu, dim=1)     # [B, num_patches]
        context_sigma = torch.stack(patch_sigma, dim=1)

        # 패치별 정규화 (원본 backbone과 동일)
        normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
        normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

        # 백본 forward (output_embeddings 추출)
        (_, output_embeddings, _, _), _ = self.backbone(normed_inputs, patched_masks)
        # output_embeddings: [B, num_patches, 1280]

        # 커스텀 헤드
        head_out = self.head(output_embeddings, covariates, return_decomposition)

        # Denormalize: 마지막 패치의 mu/sigma 사용 (전체 context의 최종 통계)
        last_mu = context_mu[:, -1:]     # [B, 1]
        last_sigma = context_sigma[:, -1:]

        if return_decomposition:
            normed_pred, decomp = head_out
            pred = normed_pred * last_sigma + last_mu
            # 각 성분도 역정규화
            decomp_denormed = {
                k: v * last_sigma + (last_mu if k == "trend" else 0)
                for k, v in decomp.items()
            }
            return pred, decomp_denormed
        else:
            pred = head_out * last_sigma + last_mu
            return pred

    def get_param_groups(self, lr_backbone=1e-5, lr_head=1e-3):
        """백본과 헤드의 learning rate를 분리한 param groups 반환."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ]
