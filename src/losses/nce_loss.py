# src/models/losses/nce_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchNCELoss(nn.Module):
    """
    Patch-wise Noise Contrastive Estimation loss.

    This loss is used to preserve content / structure consistency between
    source image features and translated image features.

    For CUT-style training:
        query feature: feature from fake_B
        key feature:   feature from real_A

    Positive pair:
        features at the same spatial / sampled patch location.

    Negative pairs:
        features from different patch locations within the same image.

    Input supports:
        1. [B, N, C]
            B: batch size
            N: number of sampled patches
            C: feature dimension

        2. [B, C, H, W]
            It will be flattened to [B, H*W, C].

    Loss:
        CrossEntropy(logits / temperature, positive_index)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.temperature = temperature
        self.normalize = normalize
        self.cross_entropy = nn.CrossEntropyLoss()

    def prepare_feature(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert feature to [B, N, C].

        Args:
            feature:
                [B, N, C] or [B, C, H, W]

        Returns:
            feature with shape [B, N, C]
        """
        if feature.dim() == 3:
            return feature

        if feature.dim() == 4:
            batch_size, channels, height, width = feature.shape

            feature = feature.view(
                batch_size,
                channels,
                height * width,
            )

            feature = feature.permute(0, 2, 1).contiguous()
            return feature

        raise ValueError(
            "PatchNCELoss expects feature with shape [B, N, C] or [B, C, H, W], "
            "but got shape {}.".format(tuple(feature.shape))
        )

    def forward(
        self,
        feat_q: torch.Tensor,
        feat_k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PatchNCE loss.

        Args:
            feat_q:
                query feature, usually from fake_B

            feat_k:
                key feature, usually from real_A

        Returns:
            scalar PatchNCE loss
        """
        feat_q = self.prepare_feature(feat_q)
        feat_k = self.prepare_feature(feat_k)

        if feat_q.shape != feat_k.shape:
            raise ValueError(
                "feat_q and feat_k must have the same shape. "
                "Got {} and {}.".format(
                    tuple(feat_q.shape),
                    tuple(feat_k.shape),
                )
            )

        if self.normalize:
            feat_q = F.normalize(feat_q, dim=-1)
            feat_k = F.normalize(feat_k, dim=-1)

        batch_size, num_patches, channels = feat_q.shape

        logits = torch.bmm(
            feat_q,
            feat_k.transpose(1, 2),
        )

        logits = logits / self.temperature

        labels = torch.arange(
            num_patches,
            device=feat_q.device,
            dtype=torch.long,
        )

        labels = labels.unsqueeze(0).repeat(batch_size, 1)
        labels = labels.view(batch_size * num_patches)

        logits = logits.view(
            batch_size * num_patches,
            num_patches,
        )

        loss = self.cross_entropy(
            logits,
            labels,
        )

        return loss


class MultiLayerPatchNCELoss(nn.Module):
    """
    Multi-layer PatchNCE loss.

    This module applies PatchNCELoss on multiple feature layers and averages them.

    Args:
        feat_q_list:
            list of query features

        feat_k_list:
            list of key features

    Example:
        loss = criterion_nce(
            feat_q_list=[fake_feat_l1, fake_feat_l2, fake_feat_l3],
            feat_k_list=[real_feat_l1, real_feat_l2, real_feat_l3],
        )
    """

    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.patch_nce = PatchNCELoss(
            temperature=temperature,
            normalize=normalize,
        )

    def forward(
        self,
        feat_q_list,
        feat_k_list,
    ) -> torch.Tensor:
        """
        Compute averaged PatchNCE loss over multiple feature layers.

        Args:
            feat_q_list: list of query features
            feat_k_list: list of key features

        Returns:
            scalar multi-layer PatchNCE loss
        """
        if len(feat_q_list) != len(feat_k_list):
            raise ValueError(
                "feat_q_list and feat_k_list must have the same length. "
                "Got {} and {}.".format(
                    len(feat_q_list),
                    len(feat_k_list),
                )
            )

        total_loss = 0.0

        for feat_q, feat_k in zip(feat_q_list, feat_k_list):
            loss = self.patch_nce(
                feat_q=feat_q,
                feat_k=feat_k,
            )

            total_loss = total_loss + loss

        total_loss = total_loss / len(feat_q_list)
        return total_loss

class PatchNCECriterion(nn.Module):
    """
    Compatibility PatchNCE criterion for the current CUTModel.

    Your current CUTModel imports and uses:

        from src.losses.nce_loss import PatchNCECriterion

        self.nce_loss = PatchNCECriterion(
            temperature=nce_temperature,
            num_patches=num_patches,
        )

        loss_NCE = self.nce_loss(
            feats_q=feats_fake_B,
            feats_k=feats_A,
        )

    This class supports:
        1. a single feature tensor
        2. a list / tuple of feature tensors from multiple generator layers

    Feature format supports:
        [B, C, H, W]
        [B, N, C]

    If the number of spatial patches is larger than num_patches,
    it randomly samples num_patches positions to reduce memory usage.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        num_patches: int = 256,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.temperature = temperature
        self.num_patches = num_patches
        self.normalize = normalize

        self.patch_nce = PatchNCELoss(
            temperature=temperature,
            normalize=normalize,
        )

    def prepare_feature(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert feature to [B, N, C].

        Args:
            feature:
                [B, C, H, W] or [B, N, C]

        Returns:
            feature with shape [B, N, C]
        """
        if feature.dim() == 3:
            return feature

        if feature.dim() == 4:
            batch_size, channels, height, width = feature.shape

            feature = feature.view(
                batch_size,
                channels,
                height * width,
            )

            feature = feature.permute(0, 2, 1).contiguous()
            return feature

        raise ValueError(
            "PatchNCECriterion expects feature with shape [B, N, C] or [B, C, H, W], "
            "but got shape {}.".format(tuple(feature.shape))
        )

    def sample_patches(
        self,
        feat_q: torch.Tensor,
        feat_k: torch.Tensor,
    ):
        """
        Randomly sample patch positions.

        Args:
            feat_q: [B, N, C]
            feat_k: [B, N, C]

        Returns:
            sampled feat_q and feat_k
        """
        if feat_q.shape != feat_k.shape:
            raise ValueError(
                "feat_q and feat_k must have the same shape. "
                "Got {} and {}.".format(
                    tuple(feat_q.shape),
                    tuple(feat_k.shape),
                )
            )

        batch_size, num_total_patches, channels = feat_q.shape

        if self.num_patches <= 0:
            return feat_q, feat_k

        if num_total_patches <= self.num_patches:
            return feat_q, feat_k

        patch_ids = torch.randperm(
            num_total_patches,
            device=feat_q.device,
        )[:self.num_patches]

        feat_q = feat_q[:, patch_ids, :]
        feat_k = feat_k[:, patch_ids, :]

        return feat_q, feat_k

    def compute_single_layer_loss(
        self,
        feat_q: torch.Tensor,
        feat_k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PatchNCE loss for one feature layer.
        """
        feat_q = self.prepare_feature(feat_q)
        feat_k = self.prepare_feature(feat_k)

        feat_q, feat_k = self.sample_patches(
            feat_q=feat_q,
            feat_k=feat_k,
        )

        loss = self.patch_nce(
            feat_q=feat_q,
            feat_k=feat_k,
        )

        return loss

    def forward(
        self,
        feats_q,
        feats_k,
    ) -> torch.Tensor:
        """
        Compute PatchNCE loss.

        Args:
            feats_q:
                query features, usually from fake_B

            feats_k:
                key features, usually from real_A

        Returns:
            scalar PatchNCE loss
        """
        if torch.is_tensor(feats_q) and torch.is_tensor(feats_k):
            return self.compute_single_layer_loss(
                feat_q=feats_q,
                feat_k=feats_k,
            )

        if isinstance(feats_q, (list, tuple)) and isinstance(feats_k, (list, tuple)):
            if len(feats_q) != len(feats_k):
                raise ValueError(
                    "feats_q and feats_k must have the same number of layers. "
                    "Got {} and {}.".format(
                        len(feats_q),
                        len(feats_k),
                    )
                )

            total_loss = 0.0

            for feat_q, feat_k in zip(feats_q, feats_k):
                loss = self.compute_single_layer_loss(
                    feat_q=feat_q,
                    feat_k=feat_k,
                )

                total_loss = total_loss + loss

            total_loss = total_loss / len(feats_q)
            return total_loss

        raise TypeError(
            "Unsupported feature type. feats_q and feats_k should both be tensors "
            "or both be list / tuple of tensors."
        )


class MaskedPatchNCELoss(nn.Module):
    """
    Mask-aware PatchNCE loss.

    This is prepared for BiSST-style masked / structure-aware contrastive learning.

    It computes PatchNCE only on valid masked regions.

    Args:
        feat_q:
            [B, C, H, W]

        feat_k:
            [B, C, H, W]

        mask:
            [B, 1, H_img, W_img] or [B, 1, H_feat, W_feat]

    Notes:
        - The mask will be resized to feature resolution.
        - Only positions with mask value > mask_threshold are used.
        - If one image has too few valid patches, it falls back to normal PatchNCE
          for numerical stability.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
        mask_threshold: float = 0.5,
        min_valid_patches: int = 8,
    ) -> None:
        super().__init__()

        self.temperature = temperature
        self.normalize = normalize
        self.mask_threshold = mask_threshold
        self.min_valid_patches = min_valid_patches

        self.patch_nce = PatchNCELoss(
            temperature=temperature,
            normalize=normalize,
        )

    def resize_mask(
        self,
        mask: torch.Tensor,
        target_size,
    ) -> torch.Tensor:
        """
        Resize mask to target spatial size.

        Args:
            mask: [B, 1, H, W]
            target_size: tuple, target height and width

        Returns:
            resized mask
        """
        mask = F.interpolate(
            mask.float(),
            size=target_size,
            mode="nearest",
        )

        return mask

    def flatten_feature(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert [B, C, H, W] to [B, H*W, C].
        """
        batch_size, channels, height, width = feature.shape

        feature = feature.view(
            batch_size,
            channels,
            height * width,
        )

        feature = feature.permute(0, 2, 1).contiguous()
        return feature

    def forward(
        self,
        feat_q: torch.Tensor,
        feat_k: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked PatchNCE loss.

        Args:
            feat_q: query feature map [B, C, H, W]
            feat_k: key feature map [B, C, H, W]
            mask: valid region mask [B, 1, H, W]

        Returns:
            scalar masked PatchNCE loss
        """
        if feat_q.dim() != 4 or feat_k.dim() != 4:
            return self.patch_nce(
                feat_q=feat_q,
                feat_k=feat_k,
            )

        if feat_q.shape != feat_k.shape:
            raise ValueError(
                "feat_q and feat_k must have the same shape. "
                "Got {} and {}.".format(
                    tuple(feat_q.shape),
                    tuple(feat_k.shape),
                )
            )

        batch_size, channels, height, width = feat_q.shape

        mask = self.resize_mask(
            mask=mask,
            target_size=(height, width),
        )

        mask = mask.view(batch_size, -1)
        valid_mask = mask > self.mask_threshold

        feat_q_flat = self.flatten_feature(feat_q)
        feat_k_flat = self.flatten_feature(feat_k)

        loss_list = []

        for batch_index in range(batch_size):
            valid_index = valid_mask[batch_index]

            num_valid = int(valid_index.sum().item())

            if num_valid < self.min_valid_patches:
                loss = self.patch_nce(
                    feat_q=feat_q_flat[batch_index:batch_index + 1],
                    feat_k=feat_k_flat[batch_index:batch_index + 1],
                )
            else:
                q_valid = feat_q_flat[batch_index][valid_index]
                k_valid = feat_k_flat[batch_index][valid_index]

                q_valid = q_valid.unsqueeze(0)
                k_valid = k_valid.unsqueeze(0)

                loss = self.patch_nce(
                    feat_q=q_valid,
                    feat_k=k_valid,
                )

            loss_list.append(loss)

        total_loss = torch.stack(loss_list).mean()
        return total_loss


def build_nce_loss(
    loss_type: str = "patchnce",
    temperature: float = 0.07,
    normalize: bool = True,
    num_patches: int = 256,
) -> nn.Module:
    """
    Build NCE loss by name.

    Args:
        loss_type:
            "patchnce"        -> PatchNCELoss
            "criterion"       -> PatchNCECriterion
            "multilayer_nce"  -> MultiLayerPatchNCELoss
            "masked_patchnce" -> MaskedPatchNCELoss

        temperature:
            softmax temperature

        normalize:
            whether to L2-normalize features

        num_patches:
            number of sampled patches for PatchNCECriterion

    Returns:
        NCE loss module
    """
    loss_type = loss_type.lower()

    if loss_type == "patchnce":
        return PatchNCELoss(
            temperature=temperature,
            normalize=normalize,
        )

    if loss_type == "criterion":
        return PatchNCECriterion(
            temperature=temperature,
            num_patches=num_patches,
            normalize=normalize,
        )

    if loss_type == "multilayer_nce":
        return MultiLayerPatchNCELoss(
            temperature=temperature,
            normalize=normalize,
        )

    if loss_type == "masked_patchnce":
        return MaskedPatchNCELoss(
            temperature=temperature,
            normalize=normalize,
        )

    raise ValueError("Unsupported NCE loss type: {}".format(loss_type))

if __name__ == "__main__":
    print("PatchNCE loss test")

    feat_q = torch.randn(2, 128, 256)
    feat_k = torch.randn(2, 128, 256)

    nce_loss = build_nce_loss("patchnce")

    loss = nce_loss(
        feat_q=feat_q,
        feat_k=feat_k,
    )

    print("feat_q shape:", feat_q.shape)
    print("feat_k shape:", feat_k.shape)
    print("loss:", loss.item())

    print("")
    print("Multi-layer PatchNCE loss test")

    feat_q_list = [
        torch.randn(2, 64, 64, 64),
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 256, 16, 16),
    ]

    feat_k_list = [
        torch.randn(2, 64, 64, 64),
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 256, 16, 16),
    ]

    multilayer_loss = build_nce_loss("multilayer_nce")

    loss_multi = multilayer_loss(
        feat_q_list=feat_q_list,
        feat_k_list=feat_k_list,
    )

    print("number of feature layers:", len(feat_q_list))
    print("multi-layer loss:", loss_multi.item())

    print("")
    print("Masked PatchNCE loss test")

    feat_q_map = torch.randn(2, 64, 32, 32)
    feat_k_map = torch.randn(2, 64, 32, 32)
    mask = torch.rand(2, 1, 256, 256)

    masked_loss = build_nce_loss("masked_patchnce")

    loss_masked = masked_loss(
        feat_q=feat_q_map,
        feat_k=feat_k_map,
        mask=mask,
    )

    print("feat_q_map shape:", feat_q_map.shape)
    print("feat_k_map shape:", feat_k_map.shape)
    print("mask shape:", mask.shape)
    print("masked loss:", loss_masked.item())