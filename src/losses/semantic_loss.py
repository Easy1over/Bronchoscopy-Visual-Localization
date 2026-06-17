# src/losses/semantic_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """
    Convert embedding to [B, C].

    Supported input:
        [B, C]
        [B, C, H, W]

    If the input is a feature map [B, C, H, W], global average pooling
    is applied to obtain a compact vector.
    """
    if embedding.dim() == 2:
        return embedding

    if embedding.dim() == 4:
        return embedding.mean(dim=(2, 3))

    raise ValueError(
        "Expected embedding with shape [B, C] or [B, C, H, W], "
        "but got {}.".format(tuple(embedding.shape))
    )


def dice_similarity(
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    eps: float = 1e-6,
    from_logits: bool = True,
) -> torch.Tensor:
    """
    Compute per-sample Dice similarity between two masks.

    Args:
        mask_a:
            First mask logits or probabilities, shape [B, 1, H, W].

        mask_b:
            Second mask logits or probabilities, shape [B, 1, H, W].

        eps:
            Numerical stability term.

        from_logits:
            If True, sigmoid will be applied to both masks.

    Returns:
        dice:
            Tensor with shape [B].
    """
    if from_logits:
        mask_a = torch.sigmoid(mask_a)
        mask_b = torch.sigmoid(mask_b)

    if mask_a.shape != mask_b.shape:
        mask_b = F.interpolate(
            mask_b,
            size=mask_a.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    mask_a = mask_a.float()
    mask_b = mask_b.float()

    batch_size = mask_a.size(0)

    mask_a_flat = mask_a.view(batch_size, -1)
    mask_b_flat = mask_b.view(batch_size, -1)

    intersection = (mask_a_flat * mask_b_flat).sum(dim=1)
    denominator = mask_a_flat.sum(dim=1) + mask_b_flat.sum(dim=1)

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return dice


class StructuralSemanticLoss(nn.Module):
    """
    Structural semantic consistency loss used in BiSST.

    This implements the paper-style semantic loss:

        L_sem = L_pos + lambda_neg * L_neg

    Positive alignment:
        L_pos = ||h_anchor - h_positive||_2^2

    Negative separation:
        L_neg = - w(anchor, negative) * ||h_anchor - h_negative||_1

    Mask-aware dynamic weight:
        w(anchor, negative) = 1 - Dice(mask_anchor, mask_negative)

    Interpretation:
        - h_anchor:
            structural embedding of source image, e.g. Eh(Es(x))

        - h_positive:
            structural embedding of translated image, e.g. Eh(Es(fake_x))

        - h_negative:
            structural embedding of another random image from the same domain

        - mask_anchor / mask_negative:
            predicted masks from frozen Fmask

    Notes:
        - This loss is designed to avoid trivial embedding collapse.
        - The positive term enforces translation-invariant structure.
        - The negative term encourages different structures to be separated.
        - The negative strength is dynamically adjusted by mask dissimilarity.
    """

    def __init__(
        self,
        lambda_neg: float = 0.1,
        eps: float = 1e-6,
        mask_from_logits: bool = True,
        detach_negative_weight: bool = True,
    ) -> None:
        super().__init__()

        self.lambda_neg = lambda_neg
        self.eps = eps
        self.mask_from_logits = mask_from_logits
        self.detach_negative_weight = detach_negative_weight

    def compute_positive_loss(
        self,
        h_anchor: torch.Tensor,
        h_positive: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute positive structural alignment loss.

        Args:
            h_anchor:
                Anchor embedding, shape [B, C] or [B, C, H, W].

            h_positive:
                Positive embedding, shape [B, C] or [B, C, H, W].

        Returns:
            Scalar positive loss.
        """
        h_anchor = flatten_embedding(h_anchor)
        h_positive = flatten_embedding(h_positive)

        if h_anchor.shape != h_positive.shape:
            raise ValueError(
                "h_anchor and h_positive must have the same shape. "
                "Got {} and {}.".format(
                    tuple(h_anchor.shape),
                    tuple(h_positive.shape),
                )
            )

        diff = h_anchor - h_positive
        loss_pos = torch.sum(diff * diff, dim=1).mean()

        return loss_pos

    def compute_negative_weight(
        self,
        mask_anchor: torch.Tensor,
        mask_negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mask-aware negative weight.

        Args:
            mask_anchor:
                Predicted mask of anchor image.

            mask_negative:
                Predicted mask of negative image.

        Returns:
            weight:
                Tensor with shape [B].
        """
        dice = dice_similarity(
            mask_a=mask_anchor,
            mask_b=mask_negative,
            eps=self.eps,
            from_logits=self.mask_from_logits,
        )

        weight = 1.0 - dice

        if self.detach_negative_weight:
            weight = weight.detach()

        return weight

    def compute_negative_loss(
        self,
        h_anchor: torch.Tensor,
        h_negative: torch.Tensor,
        mask_anchor: torch.Tensor,
        mask_negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mask-aware negative separation loss.

        Args:
            h_anchor:
                Anchor embedding, shape [B, C] or [B, C, H, W].

            h_negative:
                Negative embedding, shape [B, C] or [B, C, H, W].

            mask_anchor:
                Predicted mask of anchor image.

            mask_negative:
                Predicted mask of negative image.

        Returns:
            Scalar negative loss.
        """
        h_anchor = flatten_embedding(h_anchor)
        h_negative = flatten_embedding(h_negative)

        if h_anchor.shape != h_negative.shape:
            raise ValueError(
                "h_anchor and h_negative must have the same shape. "
                "Got {} and {}.".format(
                    tuple(h_anchor.shape),
                    tuple(h_negative.shape),
                )
            )

        weight = self.compute_negative_weight(
            mask_anchor=mask_anchor,
            mask_negative=mask_negative,
        )

        distance = torch.abs(h_anchor - h_negative).mean(dim=1)

        loss_neg = -weight * distance
        loss_neg = loss_neg.mean()

        return loss_neg

    def forward(
        self,
        h_anchor: torch.Tensor,
        h_positive: torch.Tensor,
        h_negative: torch.Tensor,
        mask_anchor: torch.Tensor,
        mask_negative: torch.Tensor,
    ):
        """
        Compute structural semantic loss.

        Args:
            h_anchor:
                Embedding of source image.

            h_positive:
                Embedding of translated image.

            h_negative:
                Embedding of random negative image.

            mask_anchor:
                Predicted mask of source image.

            mask_negative:
                Predicted mask of random negative image.

        Returns:
            total_loss:
                Scalar semantic loss.

            loss_dict:
                Dictionary containing loss components for logging.
        """
        loss_pos = self.compute_positive_loss(
            h_anchor=h_anchor,
            h_positive=h_positive,
        )

        loss_neg = self.compute_negative_loss(
            h_anchor=h_anchor,
            h_negative=h_negative,
            mask_anchor=mask_anchor,
            mask_negative=mask_negative,
        )

        total_loss = loss_pos + self.lambda_neg * loss_neg

        loss_dict = {
            "loss_sem_total": float(total_loss.detach().item()),
            "loss_sem_pos": float(loss_pos.detach().item()),
            "loss_sem_neg": float(loss_neg.detach().item()),
        }

        return total_loss, loss_dict


class SemanticCosineLoss(nn.Module):
    """
    Legacy semantic feature cosine consistency loss.

    This is kept as an optional utility / ablation loss.
    It is not the main paper BiSST semantic loss.
    """

    def __init__(
        self,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.eps = eps

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute semantic cosine loss.
        """
        if feat_a.shape != feat_b.shape:
            raise ValueError(
                "feat_a and feat_b must have the same shape. "
                "Got {} and {}.".format(
                    tuple(feat_a.shape),
                    tuple(feat_b.shape),
                )
            )

        if feat_a.dim() == 4:
            feat_a = feat_a.flatten(2)
            feat_b = feat_b.flatten(2)

            feat_a = feat_a.permute(0, 2, 1).contiguous()
            feat_b = feat_b.permute(0, 2, 1).contiguous()

        feat_a = F.normalize(
            feat_a,
            dim=-1,
            eps=self.eps,
        )

        feat_b = F.normalize(
            feat_b,
            dim=-1,
            eps=self.eps,
        )

        cosine = (feat_a * feat_b).sum(dim=-1)

        loss = 1.0 - cosine
        loss = loss.mean()

        return loss


class SemanticL1Loss(nn.Module):
    """
    Legacy semantic feature L1 consistency loss.

    This is kept as an optional utility / ablation loss.
    """

    def __init__(self) -> None:
        super().__init__()

        self.criterion = nn.L1Loss()

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute semantic feature L1 loss.
        """
        if feat_a.shape != feat_b.shape:
            raise ValueError(
                "feat_a and feat_b must have the same shape. "
                "Got {} and {}.".format(
                    tuple(feat_a.shape),
                    tuple(feat_b.shape),
                )
            )

        loss = self.criterion(
            feat_a,
            feat_b,
        )

        return loss


class SemanticKLLoss(nn.Module):
    """
    Optional semantic probability distribution consistency loss.

    This is not the main paper BiSST semantic loss.
    It is kept for future ablation or debugging.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        detach_teacher: bool = True,
    ) -> None:
        super().__init__()

        self.temperature = temperature
        self.detach_teacher = detach_teacher
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        logits_student: torch.Tensor,
        logits_teacher: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute semantic KL consistency loss.
        """
        if logits_student.shape != logits_teacher.shape:
            raise ValueError(
                "logits_student and logits_teacher must have the same shape. "
                "Got {} and {}.".format(
                    tuple(logits_student.shape),
                    tuple(logits_teacher.shape),
                )
            )

        temperature = self.temperature

        if self.detach_teacher:
            logits_teacher = logits_teacher.detach()

        log_prob_student = F.log_softmax(
            logits_student / temperature,
            dim=1,
        )

        prob_teacher = F.softmax(
            logits_teacher / temperature,
            dim=1,
        )

        loss = self.criterion(
            log_prob_student,
            prob_teacher,
        )

        loss = loss * temperature * temperature
        return loss


class SemanticCrossEntropyLoss(nn.Module):
    """
    Optional semantic cross entropy loss.

    This is used only if ground-truth or pseudo semantic labels are available.
    It is not the main paper BiSST semantic loss.
    """

    def __init__(
        self,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()

        self.ignore_index = ignore_index

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute semantic cross entropy loss.
        """
        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0, :, :]

        target = target.long()

        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        loss = self.criterion(
            logits,
            target,
        )

        return loss


class CombinedSemanticConsistencyLoss(nn.Module):
    """
    Legacy combined semantic consistency loss.

    This combines:
        - cosine semantic feature consistency
        - L1 semantic feature consistency

    This is kept as an optional ablation loss.
    It is not the main paper BiSST semantic loss.
    """

    def __init__(
        self,
        lambda_cosine: float = 1.0,
        lambda_l1: float = 0.0,
    ) -> None:
        super().__init__()

        self.lambda_cosine = lambda_cosine
        self.lambda_l1 = lambda_l1

        self.cosine_loss = SemanticCosineLoss()
        self.l1_loss = SemanticL1Loss()

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined semantic consistency loss.
        """
        total_loss = 0.0

        if self.lambda_cosine > 0:
            total_loss = total_loss + self.lambda_cosine * self.cosine_loss(
                feat_a=feat_a,
                feat_b=feat_b,
            )

        if self.lambda_l1 > 0:
            total_loss = total_loss + self.lambda_l1 * self.l1_loss(
                feat_a=feat_a,
                feat_b=feat_b,
            )

        return total_loss


def build_semantic_loss(
    loss_type: str = "structural",
    lambda_neg: float = 0.1,
    mask_from_logits: bool = True,
    detach_negative_weight: bool = True,
) -> nn.Module:
    """
    Build semantic loss by name.

    Args:
        loss_type:
            "structural":
                StructuralSemanticLoss.
                This is the paper BiSST semantic loss.

            "cosine":
                Legacy cosine consistency loss.

            "l1":
                Legacy L1 consistency loss.

            "kl":
                Optional KL distribution consistency loss.

            "ce":
                Optional cross entropy loss.

            "combined":
                Legacy cosine + L1 consistency loss.

        lambda_neg:
            Weight for negative separation term in StructuralSemanticLoss.

        mask_from_logits:
            Whether semantic structural loss receives mask logits.

        detach_negative_weight:
            Whether to detach dynamic mask-aware negative weight.

    Returns:
        Semantic loss module.
    """
    loss_type = loss_type.lower()

    if loss_type == "structural":
        return StructuralSemanticLoss(
            lambda_neg=lambda_neg,
            mask_from_logits=mask_from_logits,
            detach_negative_weight=detach_negative_weight,
        )

    if loss_type == "cosine":
        return SemanticCosineLoss()

    if loss_type == "l1":
        return SemanticL1Loss()

    if loss_type == "kl":
        return SemanticKLLoss()

    if loss_type == "ce":
        return SemanticCrossEntropyLoss()

    if loss_type == "combined":
        return CombinedSemanticConsistencyLoss(
            lambda_cosine=1.0,
            lambda_l1=0.1,
        )

    raise ValueError("Unsupported semantic loss type: {}".format(loss_type))


if __name__ == "__main__":
    print("Structural semantic loss test")

    h_anchor = torch.randn(2, 128)
    h_positive = torch.randn(2, 128)
    h_negative = torch.randn(2, 128)

    mask_anchor = torch.randn(2, 1, 128, 128)
    mask_negative = torch.randn(2, 1, 128, 128)

    semantic_loss = build_semantic_loss(
        loss_type="structural",
        lambda_neg=0.1,
        mask_from_logits=True,
        detach_negative_weight=True,
    )

    total_loss, loss_dict = semantic_loss(
        h_anchor=h_anchor,
        h_positive=h_positive,
        h_negative=h_negative,
        mask_anchor=mask_anchor,
        mask_negative=mask_negative,
    )

    print("h_anchor shape:", h_anchor.shape)
    print("h_positive shape:", h_positive.shape)
    print("h_negative shape:", h_negative.shape)
    print("mask_anchor shape:", mask_anchor.shape)
    print("mask_negative shape:", mask_negative.shape)
    print("total loss:", total_loss.item())

    for key, value in loss_dict.items():
        print("{}: {:.6f}".format(key, value))

    print("")
    print("Legacy semantic cosine loss test")

    feat_a = torch.randn(2, 128, 32, 32)
    feat_b = torch.randn(2, 128, 32, 32)

    cosine_loss = build_semantic_loss("cosine")

    loss_cosine = cosine_loss(
        feat_a=feat_a,
        feat_b=feat_b,
    )

    print("feat_a shape:", feat_a.shape)
    print("feat_b shape:", feat_b.shape)
    print("cosine loss:", loss_cosine.item())

    print("")
    print("Semantic KL loss test")

    logits_student = torch.randn(2, 4, 64, 64)
    logits_teacher = torch.randn(2, 4, 64, 64)

    kl_loss = build_semantic_loss("kl")

    loss_kl = kl_loss(
        logits_student=logits_student,
        logits_teacher=logits_teacher,
    )

    print("logits_student shape:", logits_student.shape)
    print("logits_teacher shape:", logits_teacher.shape)
    print("KL loss:", loss_kl.item())

    print("")
    print("Semantic CE loss test")

    logits = torch.randn(2, 4, 64, 64)
    target = torch.randint(
        low=0,
        high=4,
        size=(2, 64, 64),
    )

    ce_loss = build_semantic_loss("ce")

    loss_ce = ce_loss(
        logits=logits,
        target=target,
    )

    print("logits shape:", logits.shape)
    print("target shape:", target.shape)
    print("CE loss:", loss_ce.item())