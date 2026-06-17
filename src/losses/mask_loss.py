# src/losses/mask_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskDiceConsistencyLoss(nn.Module):
    """
    Dice consistency loss between two predicted soft masks.

    This is the mask-based structural loss used in BiSST.

    Paper formulation:
        M_x = Fmask(x)
        M_y = Fmask(y)

        L_mask = 1 - (2 * <M_x, M_y> + eps)
                     / (||M_x||_1 + ||M_y||_1 + eps)

    Usage in BiSST:
        mask_a:
            predicted mask of input image, e.g. Fmask(real_A)

        mask_b:
            predicted mask of translated image, e.g. Fmask(fake_B)

    Notes:
        - Inputs can be logits or probabilities.
        - If from_logits=True, sigmoid is applied first.
        - This loss does not require external ground-truth masks.
        - This is different from supervised Dice loss.
        - This is the main mask loss used during BiSST training.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        from_logits: bool = True,
    ) -> None:
        super().__init__()

        self.eps = eps
        self.from_logits = from_logits

    def forward(
        self,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Dice consistency loss.

        Args:
            mask_a:
                First mask logits or probabilities, shape [B, 1, H, W].

            mask_b:
                Second mask logits or probabilities, shape [B, 1, H, W].

        Returns:
            Scalar Dice consistency loss.
        """
        if self.from_logits:
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
            2.0 * intersection + self.eps
        ) / (
            denominator + self.eps
        )

        loss = 1.0 - dice
        loss = loss.mean()

        return loss


class DiceMaskLoss(nn.Module):
    """
    Supervised Dice loss for binary mask prediction.

    This loss is used when training the mask predictor Fmask itself.

    Dice loss:
        L = 1 - (2 * intersection + smooth) / (union + smooth)

    Args:
        pred_mask:
            Predicted mask logits or probabilities, shape [B, 1, H, W].

        target_mask:
            Target binary mask, shape [B, 1, H, W].

    Notes:
        - If from_logits=True, sigmoid will be applied to pred_mask.
        - This is for supervised / pseudo-supervised Fmask training.
        - This is not the main BiSST mask consistency loss.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        from_logits: bool = True,
    ) -> None:
        super().__init__()

        self.smooth = smooth
        self.from_logits = from_logits

    def forward(
        self,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised Dice mask loss.

        Args:
            pred_mask:
                Predicted mask.

            target_mask:
                Target binary mask.

        Returns:
            Scalar Dice loss.
        """
        if self.from_logits:
            pred_mask = torch.sigmoid(pred_mask)

        pred_mask = pred_mask.float()
        target_mask = target_mask.float()

        if pred_mask.shape != target_mask.shape:
            target_mask = F.interpolate(
                target_mask,
                size=pred_mask.shape[-2:],
                mode="nearest",
            )

        pred_flat = pred_mask.view(pred_mask.size(0), -1)
        target_flat = target_mask.view(target_mask.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        loss = 1.0 - dice
        loss = loss.mean()

        return loss


class BCEMaskLoss(nn.Module):
    """
    Binary cross entropy loss for supervised mask prediction.

    This is used when training Fmask with pseudo masks or annotated masks.

    Args:
        pred_mask:
            Predicted mask logits or probabilities, shape [B, 1, H, W].

        target_mask:
            Target binary mask, shape [B, 1, H, W].
    """

    def __init__(
        self,
        from_logits: bool = True,
    ) -> None:
        super().__init__()

        self.from_logits = from_logits

        if from_logits:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.BCELoss()

    def forward(
        self,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BCE mask loss.

        Args:
            pred_mask:
                Predicted mask.

            target_mask:
                Target binary mask.

        Returns:
            Scalar BCE loss.
        """
        target_mask = target_mask.float()

        if pred_mask.shape != target_mask.shape:
            target_mask = F.interpolate(
                target_mask,
                size=pred_mask.shape[-2:],
                mode="nearest",
            )

        loss = self.criterion(
            pred_mask,
            target_mask,
        )

        return loss


class DiceBCEMaskLoss(nn.Module):
    """
    Combined Dice + BCE loss for supervised Fmask training.

    This is useful when training the lightweight mask predictor using
    pseudo masks generated from depth estimation and morphological processing.

    Total:
        L_mask = lambda_dice * L_dice + lambda_bce * L_bce
    """

    def __init__(
        self,
        lambda_dice: float = 1.0,
        lambda_bce: float = 1.0,
        from_logits: bool = True,
    ) -> None:
        super().__init__()

        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce

        self.dice_loss = DiceMaskLoss(
            from_logits=from_logits,
        )

        self.bce_loss = BCEMaskLoss(
            from_logits=from_logits,
        )

    def forward(
        self,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Dice + BCE mask loss.

        Args:
            pred_mask:
                Predicted mask.

            target_mask:
                Target binary mask.

        Returns:
            Scalar combined mask loss.
        """
        loss_dice = self.dice_loss(
            pred_mask=pred_mask,
            target_mask=target_mask,
        )

        loss_bce = self.bce_loss(
            pred_mask=pred_mask,
            target_mask=target_mask,
        )

        loss = self.lambda_dice * loss_dice + self.lambda_bce * loss_bce
        return loss


class MaskedImageL1Loss(nn.Module):
    """
    Optional masked image L1 loss.

    This loss computes image L1 difference only inside a valid mask region:
        L = mean( |image_a - image_b| * mask )

    Important:
        - This is NOT the main BiSST paper mask loss.
        - The paper uses Dice consistency between predicted masks.
        - This class is kept only as an optional utility for future ablation.
    """

    def __init__(
        self,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.eps = eps

    def forward(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked image L1 loss.

        Args:
            image_a:
                First image, shape [B, C, H, W].

            image_b:
                Second image, shape [B, C, H, W].

            mask:
                Binary or soft mask, shape [B, 1, H, W].

        Returns:
            Scalar masked L1 loss.
        """
        if image_a.shape != image_b.shape:
            raise ValueError(
                "image_a and image_b must have the same shape. "
                "Got {} and {}.".format(
                    tuple(image_a.shape),
                    tuple(image_b.shape),
                )
            )

        if mask.shape[-2:] != image_a.shape[-2:]:
            mask = F.interpolate(
                mask.float(),
                size=image_a.shape[-2:],
                mode="nearest",
            )

        if mask.size(1) == 1 and image_a.size(1) != 1:
            mask = mask.repeat(1, image_a.size(1), 1, 1)

        diff = torch.abs(image_a - image_b)
        weighted_diff = diff * mask

        loss = weighted_diff.sum() / (mask.sum() + self.eps)
        return loss


def build_mask_loss(
    loss_type: str = "dice_consistency",
    from_logits: bool = True,
) -> nn.Module:
    """
    Build mask loss by name.

    Args:
        loss_type:
            "dice_consistency":
                MaskDiceConsistencyLoss.
                This is the paper BiSST mask loss used during BiSST training.

            "dice":
                DiceMaskLoss.
                Supervised loss for training Fmask.

            "bce":
                BCEMaskLoss.
                Supervised loss for training Fmask.

            "dice_bce":
                DiceBCEMaskLoss.
                Supervised combined loss for training Fmask.

            "masked_image_l1":
                Optional utility loss for ablation.
                Not part of the main BiSST formulation.

        from_logits:
            Whether input mask is logits.

    Returns:
        Mask loss module.
    """
    loss_type = loss_type.lower()

    if loss_type == "dice_consistency":
        return MaskDiceConsistencyLoss(
            from_logits=from_logits,
        )

    if loss_type == "dice":
        return DiceMaskLoss(
            from_logits=from_logits,
        )

    if loss_type == "bce":
        return BCEMaskLoss(
            from_logits=from_logits,
        )

    if loss_type == "dice_bce":
        return DiceBCEMaskLoss(
            from_logits=from_logits,
        )

    if loss_type == "masked_image_l1":
        return MaskedImageL1Loss()

    raise ValueError("Unsupported mask loss type: {}".format(loss_type))


if __name__ == "__main__":
    print("Mask Dice consistency loss test")

    mask_a_logits = torch.randn(2, 1, 256, 256)
    mask_b_logits = torch.randn(2, 1, 256, 256)

    dice_consistency_loss = build_mask_loss(
        loss_type="dice_consistency",
        from_logits=True,
    )

    loss_dice_consistency = dice_consistency_loss(
        mask_a=mask_a_logits,
        mask_b=mask_b_logits,
    )

    print("mask_a_logits shape:", mask_a_logits.shape)
    print("mask_b_logits shape:", mask_b_logits.shape)
    print("dice consistency loss:", loss_dice_consistency.item())

    print("")
    print("Supervised Dice + BCE mask loss test")

    pred_mask = torch.randn(2, 1, 256, 256)
    target_mask = torch.randint(
        low=0,
        high=2,
        size=(2, 1, 256, 256),
    ).float()

    supervised_mask_loss = build_mask_loss(
        loss_type="dice_bce",
        from_logits=True,
    )

    loss_supervised = supervised_mask_loss(
        pred_mask=pred_mask,
        target_mask=target_mask,
    )

    print("pred_mask shape:", pred_mask.shape)
    print("target_mask shape:", target_mask.shape)
    print("supervised dice_bce loss:", loss_supervised.item())

    print("")
    print("Optional masked image L1 loss test")

    image_a = torch.randn(2, 3, 256, 256)
    image_b = torch.randn(2, 3, 256, 256)
    mask = torch.randint(
        low=0,
        high=2,
        size=(2, 1, 256, 256),
    ).float()

    masked_l1 = build_mask_loss(
        loss_type="masked_image_l1",
        from_logits=False,
    )

    loss_masked_l1 = masked_l1(
        image_a=image_a,
        image_b=image_b,
        mask=mask,
    )

    print("image_a shape:", image_a.shape)
    print("image_b shape:", image_b.shape)
    print("mask shape:", mask.shape)
    print("masked image L1 loss:", loss_masked_l1.item())