import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeGANLoss(nn.Module):
    """
    Hinge GAN loss.

    This loss is commonly used in modern GAN training.

    Discriminator loss:
        L_D = mean(ReLU(1 - D(real))) + mean(ReLU(1 + D(fake)))

    Generator loss:
        L_G = -mean(D(fake))

    Notes:
        - D(real) and D(fake) can be patch logits with shape [B, 1, H, W].
        - The loss will average over all batch and spatial dimensions.
    """

    def __init__(self) -> None:
        super().__init__()

    def discriminator_loss(
        self,
        pred_real: torch.Tensor,
        pred_fake: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute discriminator hinge loss.

        Args:
            pred_real: discriminator output for real images
            pred_fake: discriminator output for generated images

        Returns:
            scalar discriminator loss
        """
        loss_real = torch.mean(F.relu(1.0 - pred_real))
        loss_fake = torch.mean(F.relu(1.0 + pred_fake))
        loss = loss_real + loss_fake
        return loss

    def generator_loss(
        self,
        pred_fake: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute generator hinge loss.

        Args:
            pred_fake: discriminator output for generated images

        Returns:
            scalar generator loss
        """
        loss = -torch.mean(pred_fake)
        return loss


class VanillaGANLoss(nn.Module):
    """
    Standard BCE GAN loss.

    This is not the main loss we plan to use, but it is useful for comparison
    or debugging.

    Discriminator loss:
        L_D = BCE(D(real), 1) + BCE(D(fake), 0)

    Generator loss:
        L_G = BCE(D(fake), 1)
    """

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def discriminator_loss(
        self,
        pred_real: torch.Tensor,
        pred_fake: torch.Tensor,
    ) -> torch.Tensor:
        real_targets = torch.ones_like(pred_real)
        fake_targets = torch.zeros_like(pred_fake)

        loss_real = self.criterion(pred_real, real_targets)
        loss_fake = self.criterion(pred_fake, fake_targets)

        return loss_real + loss_fake

    def generator_loss(
        self,
        pred_fake: torch.Tensor,
    ) -> torch.Tensor:
        real_targets = torch.ones_like(pred_fake)
        loss = self.criterion(pred_fake, real_targets)
        return loss


def build_gan_loss(loss_type: str = "hinge") -> nn.Module:
    """
    Build GAN loss by name.

    Args:
        loss_type:
            "hinge"   -> HingeGANLoss
            "vanilla" -> VanillaGANLoss

    Returns:
        GAN loss module
    """
    loss_type = loss_type.lower()

    if loss_type == "hinge":
        return HingeGANLoss()

    if loss_type == "vanilla":
        return VanillaGANLoss()

    raise ValueError("Unsupported GAN loss type: {}".format(loss_type))


if __name__ == "__main__":
    gan_loss = build_gan_loss("hinge")

    pred_real = torch.randn(2, 1, 30, 30)
    pred_fake = torch.randn(2, 1, 30, 30)

    loss_d = gan_loss.discriminator_loss(
        pred_real=pred_real,
        pred_fake=pred_fake,
    )

    loss_g = gan_loss.generator_loss(
        pred_fake=pred_fake,
    )

    print("GAN loss test")
    print("pred_real shape:", pred_real.shape)
    print("pred_fake shape:", pred_fake.shape)
    print("D loss:", loss_d.item())
    print("G loss:", loss_g.item())