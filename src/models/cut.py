from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.generator import ResnetGenerator
from src.models.discriminator import NLayerDiscriminator
from src.losses.gan_loss import build_gan_loss
from src.losses.nce_loss import PatchNCECriterion


class CUTModel(nn.Module):
    """
    Minimal CUT model for one-direction unpaired image translation.

    Current direction:
        virtual -> real

    Components:
        G: generator
        D: real-domain discriminator

    Losses:
        GAN loss
        PatchNCE loss
        Identity NCE loss, optional
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        ndf: int = 64,
        n_blocks: int = 9,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_gan: float = 1.0,
        lambda_nce: float = 1.0,
        lambda_idt: float = 0.5,
        nce_temperature: float = 0.07,
        num_patches: int = 256,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.lambda_gan = lambda_gan
        self.lambda_nce = lambda_nce
        self.lambda_idt = lambda_idt

        self.G = ResnetGenerator(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            n_blocks=n_blocks,
        ).to(self.device)

        self.D = NLayerDiscriminator(
            input_nc=output_nc,
            ndf=ndf,
            n_layers=3,
        ).to(self.device)

        self.gan_loss = build_gan_loss("hinge")
        self.nce_loss = PatchNCECriterion(
            temperature=nce_temperature,
            num_patches=num_patches,
        )

        self.optimizer_G = optim.Adam(
            self.G.parameters(),
            lr=lr,
            betas=(beta1, beta2),
        )

        self.optimizer_D = optim.Adam(
            self.D.parameters(),
            lr=lr,
            betas=(beta1, beta2),
        )

    def set_input(self, batch: Dict[str, Any]) -> None:
        """
        Set input batch.

        Expected batch:
            {
                "virtual": Tensor [B, 3, H, W],
                "real": Tensor [B, 3, H, W],
                "virtual_path": list[str],
                "real_path": list[str],
            }
        """
        self.real_A = batch["virtual"].to(self.device)
        self.real_B = batch["real"].to(self.device)

    def forward(self) -> None:
        """
        Generate fake real-style image.

        real_A: virtual image
        real_B: real image
        fake_B: generated real-style image
        """
        self.fake_B = self.G(self.real_A)

    def backward_D(self) -> torch.Tensor:
        """
        Update discriminator D.

        D should classify:
            real_B as real
            fake_B as fake
        """
        pred_real = self.D(self.real_B)
        pred_fake = self.D(self.fake_B.detach())

        loss_D = self.gan_loss.discriminator_loss(
            pred_real=pred_real,
            pred_fake=pred_fake,
        )

        loss_D.backward()

        return loss_D

    def backward_G(self) -> Dict[str, torch.Tensor]:
        """
        Update generator G.

        Loss:
            L_G = lambda_gan * GAN
                + lambda_nce * PatchNCE(real_A, fake_B)
                + lambda_idt * PatchNCE(real_B, G(real_B))
        """
        pred_fake = self.D(self.fake_B)

        loss_G_GAN = self.gan_loss.generator_loss(pred_fake)

        # PatchNCE between input virtual image and translated image
        _, feats_A = self.G(
            self.real_A,
            return_features=True,
        )

        _, feats_fake_B = self.G(
            self.fake_B,
            return_features=True,
        )

        loss_NCE = self.nce_loss(
            feats_q=feats_fake_B,
            feats_k=feats_A,
        )

        # Identity NCE:
        # If input is already real-domain image, generator should not change
        # its structural content too much.
        if self.lambda_idt > 0:
            idt_B, feats_idt_B = self.G(
                self.real_B,
                return_features=True,
            )

            _, feats_real_B = self.G(
                self.real_B,
                return_features=True,
            )

            loss_IDT = self.nce_loss(
                feats_q=feats_idt_B,
                feats_k=feats_real_B,
            )
        else:
            idt_B = None
            loss_IDT = torch.tensor(0.0, device=self.device)

        loss_G_total = (
            self.lambda_gan * loss_G_GAN
            + self.lambda_nce * loss_NCE
            + self.lambda_idt * loss_IDT
        )

        loss_G_total.backward()

        self.idt_B = idt_B

        return {
            "loss_G_total": loss_G_total.detach(),
            "loss_G_GAN": loss_G_GAN.detach(),
            "loss_NCE": loss_NCE.detach(),
            "loss_IDT": loss_IDT.detach(),
        }

    def optimize_parameters(self) -> Dict[str, float]:
        """
        One training step.

        Order:
            1. forward G
            2. update D
            3. update G
        """
        self.forward()

        # Update D
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()
        loss_D = self.backward_D()
        self.optimizer_D.step()

        # Update G
        self.set_requires_grad(self.D, False)
        self.optimizer_G.zero_grad()
        loss_G_dict = self.backward_G()
        self.optimizer_G.step()

        loss_dict = {
            "loss_D": float(loss_D.detach().cpu().item()),
        }

        for k, v in loss_G_dict.items():
            loss_dict[k] = float(v.cpu().item())

        return loss_dict

    def get_current_visuals(self) -> Dict[str, torch.Tensor]:
        """
        Return images for visualization.
        """
        visuals = {
            "real_A": self.real_A.detach(),
            "real_B": self.real_B.detach(),
            "fake_B": self.fake_B.detach(),
        }

        if hasattr(self, "idt_B") and self.idt_B is not None:
            visuals["idt_B"] = self.idt_B.detach()

        return visuals

    def get_model_dict(self) -> Dict[str, nn.Module]:
        """
        For checkpoint saving.
        """
        return {
            "G": self.G,
            "D": self.D,
        }

    def get_optimizer_dict(self) -> Dict[str, optim.Optimizer]:
        """
        For checkpoint saving.
        """
        return {
            "G": self.optimizer_G,
            "D": self.optimizer_D,
        }

    @staticmethod
    def set_requires_grad(
        net: nn.Module,
        requires_grad: bool = False,
    ) -> None:
        """
        Enable or disable gradients for a network.
        """
        for param in net.parameters():
            param.requires_grad = requires_grad


if __name__ == "__main__":
    model = CUTModel(
        ngf=32,
        ndf=32,
        n_blocks=3,
        device="cpu",
    )

    batch = {
        "virtual": torch.randn(1, 3, 128, 128),
        "real": torch.randn(1, 3, 128, 128),
    }

    model.set_input(batch)
    loss_dict = model.optimize_parameters()
    visuals = model.get_current_visuals()

    print("CUT model test")
    print("Loss dict:", loss_dict)

    for name, img in visuals.items():
        print("{} shape: {}".format(name, img.shape))