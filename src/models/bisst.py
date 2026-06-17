# src/models/bisst.py

from typing import Dict, Any, Optional
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.cut import CUTModel
from src.models.mask_predictor import build_mask_predictor
from src.models.embedding_extractor import build_embedding_extractor
from src.losses.mask_loss import build_mask_loss
from src.losses.semantic_loss import build_semantic_loss


class BiSSTModel(CUTModel):
    """
    Paper-aligned single-direction BiSST model built on top of CUT.

    Supported single directions:
        1. virtual2real:
            real_A = batch["virtual"]
            real_B = batch["real"]

        2. real2virtual:
            real_A = batch["real"]
            real_B = batch["virtual"]

    This file implements paper-aligned components for one selected direction:
        1. CUT base objective:
            - adversarial loss
            - PatchNCE loss
            - identity PatchNCE loss

        2. Pretrained / frozen mask predictor Fmask:
            - Fmask(real_A)
            - Fmask(fake_B)
            - Dice consistency loss between predicted soft masks

        3. Structure embedding extractor Eh:
            - h_A = Eh(Es(real_A))
            - h_fake_B = Eh(Es(fake_B))
            - h_neg = Eh(Es(real_A_negative))

        4. Structural semantic consistency loss:
            - positive alignment
            - mask-aware negative separation

    Important:
        - This is still single-direction.
        - Default direction is virtual2real to keep old scripts compatible.
        - real2virtual is enabled by swapping source and target domains in set_input().
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
        use_mask_predictor: bool = False,
        mask_predictor_type: str = "light_unet",
        mask_base_channels: int = 32,
        mask_checkpoint: Optional[str] = None,
        freeze_mask_predictor: bool = True,
        lambda_mask: float = 0.0,
        mask_loss_type: str = "dice_consistency",
        use_embedding_extractor: bool = True,
        embedding_extractor_type: str = "conv",
        embedding_input_channels: Optional[int] = None,
        embedding_hidden_channels: int = 128,
        embedding_dim: int = 128,
        lambda_semantic: float = 0.0,
        lambda_semantic_neg: float = 0.1,
        semantic_loss_type: str = "structural",
        detach_semantic_negative_weight: bool = True,
        device: str = "cuda",
        direction: str = "virtual2real",
    ) -> None:
        super().__init__(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            ndf=ndf,
            n_blocks=n_blocks,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            lambda_gan=lambda_gan,
            lambda_nce=lambda_nce,
            lambda_idt=lambda_idt,
            nce_temperature=nce_temperature,
            num_patches=num_patches,
            device=device,
        )

        if direction not in ["virtual2real", "real2virtual"]:
            raise ValueError(
                "Unsupported direction: {}. Expected 'virtual2real' or 'real2virtual'.".format(
                    direction
                )
            )

        self.direction = direction

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.ndf = ndf
        self.n_blocks = n_blocks

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.use_mask_predictor = use_mask_predictor
        self.mask_predictor_type = mask_predictor_type
        self.mask_base_channels = mask_base_channels
        self.mask_checkpoint = mask_checkpoint
        self.freeze_mask_predictor = freeze_mask_predictor

        self.lambda_mask = lambda_mask
        self.mask_loss_type = mask_loss_type

        self.use_embedding_extractor = use_embedding_extractor
        self.embedding_extractor_type = embedding_extractor_type
        self.embedding_input_channels = embedding_input_channels
        self.embedding_hidden_channels = embedding_hidden_channels
        self.embedding_dim = embedding_dim

        self.lambda_semantic = lambda_semantic
        self.lambda_semantic_neg = lambda_semantic_neg
        self.semantic_loss_type = semantic_loss_type
        self.detach_semantic_negative_weight = detach_semantic_negative_weight

        self.M = None
        self.Eh = None

        self.mask_A_logits = None
        self.mask_fake_B_logits = None
        self.mask_B_logits = None
        self.mask_A_neg_logits = None

        self.h_A = None
        self.h_fake_B = None
        self.h_A_neg = None

        if self.use_mask_predictor:
            self.M = build_mask_predictor(
                predictor_type=mask_predictor_type,
                input_nc=input_nc,
                base_channels=mask_base_channels,
                output_nc=1,
            ).to(self.device)

            if self.mask_checkpoint is not None:
                self.load_mask_predictor(self.mask_checkpoint)

            if self.freeze_mask_predictor:
                self.freeze_network(self.M)

        if self.use_embedding_extractor:
            if embedding_input_channels is None:
                embedding_input_channels = ngf * 4

            self.embedding_input_channels = embedding_input_channels

            self.Eh = build_embedding_extractor(
                extractor_type=embedding_extractor_type,
                input_channels=embedding_input_channels,
                hidden_channels=embedding_hidden_channels,
                embedding_dim=embedding_dim,
                normalize=True,
            ).to(self.device)

        self.mask_loss = build_mask_loss(
            loss_type=mask_loss_type,
            from_logits=True,
        )

        self.semantic_loss = build_semantic_loss(
            loss_type=semantic_loss_type,
            lambda_neg=lambda_semantic_neg,
            mask_from_logits=True,
            detach_negative_weight=detach_semantic_negative_weight,
        )

        self.rebuild_generator_optimizer()

    def rebuild_generator_optimizer(self) -> None:
        """
        Rebuild optimizer_G according to trainable modules.

        Trainable by default:
            G

        Optional:
            Eh is trained together with G if enabled.

        Fmask:
            - if freeze_mask_predictor=True, Fmask is not optimized.
            - if freeze_mask_predictor=False, Fmask is optimized.
              This is not the paper default, but kept for debugging.
        """
        params = list(self.G.parameters())

        if self.use_embedding_extractor and self.Eh is not None:
            params = params + list(self.Eh.parameters())

        if (
            self.use_mask_predictor
            and self.M is not None
            and not self.freeze_mask_predictor
        ):
            params = params + list(self.M.parameters())

        self.optimizer_G = optim.Adam(
            params,
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )

    def load_mask_predictor(
        self,
        checkpoint_path: str,
    ) -> None:
        """
        Load pretrained Fmask checkpoint.

        Supported formats:
            1. {"models": {"M": state_dict}}
            2. {"models": {"mask_predictor": state_dict}}
            3. {"model": state_dict}
            4. {"state_dict": state_dict}
            5. raw state_dict
        """
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        if isinstance(checkpoint, dict) and "models" in checkpoint:
            models = checkpoint["models"]

            if "M" in models:
                state_dict = models["M"]
            elif "mask_predictor" in models:
                state_dict = models["mask_predictor"]
            else:
                raise KeyError(
                    "Checkpoint['models'] does not contain 'M' or "
                    "'mask_predictor'. Available keys: {}".format(
                        list(models.keys())
                    )
                )

        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]

        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

        self.M.load_state_dict(
            state_dict,
            strict=True,
        )

        print("Loaded mask predictor from: {}".format(checkpoint_path))

    @staticmethod
    def freeze_network(
        net: nn.Module,
    ) -> None:
        """
        Freeze a network's parameters but still allow gradients to pass
        through its operations to the input tensor.

        This is important for frozen Fmask:
            loss(Fmask(fake_B), Fmask(real_A)) should update G through fake_B,
            but should not update Fmask parameters.
        """
        net.eval()

        for param in net.parameters():
            param.requires_grad = False

    def set_loss_weights(
        self,
        lambda_gan: Optional[float] = None,
        lambda_nce: Optional[float] = None,
        lambda_idt: Optional[float] = None,
        lambda_mask: Optional[float] = None,
        lambda_semantic: Optional[float] = None,
    ) -> None:
        """
        Update loss weights during training.

        This is used later by the dynamic lambda scheduler.

        Paper-inspired use:
            - early stage: small lambda_gan and lambda_semantic
            - middle stage: increase lambda_semantic
            - late stage: increase lambda_gan
        """
        if lambda_gan is not None:
            self.lambda_gan = lambda_gan

        if lambda_nce is not None:
            self.lambda_nce = lambda_nce

        if lambda_idt is not None:
            self.lambda_idt = lambda_idt

        if lambda_mask is not None:
            self.lambda_mask = lambda_mask

        if lambda_semantic is not None:
            self.lambda_semantic = lambda_semantic

    def set_input(
        self,
        batch: Dict[str, Any],
    ) -> None:
        """
        Set input batch.

        Required:
            batch["virtual"]
            batch["real"]

        Direction:
            virtual2real:
                real_A = virtual
                real_B = real

            real2virtual:
                real_A = real
                real_B = virtual

        No dataset mask is used here.

        Paper-aligned mask logic:
            masks are predicted by Fmask, not loaded as training inputs.
        """
        if "virtual" not in batch:
            raise KeyError("Batch does not contain key 'virtual'.")

        if "real" not in batch:
            raise KeyError("Batch does not contain key 'real'.")

        if self.direction == "virtual2real":
            self.real_A = batch["virtual"].to(self.device)
            self.real_B = batch["real"].to(self.device)
        elif self.direction == "real2virtual":
            self.real_A = batch["real"].to(self.device)
            self.real_B = batch["virtual"].to(self.device)
        else:
            raise ValueError("Unsupported direction: {}".format(self.direction))

    def forward(self) -> None:
        """
        Forward pass.

        1. Generate fake_B from real_A.
        2. If enabled, compute Fmask outputs.
        """
        self.fake_B = self.G(self.real_A)

        self.forward_masks()

    def forward_masks(self) -> None:
        """
        Predict masks using Fmask.

        If Fmask is frozen:
            - Fmask(real_A), Fmask(real_B), Fmask(real_A_neg) are computed
              under no_grad because they are references.
            - Fmask(fake_B) is computed with graph enabled so mask loss can
              backpropagate to fake_B and then to G.

        If Fmask is not frozen:
            all predictions are computed normally.
        """
        self.mask_A_logits = None
        self.mask_fake_B_logits = None
        self.mask_B_logits = None
        self.mask_A_neg_logits = None

        if not self.use_mask_predictor or self.M is None:
            return

        real_A_neg = self.get_negative_batch(self.real_A)

        if self.freeze_mask_predictor:
            self.M.eval()

            with torch.no_grad():
                self.mask_A_logits = self.M(self.real_A)
                self.mask_B_logits = self.M(self.real_B)
                self.mask_A_neg_logits = self.M(real_A_neg)

            self.mask_fake_B_logits = self.M(self.fake_B)

        else:
            self.mask_A_logits = self.M(self.real_A)
            self.mask_fake_B_logits = self.M(self.fake_B)
            self.mask_B_logits = self.M(self.real_B)
            self.mask_A_neg_logits = self.M(real_A_neg)

    @staticmethod
    def get_negative_batch(
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get a simple in-batch negative by rolling along batch dimension.

        If batch size is 1, this returns itself. In that case the mask-aware
        negative weight may become small, which is acceptable for debugging
        but less useful for real semantic negative learning.
        """
        if x.size(0) <= 1:
            return x

        return torch.roll(
            x,
            shifts=1,
            dims=0,
        )

    def get_last_feature(
        self,
        feats,
    ) -> torch.Tensor:
        """
        Get the last feature tensor from generator feature outputs.
        """
        if isinstance(feats, (list, tuple)):
            if len(feats) == 0:
                raise RuntimeError("Feature list is empty.")
            return feats[-1]

        if torch.is_tensor(feats):
            return feats

        raise TypeError(
            "Unsupported feature type: {}.".format(type(feats))
        )

    def compute_mask_loss(self) -> torch.Tensor:
        """
        Compute paper-style mask structural loss:

            L_mask = DiceConsistency(
                Fmask(real_A),
                Fmask(fake_B)
            )

        This is Eq. style:
            M = Fmask(x)
            M_hat = Fmask(fake_x)
            L = 1 - Dice(M, M_hat)

        If mask predictor is disabled or lambda_mask <= 0, returns zero.
        """
        if self.lambda_mask <= 0:
            return torch.tensor(0.0, device=self.device)

        if not self.use_mask_predictor or self.M is None:
            return torch.tensor(0.0, device=self.device)

        if self.mask_A_logits is None or self.mask_fake_B_logits is None:
            return torch.tensor(0.0, device=self.device)

        loss = self.mask_loss(
            mask_a=self.mask_A_logits,
            mask_b=self.mask_fake_B_logits,
        )

        loss = loss * self.lambda_mask
        return loss

    def compute_embeddings(
        self,
        feats_A,
        feats_fake_B,
    ) -> None:
        """
        Compute structural embeddings using Eh.

        h_A:
            Eh(Es(real_A))

        h_fake_B:
            Eh(Es(fake_B))

        h_A_neg:
            Eh(Es(real_A_negative))

        Here Es is approximated by the selected generator intermediate feature.
        """
        self.h_A = None
        self.h_fake_B = None
        self.h_A_neg = None

        if not self.use_embedding_extractor or self.Eh is None:
            return

        feat_A = self.get_last_feature(feats_A)
        feat_fake_B = self.get_last_feature(feats_fake_B)

        real_A_neg = self.get_negative_batch(self.real_A)

        _, feats_A_neg = self.G(
            real_A_neg,
            return_features=True,
        )

        feat_A_neg = self.get_last_feature(feats_A_neg)

        self.h_A = self.Eh(feat_A)
        self.h_fake_B = self.Eh(feat_fake_B)
        self.h_A_neg = self.Eh(feat_A_neg)

    def compute_semantic_loss(self) -> torch.Tensor:
        """
        Compute paper-style structural semantic consistency loss:

            L_sem = L_pos + lambda_neg * L_neg

        Required:
            - Eh enabled
            - Fmask enabled
            - h_A, h_fake_B, h_A_neg
            - mask_A_logits, mask_A_neg_logits

        If not available, returns zero.
        """
        if self.lambda_semantic <= 0:
            return torch.tensor(0.0, device=self.device)

        if not self.use_embedding_extractor or self.Eh is None:
            return torch.tensor(0.0, device=self.device)

        if not self.use_mask_predictor or self.M is None:
            return torch.tensor(0.0, device=self.device)

        if self.h_A is None or self.h_fake_B is None or self.h_A_neg is None:
            return torch.tensor(0.0, device=self.device)

        if self.mask_A_logits is None or self.mask_A_neg_logits is None:
            return torch.tensor(0.0, device=self.device)

        loss, loss_dict = self.semantic_loss(
            h_anchor=self.h_A,
            h_positive=self.h_fake_B,
            h_negative=self.h_A_neg,
            mask_anchor=self.mask_A_logits,
            mask_negative=self.mask_A_neg_logits,
        )

        self.latest_semantic_loss_dict = loss_dict

        loss = loss * self.lambda_semantic
        return loss

    def backward_G(self) -> Dict[str, torch.Tensor]:
        """
        Update generator side.

        Optimized networks:
            - G
            - Eh if enabled
            - Fmask only if use_mask_predictor=True and freeze_mask_predictor=False

        Paper-aligned losses in this single direction:
            - GAN
            - PatchNCE
            - Identity PatchNCE
            - Dice mask consistency
            - Structural semantic consistency
        """
        pred_fake = self.D(self.fake_B)

        loss_G_GAN = self.gan_loss.generator_loss(pred_fake)

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

        self.compute_embeddings(
            feats_A=feats_A,
            feats_fake_B=feats_fake_B,
        )

        loss_mask = self.compute_mask_loss()
        loss_semantic = self.compute_semantic_loss()

        loss_G_total = (
            self.lambda_gan * loss_G_GAN
            + self.lambda_nce * loss_NCE
            + self.lambda_idt * loss_IDT
            + loss_mask
            + loss_semantic
        )

        loss_G_total.backward()

        self.idt_B = idt_B

        loss_sem_pos = torch.tensor(0.0, device=self.device)
        loss_sem_neg = torch.tensor(0.0, device=self.device)

        if hasattr(self, "latest_semantic_loss_dict"):
            sem_dict = self.latest_semantic_loss_dict
            if "loss_sem_pos" in sem_dict:
                loss_sem_pos = torch.tensor(
                    sem_dict["loss_sem_pos"],
                    device=self.device,
                )
            if "loss_sem_neg" in sem_dict:
                loss_sem_neg = torch.tensor(
                    sem_dict["loss_sem_neg"],
                    device=self.device,
                )

        return {
            "loss_G_total": loss_G_total.detach(),
            "loss_G_GAN": loss_G_GAN.detach(),
            "loss_NCE": loss_NCE.detach(),
            "loss_IDT": loss_IDT.detach(),
            "loss_mask": loss_mask.detach(),
            "loss_semantic": loss_semantic.detach(),
            "loss_sem_pos": loss_sem_pos.detach(),
            "loss_sem_neg": loss_sem_neg.detach(),
        }

    def get_current_visuals(self) -> Dict[str, torch.Tensor]:
        """
        Return images and predicted masks for visualization.
        """
        visuals = super().get_current_visuals()

        if self.mask_A_logits is not None:
            visuals["mask_A_pred"] = torch.sigmoid(
                self.mask_A_logits
            ).detach()

        if self.mask_fake_B_logits is not None:
            visuals["mask_fake_B_pred"] = torch.sigmoid(
                self.mask_fake_B_logits
            ).detach()

        if self.mask_B_logits is not None:
            visuals["mask_B_pred"] = torch.sigmoid(
                self.mask_B_logits
            ).detach()

        return visuals

    def get_model_dict(self) -> Dict[str, nn.Module]:
        """
        Return model dict for checkpoint saving.

        Fmask is included only when enabled.
        Even if frozen, saving it is useful for experiment reproducibility.
        """
        model_dict = {
            "G": self.G,
            "D": self.D,
        }

        if self.use_embedding_extractor and self.Eh is not None:
            model_dict["Eh"] = self.Eh

        if self.use_mask_predictor and self.M is not None:
            model_dict["M"] = self.M

        return model_dict

    def get_optimizer_dict(self) -> Dict[str, optim.Optimizer]:
        """
        Return optimizer dict for checkpoint saving.
        """
        return {
            "G": self.optimizer_G,
            "D": self.optimizer_D,
        }


if __name__ == "__main__":
    print("BiSST model test: virtual2real, mask branch disabled")

    model = BiSSTModel(
        ngf=32,
        ndf=32,
        n_blocks=3,
        use_mask_predictor=False,
        use_embedding_extractor=True,
        embedding_input_channels=128,
        lambda_mask=0.0,
        lambda_semantic=0.0,
        device="cpu",
        direction="virtual2real",
    )

    batch = {
        "virtual": torch.randn(1, 3, 128, 128),
        "real": torch.randn(1, 3, 128, 128),
    }

    model.set_input(batch)
    loss_dict = model.optimize_parameters()
    visuals = model.get_current_visuals()

    print("Direction:", model.direction)
    print("Loss dict:", loss_dict)

    for name, tensor in visuals.items():
        print("{} shape: {}".format(name, tensor.shape))

    print("")
    print("BiSST model test: real2virtual, mask branch disabled")

    model_r2v = BiSSTModel(
        ngf=32,
        ndf=32,
        n_blocks=3,
        use_mask_predictor=False,
        use_embedding_extractor=True,
        embedding_input_channels=128,
        lambda_mask=0.0,
        lambda_semantic=0.0,
        device="cpu",
        direction="real2virtual",
    )

    batch = {
        "virtual": torch.randn(1, 3, 128, 128),
        "real": torch.randn(1, 3, 128, 128),
    }

    model_r2v.set_input(batch)
    loss_dict = model_r2v.optimize_parameters()
    visuals = model_r2v.get_current_visuals()

    print("Direction:", model_r2v.direction)
    print("Loss dict:", loss_dict)

    for name, tensor in visuals.items():
        print("{} shape: {}".format(name, tensor.shape))

    print("")
    print("BiSST model test: mask branch enabled without checkpoint")

    model_with_mask = BiSSTModel(
        ngf=32,
        ndf=32,
        n_blocks=3,
        use_mask_predictor=True,
        mask_base_channels=16,
        mask_checkpoint=None,
        freeze_mask_predictor=True,
        use_embedding_extractor=True,
        embedding_input_channels=128,
        lambda_mask=0.1,
        lambda_semantic=0.1,
        lambda_semantic_neg=0.1,
        device="cpu",
        direction="virtual2real",
    )

    batch = {
        "virtual": torch.randn(2, 3, 128, 128),
        "real": torch.randn(2, 3, 128, 128),
    }

    model_with_mask.set_input(batch)
    loss_dict = model_with_mask.optimize_parameters()
    visuals = model_with_mask.get_current_visuals()

    print("Direction:", model_with_mask.direction)
    print("Loss dict:", loss_dict)

    for name, tensor in visuals.items():
        print("{} shape: {}".format(name, tensor.shape))