# src/models/mask_predictor.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic convolution block for lightweight mask prediction.

    Structure:
        Conv2d -> InstanceNorm2d -> ReLU
        Conv2d -> InstanceNorm2d -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class LightweightMaskPredictor(nn.Module):
    """
    Lightweight U-Net style mask predictor.

    Input:
        image tensor [B, 3, H, W]

    Output:
        mask logits [B, 1, H, W]

    Notes:
        - The output is logits, not sigmoid probability.
        - Use torch.sigmoid(mask_logits) when visualizing or using as soft mask.
        - This module is intentionally lightweight for early BiSST reproduction.
    """

    def __init__(
        self,
        input_nc: int = 3,
        base_channels: int = 32,
        output_nc: int = 1,
    ) -> None:
        super().__init__()

        c = base_channels

        self.enc1 = ConvBlock(input_nc, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(c * 4, c * 8)

        self.up3 = nn.ConvTranspose2d(
            c * 8,
            c * 4,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = ConvBlock(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(
            c * 4,
            c * 2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = ConvBlock(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(
            c * 2,
            c,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = ConvBlock(c * 2, c)

        self.out_conv = nn.Conv2d(
            c,
            output_nc,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict mask logits.

        Args:
            x: input image [B, 3, H, W]

        Returns:
            mask logits [B, 1, H, W]
        """
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        b = self.bottleneck(
            self.pool(e3)
        )

        d3 = self.up3(b)

        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(
                d3,
                size=e3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(
                d2,
                size=e2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(
                d1,
                size=e1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        mask_logits = self.out_conv(d1)

        if mask_logits.shape[-2:] != x.shape[-2:]:
            mask_logits = F.interpolate(
                mask_logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return mask_logits


def build_mask_predictor(
    predictor_type: str = "light_unet",
    input_nc: int = 3,
    base_channels: int = 32,
    output_nc: int = 1,
) -> nn.Module:
    """
    Build mask predictor by name.

    Args:
        predictor_type:
            "light_unet" -> LightweightMaskPredictor

    Returns:
        mask predictor network
    """
    predictor_type = predictor_type.lower()

    if predictor_type == "light_unet":
        return LightweightMaskPredictor(
            input_nc=input_nc,
            base_channels=base_channels,
            output_nc=output_nc,
        )

    raise ValueError(
        "Unsupported mask predictor type: {}".format(predictor_type)
    )


if __name__ == "__main__":
    net = build_mask_predictor(
        predictor_type="light_unet",
        input_nc=3,
        base_channels=16,
        output_nc=1,
    )

    x = torch.randn(2, 3, 128, 128)

    mask_logits = net(x)
    mask_prob = torch.sigmoid(mask_logits)

    print("Mask predictor test")
    print("input shape:", x.shape)
    print("mask_logits shape:", mask_logits.shape)
    print("mask_prob min:", mask_prob.min().item())
    print("mask_prob max:", mask_prob.max().item())