from typing import List

import torch
import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator.

    Input:
        x: image tensor with shape [B, 3, H, W]

    Output:
        patch logits with shape [B, 1, H', W']

    Unlike a standard image classifier, PatchGAN does not output one scalar.
    It outputs a grid of local real/fake predictions.
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()

        if n_layers < 1:
            raise ValueError("n_layers must be >= 1.")

        kw = 4
        padw = 1

        sequence: List[nn.Module] = []

        # First layer: no normalization
        sequence += [
            nn.Conv2d(
                input_nc,
                ndf,
                kernel_size=kw,
                stride=2,
                padding=padw,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        nf_mult_prev = 1

        # Intermediate downsampling layers
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=False,
                ),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # One more layer with stride 1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=False,
            ),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final prediction layer
        sequence += [
            nn.Conv2d(
                ndf * nf_mult,
                1,
                kernel_size=kw,
                stride=1,
                padding=padw,
            )
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    net = NLayerDiscriminator(
        input_nc=3,
        ndf=32,
        n_layers=3,
    )

    x = torch.randn(1, 3, 128, 128)
    y = net(x)

    print("Discriminator test")
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Output min:", y.min().item())
    print("Output max:", y.max().item())