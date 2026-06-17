from typing import List, Tuple, Union

import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    """
    Standard ResNet block used in image-to-image translation generators.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """
    ResNet generator for image-to-image translation.

    Input:
        x: image tensor with shape [B, 3, H, W], usually in range [-1, 1]

    Output:
        y: translated image tensor with shape [B, 3, H, W], range [-1, 1]

    If return_features=True:
        return y, features

        features is a list of intermediate feature maps.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
    ) -> None:
        super().__init__()

        if n_blocks < 0:
            raise ValueError("n_blocks must be non-negative.")

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks

        layers: List[nn.Module] = []

        # layer 0: initial convolution
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                input_nc,
                ngf,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # downsampling
        mult = 1
        for _ in range(2):
            layers += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]
            mult *= 2

        # ResNet blocks
        for _ in range(n_blocks):
            layers += [
                ResnetBlock(ngf * mult)
            ]

        # upsampling
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(inplace=True),
            ]
            mult = int(mult / 2)

        # output layer
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                ngf,
                output_nc,
                kernel_size=7,
                stride=1,
                padding=0,
            ),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

        # These indices are selected after major stages.
        # They are used for PatchNCE feature extraction.
        #
        # For n_blocks=9, the approximate stages are:
        #   3  : after initial conv block
        #   6  : after first downsampling
        #   9  : after second downsampling
        #   9+n_blocks : after ResNet blocks
        #
        # Since model is a flat Sequential, every module has its own index.
        self.default_feature_layers = [
            3,
            6,
            9,
            9 + n_blocks,
        ]

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        feature_layers: Union[None, List[int]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: input image tensor
            return_features: whether to return intermediate features
            feature_layers: layer indices where features are collected

        Returns:
            If return_features=False:
                output image

            If return_features=True:
                output image, list of feature maps
        """
        if feature_layers is None:
            feature_layers = self.default_feature_layers

        features: List[torch.Tensor] = []
        out = x

        for layer_id, layer in enumerate(self.model):
            out = layer(out)

            if return_features and layer_id in feature_layers:
                features.append(out)

        if return_features:
            return out, features

        return out


if __name__ == "__main__":
    net = ResnetGenerator(
        input_nc=3,
        output_nc=3,
        ngf=32,
        n_blocks=3,
    )

    x = torch.randn(1, 3, 128, 128)

    y = net(x)
    y2, feats = net(x, return_features=True)

    print("Generator test")
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Output min:", y.min().item())
    print("Output max:", y.max().item())

    print("")
    print("Generator feature test")
    print("Output shape:", y2.shape)
    print("Number of features:", len(feats))

    for i, feat in enumerate(feats):
        print("Feature {} shape: {}".format(i, feat.shape))