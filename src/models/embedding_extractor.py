# src/models/embedding_extractor.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormReLU(nn.Module):
    """
    Lightweight Conv -> InstanceNorm -> ReLU block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
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


class StructureEmbeddingExtractor(nn.Module):
    """
    Lightweight structure embedding extractor Eh.

    In the BiSST paper, the structure encoder Es extracts structural
    feature maps, and the embedding extractor Eh maps them into a compact
    normalized 128-dimensional vector.

    Input:
        structural feature map S = Es(x), shape [B, C, H, W]

    Output:
        normalized structure embedding h, shape [B, embedding_dim]

    Default:
        embedding_dim = 128

    Notes:
        - This module should be trained together with BiSST.
        - The output is L2-normalized by default.
        - It is used by StructuralSemanticLoss.
    """

    def __init__(
        self,
        input_channels: int = 256,
        hidden_channels: int = 128,
        embedding_dim: int = 128,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.normalize = normalize

        self.encoder = nn.Sequential(
            ConvNormReLU(
                in_channels=input_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvNormReLU(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.projector = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, embedding_dim),
        )

    def forward(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract normalized structural embedding.

        Args:
            feature:
                Structural feature map, shape [B, C, H, W].

        Returns:
            embedding:
                Structure embedding, shape [B, embedding_dim].
        """
        if feature.dim() != 4:
            raise ValueError(
                "StructureEmbeddingExtractor expects feature map with shape "
                "[B, C, H, W], but got {}.".format(tuple(feature.shape))
            )

        if feature.size(1) != self.input_channels:
            raise ValueError(
                "Input channel mismatch. Expected {}, but got {}. "
                "Please set input_channels according to the selected "
                "structure feature layer.".format(
                    self.input_channels,
                    feature.size(1),
                )
            )

        x = self.encoder(feature)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        embedding = self.projector(x)

        if self.normalize:
            embedding = F.normalize(
                embedding,
                p=2,
                dim=1,
                eps=1e-8,
            )

        return embedding


class GlobalPoolEmbeddingExtractor(nn.Module):
    """
    Simpler embedding extractor.

    This version directly applies global average pooling and an MLP projection.

    Compared with StructureEmbeddingExtractor:
        - fewer parameters
        - easier to use for debugging
        - weaker spatial processing capability

    Input:
        feature map [B, C, H, W]

    Output:
        normalized embedding [B, embedding_dim]
    """

    def __init__(
        self,
        input_channels: int = 256,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.normalize = normalize

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.projector = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract normalized structural embedding.

        Args:
            feature:
                Feature map, shape [B, C, H, W].

        Returns:
            embedding:
                Structure embedding, shape [B, embedding_dim].
        """
        if feature.dim() != 4:
            raise ValueError(
                "GlobalPoolEmbeddingExtractor expects feature map with shape "
                "[B, C, H, W], but got {}.".format(tuple(feature.shape))
            )

        if feature.size(1) != self.input_channels:
            raise ValueError(
                "Input channel mismatch. Expected {}, but got {}.".format(
                    self.input_channels,
                    feature.size(1),
                )
            )

        x = self.pool(feature)
        x = x.view(x.size(0), -1)

        embedding = self.projector(x)

        if self.normalize:
            embedding = F.normalize(
                embedding,
                p=2,
                dim=1,
                eps=1e-8,
            )

        return embedding


def build_embedding_extractor(
    extractor_type: str = "conv",
    input_channels: int = 256,
    hidden_channels: int = 128,
    embedding_dim: int = 128,
    normalize: bool = True,
) -> nn.Module:
    """
    Build structure embedding extractor Eh.

    Args:
        extractor_type:
            "conv":
                Conv-based lightweight extractor.

            "pool":
                Global-pooling MLP extractor.

        input_channels:
            Channel number of the selected structural feature map.

        hidden_channels:
            Hidden channel / hidden dimension.

        embedding_dim:
            Output embedding dimension. Paper default is 128.

        normalize:
            Whether to L2-normalize output embedding.

    Returns:
        Embedding extractor module.
    """
    extractor_type = extractor_type.lower()

    if extractor_type == "conv":
        return StructureEmbeddingExtractor(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            normalize=normalize,
        )

    if extractor_type == "pool":
        return GlobalPoolEmbeddingExtractor(
            input_channels=input_channels,
            hidden_dim=hidden_channels,
            embedding_dim=embedding_dim,
            normalize=normalize,
        )

    raise ValueError(
        "Unsupported embedding extractor type: {}".format(extractor_type)
    )


if __name__ == "__main__":
    print("Structure embedding extractor test")

    feature = torch.randn(2, 256, 32, 32)

    extractor = build_embedding_extractor(
        extractor_type="conv",
        input_channels=256,
        hidden_channels=128,
        embedding_dim=128,
        normalize=True,
    )

    embedding = extractor(feature)

    norm = torch.norm(
        embedding,
        p=2,
        dim=1,
    )

    print("feature shape:", feature.shape)
    print("embedding shape:", embedding.shape)
    print("embedding norm:", norm)

    print("")
    print("Global pool embedding extractor test")

    pool_extractor = build_embedding_extractor(
        extractor_type="pool",
        input_channels=256,
        hidden_channels=128,
        embedding_dim=128,
        normalize=True,
    )

    pool_embedding = pool_extractor(feature)

    pool_norm = torch.norm(
        pool_embedding,
        p=2,
        dim=1,
    )

    print("pool embedding shape:", pool_embedding.shape)
    print("pool embedding norm:", pool_norm)