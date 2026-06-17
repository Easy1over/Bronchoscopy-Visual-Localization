# src/losses/__init__.py

from src.losses.gan_loss import build_gan_loss
from src.losses.nce_loss import build_nce_loss
from src.losses.mask_loss import build_mask_loss
from src.losses.semantic_loss import build_semantic_loss