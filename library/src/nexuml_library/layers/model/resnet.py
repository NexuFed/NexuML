"""ResNet pipeline layer for image classification backbones."""

from __future__ import annotations
from nexuml.core.discovery import layer

import logging
from typing import Any, cast

import torch
import torch.nn as nn

from nexuml.core.base_layer import PipelineLayer

logger = logging.getLogger(__name__)

_RESNET_TYPES = {
    "resnet18": ("resnet18", 512),
    "resnet34": ("resnet34", 512),
    "resnet50": ("resnet50", 2048),
    "resnet101": ("resnet101", 2048),
    "resnet152": ("resnet152", 2048),
}


@layer("ResNet")
class ResNet(PipelineLayer):
    """ResNet backbone that emits feature embeddings.

    Wraps ``torchvision.models.resnet*`` and replaces the final
    fully-connected layer with an identity so downstream heads
    can operate on embeddings.

    Args:
        resnet_type: Architecture variant (e.g. ``"resnet18"``).
        pretrained: Whether to load pretrained ImageNet weights.
        cifar_stem: Use a CIFAR-friendly stem (3x3 stride-1, no maxpool).
            Defaults to ``False`` when pretrained, ``True`` otherwise.
    """

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        resnet_type: str = "resnet18",
        pretrained: bool = False,
        cifar_stem: bool | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            input_sizes=input_sizes,
            keys_in=keys_in,
            keys_out=keys_out,
            **kwargs,
        )

        if resnet_type not in _RESNET_TYPES:
            available = ", ".join(sorted(_RESNET_TYPES.keys()))
            raise ValueError(f"Unsupported resnet_type '{resnet_type}'. Available: [{available}]")

        self.resnet_type = resnet_type
        self.pretrained = pretrained

        if cifar_stem is None:
            cifar_stem = not pretrained
        self.cifar_stem = cifar_stem

        if cifar_stem and pretrained:
            logger.warning(
                "cifar_stem=True with pretrained=True: pretrained weights "
                "were trained with the ImageNet stem; stem mismatch may "
                "degrade transfer performance."
            )

        self.in_channels = self._infer_in_channels()
        self.model = self._build_model()
        logger.info(
            "ResNet: type=%s, pretrained=%s, cifar_stem=%s, in_channels=%s",
            resnet_type,
            pretrained,
            cifar_stem,
            self.in_channels,
        )

    def _infer_in_channels(self) -> int:
        """Infer the number of input channels from the incoming feature shape.

        Returns:
            int: Number of input channels (defaults to 3 if not determinable).
        """
        if isinstance(self.keys_in, dict):
            key = next(iter(self.keys_in.values()))
        else:
            key = self.keys_in[0]
        shape = self.input_sizes.get(key, (3,))
        return int(shape[0]) if shape else 3

    def _build_model(self) -> nn.Module:
        from torchvision import models

        builder_name = self.resnet_type
        builder = getattr(models, builder_name)

        # Load weights if requested (torchvision >= 0.13 API)
        try:
            weights_module = getattr(models, f"{builder_name}_weights")
            weights = weights_module.DEFAULT if self.pretrained else None
        except AttributeError:
            weights = "DEFAULT" if self.pretrained else None

        # Pretrained weights are incompatible with non-RGB inputs.
        if weights is not None and self.in_channels != 3:
            logger.warning(
                "Pretrained weights require 3 input channels; disabling weights "
                "because input has %d channels.",
                self.in_channels,
            )
            weights = None

        if weights is not None:
            model = builder(weights=weights)
        else:
            model = builder(weights=None)

        # Replace final fc with identity for embeddings
        model.fc = nn.Identity()

        # Adapt the first conv layer when input channels differ from torchvision default.
        conv1 = cast(nn.Conv2d, model.conv1)
        if self.in_channels != conv1.in_channels:
            out_planes = conv1.out_channels
            model.conv1 = nn.Conv2d(
                self.in_channels,
                out_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        if self.cifar_stem:
            self._apply_cifar_stem(model)

        return model

    @staticmethod
    def _apply_cifar_stem(model: nn.Module) -> None:
        """Replace ImageNet stem with CIFAR-friendly stem."""
        # Replace 7x7 stride=2 conv with 3x3 stride=1 conv
        conv1 = cast(nn.Conv2d, model.conv1)
        in_planes = conv1.in_channels
        out_planes = conv1.out_channels
        model.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Remove initial maxpool
        model.maxpool = nn.Identity()

    def forward_tensor(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(x)
