"""Reconstruction loss layer."""

from __future__ import annotations
from nexuml.core.discovery import layer

import math
from typing import cast

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer


@layer("ReconstructionLoss")
class ReconstructionLoss(PipelineLayer):
    """Computes MSE reconstruction loss between original and reconstructed features.

    keys_in should be [original_key, reconstructed_key].
    keys_out should be [loss_key].
    """

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        norm_pix_loss: bool = False,
        match_min_length: bool = False,
        patch_size: int | tuple[int, int] | None = None,
        use_mask_loss: bool = False,
        inverse_mask: bool = True,
        num_tokens: int = 0,
        **kwargs,
    ):
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)
        self.norm_pix_loss = norm_pix_loss
        self.match_min_length = match_min_length
        self.patch_size = patch_size
        self.use_mask_loss = use_mask_loss
        self.inverse_mask = inverse_mask
        self.num_tokens = num_tokens

    def _resolve_patch_size(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> tuple[int, int] | None:
        if self.patch_size is not None:
            if isinstance(self.patch_size, int):
                return (self.patch_size, self.patch_size)
            return self.patch_size

        if original.ndim != 4 or reconstructed.ndim != 3:
            return None

        channels = original.shape[1]
        patch_area = reconstructed.shape[-1] // max(channels, 1)
        side = int(round(patch_area**0.5))
        if side > 0 and side * side * channels == reconstructed.shape[-1]:
            return (side, side)
        return None

    def _patchify_original(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_size = self._resolve_patch_size(original, reconstructed)
        if patch_size is None or original.ndim != 4 or reconstructed.ndim != 3:
            return original, reconstructed

        batch_size, channels, height, width = original.shape
        patch_h, patch_w = patch_size
        if patch_h <= 0 or patch_w <= 0:
            return original, reconstructed

        n_patches_h, n_patches_w = self._resolve_patch_grid(
            height=height,
            width=width,
            patch_h=patch_h,
            patch_w=patch_w,
            expected_tokens=reconstructed.shape[1],
        )
        if n_patches_h == 0 or n_patches_w == 0:
            return original, reconstructed

        padded = self._pad_to_patch_grid(original, n_patches_h, n_patches_w, patch_h, patch_w)
        target = (
            padded.reshape(batch_size, channels, n_patches_h, patch_h, n_patches_w, patch_w)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch_size, n_patches_h * n_patches_w, channels * patch_h * patch_w)
        )

        if target.shape[1:] != reconstructed.shape[1:]:
            return original, reconstructed

        return target, reconstructed

    def _resolve_patch_grid(
        self,
        *,
        height: int,
        width: int,
        patch_h: int,
        patch_w: int,
        expected_tokens: int,
    ) -> tuple[int, int]:
        floor_h = height // patch_h
        floor_w = width // patch_w
        ceil_h = math.ceil(height / patch_h)
        ceil_w = math.ceil(width / patch_w)
        candidates = [
            (ceil_h, ceil_w),
            (floor_h, floor_w),
        ]
        for n_patches_h, n_patches_w in candidates:
            if n_patches_h > 0 and n_patches_w > 0 and n_patches_h * n_patches_w == expected_tokens:
                return n_patches_h, n_patches_w
        if ceil_h > 0 and expected_tokens % ceil_h == 0:
            return ceil_h, expected_tokens // ceil_h
        if floor_h > 0 and expected_tokens % floor_h == 0:
            return floor_h, expected_tokens // floor_h
        return floor_h, floor_w

    def _pad_to_patch_grid(
        self,
        original: torch.Tensor,
        n_patches_h: int,
        n_patches_w: int,
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        target_height = n_patches_h * patch_h
        target_width = n_patches_w * patch_w
        pad_h = max(0, target_height - original.shape[-2])
        pad_w = max(0, target_width - original.shape[-1])
        if pad_h == 0 and pad_w == 0:
            return original[:, :, :target_height, :target_width]
        padded = F.pad(original, (0, pad_w, 0, pad_h))
        return padded[:, :, :target_height, :target_width]

    def _align_mask(self, mask: torch.Tensor, num_items: int) -> torch.Tensor:
        mask = mask.bool()
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask tensor, got shape {tuple(mask.shape)}.")

        if mask.shape[1] == num_items + self.num_tokens:
            mask = mask[:, self.num_tokens :]
        elif mask.shape[1] != num_items:
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} does not match target item count {num_items} "
                f"with num_tokens={self.num_tokens}."
            )

        return ~mask if self.inverse_mask else mask

    def forward(
        self,
        x: TensorDict | torch.Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | torch.Tensor, TensorDict | None]:
        assert isinstance(x, TensorDict)
        keys_in: list[str] = cast(list[str], self.keys_in)
        original = cast(torch.Tensor, x[keys_in[0]])
        reconstructed = cast(torch.Tensor, x[keys_in[1]])
        mask = cast(torch.Tensor, x[keys_in[2]]) if len(keys_in) >= 3 else None

        original_cmp, reconstructed_cmp = self._patchify_original(original, reconstructed)
        batch_size = original_cmp.shape[0]

        if original_cmp.shape != reconstructed_cmp.shape:
            original_flat = original_cmp.reshape(batch_size, -1)
            reconstructed_flat = reconstructed_cmp.reshape(batch_size, -1)

            if original_flat.shape[1] != reconstructed_flat.shape[1]:
                if not self.match_min_length:
                    raise ValueError(
                        "ReconstructionLoss input shape mismatch after flattening: "
                        f"original={tuple(original_flat.shape)} "
                        f"reconstructed={tuple(reconstructed_flat.shape)}. "
                        "Set match_min_length=True to align by truncating to the shared prefix."
                    )
                min_len = min(original_flat.shape[1], reconstructed_flat.shape[1])
                original_flat = original_flat[:, :min_len]
                reconstructed_flat = reconstructed_flat[:, :min_len]

            if self.norm_pix_loss:
                mean = original_flat.mean(dim=-1, keepdim=True)
                var = original_flat.var(dim=-1, keepdim=True)
                original_flat = (original_flat - mean) / (var + 1e-6).sqrt()

            loss = (reconstructed_flat - original_flat).pow(2).mean(dim=-1)
            x[self.keys_out[0]] = loss
            return x, y

        if self.norm_pix_loss:
            mean = original_cmp.mean(dim=-1, keepdim=True)
            var = original_cmp.var(dim=-1, keepdim=True)
            original_cmp = (original_cmp - mean) / (var + 1e-6).sqrt()

        sq_error = (reconstructed_cmp - original_cmp).pow(2)

        if mask is not None and self.use_mask_loss:
            item_error = sq_error.mean(dim=-1)
            item_mask = self._align_mask(mask, item_error.shape[1]).float()

            # When no patches are masked (e.g. validation with drop_only_train=True),
            # the aligned mask is all zeros. Fall back to full reconstruction loss
            # so that validation loss remains meaningful.
            mask_sums = item_mask.sum(dim=-1)
            if mask_sums.sum() == 0:
                loss = sq_error.reshape(batch_size, -1).mean(dim=-1)
            else:
                loss = (item_error * item_mask).sum(dim=-1) / mask_sums.clamp_min(1.0)
        else:
            loss = sq_error.reshape(batch_size, -1).mean(dim=-1)

        x[self.keys_out[0]] = loss

        return x, y

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        # Not used — forward() is overridden directly
        raise NotImplementedError
