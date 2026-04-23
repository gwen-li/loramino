"""Grouped LoRA kernel backends.

The runtime and scheduler decide which adapters execute together. This module
only computes the per-token LoRA delta for a bank of private adapters over a
shared frozen linear layer.
"""

from __future__ import annotations

import torch

from .cuda_extension import grouped_lora_cuda_error, load_grouped_lora_cuda_extension

_CUDA_DTYPES = {torch.float16, torch.float32, torch.float64}
_AUTO_CUDA_DTYPES = {torch.float16}
_AUTO_CUDA_MIN_ACTIVE_ADAPTERS = 8
_AUTO_CUDA_MIN_TOKENS = 2048
_AUTO_CUDA_MIN_WORK_UNITS = 128_000_000


def _validate_inputs(
    flat_x: torch.Tensor,
    token_adapter_ids: torch.Tensor | None,
    A: torch.Tensor,
    B: torch.Tensor,
    scales: torch.Tensor,
    ranks: torch.Tensor,
    segment_adapter_ids: torch.Tensor | None = None,
    segment_token_starts: torch.Tensor | None = None,
    segment_token_counts: torch.Tensor | None = None,
) -> None:
    if flat_x.ndim != 2:
        raise ValueError(f"flat_x must have shape [tokens, hidden], got {tuple(flat_x.shape)}.")
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("A and B must have shapes [adapters, rank, in] and [adapters, out, rank].")
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must contain the same number of adapters.")
    if A.shape[1] != B.shape[2]:
        raise ValueError("A and B must agree on max rank.")
    if flat_x.shape[1] != A.shape[2]:
        raise ValueError("flat_x hidden size must match A in_features.")
    if scales.shape != (A.shape[0],):
        raise ValueError(f"scales must have shape ({A.shape[0]},), got {tuple(scales.shape)}.")
    if ranks.shape != (A.shape[0],):
        raise ValueError(f"ranks must have shape ({A.shape[0]},), got {tuple(ranks.shape)}.")
    if token_adapter_ids is None and segment_adapter_ids is None:
        raise ValueError("Provide either token_adapter_ids or a contiguous segment layout.")
    if token_adapter_ids is not None:
        if token_adapter_ids.shape != (flat_x.shape[0],):
            raise ValueError(
                f"token_adapter_ids must have shape ({flat_x.shape[0]},), got {tuple(token_adapter_ids.shape)}."
            )
        if token_adapter_ids.dtype != torch.long:
            raise ValueError("token_adapter_ids must use dtype torch.long.")
    if ranks.dtype != torch.long:
        raise ValueError("ranks must use dtype torch.long.")
    if flat_x.dtype != A.dtype or A.dtype != B.dtype or scales.dtype != A.dtype:
        raise ValueError("flat_x, A, B, and scales must share the same dtype.")
    devices = {flat_x.device, A.device, B.device, scales.device, ranks.device}
    if token_adapter_ids is not None:
        devices.add(token_adapter_ids.device)
    if segment_adapter_ids is not None or segment_token_starts is not None or segment_token_counts is not None:
        if segment_adapter_ids is None or segment_token_starts is None or segment_token_counts is None:
            raise ValueError("segment_adapter_ids, segment_token_starts, and segment_token_counts must be provided together.")
        if segment_adapter_ids.ndim != 1 or segment_token_starts.ndim != 1 or segment_token_counts.ndim != 1:
            raise ValueError("Contiguous segment layout tensors must be one-dimensional.")
        if not (
            segment_adapter_ids.shape == segment_token_starts.shape == segment_token_counts.shape
        ):
            raise ValueError("Contiguous segment layout tensors must share the same shape.")
        if (
            segment_adapter_ids.dtype != torch.long
            or segment_token_starts.dtype != torch.long
            or segment_token_counts.dtype != torch.long
        ):
            raise ValueError("Contiguous segment layout tensors must use dtype torch.long.")
        devices.update({segment_adapter_ids.device, segment_token_starts.device, segment_token_counts.device})
        if segment_adapter_ids.numel() == 0:
            if flat_x.shape[0] != 0:
                raise ValueError("Contiguous segment layout must cover every token in flat_x.")
        else:
            if int(segment_token_starts[0].item()) != 0:
                raise ValueError("Contiguous segment layout must start at token offset 0.")
            if torch.any(segment_token_counts < 0):
                raise ValueError("segment_token_counts must be non-negative.")
            if torch.any(segment_adapter_ids < 0) or torch.any(segment_adapter_ids >= A.shape[0]):
                raise ValueError("segment_adapter_ids must be in range [0, num_adapters).")
            expected_starts = torch.cumsum(
                torch.cat((segment_token_counts.new_zeros(1), segment_token_counts[:-1])),
                dim=0,
            )
            if not torch.equal(segment_token_starts, expected_starts):
                raise ValueError("Contiguous segment layout must be packed without gaps or overlaps.")
            if int((segment_token_starts[-1] + segment_token_counts[-1]).item()) != flat_x.shape[0]:
                raise ValueError("Contiguous segment layout must cover every token in flat_x.")
    if len(devices) != 1:
        raise ValueError("All grouped LoRA tensors must live on the same device.")


def _active_adapter_ids(
    token_adapter_ids: torch.Tensor,
    num_adapters: int,
    adapter_order: torch.Tensor | None,
) -> torch.Tensor:
    active_mask = torch.zeros(num_adapters, dtype=torch.bool, device=token_adapter_ids.device)
    active_mask.scatter_(0, token_adapter_ids, True)
    if adapter_order is None:
        return active_mask.nonzero(as_tuple=True)[0]
    order = adapter_order.to(device=token_adapter_ids.device, dtype=torch.long)
    return order[active_mask.index_select(0, order)]


def _active_segment_adapter_ids(
    segment_adapter_ids: torch.Tensor,
    num_adapters: int,
    adapter_order: torch.Tensor | None,
) -> torch.Tensor:
    active_mask = torch.zeros(num_adapters, dtype=torch.bool, device=segment_adapter_ids.device)
    active_mask.scatter_(0, segment_adapter_ids, True)
    if adapter_order is None:
        return active_mask.nonzero(as_tuple=True)[0]
    order = adapter_order.to(device=segment_adapter_ids.device, dtype=torch.long)
    return order[active_mask.index_select(0, order)]


def _reference_grouped_lora_forward(
    flat_x: torch.Tensor,
    token_adapter_ids: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scales: torch.Tensor,
    ranks: torch.Tensor,
    adapter_order: torch.Tensor | None,
) -> torch.Tensor:
    output = flat_x.new_zeros(flat_x.shape[0], B.shape[1])
    for adapter_id in _active_adapter_ids(token_adapter_ids, A.shape[0], adapter_order).tolist():
        token_indices = (token_adapter_ids == adapter_id).nonzero(as_tuple=True)[0]
        rank = int(ranks[adapter_id].item())
        if token_indices.numel() == 0 or rank == 0:
            continue

        x_adapter = flat_x.index_select(0, token_indices)
        A_adapter = A[adapter_id, :rank, :]
        B_adapter = B[adapter_id, :, :rank]
        delta = (x_adapter @ A_adapter.transpose(0, 1)) @ B_adapter.transpose(0, 1)
        output.index_copy_(0, token_indices, delta * scales[adapter_id])
    return output


def _reference_grouped_lora_backward(
    flat_x: torch.Tensor,
    token_adapter_ids: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scales: torch.Tensor,
    ranks: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_x = torch.zeros_like(flat_x)
    grad_A = torch.zeros_like(A)
    grad_B = torch.zeros_like(B)

    active_adapter_ids = token_adapter_ids.unique(sorted=True)
    for adapter_id in active_adapter_ids.tolist():
        token_indices = (token_adapter_ids == adapter_id).nonzero(as_tuple=True)[0]
        rank = int(ranks[adapter_id].item())
        if token_indices.numel() == 0 or rank == 0:
            continue

        x_adapter = flat_x.index_select(0, token_indices)
        grad_adapter = grad_output.index_select(0, token_indices) * scales[adapter_id]
        A_adapter = A[adapter_id, :rank, :]
        B_adapter = B[adapter_id, :, :rank]

        down_projection = x_adapter @ A_adapter.transpose(0, 1)
        grad_B[adapter_id, :, :rank] = grad_adapter.transpose(0, 1) @ down_projection

        grad_down_projection = grad_adapter @ B_adapter
        grad_A[adapter_id, :rank, :] = grad_down_projection.transpose(0, 1) @ x_adapter
        grad_x.index_copy_(0, token_indices, grad_down_projection @ A_adapter)

    return grad_x, grad_A, grad_B


def _reference_grouped_lora_forward_from_segments(
    flat_x: torch.Tensor,
    segment_adapter_ids: torch.Tensor,
    segment_token_starts: torch.Tensor,
    segment_token_counts: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scales: torch.Tensor,
    ranks: torch.Tensor,
) -> torch.Tensor:
    output = flat_x.new_zeros(flat_x.shape[0], B.shape[1])
    for segment_index, adapter_id in enumerate(segment_adapter_ids.tolist()):
        token_start = int(segment_token_starts[segment_index].item())
        token_count = int(segment_token_counts[segment_index].item())
        rank = int(ranks[adapter_id].item())
        if token_count == 0 or rank == 0:
            continue

        x_adapter = flat_x.narrow(0, token_start, token_count)
        A_adapter = A[adapter_id, :rank, :]
        B_adapter = B[adapter_id, :, :rank]
        delta = (x_adapter @ A_adapter.transpose(0, 1)) @ B_adapter.transpose(0, 1)
        output.narrow(0, token_start, token_count).copy_(delta * scales[adapter_id])
    return output


def _reference_grouped_lora_backward_from_segments(
    flat_x: torch.Tensor,
    segment_adapter_ids: torch.Tensor,
    segment_token_starts: torch.Tensor,
    segment_token_counts: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scales: torch.Tensor,
    ranks: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_x = torch.zeros_like(flat_x)
    grad_A = torch.zeros_like(A)
    grad_B = torch.zeros_like(B)

    for segment_index, adapter_id in enumerate(segment_adapter_ids.tolist()):
        token_start = int(segment_token_starts[segment_index].item())
        token_count = int(segment_token_counts[segment_index].item())
        rank = int(ranks[adapter_id].item())
        if token_count == 0 or rank == 0:
            continue

        x_adapter = flat_x.narrow(0, token_start, token_count)
        grad_adapter = grad_output.narrow(0, token_start, token_count) * scales[adapter_id]
        A_adapter = A[adapter_id, :rank, :]
        B_adapter = B[adapter_id, :, :rank]

        down_projection = x_adapter @ A_adapter.transpose(0, 1)
        grad_B[adapter_id, :, :rank].add_(grad_adapter.transpose(0, 1) @ down_projection)

        grad_down_projection = grad_adapter @ B_adapter
        grad_A[adapter_id, :rank, :].add_(grad_down_projection.transpose(0, 1) @ x_adapter)
        grad_x.narrow(0, token_start, token_count).copy_(grad_down_projection @ A_adapter)

    return grad_x, grad_A, grad_B


def _estimate_grouped_lora_work(
    flat_x: torch.Tensor,
    token_adapter_ids: torch.Tensor | None,
    B: torch.Tensor,
    ranks: torch.Tensor,
    segment_adapter_ids: torch.Tensor | None = None,
) -> tuple[int, int, int]:
    if token_adapter_ids is not None:
        active_adapter_ids = token_adapter_ids.unique(sorted=True)
    elif segment_adapter_ids is not None:
        active_adapter_ids = segment_adapter_ids.unique(sorted=True)
    else:
        raise ValueError("Provide token_adapter_ids or segment_adapter_ids to estimate grouped LoRA work.")

    active_adapter_count = int(active_adapter_ids.numel())
    max_rank = int(ranks.index_select(0, active_adapter_ids).max().item())
    work_units = flat_x.shape[0] * max_rank * (flat_x.shape[1] + B.shape[1])
    return active_adapter_count, max_rank, work_units


def _should_use_auto_cuda(
    flat_x: torch.Tensor,
    token_adapter_ids: torch.Tensor | None,
    B: torch.Tensor,
    ranks: torch.Tensor,
    segment_adapter_ids: torch.Tensor | None = None,
) -> bool:
    if not flat_x.is_cuda or flat_x.dtype not in _AUTO_CUDA_DTYPES:
        return False

    active_adapter_count, _max_rank, work_units = _estimate_grouped_lora_work(
        flat_x,
        token_adapter_ids,
        B,
        ranks,
        segment_adapter_ids=segment_adapter_ids,
    )
    return (
        active_adapter_count >= _AUTO_CUDA_MIN_ACTIVE_ADAPTERS
        and flat_x.shape[0] >= _AUTO_CUDA_MIN_TOKENS
        and work_units >= _AUTO_CUDA_MIN_WORK_UNITS
    )


def resolve_grouped_lora_backend(
    requested_backend: str,
    flat_x: torch.Tensor,
    token_adapter_ids: torch.Tensor | None,
    B: torch.Tensor,
    ranks: torch.Tensor,
    adapter_group_ids: torch.Tensor | None = None,
    segment_adapter_ids: torch.Tensor | None = None,
) -> str:
    if requested_backend not in {"auto", "cuda", "torch", "legacy"}:
        raise ValueError("kernel backend must be one of: auto, cuda, torch, legacy.")

    if requested_backend == "legacy":
        return "legacy"

    if requested_backend == "auto":
        if _should_use_auto_cuda(
            flat_x,
            token_adapter_ids,
            B,
            ranks,
            segment_adapter_ids=segment_adapter_ids,
        ):
            extension = load_grouped_lora_cuda_extension()
            if extension is not None:
                return "cuda"
        return "torch"

    if requested_backend == "torch":
        return "torch"

    if not flat_x.is_cuda:
        raise RuntimeError("kernel_backend='cuda' requires CUDA tensors.")

    if flat_x.dtype not in _CUDA_DTYPES:
        raise RuntimeError(f"CUDA kernel does not support dtype {flat_x.dtype}.")

    extension = load_grouped_lora_cuda_extension()
    if extension is None:
        error_message = grouped_lora_cuda_error() or "CUDA extension failed to load."
        raise RuntimeError(error_message)

    return "cuda"


class _GroupedLoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        flat_x: torch.Tensor,
        token_adapter_ids: torch.Tensor | None,
        A: torch.Tensor,
        B: torch.Tensor,
        scales: torch.Tensor,
        ranks: torch.Tensor,
        adapter_order: torch.Tensor | None,
        segment_adapter_ids: torch.Tensor | None,
        segment_token_starts: torch.Tensor | None,
        segment_token_counts: torch.Tensor | None,
        backend: str,
    ) -> torch.Tensor:
        empty_long = torch.empty(0, dtype=torch.long, device=flat_x.device)
        saved_token_adapter_ids = empty_long if token_adapter_ids is None else token_adapter_ids
        saved_segment_adapter_ids = empty_long if segment_adapter_ids is None else segment_adapter_ids
        saved_segment_token_starts = empty_long if segment_token_starts is None else segment_token_starts
        saved_segment_token_counts = empty_long if segment_token_counts is None else segment_token_counts
        ctx.use_segment_layout = segment_adapter_ids is not None

        if backend == "cuda":
            extension = load_grouped_lora_cuda_extension()
            if extension is None:
                raise RuntimeError(grouped_lora_cuda_error() or "CUDA extension failed to load.")
            ctx.backend = "cuda"
            ctx.save_for_backward(
                flat_x,
                saved_token_adapter_ids,
                A,
                B,
                scales,
                ranks,
                saved_segment_adapter_ids,
                saved_segment_token_starts,
                saved_segment_token_counts,
            )
            if ctx.use_segment_layout:
                return extension.forward_contiguous(
                    flat_x.contiguous(),
                    saved_segment_adapter_ids.contiguous(),
                    saved_segment_token_starts.contiguous(),
                    saved_segment_token_counts.contiguous(),
                    A.contiguous(),
                    B.contiguous(),
                    scales.contiguous(),
                    ranks.contiguous(),
                )
            return extension.forward(
                flat_x.contiguous(),
                saved_token_adapter_ids.contiguous(),
                A.contiguous(),
                B.contiguous(),
                scales.contiguous(),
                ranks.contiguous(),
            )

        ctx.backend = "torch"
        ctx.save_for_backward(
            flat_x,
            saved_token_adapter_ids,
            A,
            B,
            scales,
            ranks,
            saved_segment_adapter_ids,
            saved_segment_token_starts,
            saved_segment_token_counts,
        )
        if ctx.use_segment_layout:
            return _reference_grouped_lora_forward_from_segments(
                flat_x=flat_x,
                segment_adapter_ids=saved_segment_adapter_ids,
                segment_token_starts=saved_segment_token_starts,
                segment_token_counts=saved_segment_token_counts,
                A=A,
                B=B,
                scales=scales,
                ranks=ranks,
            )
        return _reference_grouped_lora_forward(
            flat_x=flat_x,
            token_adapter_ids=saved_token_adapter_ids,
            A=A,
            B=B,
            scales=scales,
            ranks=ranks,
            adapter_order=adapter_order,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (
            flat_x,
            token_adapter_ids,
            A,
            B,
            scales,
            ranks,
            segment_adapter_ids,
            segment_token_starts,
            segment_token_counts,
        ) = ctx.saved_tensors
        if getattr(ctx, "backend", None) == "cuda":
            extension = load_grouped_lora_cuda_extension()
            if extension is None:
                raise RuntimeError(grouped_lora_cuda_error() or "CUDA extension failed to load.")
            if ctx.use_segment_layout:
                grad_x, grad_A, grad_B = extension.backward_contiguous(
                    flat_x.contiguous(),
                    segment_adapter_ids.contiguous(),
                    segment_token_starts.contiguous(),
                    segment_token_counts.contiguous(),
                    A.contiguous(),
                    B.contiguous(),
                    scales.contiguous(),
                    ranks.contiguous(),
                    grad_output.contiguous(),
                )
                return grad_x, None, grad_A, grad_B, None, None, None, None, None, None, None
            grad_x, grad_A, grad_B = extension.backward(
                flat_x.contiguous(),
                token_adapter_ids.contiguous(),
                A.contiguous(),
                B.contiguous(),
                scales.contiguous(),
                ranks.contiguous(),
                grad_output.contiguous(),
            )
            return grad_x, None, grad_A, grad_B, None, None, None, None, None, None, None

        if ctx.use_segment_layout:
            grad_x, grad_A, grad_B = _reference_grouped_lora_backward_from_segments(
                flat_x=flat_x,
                segment_adapter_ids=segment_adapter_ids,
                segment_token_starts=segment_token_starts,
                segment_token_counts=segment_token_counts,
                A=A,
                B=B,
                scales=scales,
                ranks=ranks,
                grad_output=grad_output.contiguous(),
            )
        else:
            grad_x, grad_A, grad_B = _reference_grouped_lora_backward(
                flat_x=flat_x,
                token_adapter_ids=token_adapter_ids,
                A=A,
                B=B,
                scales=scales,
                ranks=ranks,
                grad_output=grad_output.contiguous(),
            )
        return grad_x, None, grad_A, grad_B, None, None, None, None, None, None, None


def grouped_lora_forward(
    flat_x: torch.Tensor,
    token_adapter_ids: torch.Tensor | None,
    A: torch.Tensor,
    B: torch.Tensor,
    scales: torch.Tensor,
    ranks: torch.Tensor,
    *,
    adapter_order: torch.Tensor | None = None,
    adapter_group_ids: torch.Tensor | None = None,
    segment_adapter_ids: torch.Tensor | None = None,
    segment_token_starts: torch.Tensor | None = None,
    segment_token_counts: torch.Tensor | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    _validate_inputs(
        flat_x,
        token_adapter_ids,
        A,
        B,
        scales,
        ranks,
        segment_adapter_ids=segment_adapter_ids,
        segment_token_starts=segment_token_starts,
        segment_token_counts=segment_token_counts,
    )
    resolved_backend = resolve_grouped_lora_backend(
        backend,
        flat_x,
        token_adapter_ids,
        B,
        ranks,
        adapter_group_ids=adapter_group_ids,
        segment_adapter_ids=segment_adapter_ids,
    )
    if resolved_backend == "legacy":
        if segment_adapter_ids is not None:
            return _reference_grouped_lora_forward_from_segments(
                flat_x=flat_x,
                segment_adapter_ids=segment_adapter_ids,
                segment_token_starts=segment_token_starts,
                segment_token_counts=segment_token_counts,
                A=A,
                B=B,
                scales=scales,
                ranks=ranks,
            )
        return _reference_grouped_lora_forward(
            flat_x=flat_x,
            token_adapter_ids=token_adapter_ids,
            A=A,
            B=B,
            scales=scales,
            ranks=ranks,
            adapter_order=adapter_order,
        )
    return _GroupedLoRAFunction.apply(
        flat_x,
        token_adapter_ids,
        A,
        B,
        scales,
        ranks,
        adapter_order,
        segment_adapter_ids,
        segment_token_starts,
        segment_token_counts,
        resolved_backend,
    )
