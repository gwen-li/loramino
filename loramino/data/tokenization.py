import torch


def _coerce_token_id(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_padding_token_id(tokenizer) -> int:
    for attr_name in (
        "pad_token_id",
        "eos_token_id",
        "bos_token_id",
        "unk_token_id",
        "sep_token_id",
        "cls_token_id",
    ):
        token_id = _coerce_token_id(getattr(tokenizer, attr_name, None))
        if token_id is not None:
            return token_id
    return 0


def _flatten_token_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone().to(dtype=torch.long)
    else:
        tensor = torch.as_tensor(value, dtype=torch.long)

    if tensor.ndim == 0:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor.reshape(-1)


def _normalize_tokenized_output(encoded) -> dict[str, torch.Tensor]:
    if isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
        attention_mask = encoded.get("attention_mask")
    else:
        input_ids = getattr(encoded, "input_ids", None)
        attention_mask = getattr(encoded, "attention_mask", None)

    if input_ids is None:
        input_ids = getattr(encoded, "ids", encoded)

    input_ids_tensor = _flatten_token_tensor(input_ids)
    if attention_mask is None:
        attention_mask_tensor = torch.ones_like(input_ids_tensor)
    else:
        attention_mask_tensor = _flatten_token_tensor(attention_mask)

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
    }


def _call_tokenizer(tokenizer, text: str, *, max_length: int):
    if not callable(tokenizer):
        raise TypeError(f"Tokenizer {type(tokenizer).__name__} is not callable.")

    attempts = [
        {"truncation": True, "padding": False, "max_length": max_length, "return_tensors": "pt"},
        {"truncation": True, "padding": False, "max_length": max_length},
        {"truncation": True, "max_length": max_length},
        {},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return tokenizer(text, **kwargs)
        except TypeError as exc:
            last_error = exc

    if hasattr(tokenizer, "encode"):
        try:
            return tokenizer.encode(text)
        except TypeError as exc:
            last_error = exc

    raise TypeError(
        f"Tokenizer {type(tokenizer).__name__} does not support the expected call interface."
    ) from last_error


def tokenize_fixed_length(tokenizer, text: str, max_length: int) -> dict[str, torch.Tensor]:
    encoded = _normalize_tokenized_output(_call_tokenizer(tokenizer, text, max_length=max_length))
    input_ids = encoded["input_ids"][:max_length]
    attention_mask = encoded["attention_mask"][:max_length]

    if input_ids.shape[0] < max_length:
        pad_length = max_length - input_ids.shape[0]
        pad_token_id = _resolve_padding_token_id(tokenizer)
        input_ids = torch.cat(
            [input_ids, torch.full((pad_length,), pad_token_id, dtype=torch.long)]
        )
        attention_mask = torch.cat(
            [attention_mask, torch.zeros(pad_length, dtype=torch.long)]
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
