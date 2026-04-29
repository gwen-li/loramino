import os
from pathlib import Path

import torch.distributed as dist
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from .base import Model


class Pythia(Model):
    def __init__(self, num_params: str):
        super().__init__()
        self.model_name_or_path = f"EleutherAI/pythia-{num_params}-deduped"
        self.model_revision = "step3000"
        self.model_cache_dir = self._resolve_cache_dir(num_params)

        if not self._use_distributed_cache_coordination():
            self.model = self._load_model()
            self.tokenizer = self.build_tokenizer()
        elif self._distributed_rank() == 0:
            self.model = self._load_model()
            self.tokenizer = self.build_tokenizer()
            self._distributed_barrier()
        else:
            self._distributed_barrier()
            self.model = self._load_model(local_files_only=True)
            self.tokenizer = self.build_tokenizer(local_files_only=True)
        self._distributed_barrier()

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if (
            getattr(self.model, "generation_config", None) is not None
            and self.model.generation_config.pad_token_id is None
        ):
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def build_tokenizer(self, *, local_files_only: bool = False):
        return AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            revision=self.model_revision,
            local_files_only=local_files_only,
            **self._cache_kwargs(),
        )

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def _load_model(self, *, local_files_only: bool = False):
        return GPTNeoXForCausalLM.from_pretrained(
            self.model_name_or_path,
            revision=self.model_revision,
            local_files_only=local_files_only,
            **self._cache_kwargs(),
        )

    def _resolve_cache_dir(self, num_params: str) -> str | None:
        explicit_cache_dir = os.environ.get("LORAMINO_MODEL_CACHE_DIR")
        if explicit_cache_dir:
            Path(explicit_cache_dir).mkdir(parents=True, exist_ok=True)
            return explicit_cache_dir

        cache_root = os.environ.get("LORAMINO_MODEL_CACHE_ROOT")
        if cache_root:
            cache_dir = Path(cache_root) / f"pythia-{num_params}-deduped" / self.model_revision
            cache_dir.mkdir(parents=True, exist_ok=True)
            return str(cache_dir)

        # Let Hugging Face honor HF_HOME/HF_HUB_CACHE/TRANSFORMERS_CACHE.
        # Avoid defaulting to ./pythia-... because project directories are often
        # small quota-backed home filesystems on clusters.
        return None

    def _cache_kwargs(self) -> dict:
        if self.model_cache_dir is None:
            return {}
        return {"cache_dir": self.model_cache_dir}

    def _distributed_rank(self) -> int:
        return dist.get_rank()

    def _use_distributed_cache_coordination(self) -> bool:
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    def _distributed_barrier(self) -> None:
        if self._use_distributed_cache_coordination():
            dist.barrier()
