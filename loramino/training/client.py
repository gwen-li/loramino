import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
from tqdm import tqdm

from loramino.models.registry import model_dict

from .runtime import (
    build_dataloader,
    build_lora_checkpoint,
    create_training_state,
    forward_backward as runtime_forward_backward,
    get_model_tokenizer,
    optim_step as runtime_optim_step,
    summarize_training,
    train_epoch,
)


@dataclass(frozen=True)
class SupportedModel:
    model_name: str


@dataclass(frozen=True)
class ServerCapabilities:
    supported_models: list[SupportedModel]


@dataclass
class ForwardBackwardResult:
    outputs: object
    loss: object
    examples_seen: int
    tokens_seen: int


@dataclass(frozen=True)
class OptimStepResult:
    optimizer_steps: int


class ServiceClient:
    def get_server_capabilities(self) -> ServerCapabilities:
        supported_models = [SupportedModel(model_name=name) for name in sorted(model_dict)]
        return ServerCapabilities(supported_models=supported_models)

    def create_lora_training_client(self, config_options: dict):
        return TrainingClient(config_options)


class TrainingClient:
    def __init__(self, config_options: dict):
        self.config_options = deepcopy(config_options)
        self.state = create_training_state(self.config_options)

    @property
    def model(self):
        return self.state.model

    @property
    def tokenizer(self):
        return get_model_tokenizer(self.state.model)

    @property
    def device(self):
        return self.state.device

    @property
    def jobs(self):
        return self.state.jobs

    def get_tokenizer(self):
        return self.tokenizer

    def build_dataloader(self, shuffle: bool = True):
        return build_dataloader(
            self.config_options,
            self.tokenizer,
            shuffle=shuffle,
            jobs=self.state.jobs,
            job_groups=self.state.job_groups,
        )

    def forward_backward(self, batch: dict) -> ForwardBackwardResult:
        result = runtime_forward_backward(self.state, batch)
        return ForwardBackwardResult(
            outputs=result["outputs"],
            loss=result["loss"],
            examples_seen=result["examples_seen"],
            tokens_seen=result["tokens_seen"],
        )

    def optim_step(self) -> OptimStepResult:
        return OptimStepResult(optimizer_steps=runtime_optim_step(self.state))

    def train_batch(self, batch: dict) -> dict:
        forward_result = self.forward_backward(batch)
        step_result = self.optim_step()
        return {
            "loss": forward_result.loss,
            "examples_seen": forward_result.examples_seen,
            "tokens_seen": forward_result.tokens_seen,
            "optimizer_steps": step_result.optimizer_steps,
        }

    def save(self, output_dir: str, metrics: dict | None = None) -> dict:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint = build_lora_checkpoint(self.state, self.config_options, metrics=metrics)
        checkpoint_path = output_path / "lora_weights.pt"
        config_path = output_path / "config.json"

        torch.save(checkpoint, checkpoint_path)
        config_path.write_text(json.dumps(self.config_options, indent=2) + "\n")

        saved = {
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path),
        }
        if metrics is not None:
            metrics_path = output_path / "metrics.json"
            metrics_with_saved = dict(metrics)
            metrics_with_saved["saved"] = saved
            metrics_path.write_text(json.dumps(metrics_with_saved, indent=2) + "\n")
            saved["metrics_path"] = str(metrics_path)
        return saved

    def fit(self, dataloader=None, num_epochs: int | None = None) -> dict:
        if dataloader is None:
            dataloader = self.build_dataloader(shuffle=True)

        num_epochs = self.config_options["num_epochs"] if num_epochs is None else num_epochs
        step_times = []
        examples_seen = 0
        tokens_seen = 0
        macro_steps = 0
        optimizer_steps = 0
        last_loss = None

        if self.config_options["verbose"]:
            print(f"Starting training for {num_epochs} epochs...")

        train_start = perf_counter()
        for epoch in range(num_epochs):
            if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch}") if self.config_options["verbose"] else dataloader
            epoch_metrics = train_epoch(self.state, epoch_iterator)
            step_times.extend(epoch_metrics["step_times"])
            examples_seen += epoch_metrics["examples_seen"]
            tokens_seen += epoch_metrics["tokens_seen"]
            macro_steps += epoch_metrics["macro_steps"]
            optimizer_steps += epoch_metrics["optimizer_steps"]
            last_loss = epoch_metrics["last_loss"]

            if self.config_options["verbose"] and last_loss is not None:
                print(f"Epoch {epoch + 1}/{num_epochs} completed. Loss: {last_loss.item()}")

        metrics = summarize_training(
            adapter_type=self.state.adapter_type,
            num_adapters=self.state.num_adapters,
            step_times=step_times,
            total_time=perf_counter() - train_start,
            examples_seen=examples_seen,
            tokens_seen=tokens_seen,
            macro_steps=macro_steps,
            optimizer_steps=optimizer_steps,
            last_loss=last_loss,
        )
        if self.config_options["verbose"]:
            print(
                "Training time: "
                f"{metrics['total_time_s']:.4f}s total, "
                f"{metrics['avg_step_time_s']:.4f}s/macro-step, "
                f"{metrics['examples_per_second']:.2f} examples/s"
            )
        output_dir = self.config_options.get("output_dir")
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = str(Path("outputs") / f"{self.config_options['base_model']}_{self.state.adapter_type}_{timestamp}")

        save_paths = self.save(output_dir, metrics=metrics)
        metrics["saved"] = save_paths
        if self.config_options["verbose"]:
            print(f"Saved LoRA weights to {save_paths['checkpoint_path']}")
        return metrics
