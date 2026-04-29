"""Small benchmark harness built on top of the shared training runtime.

It measures either:
- one repeated sample batch for quick local comparisons
- the whole dataset for a more realistic pass-level benchmark

The key design choice is that benchmarking reuses the same runtime as training,
so correctness and performance are measured through the same execution path.
"""

import gc
from copy import deepcopy
from dataclasses import dataclass
import math
from time import perf_counter

from loramino.training.distributed import (
    barrier,
    get_rank,
    is_distributed,
    is_primary_rank,
    reduce_max,
    reduce_optional_mean,
    reduce_sum,
)
from loramino.training.runtime import (
    batch_size,
    build_dataloader,
    clear_pending_step,
    create_training_state,
    forward_backward,
    move_batch_to_device,
    optim_step,
    set_seed,
    synchronize_device,
)


def _release_case_memory(device=None) -> None:
    gc.collect()
    if device is not None and getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.empty_cache()


def _clone_batch(batch: dict) -> dict:
    return {key: value.clone() for key, value in batch.items()}


def _ensure_batch_size(batch: dict, min_batch_size: int) -> dict:
    current_size = batch_size(batch)
    if current_size >= min_batch_size:
        return batch

    repeats = -(-min_batch_size // current_size)
    return {
        key: value.repeat((repeats,) + (1,) * (value.ndim - 1))[:min_batch_size].clone()
        for key, value in batch.items()
    }


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    config: dict


class BatchSource:
    def __init__(
        self,
        config_options: dict,
        tokenizer,
        *,
        scope: str = "single_batch",
        min_batch_size: int = 1,
    ):
        if scope not in {"single_batch", "whole_dataset"}:
            raise ValueError("benchmark scope must be 'single_batch' or 'whole_dataset'.")

        self.scope = scope
        self.config_options = deepcopy(config_options)
        self.tokenizer = tokenizer
        self.min_batch_size = min_batch_size

        # For quick comparisons we cache a single batch and optionally upsize it
        # so every configured adapter has at least one example to process.
        first_batch = next(iter(build_dataloader(self.config_options, self.tokenizer, shuffle=False)))
        self.summary_batch_size = batch_size(first_batch)
        self.sample_batch = _ensure_batch_size(first_batch, self.min_batch_size) if scope == "single_batch" else None
        self.benchmark_batch_size = self.summary_batch_size if self.sample_batch is None else batch_size(self.sample_batch)

    def iter_batches(self, device):
        if self.scope == "single_batch":
            yield move_batch_to_device(_clone_batch(self.sample_batch), device)
            return

        dataloader = build_dataloader(self.config_options, self.tokenizer, shuffle=False)
        for batch in dataloader:
            yield move_batch_to_device(batch, device)


class BenchmarkRunner:
    def __init__(self, batch_source: BatchSource, *, seed: int = 0):
        self.batch_source = batch_source
        self.seed = seed

    def create_state(self, case: BenchmarkCase):
        case_config = deepcopy(case.config)
        case_config["seed"] = self.seed
        return create_training_state(case_config)

    def _case_label(self, case: BenchmarkCase) -> str:
        jobs = case.config.get("jobs", [])
        if len(jobs) == 1:
            return f"{case.name}:{jobs[0]['name']}"
        if jobs:
            return f"{case.name}:{len(jobs)}jobs"
        return case.name

    def _should_print_progress(self, state) -> bool:
        if not is_distributed():
            return True
        if state.ddp_enabled:
            return is_primary_rank()
        return True

    def _progress_prefix(self, state) -> str:
        if is_distributed() and not state.ddp_enabled:
            return f"[rank{get_rank()}] "
        return ""

    def _print_progress(self, state, message: str) -> None:
        if not self._should_print_progress(state):
            return
        print(f"{self._progress_prefix(state)}{message}", flush=True)

    @staticmethod
    def _format_pass_summary(run: dict) -> str:
        elapsed_s = run["elapsed_s"]
        examples_per_second = (run["examples_seen"] / elapsed_s) if elapsed_s else 0.0
        tokens_per_second = (run["tokens_seen"] / elapsed_s) if elapsed_s else 0.0
        peak_memory_mb = run.get("peak_memory_mb")
        peak_memory_text = f", peak {peak_memory_mb:.0f} MB" if peak_memory_mb is not None else ""
        return (
            f"{elapsed_s:.2f}s, {examples_per_second:.1f} ex/s, "
            f"{tokens_per_second:.1f} tok/s{peak_memory_text}"
        )

    def run_pass(self, state, *, execution_mode: str = "forward_backward") -> dict:
        if execution_mode not in {"forward_backward", "train_step"}:
            raise ValueError("execution_mode must be 'forward_backward' or 'train_step'.")

        if state.device.type == "cuda":
            import torch

            torch.cuda.reset_peak_memory_stats(state.device)

        synchronize_device(state.device)
        if state.ddp_enabled:
            barrier()
        start_time = perf_counter()
        examples_seen = 0
        tokens_seen = 0
        last_loss = None
        last_finite_loss = None
        non_finite_loss_steps = 0
        optimizer_steps = 0

        for batch in self.batch_source.iter_batches(state.device):
            step_result = forward_backward(state, _clone_batch(batch))
            loss_value = None if step_result["loss"] is None else step_result["loss"].item()
            if loss_value is None or not math.isfinite(loss_value):
                non_finite_loss_steps += 1
            else:
                last_finite_loss = loss_value
            last_loss = loss_value

            if execution_mode == "train_step":
                optimizer_steps += optim_step(state)
            else:
                clear_pending_step(state)
            examples_seen += step_result["examples_seen"]
            tokens_seen += step_result["tokens_seen"]

        synchronize_device(state.device)
        if state.ddp_enabled:
            barrier()
        elapsed_s = perf_counter() - start_time
        peak_memory_mb = None
        if state.device.type == "cuda":
            import torch

            peak_memory_mb = torch.cuda.max_memory_allocated(state.device) / (1024 * 1024)
        if state.ddp_enabled:
            elapsed_s = reduce_max(elapsed_s, device=state.device)
            examples_seen = int(reduce_sum(examples_seen, device=state.device))
            tokens_seen = int(reduce_sum(tokens_seen, device=state.device))
            non_finite_loss_steps = int(reduce_sum(non_finite_loss_steps, device=state.device))
            optimizer_steps = int(reduce_sum(optimizer_steps, device=state.device))
            peak_memory_mb = reduce_max(peak_memory_mb or 0.0, device=state.device) if peak_memory_mb is not None else None
            last_loss = reduce_optional_mean(last_loss, device=state.device)
            last_finite_loss = reduce_optional_mean(last_finite_loss, device=state.device)
        return {
            "elapsed_s": elapsed_s,
            "examples_seen": examples_seen,
            "tokens_seen": tokens_seen,
            "last_loss": last_loss,
            "last_finite_loss": last_finite_loss,
            "non_finite_loss_steps": non_finite_loss_steps,
            "optimizer_steps": optimizer_steps,
            "peak_memory_mb": peak_memory_mb,
        }

    def benchmark(
        self,
        case: BenchmarkCase,
        *,
        benchmark_steps: int = 5,
        warmup_steps: int = 1,
        execution_mode: str = "forward_backward",
    ) -> dict:
        set_seed(self.seed)
        state = None
        try:
            state = self.create_state(case)
            case_label = self._case_label(case)
            self._print_progress(
                state,
                (
                    f"[benchmark] {case_label} start "
                    f"(scope={self.batch_source.scope}, mode={execution_mode}, "
                    f"batch={self.batch_source.benchmark_batch_size}, "
                    f"warmup={warmup_steps}, measured={benchmark_steps})"
                ),
            )

            for warmup_index in range(warmup_steps):
                run = self.run_pass(state, execution_mode=execution_mode)
                self._print_progress(
                    state,
                    f"[benchmark] {case_label} warmup {warmup_index + 1}/{warmup_steps}: {self._format_pass_summary(run)}",
                )

            measured_passes = []
            for pass_index in range(benchmark_steps):
                run = self.run_pass(state, execution_mode=execution_mode)
                measured_passes.append(run)
                running_elapsed = sum(item["elapsed_s"] for item in measured_passes)
                running_examples = sum(item["examples_seen"] for item in measured_passes)
                running_tokens = sum(item["tokens_seen"] for item in measured_passes)
                running_examples_per_second = (running_examples / running_elapsed) if running_elapsed else 0.0
                running_tokens_per_second = (running_tokens / running_elapsed) if running_elapsed else 0.0
                self._print_progress(
                    state,
                    (
                        f"[benchmark] {case_label} pass {pass_index + 1}/{benchmark_steps}: "
                        f"{self._format_pass_summary(run)} | running avg {run['elapsed_s'] if len(measured_passes) == 1 else running_elapsed / len(measured_passes):.2f}s, "
                        f"{running_examples_per_second:.1f} ex/s, {running_tokens_per_second:.1f} tok/s"
                    ),
                )
            elapsed_times = [run["elapsed_s"] for run in measured_passes]
            total_elapsed = sum(elapsed_times)
            total_examples = sum(run["examples_seen"] for run in measured_passes)
            total_tokens = sum(run["tokens_seen"] for run in measured_passes)
            total_non_finite_loss_steps = sum(run["non_finite_loss_steps"] for run in measured_passes)
            peak_memory_values = [run["peak_memory_mb"] for run in measured_passes if run["peak_memory_mb"] is not None]
            final_loss = measured_passes[-1]["last_loss"] if measured_passes else None

            self._print_progress(
                state,
                (
                    f"[benchmark] {case_label} done "
                    f"(avg {total_elapsed / len(measured_passes):.2f}s, "
                    f"{(total_examples / total_elapsed) if total_elapsed else 0.0:.1f} ex/s, "
                    f"{(total_tokens / total_elapsed) if total_elapsed else 0.0:.1f} tok/s)"
                ),
            )

            return {
                "avg_pass_time_s": total_elapsed / len(measured_passes),
                "pass_times_s": elapsed_times,
                "examples_per_second": (total_examples / total_elapsed) if total_elapsed else 0.0,
                "tokens_per_second": (total_tokens / total_elapsed) if total_elapsed else 0.0,
                "final_loss": final_loss,
                "final_loss_is_finite": final_loss is not None and math.isfinite(final_loss),
                "last_finite_loss": measured_passes[-1]["last_finite_loss"] if measured_passes else None,
                "non_finite_loss_steps": total_non_finite_loss_steps,
                "optimizer_steps": sum(run["optimizer_steps"] for run in measured_passes),
                "peak_memory_mb_per_pass": peak_memory_values,
                "avg_peak_memory_mb": (
                    sum(peak_memory_values) / len(peak_memory_values) if peak_memory_values else None
                ),
                "max_peak_memory_mb": max(peak_memory_values) if peak_memory_values else None,
            }
        finally:
            device = state.device if state is not None else None
            del state
            _release_case_memory(device)

    def benchmark_all(
        self,
        cases: list[BenchmarkCase],
        *,
        benchmark_steps: int = 5,
        warmup_steps: int = 1,
        execution_mode: str = "forward_backward",
    ) -> dict:
        return {
            case.name: self.benchmark(
                case,
                benchmark_steps=benchmark_steps,
                warmup_steps=warmup_steps,
                execution_mode=execution_mode,
            )
            for case in cases
        }
