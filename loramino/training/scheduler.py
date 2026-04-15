from dataclasses import dataclass

from loramino.adapters.scheduler import compute_rank_groups
from loramino.data import TrainingJob


@dataclass(frozen=True)
class ScheduledJobGroup:
    """A lightweight runtime view of jobs co-scheduled together."""

    job_indices: tuple[int, ...]
    adapter_ids: tuple[int, ...]
    ranks: tuple[int, ...]
    job_names: tuple[str, ...]


class RankAwareScheduler:
    """Group jobs with similar adapter ranks to reduce padded batched work."""

    def __init__(
        self,
        *,
        min_group_size: int = 1,
        max_group_size: int = 16,
        max_rank_difference: int = 8,
    ):
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.max_rank_difference = max_rank_difference

    @classmethod
    def from_config(cls, config_options: dict):
        scheduler_config = config_options.get("job_scheduler", {})
        return cls(
            min_group_size=scheduler_config.get("min_group_size", 1),
            max_group_size=scheduler_config.get("max_group_size", 16),
            max_rank_difference=scheduler_config.get("max_rank_difference", 8),
        )

    def group_jobs(self, jobs: list[TrainingJob]) -> list[ScheduledJobGroup]:
        if not jobs:
            return []

        rank_groups = compute_rank_groups(
            [job.rank for job in jobs],
            min_group_size=self.min_group_size,
            max_group_size=self.max_group_size,
            max_rank_difference=self.max_rank_difference,
        )
        return [
            ScheduledJobGroup(
                job_indices=tuple(group),
                adapter_ids=tuple(jobs[index].adapter_id for index in group),
                ranks=tuple(jobs[index].rank for index in group),
                job_names=tuple(jobs[index].name for index in group),
            )
            for group in rank_groups
        ]
