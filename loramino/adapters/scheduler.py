from copy import deepcopy

def compute_memory_cost(rank: int, parameter_bytes: int = 4, layer_size: int = 4096) -> int:
    """ Computes the memory cost of a LoRA adaptor given its rank. This is a simple
        model that assumes the memory cost is proportional to the number of parameters
        in the adaptor, which is determined by the rank and the size of the layers
        it is applied to.
        
        Args:
            rank (int): The rank of the LoRA adaptor.
            parameter_bytes (int): The number of bytes per parameter. Default is 4 (float32).
            layer_size (int): The size of the layers the adaptor is applied to. Default is 4096.
        
        Returns:
            int: The estimated memory cost in bytes.
    """
    return 2 * rank * layer_size * parameter_bytes


def compute_group_memory_cost(ranks: list[tuple[int, int]], start_rank: int, end_rank: int, parameter_bytes: int = 4, layer_sizes: list[int]) -> int:
    """ Computes the total memory cost of a group of LoRA adaptors. This is done by summing
        the memory costs of each individual adaptor in the group, which are computed using
        the compute_memory_cost function.
        
        Args:
            ranks (list[int]): A list of ranks for the LoRA adaptors in the group.
            start_rank (int): The rank of the first adaptor in the group.
            end_rank (int): The rank of the last adaptor in the group.
            parameter_bytes (int): The number of bytes per parameter. Default is 4 (float32).
            layer_sizes (list[int] | None): A list of layer sizes corresponding to each adaptor. If None, a default layer size will be used for all adaptors.
        
        Returns:
            int: The total estimated memory cost in bytes for the group.
    """
    total_cost = 0
    for i in range(start_rank, end_rank):
        rank = ranks[i][1]
        for layer_size in layer_sizes:
            total_cost += compute_memory_cost(rank, parameter_bytes, layer_size)
    return total_cost

def compute_rank_groups(ranks: list[int],
                        min_group_size: int = 1,
                        max_group_size: int = 16,
                        max_rank_difference: int = 8,
                        parameter_bytes: int = 4,
                        layer_sizes: list[int] | None = None,
                        max_memory_usage: int | None = None) -> list[list[int]]:
    """ Partitions a list of LoRA adaptor ranks into groups. Partitioning is
        done using dynamic programming in order to maximize the minimum group
        size. These groups are subject to a constraint on the maxmimum rank difference
        within each group, as well as a maximum group size, in order to mimimize wasted
        computation due to padding and to ensure that groups are reasonably sized.
        
        Args:
            ranks (list[int]): A list of LoRA adaptor ranks.
            min_group_size (int): The minimum size of each group. Default is 1. This
                argument will be ignored if it is impossible to satisfy the max_rank_difference
                constraint while maintaining this minimum group size.
            max_group_size (int): The maximum size of each group. Default is 16.
            max_rank_difference (int): The maximum allowed difference in ranks within a group. Default is 8.
        
        Returns:
            list[list[int]]: A list of groups, where each group is a list of indices corresponding to the input adaptors.
    """
    ranks_with_indices = list(enumerate(ranks))
    sorted_ranks = sorted(ranks_with_indices, key=lambda x: x[1])
    partition_groups = []
    dp_cache = [None for _ in range(len(sorted_ranks))]
    if max_memory_usage and not layer_sizes:
        raise ValueError("If max_memory_usage is specified, layer_sizes must also be provided.")
    
    def dp_helper(current_index: int) -> tuple[int, int]:
        curr_min_size = min_group_size
        curr_max_size = max_group_size
        if current_index >= len(sorted_ranks): return (max_group_size, -1)
        if dp_cache[current_index]:
            return dp_cache[current_index]
        while curr_min_size >= 1:
            min_size = min(curr_min_size, len(sorted_ranks) - current_index)
            best_score = -1
            best_partition = -1
            for group_size in range(min_size, curr_max_size + 1):
                if current_index + group_size > len(sorted_ranks):
                    break
                if max_memory_usage:
                    group_memory_cost = compute_group_memory_cost(sorted_ranks, current_index, current_index + group_size, parameter_bytes, layer_sizes)
                    if group_memory_cost > max_memory_usage:
                        break
                group_max_rank = sorted_ranks[current_index + group_size - 1][1]
                group_min_rank = sorted_ranks[current_index][1]
                if group_max_rank - group_min_rank > max_rank_difference:
                    break
                min_size_found, partition = dp_helper(current_index + group_size)
                score = min(group_size, min_size_found)
                if score >= best_score:
                    best_score = score
                    best_partition = current_index + group_size
            if best_score != -1:
                dp_cache[current_index] = (best_score, best_partition)
                return best_score, best_partition
            curr_max_size = curr_min_size
            curr_min_size //= 2
        return 1, current_index + 1
    dp_helper(0)
    current_index = 0
    while current_index < len(sorted_ranks):
        _, next_index = dp_cache[current_index]
        partition_groups.append([index for index, _ in sorted_ranks[current_index:next_index]])
        current_index = next_index
    return partition_groups
    
