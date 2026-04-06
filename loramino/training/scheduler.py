from copy import deepcopy

def compute_rank_groups(ranks: list[int],
                        min_group_size: int = 1,
                        max_group_size: int = 16,
                        max_rank_difference: int = 8) -> list[list[int]]:
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
    def dp_helper(current_index: int) -> tuple[int, int]:
        curr_min_size = min_group_size
        curr_max_size = max_group_size
        if current_index >= len(sorted_ranks): return (max_group_size, [])
        if dp_cache[current_index]:
            return dp_cache[current_index]
        while curr_min_size > 1:
            min_size = min(curr_min_size, len(sorted_ranks) - current_index)
            best_score = -1
            best_partition = -1
            for group_size in range(min_size, curr_max_size + 1):
                if current_index + group_size > len(sorted_ranks):
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
    
