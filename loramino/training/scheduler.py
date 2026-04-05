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
    def dp_helper(current_index: int) -> tuple[int, list]:
        if current_index >= len(sorted_ranks): return (max_group_size, [])
        min_size = min(min_group_size, len(sorted_ranks) - current_index)
        best_score = -1
        best_partition = []
        for group_size in range(min_size, max_group_size + 1):
            if current_index + group_size > len(sorted_ranks):
                break
            group_max_rank = sorted_ranks[current_index + group_size - 1][1]
            group_min_rank = sorted_ranks[current_index][1]
            if group_max_rank - group_min_rank > max_rank_difference:
                break
            min_size_found, partition = dp_helper(current_index + group_size)
            score = min(group_size, min_size_found)
            if score > best_score:
                best_score = score
                best_partition = partition
                best_partition.append([idx for idx, _ in sorted_ranks[current_index:current_index + group_size]])
        if best_score == -1:
            return 1, [[sorted_ranks[current_index][0]]] + dp_helper(current_index + 1)[1]
        return best_score, best_partition
    _, partition_groups = dp_helper(0)
    return partition_groups[::-1]
    
