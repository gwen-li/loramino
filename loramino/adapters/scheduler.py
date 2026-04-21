from typing import Callable


def build_default_memory_model(layer_sizes: list[int],
                              total_activation_bytes: int,
                              total_model_memory_bytes: int,
                              weight_bytes: int = 4,
                              gradient_bytes: int = 4,
                              optimizer_bytes: int = 4,
                              ) -> Callable:
    """ Builds a default memory model function that can be used to compute the memory cost of a group of LoRA adaptors.
        This function returns a closure that captures the provided layer sizes and memory cost parameters, and can be used as the memory_model argument in the compute_rank_groups function.
        Args:
            layer_sizes (list[int]): A list of the sizes of the layers the adaptors are applied to.
            total_activation_bytes (int): The total memory cost of the activations for the model:
            weight_bytes (int): The number of bytes per parameter for the weights. Default is 4 (float32).
            gradient_bytes (int): The number of bytes per parameter for the gradients. Default is 4 (float32).
            optimizer_bytes (int): The number of bytes per parameter for the optimizer states. Default is 4 (float32).
        Returns:
            Callable: A function that takes a list of ranks and start/end indices, and returns the total memory cost for that group of adaptors.
    """
    total_layer_size = sum(layer_sizes)
    weight_cost_per_parameter = weight_bytes * total_layer_size
    gradient_cost_per_parameter = gradient_bytes * total_layer_size
    optimizer_cost_per_parameter = 2 * optimizer_bytes * total_layer_size
    total_fixed_cost = total_activation_bytes + total_model_memory_bytes
    def memory_model(ranks: list[tuple[int, int]], start_rank: int, end_rank: int) -> int:
        max_rank = ranks[end_rank - 1][1]
        num_adaptors = end_rank - start_rank
        total_adaptor_size = max_rank * num_adaptors
        weight_cost = total_adaptor_size * weight_cost_per_parameter
        gradient_cost = total_adaptor_size * gradient_cost_per_parameter
        optimizer_cost = total_adaptor_size * optimizer_cost_per_parameter
        return weight_cost + gradient_cost + optimizer_cost + total_fixed_cost
    return memory_model
    
                               

def compute_rank_groups(ranks: list[int],
                        min_group_size: int = 1,
                        max_group_size: int = 16,
                        max_rank_difference: int = 8,
                        layer_sizes: list[int] | None = None,
                        total_activation_bytes: int = 0,
                        total_model_memory_bytes: int = 0,
                        weight_bytes: int = 4,
                        gradient_bytes: int = 4,
                        optimizer_bytes: int = 4,
                        memory_model: Callable[[list[tuple[int, int]], int, int], int] | None = None,
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
            layer_sizes (list[int] | None): A list of the sizes of the layers the adaptors are applied to.
              This is required if max_memory_usage is specified and no custom memory_model is provided,
              and is used to build a default memory model.
            total_activation_bytes (int): The total memory cost of the activations for the model.
            This is used in the default memory model to compute the total memory cost of a group of adaptors.
            total_model_memory_bytes (int): The total memory cost of the model parameters and optimizer states.
            This is used in the default memory model to compute the total memory cost of a group of adaptors.
            weight_bytes (int): The number of bytes per parameter for the weights. Default is 4 (float32).
            gradient_bytes (int): The number of bytes per parameter for the gradients. Default is 4 (float32).
            optimizer_bytes (int): The number of bytes per parameter for the optimizer states. Default is 4 (float32).
            memory_model (Callable[[list[tuple[int, int]], int, int], int] | None): A custom memory model function that
              takes a list of ranks with indices, a start index, and an end index, and returns the total memory cost for
              the model during fine tuning with that group of adaptors.
            max_memory_usage (int | None): An optional maximum memory usage constraint for each group. If specified, groups
              that exceed this memory usage will not be considered valid partitions.
            
        Returns:
            list[list[int]]: A list of groups, where each group is a list of indices corresponding to the input adaptors.
    """
    ranks_with_indices = list(enumerate(ranks))
    sorted_ranks = sorted(ranks_with_indices, key=lambda x: x[1])
    partition_groups = []
    dp_cache = [None for _ in range(len(sorted_ranks))]
    # just to remove type issue
    if isinstance(memory_model, Callable):
        mem_model = memory_model
    if not memory_model and max_memory_usage:
        if not layer_sizes:
            raise ValueError("""If max_memory_usage is specified and no custom memory_model is provided,
                              layer_sizes must be provided to build a default memory model.""")
        mem_model = build_default_memory_model(layer_sizes,
                                                total_activation_bytes,
                                                total_model_memory_bytes,
                                                weight_bytes, 
                                                gradient_bytes, 
                                                optimizer_bytes)
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
                    group_memory_cost = mem_model(sorted_ranks, current_index, current_index + group_size)
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
    
