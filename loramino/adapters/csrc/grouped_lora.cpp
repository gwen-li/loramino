#include <torch/extension.h>

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

namespace {

using AdapterGroups = std::map<int64_t, std::vector<int64_t>>;

struct GroupingState {
  AdapterGroups adapters_by_rank;
  std::vector<torch::Tensor> token_indices_by_adapter;
  std::vector<int64_t> token_counts_by_adapter;
};

// Validate the public extension contract before we touch any grouped logic.
//
// This extension assumes:
// - every tensor already lives on CUDA
// - token routing ids and per-adapter ranks use int64
// - `flat_x` is flattened to [tokens, hidden]
// - `A` and `B` store the LoRA bank in padded form
//
// Keeping these checks in one place makes the rest of the file easier to read:
// every later helper can focus on grouping/padding/math rather than re-checking
// shapes and devices.
void validate_inputs(
    const torch::Tensor& flat_x,
    const torch::Tensor& token_adapter_ids,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& scales,
    const torch::Tensor& ranks) {
  TORCH_CHECK(flat_x.is_cuda(), "flat_x must be a CUDA tensor.");
  TORCH_CHECK(token_adapter_ids.is_cuda(), "token_adapter_ids must be a CUDA tensor.");
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors.");
  TORCH_CHECK(scales.is_cuda() && ranks.is_cuda(), "scales and ranks must be CUDA tensors.");
  TORCH_CHECK(token_adapter_ids.scalar_type() == torch::kLong, "token_adapter_ids must use torch.long.");
  TORCH_CHECK(ranks.scalar_type() == torch::kLong, "ranks must use torch.long.");
  TORCH_CHECK(flat_x.dim() == 2, "flat_x must have shape [tokens, hidden].");
  TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "A and B must be rank-3 tensors.");
  TORCH_CHECK(A.size(0) == B.size(0), "A and B must agree on the number of adapters.");
  TORCH_CHECK(A.size(1) == B.size(2), "A and B must agree on max rank.");
  TORCH_CHECK(flat_x.size(1) == A.size(2), "flat_x hidden size must match A in_features.");
  TORCH_CHECK(scales.dim() == 1 && scales.size(0) == A.size(0), "scales must be shape [num_adapters].");
  TORCH_CHECK(ranks.dim() == 1 && ranks.size(0) == A.size(0), "ranks must be shape [num_adapters].");
}

// Build the execution plan for one forward/backward call.
//
// The Python runtime has already decided which token belongs to which adapter.
// Here we convert that routing vector into two pieces of reusable metadata:
// 1. `token_indices_by_adapter`: where each adapter's tokens live in the flat batch
// 2. `adapters_by_rank`: adapters grouped by their active rank
//
// Grouping by exact rank lets us run one batched GEMM per rank bucket instead of
// one tiny GEMM per adapter. That is the main performance idea in this file.
GroupingState build_grouping(
    const torch::Tensor& token_adapter_ids,
    const torch::Tensor& ranks,
    int64_t num_adapters) {
  GroupingState state;
  state.token_indices_by_adapter.resize(num_adapters);
  state.token_counts_by_adapter.resize(num_adapters, 0);

  for (int64_t adapter_id = 0; adapter_id < num_adapters; ++adapter_id) {
    auto token_indices = token_adapter_ids.eq(adapter_id).nonzero().reshape({-1});
    const int64_t token_count = token_indices.numel();
    state.token_indices_by_adapter[adapter_id] = token_indices;
    state.token_counts_by_adapter[adapter_id] = token_count;
    if (token_count == 0) {
      continue;
    }

    const int64_t rank = ranks[adapter_id].item<int64_t>();
    if (rank > 0) {
      state.adapters_by_rank[rank].push_back(adapter_id);
    }
  }

  return state;
}

// Small helper to turn a C++ adapter id vector into a torch.long tensor on the
// same device as the current workload. We use this before index_select calls.
torch::Tensor adapter_ids_tensor(
    const std::vector<int64_t>& adapter_ids,
    const torch::TensorOptions& options) {
  return torch::tensor(adapter_ids, options.dtype(torch::kLong));
}

// Pack one rank group into a dense [group, max_tokens, feature] tensor.
//
// Adapters inside the same rank bucket still have different token counts. GEMM
// kernels want dense rectangular tensors, so we pad each adapter's token slice
// up to the maximum token count in the group.
//
// Returned values:
// - padded tensor ready for batched GEMM
// - original per-adapter token counts, used later to ignore padded rows
std::tuple<torch::Tensor, std::vector<int64_t>> build_padded_token_batch(
    const torch::Tensor& source,
    const std::vector<int64_t>& adapter_ids,
    const GroupingState& grouping) {
  const int64_t group_size = static_cast<int64_t>(adapter_ids.size());
  const int64_t feature_dim = source.size(1);

  int64_t max_tokens = 0;
  std::vector<int64_t> lengths;
  lengths.reserve(group_size);
  for (int64_t adapter_id : adapter_ids) {
    const int64_t token_count = grouping.token_counts_by_adapter[adapter_id];
    max_tokens = std::max(max_tokens, token_count);
    lengths.push_back(token_count);
  }

  auto padded = torch::zeros({group_size, max_tokens, feature_dim}, source.options());
  for (int64_t group_index = 0; group_index < group_size; ++group_index) {
    const int64_t adapter_id = adapter_ids[group_index];
    const int64_t token_count = lengths[group_index];
    if (token_count == 0) {
      continue;
    }

    auto token_values = source.index_select(0, grouping.token_indices_by_adapter[adapter_id]);
    padded.select(0, group_index).narrow(0, 0, token_count).copy_(token_values);
  }

  return {padded, lengths};
}

// Scatter a dense grouped result back to the original token order.
//
// After a batched GEMM, results are still organized as
//   [adapter_in_group, padded_token_index, feature]
// but the caller needs
//   [original_token_index, feature].
//
// `lengths` tells us how many rows in each padded slice are real. We copy only
// those real rows back to the output tensor and drop the padding.
void scatter_group_output(
    torch::Tensor output,
    const torch::Tensor& grouped_values,
    const std::vector<int64_t>& adapter_ids,
    const std::vector<int64_t>& lengths,
    const GroupingState& grouping) {
  for (int64_t group_index = 0; group_index < static_cast<int64_t>(adapter_ids.size()); ++group_index) {
    const int64_t adapter_id = adapter_ids[group_index];
    const int64_t token_count = lengths[group_index];
    if (token_count == 0) {
      continue;
    }

    output.index_copy_(
        0,
        grouping.token_indices_by_adapter[adapter_id],
        grouped_values.select(0, group_index).narrow(0, 0, token_count));
  }
}

// Forward kernel for grouped LoRA on CUDA tensors.
//
// High-level flow:
// 1. validate inputs
// 2. group active adapters by rank
// 3. pack each rank group into dense padded batches
// 4. run two batched GEMMs with ATen/cuBLAS:
//      x @ A^T   then   down @ B^T
// 5. apply LoRA scaling
// 6. scatter each group's result back to original token order
//
// This is not a handwritten CUDA kernel. Instead, it is a thin orchestration
// layer around highly optimized GEMM kernels that already exist in PyTorch/ATen.
torch::Tensor grouped_lora_forward_cuda(
    const torch::Tensor& flat_x,
    const torch::Tensor& token_adapter_ids,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& scales,
    const torch::Tensor& ranks) {
  validate_inputs(flat_x, token_adapter_ids, A, B, scales, ranks);

  auto output = torch::zeros({flat_x.size(0), B.size(1)}, flat_x.options());
  if (flat_x.size(0) == 0 || B.size(1) == 0 || A.size(1) == 0) {
    return output;
  }

  const auto grouping = build_grouping(token_adapter_ids, ranks, A.size(0));
  for (const auto& [rank, adapter_ids] : grouping.adapters_by_rank) {
    auto adapters = adapter_ids_tensor(adapter_ids, flat_x.options());
    auto A_group = A.index_select(0, adapters).narrow(1, 0, rank);
    auto B_group = B.index_select(0, adapters).narrow(2, 0, rank);
    auto scales_group = scales.index_select(0, adapters).view({static_cast<int64_t>(adapter_ids.size()), 1, 1});

    auto [x_group, lengths] = build_padded_token_batch(flat_x, adapter_ids, grouping);
    auto down_projection = torch::bmm(x_group, A_group.transpose(1, 2));
    auto delta = torch::bmm(down_projection, B_group.transpose(1, 2)) * scales_group;
    scatter_group_output(output, delta, adapter_ids, lengths, grouping);
  }

  return output;
}

// Backward kernel for grouped LoRA on CUDA tensors.
//
// We reuse the same grouping/packing strategy as the forward pass so backward
// stays aligned with forward execution:
// 1. rebuild the rank groups
// 2. pack `flat_x` and `grad_output` for each group
// 3. compute the standard LoRA gradients with batched GEMMs
//      grad_B = grad^T @ down
//      grad_down = grad @ B
//      grad_A = grad_down^T @ x
//      grad_x = grad_down @ A
// 4. scatter `grad_x` back to token order
// 5. copy per-adapter `grad_A` / `grad_B` slices back into the padded bank
//
// Padding is only a temporary execution detail. The returned gradients match the
// original bank layout expected by autograd: full-size `grad_A` and `grad_B`
// tensors with active-rank slices filled in for each adapter.
std::vector<torch::Tensor> grouped_lora_backward_cuda(
    const torch::Tensor& flat_x,
    const torch::Tensor& token_adapter_ids,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& scales,
    const torch::Tensor& ranks,
    const torch::Tensor& grad_output) {
  validate_inputs(flat_x, token_adapter_ids, A, B, scales, ranks);
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor.");
  TORCH_CHECK(grad_output.dim() == 2 && grad_output.size(0) == flat_x.size(0) && grad_output.size(1) == B.size(1),
              "grad_output must have shape [tokens, out_features].");

  auto grad_x = torch::zeros_like(flat_x);
  auto grad_A = torch::zeros_like(A);
  auto grad_B = torch::zeros_like(B);
  if (flat_x.size(0) == 0 || B.size(1) == 0 || A.size(1) == 0) {
    return {grad_x, grad_A, grad_B};
  }

  const auto grouping = build_grouping(token_adapter_ids, ranks, A.size(0));
  for (const auto& [rank, adapter_ids] : grouping.adapters_by_rank) {
    auto adapters = adapter_ids_tensor(adapter_ids, flat_x.options());
    auto A_group = A.index_select(0, adapters).narrow(1, 0, rank);
    auto B_group = B.index_select(0, adapters).narrow(2, 0, rank);
    auto scales_group = scales.index_select(0, adapters).view({static_cast<int64_t>(adapter_ids.size()), 1, 1});

    auto [x_group, lengths] = build_padded_token_batch(flat_x, adapter_ids, grouping);
    auto [grad_group, _] = build_padded_token_batch(grad_output, adapter_ids, grouping);

    auto down_projection = torch::bmm(x_group, A_group.transpose(1, 2));
    auto scaled_grad = grad_group * scales_group;
    auto grad_B_group = torch::bmm(scaled_grad.transpose(1, 2), down_projection);
    auto grad_down_projection = torch::bmm(scaled_grad, B_group);
    auto grad_A_group = torch::bmm(grad_down_projection.transpose(1, 2), x_group);
    auto grad_x_group = torch::bmm(grad_down_projection, A_group);

    scatter_group_output(grad_x, grad_x_group, adapter_ids, lengths, grouping);
    for (int64_t group_index = 0; group_index < static_cast<int64_t>(adapter_ids.size()); ++group_index) {
      const int64_t adapter_id = adapter_ids[group_index];
      grad_A[adapter_id].narrow(0, 0, rank).copy_(grad_A_group.select(0, group_index));
      grad_B[adapter_id].narrow(1, 0, rank).copy_(grad_B_group.select(0, group_index));
    }
  }

  return {grad_x, grad_A, grad_B};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &grouped_lora_forward_cuda, "Grouped LoRA forward (CUDA via ATen GEMM)");
  m.def("backward", &grouped_lora_backward_cuda, "Grouped LoRA backward (CUDA via ATen GEMM)");
}
