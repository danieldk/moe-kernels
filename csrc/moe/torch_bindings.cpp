#include "core/registration.h"
#include "core/scalar_type.hpp"

#include "moe_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // Activation ops
  // Activation function used in SwiGLU.
  m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  // Activation function used in GeGLU with `none` approximation.
  m.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  // Activation function used in GeGLU with `tanh` approximation.
  m.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  // FATReLU implementation.
  m.def("fatrelu_and_mul(Tensor! out, Tensor input, float threshold) -> ()");
  m.impl("fatrelu_and_mul", torch::kCUDA, &fatrelu_and_mul);

  // GELU implementation used in GPT-2.
  m.def("gelu_new(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_new", torch::kCUDA, &gelu_new);

  // Approximate GELU implementation.
  m.def("gelu_fast(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_fast", torch::kCUDA, &gelu_fast);

  // Quick GELU implementation.
  m.def("gelu_quick(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_quick", torch::kCUDA, &gelu_quick);

  // Apply topk softmax to the gating outputs.
  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);

  // Calculate the result of moe by summing up the partial results
  // from all selected experts.
  m.def("moe_sum(Tensor! input, Tensor output) -> ()");
  m.impl("moe_sum", torch::kCUDA, &moe_sum);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

#ifndef USE_ROCM
  m.def(
      "marlin_gemm_moe(Tensor! a, Tensor! b_q_weights, Tensor! sorted_ids, "
      "Tensor! topk_weights, Tensor! topk_ids, Tensor! b_scales, Tensor! "
      "b_zeros, Tensor! g_idx, Tensor! perm, Tensor! workspace, "
      "int b_q_type, SymInt size_m, "
      "SymInt size_n, SymInt size_k, bool is_k_full, int num_experts, int "
      "topk, "
      "int moe_block_size, bool replicate_input, bool apply_weights)"
      " -> Tensor");
  // conditionally compiled so impl registration is in source file
#endif

  // Compute FP8 quantized tensor for given scaling factor.
  m.def(
      "static_scaled_fp8_quant(Tensor! out, Tensor input, Tensor scale) -> ()");
  m.impl("static_scaled_fp8_quant", torch::kCUDA, &static_scaled_fp8_quant);

  // Compute dynamic-per-tensor FP8 quantized tensor and scaling factor.
  m.def(
      "dynamic_scaled_fp8_quant(Tensor! out, Tensor input, Tensor! scale) -> "
      "()");
  m.impl("dynamic_scaled_fp8_quant", torch::kCUDA, &dynamic_scaled_fp8_quant);

  // Compute dynamic-per-token FP8 quantized tensor and scaling factor.
  m.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor! out, Tensor input, "
      "Tensor! scale, Tensor? scale_ub) -> "
      "()");
  m.impl("dynamic_per_token_scaled_fp8_quant", torch::kCUDA,
           &dynamic_per_token_scaled_fp8_quant);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
