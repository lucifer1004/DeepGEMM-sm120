// Representative template instantiations for the SM120 compile gate.
//
// One explicit instantiation per vendored kernel template, taken by address
// at namespace scope so the device code is fully codegen'd without a GPU.
// Configs are the minimal shapes actually used by the forks' host glue
// (see host-glue/*/csrc/jit_kernels/impls/sm120_*.hpp) and satisfy every
// DG_STATIC_ASSERT in the kernel headers.

#include <deep_gemm/impls/sm120_bf16_gemm.cuh>
#include <deep_gemm/impls/sm120_bmk_bnk_mn.cuh>
#include <deep_gemm/impls/sm120_fp4_mqa_logits.cuh>
#include <deep_gemm/impls/sm120_fp4_paged_mqa_logits.cuh>
#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>
#include <deep_gemm/impls/sm120_fp8_fp4_sparse_mqa_logits.cuh>
#include <deep_gemm/impls/sm120_fp8_mqa_logits.cuh>
#include <deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh>
#include <deep_gemm/impls/sm120_padding.cuh>
#include <deep_gemm/impls/sm120_split_k_reduce.cuh>
#include <deep_gemm/impls/sm120_tf32_hc_prenorm_gemm.cuh>
#include <deep_gemm/scheduler/sm120_paged_mqa_logits.cuh>

namespace deep_gemm {

// ---------------------------------------------------------------------------
// BF16 GEMM (Normal). BLOCK_M=64/BLOCK_N=128/BLOCK_K=64, 128B swizzles,
// 4 stages, 32 TMA + 128 math threads, 24 SMs.
static auto k_bf16_gemm = &sm120_bf16_gemm_impl<
    0, 0, 0, 1, 64, 128, 64, 128, 128, 128, 4, 32, 128, 24,
    GemmType::Normal, false, cutlass::bfloat16_t>;

// ---------------------------------------------------------------------------
// FP8 block-scaled 1D1D GEMM, gran_k_a = gran_k_b = 128, BLOCK_K = 128
// (4 SF stages per load), same tile/threads as above.
static auto k_fp8_gemm_1d1d = &sm120_fp8_fp4_gemm_1d1d_impl<
    0, 0, 0, 128, 128, 1, 64, 128, 128, 128, 128, 128, 4, 32, 128, 24,
    GemmType::Normal, false, cutlass::bfloat16_t>;

// Symmetric FP4 (kIsFP4 = true): FP4 packs to half bytes, so SMEM rows are
// BLOCK_K / 2 = 64 bytes and the A/B swizzle drops to 64B.
static auto k_fp4_gemm_1d1d = &sm120_fp8_fp4_gemm_1d1d_impl<
    0, 0, 0, 128, 128, 1, 64, 128, 128, 64, 64, 128, 4, 32, 128, 24,
    GemmType::Normal, false, cutlass::bfloat16_t,
    epilogue::transform::EpilogueIdentity, true>;

// ---------------------------------------------------------------------------
// MQA logits (non-paged): 64 heads x head_dim 128, BLOCK_Q = 128 / heads = 2,
// BLOCK_KV = 8 warps * 16 = 128 (required), 2 Q stages, 3 KV stages.
static auto k_fp8_mqa_logits = &sm120_fp8_mqa_logits<
    64, 128, false, 2, 128, 2, 3, 24, 128, 256, float>;

// FP4 MQA logits: head_dim must be 128; host uses 5 KV stages for FP4.
static auto k_fp4_mqa_logits = &sm120_fp4_mqa_logits<
    64, 128, false, 2, 128, 2, 5, 24, 128, 256, cutlass::bfloat16_t>;

// ---------------------------------------------------------------------------
// Paged MQA logits. SPLIT_KV = 128 = BLOCK_KV * groups; BLOCK_KV = 64.
// FP8 supports PAGE_KV in {64, 128, 256}; instantiate 64 and 256.
static auto k_fp8_paged_mqa_logits_p64 = &sm120_fp8_paged_mqa_logits<
    1, 64, 128, 64, false, false, 2, 3, 128, 128, 256, float>;
static auto k_fp8_paged_mqa_logits_p256 = &sm120_fp8_paged_mqa_logits<
    2, 64, 128, 256, false, false, 2, 3, 128, 128, 256, float>;

// FP4 paged adds PAGE_KV = 32; head_dim must be 128.
static auto k_fp4_paged_mqa_logits_p32 = &sm120_fp4_paged_mqa_logits<
    1, 64, 128, 32, false, false, 2, 3, 128, 128, 256, float>;
static auto k_fp4_paged_mqa_logits_p256 = &sm120_fp4_paged_mqa_logits<
    1, 64, 128, 256, false, false, 2, 3, 128, 128, 256, float>;

// ---------------------------------------------------------------------------
// Sparse MQA logits: FP8 contiguous (SPARSE_BLOCK_KV = 8) and FP4 paged
// (PAGE_KV = 128, a multiple of SPARSE_BLOCK_KV as required).
static auto k_sparse_mqa_fp8 = &sm120_fp8_fp4_sparse_mqa_logits_kernel<
    false, false, 8, 0, 24, false>;
static auto k_sparse_mqa_fp4_paged = &sm120_fp8_fp4_sparse_mqa_logits_kernel<
    true, true, 8, 128, 24, false>;

// ---------------------------------------------------------------------------
// bmk_bnk_mn (batched BF16 "einsum" GEMM): BLOCK_M = 128, 128 TMA + 256 math
// threads (required), BLOCK_N = 128 -> kNTiles = 16 (even), BLOCK_K = 64,
// 128B swizzle, split factor 2. NOTE: this kernel indexes with
// ceil_div(SHAPE_*, BLOCK_*) directly (no runtime-shape fallback), so zero
// SHAPE_* makes the mainloop unreachable; use one block per dim.
static auto k_bmk_bnk_mn = &sm120_bmn_bnk_mn_gemm_impl<
    128, 128, 64, 128, 128, 64, 2, 128, 4, 128, 256>;

// ---------------------------------------------------------------------------
// TF32 hyperconnection prenorm GEMM: BLOCK_K = 64 (required),
// one M-tile per warp -> BLOCK_M = 8 warps * 16 = 128, 2 K-splits.
static auto k_tf32_hc_prenorm = &sm120_tf32_hc_prenorm_gemm_impl<
    4096, 1024, 128, 64, 64, 2, 4, 256, 128>;

// ---------------------------------------------------------------------------
// Padding helpers and the split-K reduce epilogue.
static auto k_clear_padding = &sm120_clear_padding<cutlass::bfloat16_t, false, false>;
static auto k_clear_padding_grouped = &sm120_clear_padding<cutlass::bfloat16_t, true, false>;
static auto k_copy_grouped_output = &sm120_copy_grouped_output<false, false>;
static auto k_copy_grouped_output_masked = &sm120_copy_grouped_output<true, false>;
static auto k_split_k_reduce = &sm120_split_k_reduce_impl<cutlass::bfloat16_t, 2>;
static auto k_split_k_reduce_f32 = &sm120_split_k_reduce_impl<float, 4>;

// ---------------------------------------------------------------------------
// Paged-MQA scheduler metadata kernel (lives in scheduler/, not impls/).
static auto k_paged_mqa_metadata = &sched::sm120_paged_mqa_logits_metadata<32, 128, 24, false>;

} // namespace deep_gemm
