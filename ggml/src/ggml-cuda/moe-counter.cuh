
#pragma once

#include "common.cuh"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// llama.moe: GGML_OP_MOE_COUNTER
//
// dst->src[0]: selected experts (I32)  [n_expert_used, n_tokens]
// dst->src[1]: weights (F32)          [1, n_expert_used, n_tokens]
//
// op_params layout (int32):
//   [0] layer_idx
//   [1] num_experts
//   [2] d_counts ptr lo (uint64)
//   [3] d_counts ptr hi
//   [4] d_weights ptr lo (uint64)
//   [5] d_weights ptr hi

static inline uint64_t ggml_moe_counter_get_u64_op_param(const ggml_tensor * t, uint32_t i32_index) {
	const uint32_t lo = (uint32_t) ggml_get_op_params_i32(t, i32_index + 0);
	const uint32_t hi = (uint32_t) ggml_get_op_params_i32(t, i32_index + 1);
	return (uint64_t) lo | ((uint64_t) hi << 32);
}

__global__ void ggml_cuda_moe_counter_kernel(
	const char * __restrict__ selected_data,
	int64_t nb_sel0,
	int64_t nb_sel1,
	const char * __restrict__ weights_data,
	int64_t nb_w0,
	int64_t nb_w1,
	int64_t nb_w2,
	int64_t n_expert_used,
	int64_t n_tokens,
	int32_t layer_idx,
	int32_t n_experts,
	uint64_t * __restrict__ d_counts,
	double   * __restrict__ d_weights) {

	const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
	const int64_t n   = n_expert_used * n_tokens;
	if (tid >= n) {
		return;
	}

	const int64_t i0 = tid % n_expert_used; // expert slot
	const int64_t i1 = tid / n_expert_used; // token

	const int32_t expert_idx = *(const int32_t *) (selected_data + i0 * nb_sel0 + i1 * nb_sel1);
	if (expert_idx < 0 || expert_idx >= n_experts) {
		return;
	}

	const float w = *(const float *) (weights_data + 0 * nb_w0 + i0 * nb_w1 + i1 * nb_w2);

	const int64_t off = (int64_t) layer_idx * (int64_t) n_experts + (int64_t) expert_idx;

	atomicAdd((unsigned long long *) &d_counts[off], 1ULL);
	atomicAdd(&d_weights[off], (double) w);
}

static inline void ggml_cuda_op_moe_counter(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
	const ggml_tensor * selected = dst->src[0];
	const ggml_tensor * weights  = dst->src[1];

	static bool dbg_inited = false;
	static bool dbg_enabled = false;
	static int  dbg_left = 0;
	if (!dbg_inited) {
		dbg_inited = true;
		const char * env_dbg = std::getenv("LLAMA_MOE_COUNTER_DEBUG");
		dbg_enabled = env_dbg && std::strcmp(env_dbg, "1") == 0;
		if (dbg_enabled) {
			const char * env_lim = std::getenv("LLAMA_MOE_COUNTER_DEBUG_LIMIT");
			dbg_left = env_lim ? std::atoi(env_lim) : 16;
			if (dbg_left < 0) dbg_left = 0;
		}
	}

	if (!selected || !weights) {
		return;
	}

	if (selected->type != GGML_TYPE_I32 || weights->type != GGML_TYPE_F32) {
		return;
	}

	const int32_t layer_idx = ggml_get_op_params_i32(dst, 0);
	const int32_t n_experts = ggml_get_op_params_i32(dst, 1);

	const uint64_t d_counts_u64  = ggml_moe_counter_get_u64_op_param(dst, 2);
	const uint64_t d_weights_u64 = ggml_moe_counter_get_u64_op_param(dst, 4);

	if (d_counts_u64 == 0 || d_weights_u64 == 0) {
		if (dbg_enabled && dbg_left-- > 0) {
			fprintf(stderr,
				"[moe-counter dbg] cuda op: pointers are null layer=%d n_experts=%d d_counts=0x%llx d_weights=0x%llx\n",
				(int) layer_idx, (int) n_experts,
				(unsigned long long) d_counts_u64,
				(unsigned long long) d_weights_u64);
		}
		return;
	}

	auto * d_counts  = (uint64_t *) (uintptr_t) d_counts_u64;
	auto * d_weights = (double   *) (uintptr_t) d_weights_u64;

	const int64_t n_expert_used = selected->ne[0];
	const int64_t n_tokens      = selected->ne[1];
	if (n_expert_used <= 0 || n_tokens <= 0) {
		return;
	}

	if (dbg_enabled && dbg_left-- > 0) {
		fprintf(stderr,
			"[moe-counter dbg] cuda op: layer=%d n_experts=%d n_expert_used=%lld n_tokens=%lld sel_type=%d w_type=%d d_counts=0x%llx d_weights=0x%llx\n",
			(int) layer_idx, (int) n_experts,
			(long long) n_expert_used, (long long) n_tokens,
			(int) selected->type, (int) weights->type,
			(unsigned long long) d_counts_u64,
			(unsigned long long) d_weights_u64);
	}

	const int64_t n = n_expert_used * n_tokens;

	const int threads = 256;
	const int blocks  = (int) ((n + threads - 1) / threads);

	const cudaStream_t stream = ctx.stream();

	ggml_cuda_moe_counter_kernel<<<blocks, threads, 0, stream>>>(
		(const char *) selected->data,
		(int64_t) selected->nb[0],
		(int64_t) selected->nb[1],
		(const char *) weights->data,
		(int64_t) weights->nb[0],
		(int64_t) weights->nb[1],
		(int64_t) weights->nb[2],
		n_expert_used,
		n_tokens,
		layer_idx,
		n_experts,
		d_counts,
		d_weights);

	CUDA_CHECK(cudaGetLastError());
}

