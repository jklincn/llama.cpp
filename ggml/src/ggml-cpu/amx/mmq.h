#pragma once
#include "common.h"

// 预估 workspace 大小（以字节计）。在推理时 GGML 给线程分配临时缓冲，需要先知道多大。
size_t ggml_backend_amx_desired_wsize(const struct ggml_tensor * dst);

// 量化权重在 AMX‑VNNI 布局 下真实占用多少内存（含 padding / 重排）。加载权重文件时用来 malloc。
size_t ggml_backend_amx_get_alloc_size(const struct ggml_tensor * tensor);

// 把原始 GGUF/GGML 权重（列主序）重新打包成 VNNI + 32B/64B 对齐格式，并写进 AMX 专用 buffer。
void ggml_backend_amx_convert_weight(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);

// 核心 GEMM 调度函数：把 dst = src1 @ src0ᵀ 分块丢给 AVX512/AMX tiny‑kernel；支持 F16×F32 和多种整数量化。
void ggml_backend_amx_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
