#pragma once

#include <stdbool.h> 
#include <stddef.h>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// 前向声明 ggml_tensor 结构体
struct ggml_tensor;

/**
 * @struct MoeActivationCounter
 * MoE 专家激活次数计数器 (不透明结构体)
 *
 * 这是一个不透明的指针，其C++实现细节被隐藏。
 * 使用 create_moe_activation_counter() 来创建实例，
 * 使用 destroy_moe_activation_counter() 来释放实例。
 */
struct MoeActivationCounter;
typedef struct MoeActivationCounter MoeActivationCounter;

/**
 * @brief 创建一个新的 MoE 专家激活计数器实例。
 */
MoeActivationCounter * create_moe_activation_counter();


bool setup_moe_activation_counter(MoeActivationCounter * counter, int layers, int experts);


/**
 * @brief 销毁 MoE 激活计数器实例并释放所有相关资源。
 *
 * @param counter 指向要销毁的计数器实例的指针。
 */
void destroy_moe_activation_counter(MoeActivationCounter * counter);

/**
 * @brief 将收集到的激活次数统计数据保存到一个CSV报告文件中。
 *
 * 这个函数应该在整个推理过程完全结束后调用一次，以生成最终的报告。
 *
 * @param counter 包含累计激活数据的计数器实例指针。
 * @param output_dir 一个C字符串，指定用于存放报告文件的目录路径。
 */
void save_activation_report(MoeActivationCounter * counter);

// Build a side-effect ggml op that accumulates MoE expert activations/weights.
// Intended to be inserted from llama graph build code (e.g. in build_moe_ffn).
//
// selected_experts: I32 [n_expert_used, n_tokens]
// weights:          F32 [1, n_expert_used, n_tokens]
// layer_idx:        block/layer index (il)
struct ggml_tensor * ggml_moe_counter(
	struct ggml_context * ctx,
	struct ggml_tensor  * selected_experts,
	struct ggml_tensor  * weights,
	MoeActivationCounter * counter,
	int32_t layer_idx);


#ifdef __cplusplus
}
#endif
