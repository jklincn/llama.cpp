#pragma once

// 只包含C兼容的头文件
#include <stdbool.h> // for bool
#include <stddef.h>  // for size_t

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


bool setup_moe_activation_counter(MoeActivationCounter * counter, int layers, int experts, int expert_used);


/**
 * @brief 销毁 MoE 激活计数器实例并释放所有相关资源。
 *
 * @param counter 指向要销毁的计数器实例的指针。
 */
void destroy_moe_activation_counter(MoeActivationCounter * counter);

/**
 * @brief MoE 专家激活计数回调函数。
 *
 * 这是核心的回调函数，应通过 ggml_set_debug_callback() 设置，
 * 它会在计算图执行期间被GGML调度器调用。
 *
 * @param t 当前正在处理的ggml_tensor。
 * @param ask 如果为 true，表示GGML在询问是否对该张量感兴趣。
 * 如果为 false，表示正在处理一个先前已标记为感兴趣的张量。
 * @param user_data 用户数据指针（应指向 MoeActivationCounter 实例）。
 * @return 应该始终返回 true 以继续图的计算。
 */
bool moe_activation_counter_callback(struct ggml_tensor * t, bool ask, void * user_data);

/**
 * @brief 将收集到的激活次数统计数据保存到一个CSV报告文件中。
 *
 * 这个函数应该在整个推理过程完全结束后调用一次，以生成最终的报告。
 *
 * @param counter 包含累计激活数据的计数器实例指针。
 * @param output_dir 一个C字符串，指定用于存放报告文件的目录路径。
 */
void save_activation_report(const MoeActivationCounter * counter);


#ifdef __cplusplus
}
#endif
