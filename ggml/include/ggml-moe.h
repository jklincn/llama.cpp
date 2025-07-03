#pragma once

// 只包含C兼容的头文件
#include <stdbool.h>  // for bool
#include <stddef.h>   // for size_t

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// 前向声明 - 需要包含相应的GGML/Llama头文件
struct ggml_tensor;

/**
 * @struct MoeTopkCollector
 * MoE TopK 张量数据采集器 (不透明结构体)
 *
 * 这是一个不透明的指针，其实现细节隐藏在C++源文件中。
 * 使用 create_moe_topk_collector() 来创建实例，
 * 使用 destroy_moe_topk_collector() 来释放实例。
 */
struct MoeTopkCollector;
typedef struct MoeTopkCollector MoeTopkCollector;

/**
 * 创建一个新的 MoE TopK 收集器实例
 *
 * @return 指向新创建的收集器实例的指针，如果失败则返回NULL。
 *         使用完毕后，必须调用 destroy_moe_topk_collector() 来释放内存。
 */
MoeTopkCollector * create_moe_topk_collector(void);

/**
 * 销毁 MoE TopK 收集器实例并释放所有相关资源
 *
 * @param collector 指向要销毁的收集器实例的指针。
 */
void destroy_moe_topk_collector(MoeTopkCollector * collector);

/**
 * 初始化MoE TopK数据收集器
 *
 * @param collector 收集器实例指针
 * @param output_dir 输出目录路径 (C 字符串)
 * @return true 如果初始化成功
 */
bool init_moe_topk_collector(MoeTopkCollector * collector, const char * output_dir);

/**
 * MoE TopK张量采集回调函数
 *
 * 这是核心的回调函数，会被GGML调度器调用
 *
 * @param t 当前张量
 * @param ask 询问阶段标志
 * @param user_data 用户数据指针（应指向 MoeTopkCollector 实例）
 * @return true 继续执行，false 停止执行
 */
bool moe_topk_collector_callback(struct ggml_tensor * t, bool ask, void * user_data);

/**
 * 打印收集统计信息
 *
 * @param collector 收集器实例指针
 */
void print_collection_summary(const MoeTopkCollector * collector);

#ifdef __cplusplus
}
#endif
