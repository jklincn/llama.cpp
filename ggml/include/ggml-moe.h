#pragma once

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "ggml.h"


// 前向声明 - 需要包含相应的GGML/Llama头文件
struct ggml_tensor;

/**
 * MoE TopK 张量数据采集器
 * 
 * 专门用于采集和导出以 "ffn_moe_topk" 开头的张量数据
 * 支持NPY格式导出和CSV元数据记录
 */
struct MoeTopkCollector {
    std::vector<uint8_t> buffer;  // 临时数据缓冲区
    std::string          output_dir = "moe_topk_data";
    std::ofstream        metadata_file;
    int                  tensor_counter = 0;
    bool                 initialized    = false;

    // 统计信息
    int    total_collected   = 0;
    size_t total_bytes_saved = 0;

    /**
     * 构造函数
     */
    MoeTopkCollector() = default;

    /**
     * 析构函数 - 自动清理资源
     */
    ~MoeTopkCollector() {
        if (metadata_file.is_open()) {
            metadata_file.close();
        }
    }

    /**
     * 禁用拷贝构造和赋值（因为包含文件流）
     */
    MoeTopkCollector(const MoeTopkCollector &)             = delete;
    MoeTopkCollector & operator=(const MoeTopkCollector &) = delete;

    /**
     * 支持移动构造
     */
    MoeTopkCollector(MoeTopkCollector && other) noexcept :
        buffer(std::move(other.buffer)),
        output_dir(std::move(other.output_dir)),
        metadata_file(std::move(other.metadata_file)),
        tensor_counter(other.tensor_counter),
        initialized(other.initialized),
        total_collected(other.total_collected),
        total_bytes_saved(other.total_bytes_saved) {
        other.tensor_counter    = 0;
        other.initialized       = false;
        other.total_collected   = 0;
        other.total_bytes_saved = 0;
    }
};

/**
 * 检查张量名称是否匹配目标前缀
 * 
 * @param tensor_name 张量名称
 * @return true 如果名称以"ffn_moe_topk"开头
 */
bool is_target_tensor(const char * tensor_name);

/**
 * 保存张量为NPY格式文件
 * 
 * @param filepath 输出文件路径
 * @param t 张量指针
 * @param data 张量数据指针
 * @return true 如果保存成功
 */
bool save_tensor_npy(const std::string & filepath, ggml_tensor * t, uint8_t * data);

/**
 * 保存张量元数据到CSV文件
 * 
 * @param collector 收集器实例
 * @param t 张量指针
 * @param filename 对应的NPY文件名
 */
void save_metadata(MoeTopkCollector * collector, ggml_tensor * t, const std::string & filename);

/**
 * MoE TopK张量采集回调函数
 * 
 * 这是核心的回调函数，会被GGML调度器调用
 * 
 * @param t 当前张量
 * @param ask 询问阶段标志
 * @param user_data 用户数据指针（指向MoeTopkCollector）
 * @return true 继续执行，false 停止执行
 */
bool moe_topk_collector_callback(struct ggml_tensor * t, bool ask, void * user_data);

/**
 * 初始化MoE TopK数据收集器
 * 
 * @param collector 收集器实例指针
 * @param output_dir 输出目录路径
 * @return true 如果初始化成功
 */
bool init_moe_topk_collector(MoeTopkCollector * collector, const std::string & output_dir = "moe_topk_data");

/**
 * 打印收集统计信息
 * 
 * @param collector 收集器实例指针
 */
void print_collection_summary(const MoeTopkCollector * collector);

/**
 * 清理收集器资源并打印统计报告
 * 
 * @param collector 收集器实例指针
 */
void cleanup_moe_topk_collector(MoeTopkCollector * collector);

// 内联工具函数
namespace moe_topk_utils {

/**
     * 获取张量的维度字符串表示
     * 注意：这个函数假设存在 ggml_ne_string 函数
     * 如果不存在，需要自行实现
     */
inline std::string get_tensor_shape_string(ggml_tensor * t) {
    // 这里需要根据实际的GGML API来实现
    // 临时实现，可能需要根据实际情况调整
    std::string str;
    for (int i = 0; i < 4; ++i) {  // 假设GGML_MAX_DIMS是4
        if (i > 0) {
            str += ", ";
        }
        str += std::to_string(t->ne[i]);
    }
    return str;
}

/**
     * 计算张量的字节大小
     * 注意：这个函数假设存在 ggml_nbytes 函数
     */
inline size_t get_tensor_bytes(ggml_tensor * t) {
    // 这里需要根据实际的GGML API来实现
    // 临时实现，可能需要根据实际情况调整
    size_t element_size   = 4;     // 假设默认是float32
    size_t total_elements = 1;
    for (int i = 0; i < 4; ++i) {  // 假设GGML_MAX_DIMS是4
        if (t->ne[i] > 1 || i == 0) {
            total_elements *= t->ne[i];
        }
    }
    return total_elements * element_size;
}

/**
     * 创建带时间戳的备份目录名
     */
inline std::string create_timestamped_dir(const std::string & base_dir) {
    auto now    = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::ostringstream oss;
    oss << base_dir << "_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    return oss.str();
}
}  // namespace moe_topk_utils


// 使用示例的内联注释
/*
使用示例:

#include "moe_topk_collector.hpp"

int main() {
    // 1. 创建收集器
    MoeTopkCollector collector;
    
    // 2. 初始化
    if (!init_moe_topk_collector(&collector, "my_moe_data")) {
        return -1;
    }
    
    // 3. 设置回调（在llama参数中）
    params.cb_eval = moe_topk_collector_callback;
    params.cb_eval_user_data = &collector;
    
    // 4. 运行推理...
    
    // 5. 清理和统计
    cleanup_moe_topk_collector(&collector);
    
    return 0;
}

Python分析代码:
    import numpy as np
    import pandas as pd
    
    # 加载元数据
    df = pd.read_csv('my_moe_data/metadata.csv')
    
    # 加载第一个张量
    first_tensor = np.load(f'my_moe_data/{df.iloc[0]["文件名"]}')
    print(f'Shape: {first_tensor.shape}')
    print(f'Data preview: {first_tensor.flatten()[:10]}')
*/
