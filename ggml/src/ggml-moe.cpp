#include "ggml-moe.h"

#include "ggml-backend.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <vector>

/**
 * @struct MoeActivationCounter
 * 用于收集和统计MoE模型中专家激活次数的C++实现。
 * 此定义对C代码隐藏。
 */
struct MoeActivationCounter {
    int num_layers  = 0;
    int num_experts = 0;

    // 激活计数器
    std::vector<std::vector<uint64_t>> expert_activation_counts;
    // 激活权重累加器
    std::vector<std::vector<double>> expert_activation_weights;

    // 临时缓冲：用于匹配同一层的 topk 和 weights
    struct LayerData {
        std::vector<int32_t> indices;
        std::vector<float>   weights;
    };
    std::map<int, LayerData> layer_buffers;

    // 用于从GPU复制数据的临时缓冲区
    std::vector<uint8_t> buffer;

    bool initialized = false;
    bool enabled = true;

    MoeActivationCounter()  = default;
    ~MoeActivationCounter() = default;
};

// C-compatible API implementations

MoeActivationCounter * create_moe_activation_counter() {
    auto * counter = new (std::nothrow) MoeActivationCounter();
    if (!counter) {
        GGML_LOG_ERROR("无法分配 MoeActivationCounter 对象。\n");
    }
    return counter;
}

bool setup_moe_activation_counter(MoeActivationCounter * counter, int layers, int experts, int expert_used) {
    if (!counter) {
        return false;
    }
    if (layers <= 0 || experts <= 0) {
        GGML_LOG_ERROR("setup_moe_activation_counter: 层数和专家数必须为正数。\n");
        return false;
    }
    counter->num_layers  = layers;
    counter->num_experts = experts;
    counter->expert_activation_counts.assign(layers, std::vector<uint64_t>(experts, 0));
    counter->expert_activation_weights.assign(layers, std::vector<double>(experts, 0.0));
    counter->initialized = true;

    const char * env_p = std::getenv("LLAMA_MOE_COUNTER");
    if (env_p && strcmp(env_p, "0") == 0) {
        counter->enabled = false;
        GGML_LOG_INFO("MoE激活计数器已禁用 (LLAMA_MOE_COUNTER=0)\n");
    } else {
        counter->enabled = true;
        GGML_LOG_INFO("MoE激活计数器已启用 (模型层数: %d 层, 每层专家数量: %d, 激活专家数: %d)\n", layers, experts,
                  expert_used);
    }
    return true;
}

void destroy_moe_activation_counter(MoeActivationCounter * counter) {
    delete counter;
}

// --- Helper function prototypes (internal to this file) ---

static bool is_target_tensor(const char * tensor_name);
static int  parse_layer_index_from_name(const char * tensor_name);
static void accumulate_weights(MoeActivationCounter * counter, int layer_idx, const std::vector<int32_t> & indices,
                               const std::vector<float> & weights);

// --- Function Implementations ---

static bool is_target_tensor(const char * tensor_name) {
    if (!tensor_name) {
        return false;
    }
    // 匹配 topk
    if (strstr(tensor_name, "ffn_moe_topk") != nullptr) {
        return true;
    }
    // 匹配 weights (只统计被选中的权重)
    // 排除 sum, norm, scaled, softmax 等后缀，尽量只匹配基础的 weights 张量
    // 注意：不同模型架构命名可能不同，这里尽量通用
    if (strstr(tensor_name, "ffn_moe_weights") != nullptr) {
        if (strstr(tensor_name, "sum") != nullptr)
            return false;
        if (strstr(tensor_name, "norm") != nullptr)
            return false;
        if (strstr(tensor_name, "scaled") != nullptr)
            return false;
        if (strstr(tensor_name, "softmax") != nullptr)
            return false;
        return true;
    }
    return false;
}

static void accumulate_weights(MoeActivationCounter * counter, int layer_idx, const std::vector<int32_t> & indices,
                               const std::vector<float> & weights) {
    if (indices.size() != weights.size()) {
        GGML_LOG_WARN("Layer %d: indices size %zu != weights size %zu\n", layer_idx, indices.size(), weights.size());
        return;
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        int expert_idx = indices[i];
        if (expert_idx >= 0 && expert_idx < counter->num_experts) {
            counter->expert_activation_weights[layer_idx][expert_idx] += (double) weights[i];
        }
    }
}

/**
 * 从张量名称中解析出层索引。
 * 假设张量名称格式为 "blk.XX.*" 或 "layers.XX.*"，其中 XX 是数字。
 */
static int parse_layer_index_from_name(const char * tensor_name) {
    try {
        // 使用正则表达式查找第一个出现的数字序列
        static const std::regex re("\\d+");
        std::smatch match;
        std::string s(tensor_name);
        if (std::regex_search(s, match, re)) {
            return std::stoi(match.str(0));
        }
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: 解析层索引失败: %s\n", __func__, e.what());
    }
    GGML_LOG_WARN("%s: 无法从 '%s' 中解析层索引。\n", __func__, tensor_name);
    return -1;
}

/**
 * MoE 专家激活计数回调函数
 */
bool moe_activation_counter_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * counter = (MoeActivationCounter *) user_data;

    if (!counter || !counter->initialized) {
        return false;
    }

    if (!counter->enabled) {
        return false;
    }

    if (ask) {
        // 第一阶段：询问是否对该张量感兴趣
        return is_target_tensor(t->name);
    }

    // 第二阶段：处理感兴趣的张量数据
    // GGML_LOG_INFO("[MoE Counter] 捕获到目标张量: %s\n", t->name);

    // 1. 解析层索引
    int layer_idx = parse_layer_index_from_name(t->name);
    if (layer_idx < 0 || layer_idx >= counter->num_layers) {
        GGML_LOG_ERROR("从 '%s' 解析到无效的层索引 %d。\n", t->name, layer_idx);
        return true;  // 继续执行
    }

    // 2. 处理不同类型的张量
    if (strstr(t->name, "ffn_moe_topk") != nullptr) {
        if (t->type != GGML_TYPE_I32) {
            GGML_LOG_WARN("跳过张量 '%s'，因为其类型不是 I32 (而是 %s)，无法解析为专家索引。\n", t->name,
                          ggml_type_name(t->type));
            return true;
        }

        // 获取数据
        uint8_t *    data_ptr = nullptr;
        const size_t n_bytes  = ggml_nbytes(t);

        if (!ggml_backend_buffer_is_host(t->buffer)) {
            counter->buffer.resize(n_bytes);
            ggml_backend_tensor_get(t, counter->buffer.data(), 0, n_bytes);
            data_ptr = counter->buffer.data();
        } else {
            data_ptr = (uint8_t *) t->data;
        }

        const int32_t * expert_indices = (const int32_t *) data_ptr;
        const size_t    num_indices    = ggml_nelements(t);

        std::vector<int32_t> current_indices(expert_indices, expert_indices + num_indices);

        // 1. 统计 counts (保留原有逻辑)
        for (int32_t idx : current_indices) {
            if (idx >= 0 && idx < counter->num_experts) {
                counter->expert_activation_counts[layer_idx][idx]++;
            } else {
                // GGML_LOG_ERROR("在张量 '%s' 中发现无效的专家索引 %d。\n", t->name, idx);
            }
        }

        // 2. 尝试与 weights 匹配进行累加
        if (!counter->layer_buffers[layer_idx].weights.empty()) {
            accumulate_weights(counter, layer_idx, current_indices, counter->layer_buffers[layer_idx].weights);
            counter->layer_buffers[layer_idx].weights.clear();
        } else {
            counter->layer_buffers[layer_idx].indices = std::move(current_indices);
        }

    } else if (strstr(t->name, "ffn_moe_weights") != nullptr) {
        // 处理被选中的权重 (Effective Weights)

        if (t->type != GGML_TYPE_F32) {
            GGML_LOG_WARN("跳过张量 '%s'，因为其类型不是 F32 (而是 %s)。\n", t->name, ggml_type_name(t->type));
            return true;
        }

        // 获取数据
        uint8_t *    data_ptr = nullptr;
        const size_t n_bytes  = ggml_nbytes(t);

        if (!ggml_backend_buffer_is_host(t->buffer)) {
            counter->buffer.resize(n_bytes);
            ggml_backend_tensor_get(t, counter->buffer.data(), 0, n_bytes);
            data_ptr = counter->buffer.data();
        } else {
            data_ptr = (uint8_t *) t->data;
        }

        const float * weights_ptr  = (const float *) data_ptr;
        const size_t  num_elements = ggml_nelements(t);

        std::vector<float> current_weights(weights_ptr, weights_ptr + num_elements);

        // 尝试与 indices 匹配进行累加
        if (!counter->layer_buffers[layer_idx].indices.empty()) {
            accumulate_weights(counter, layer_idx, counter->layer_buffers[layer_idx].indices, current_weights);
            counter->layer_buffers[layer_idx].indices.clear();
        } else {
            counter->layer_buffers[layer_idx].weights = std::move(current_weights);
        }
    }

    return true;
}

/**
 * 将收集到的激活次数统计数据保存到CSV文件中。
 */
void save_activation_report(MoeActivationCounter * counter) {
    if (!counter || !counter->initialized) {
    if (!counter->enabled) {
        return;
    }

        GGML_LOG_ERROR("%s: 计数器未初始化。\n", __func__);
        return;
    }

    counter->initialized = false;

    // 从环境变量读取 WORK_DIR
    const char * work_dir_env = std::getenv("WORK_DIR");
    std::string  work_dir     = work_dir_env ? work_dir_env : ".";

    // 确保目录末尾有 '/'
    if (!work_dir.empty() && work_dir.back() != '/') {
        work_dir += "/";
    }

    const std::string filepath = work_dir + "expert_activations.csv";

    std::ofstream file(filepath);
    if (!file.is_open()) {
        GGML_LOG_ERROR("%s: 无法创建报告文件: %s\n", __func__, filepath.c_str());
        return;
    }

    GGML_LOG_INFO("\n=== MoE 激活次数统计报告 ===\n");
    GGML_LOG_INFO("正在保存报告到: %s\n", filepath.c_str());

    // 写入CSV表头
    file << "layer_index";
    for (int i = 0; i < counter->num_experts; ++i) {
        file << ",expert_" << i;
    }
    file << "\n";

    // 写入数据
    unsigned long long total_activations = 0;
    for (int layer = 0; layer < counter->num_layers; ++layer) {
        file << layer;
        for (int expert = 0; expert < counter->num_experts; ++expert) {
            uint64_t cnt = counter->expert_activation_counts[layer][expert];
            file << "," << cnt;
            total_activations += cnt;
        }
        file << "\n";
    }

    file.close();

    // 保存权重报告
    const std::string weights_filepath = work_dir + "expert_weights.csv";
    std::ofstream wfile(weights_filepath);
    if (wfile.is_open()) {
        GGML_LOG_INFO("正在保存权重报告到: %s\n", weights_filepath.c_str());
        wfile << "layer_index";
        for (int i = 0; i < counter->num_experts; ++i) {
            wfile << ",expert_" << i;
        }
        wfile << "\n";

        for (int layer = 0; layer < counter->num_layers; ++layer) {
            wfile << layer;
            for (int expert = 0; expert < counter->num_experts; ++expert) {
                wfile << "," << counter->expert_activation_weights[layer][expert];
            }
            wfile << "\n";
        }
        wfile.close();
        GGML_LOG_INFO("权重报告保存成功。\n");
    } else {
        GGML_LOG_ERROR("%s: 无法创建权重报告文件: %s\n", __func__, weights_filepath.c_str());
    }

    GGML_LOG_INFO("报告保存成功。\n");
    GGML_LOG_INFO("总计 %d 层, %d 个专家/层。\n", counter->num_layers, counter->num_experts);
    GGML_LOG_INFO("在本次运行中，总共记录到 %llu 次专家激活。\n", total_activations);
    GGML_LOG_INFO("执行 python scripts/expert_activation_analysis.py 进行数据分析。\n");
    GGML_LOG_INFO("==============================\n");
}
