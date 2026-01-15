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
#include <algorithm>
#include <map>
#include <regex>
#include <string>
#include <vector>

static inline int moe_counter_debug_get_limit() {
    static int limit = -1;
    static bool inited = false;
    if (!inited) {
        inited = true;
        const char * env_dbg = std::getenv("LLAMA_MOE_COUNTER_DEBUG");
        if (env_dbg && std::strcmp(env_dbg, "1") == 0) {
            const char * env_lim = std::getenv("LLAMA_MOE_COUNTER_DEBUG_LIMIT");
            limit = env_lim ? std::atoi(env_lim) : 64;
            limit = std::max(0, limit);
        } else {
            limit = 0;
        }
    }
    return limit;
}

static inline bool moe_counter_debug_take() {
    static int remaining = -1;
    static bool inited = false;
    if (!inited) {
        inited = true;
        remaining = moe_counter_debug_get_limit();
    }
    if (remaining <= 0) {
        return false;
    }
    --remaining;
    return true;
}

static inline bool moe_counter_debug_enabled() {
    return moe_counter_debug_get_limit() > 0;
}

// NOTE: in this build setup, GGML_USE_CUDA may only be defined for CUDA compilation units.
// We still want to use the CUDA runtime from this .cpp for GPU-side accumulation.
// Some toolchains don't add the CUDA include path for C++ targets, so we provide a minimal
// fallback declaration set if <cuda_runtime.h> is not available.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#define LLAMA_MOE_HAVE_CUDA_RUNTIME 1
#include <cuda_runtime.h>
#endif
#endif

#if !defined(LLAMA_MOE_HAVE_CUDA_RUNTIME)
#define LLAMA_MOE_HAVE_CUDA_RUNTIME 1
extern "C" {
    typedef int cudaError_t;

    cudaError_t cudaGetDevice(int * device);
    cudaError_t cudaSetDevice(int device);
    cudaError_t cudaDeviceSynchronize(void);
    cudaError_t cudaMalloc(void ** devPtr, size_t size);
    cudaError_t cudaFree(void * devPtr);
    cudaError_t cudaMemset(void * devPtr, int value, size_t count);
    cudaError_t cudaMemcpy(void * dst, const void * src, size_t count, int kind);
    const char * cudaGetErrorString(cudaError_t error);
}

// cudaMemcpyKind values from the CUDA Runtime API
enum {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
};

static constexpr cudaError_t cudaSuccess = 0;
#endif
#endif

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
    std::vector<std::vector<double>>   expert_activation_weights;

    // 临时缓冲：用于匹配同一层的 topk 和 weights
    struct LayerData {
        std::vector<int32_t> indices;
        std::vector<float>   weights;
    };

    std::map<int, LayerData> layer_buffers;

    // 用于从GPU复制数据的临时缓冲区
    std::vector<uint8_t> buffer;

    // 默认为禁用，直到 setup 被调用。防止在 setup 之前（如 warmup）触发回调报错
    bool enabled = false;

    // 是否启用 GGML_OP_MOE_COUNTER 路径（GPU侧累加，save时一次性回传）
    bool use_gpu_op = false;

    // 构图阶段是否实际插入过 GGML_OP_MOE_COUNTER（不同模型/不同 FFN 路径可能不走 build_moe_ffn）
    // 若 GPU-op 被启用但该值为 0，则需要回退到回调统计，避免最终 0 记录。
    int gpu_op_nodes_built = 0;

#if defined(LLAMA_MOE_HAVE_CUDA_RUNTIME)
    int cuda_device = 0;
    uint64_t * d_counts  = nullptr; // [num_layers * num_experts]
    double   * d_weights = nullptr; // [num_layers * num_experts]
#endif

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
    (void) expert_used;

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
    counter->gpu_op_nodes_built = 0;

    const char * env_p = std::getenv("LLAMA_MOE_COUNTER");
    if (env_p && strcmp(env_p, "1") == 0) {
        counter->enabled = true;
        GGML_LOG_INFO("MoE激活计数器已启用 (LLAMA_MOE_COUNTER=1)\n");
    } else {
        counter->enabled = false;
        GGML_LOG_INFO("MoE激活计数器已禁用 (LLAMA_MOE_COUNTER=0)\n");
        return true;
    }

    // GPU op 开关：默认在 CUDA 构建下启用（可用 LLAMA_MOE_COUNTER_GPU=0 禁用）
    bool want_gpu = true;
    const char * env_gpu = std::getenv("LLAMA_MOE_COUNTER_GPU");
    if (env_gpu && strcmp(env_gpu, "0") == 0) {
        want_gpu = false;
    }

    if (moe_counter_debug_take()) {
        GGML_LOG_INFO("[moe-counter dbg] setup: layers=%d experts=%d enabled=%d want_gpu=%d\n",
                      layers, experts, (int) counter->enabled, (int) want_gpu);
    }

#if defined(LLAMA_MOE_HAVE_CUDA_RUNTIME)
    if (want_gpu) {
        cudaError_t cerr = cudaGetDevice(&counter->cuda_device);
        if (cerr != cudaSuccess) {
            GGML_LOG_WARN("setup_moe_activation_counter: cudaGetDevice 失败: %s\n", cudaGetErrorString(cerr));
            counter->use_gpu_op = false;
        } else {
            const size_t n = (size_t) layers * (size_t) experts;

            cerr = cudaMalloc((void **) &counter->d_counts, n * sizeof(uint64_t));
            if (cerr != cudaSuccess) {
                GGML_LOG_ERROR("setup_moe_activation_counter: cudaMalloc(d_counts) 失败: %s\n", cudaGetErrorString(cerr));
                counter->d_counts = nullptr;
            }

            cerr = cudaMalloc((void **) &counter->d_weights, n * sizeof(double));
            if (cerr != cudaSuccess) {
                GGML_LOG_ERROR("setup_moe_activation_counter: cudaMalloc(d_weights) 失败: %s\n", cudaGetErrorString(cerr));
                counter->d_weights = nullptr;
            }

            if (counter->d_counts && counter->d_weights) {
                cerr = cudaMemset(counter->d_counts,  0, n * sizeof(uint64_t));
                if (cerr != cudaSuccess) {
                    GGML_LOG_ERROR("setup_moe_activation_counter: cudaMemset(d_counts) 失败: %s\n", cudaGetErrorString(cerr));
                }
                cerr = cudaMemset(counter->d_weights, 0, n * sizeof(double));
                if (cerr != cudaSuccess) {
                    GGML_LOG_ERROR("setup_moe_activation_counter: cudaMemset(d_weights) 失败: %s\n", cudaGetErrorString(cerr));
                }
                counter->use_gpu_op = true;
                GGML_LOG_INFO("MoE激活计数器 GPU 模式已启用 (LLAMA_MOE_COUNTER_GPU=%s)\n", env_gpu ? env_gpu : "<default 1>");

                if (moe_counter_debug_take()) {
                    GGML_LOG_INFO("[moe-counter dbg] gpu alloc ok: device=%d d_counts=%p d_weights=%p bytes=%zu\n",
                                  counter->cuda_device,
                                  (void *) counter->d_counts,
                                  (void *) counter->d_weights,
                                  n * sizeof(uint64_t) + n * sizeof(double));
                }
            } else {
                if (counter->d_counts) {
                    cudaFree(counter->d_counts);
                    counter->d_counts = nullptr;
                }
                if (counter->d_weights) {
                    cudaFree(counter->d_weights);
                    counter->d_weights = nullptr;
                }
                counter->use_gpu_op = false;
                GGML_LOG_WARN("MoE激活计数器 GPU 模式初始化失败，回退到回调统计（可能会触发拷贝）。\n");
            }
        }
    } else {
        counter->use_gpu_op = false;
        GGML_LOG_INFO("MoE激活计数器 GPU 模式已禁用 (LLAMA_MOE_COUNTER_GPU=0)\n");
    }
#else
    (void) want_gpu;
    counter->use_gpu_op = false;
#endif
    return true;
}

bool moe_activation_counter_use_gpu_op(MoeActivationCounter * counter) {
    return counter && counter->enabled && counter->use_gpu_op;
}

void destroy_moe_activation_counter(MoeActivationCounter * counter) {
#if defined(LLAMA_MOE_HAVE_CUDA_RUNTIME)
    if (counter) {
        if (counter->d_counts) {
            cudaFree(counter->d_counts);
            counter->d_counts = nullptr;
        }
        if (counter->d_weights) {
            cudaFree(counter->d_weights);
            counter->d_weights = nullptr;
        }
    }
#endif
    delete counter;
}

// --- Helper function prototypes (internal to this file) ---

static bool is_target_tensor(const char * tensor_name);
static int  parse_layer_index_from_name(const char * tensor_name);
static void accumulate_weights(MoeActivationCounter *       counter,
                               int                          layer_idx,
                               const std::vector<int32_t> & indices,
                               const std::vector<float> &   weights);

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
        if (strstr(tensor_name, "sum") != nullptr) {
            return false;
        }
        if (strstr(tensor_name, "norm") != nullptr) {
            return false;
        }
        if (strstr(tensor_name, "scaled") != nullptr) {
            return false;
        }
        if (strstr(tensor_name, "softmax") != nullptr) {
            return false;
        }
        return true;
    }
    return false;
}

static bool is_topk_tensor(const char * tensor_name) {
    return tensor_name && std::strstr(tensor_name, "ffn_moe_topk") != nullptr;
}

static bool is_weights_tensor(const char * tensor_name) {
    if (!tensor_name) {
        return false;
    }
    if (std::strstr(tensor_name, "ffn_moe_weights") == nullptr) {
        return false;
    }
    // 排除 sum, norm, scaled, softmax 等后缀，尽量只匹配基础的 weights 张量
    if (std::strstr(tensor_name, "sum") != nullptr) {
        return false;
    }
    if (std::strstr(tensor_name, "norm") != nullptr) {
        return false;
    }
    if (std::strstr(tensor_name, "scaled") != nullptr) {
        return false;
    }
    if (std::strstr(tensor_name, "softmax") != nullptr) {
        return false;
    }
    return true;
}

static void accumulate_weights(MoeActivationCounter *       counter,
                               int                          layer_idx,
                               const std::vector<int32_t> & indices,
                               const std::vector<float> &   weights) {
    if (indices.size() != weights.size()) {
        // legacy callback path assumes indices and weights are 1:1 aligned.
        // Some architectures expose different shapes here (e.g. indices only per expert slot, weights per slot x tokens),
        // so this mismatch is not necessarily an error. In GPU-op mode, weights are expected to be handled by GGML_OP_MOE_COUNTER.
        if (!counter || !counter->use_gpu_op) {
            GGML_LOG_WARN("Layer %d: indices size %zu != weights size %zu\n", layer_idx, indices.size(), weights.size());
        } else if (moe_counter_debug_take()) {
            GGML_LOG_INFO("[moe-counter dbg] legacy weights mismatch: layer=%d indices=%zu weights=%zu (ignored in gpu-op mode)\n",
                          layer_idx, indices.size(), weights.size());
        }
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
        std::smatch             match;
        std::string             s(tensor_name);
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

    if (!counter) {
        return false;
    }

    if (!counter->enabled) {
        return false;
    }

    if (ask) {
        // 调试：只对目标张量消耗 debug 配额，避免被大量无关张量的 ask() 调用提前耗尽。
        if (moe_counter_debug_enabled() && t && (is_target_tensor(t->name) || std::strstr(t->name, "ffn_moe_counter") != nullptr)) {
            if (moe_counter_debug_take()) {
                const char * buf_name = t->buffer ? ggml_backend_buffer_name(t->buffer) : "<null>";
                const int is_host = t->buffer ? (int) ggml_backend_buffer_is_host(t->buffer) : 0;
                GGML_LOG_INFO("[moe-counter dbg] ask: name=%s op=%s type=%s buf=%s is_host=%d ne=[%lld,%lld,%lld,%lld]\n",
                              t->name,
                              ggml_op_name(t->op),
                              ggml_type_name(t->type),
                              buf_name,
                              is_host,
                              (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
            }
        }
        // 第一阶段：询问是否对该张量感兴趣
        if (counter->use_gpu_op) {
            // GPU-op 模式下：
            // - 正常情况：图中插入 GGML_OP_MOE_COUNTER，device 张量不走回调，避免逐步 D2H。
            // - 兼容情况：如果该模型/路径没有插入 GGML_OP_MOE_COUNTER（gpu_op_nodes_built==0），
            //   至少订阅 topk indices 做回退统计，避免最终 0 记录（必要时会发生 D2H）。
            if (counter->gpu_op_nodes_built == 0) {
                return is_topk_tensor(t->name);
            }

            // 允许 host/CPU buffer 在回调里直接统计（不触发任何拷贝）
            if (is_target_tensor(t->name) && ggml_backend_buffer_is_host(t->buffer)) {
                return true;
            }
            return false;
        }

        return is_target_tensor(t->name);
    }

    if (counter->use_gpu_op && counter->gpu_op_nodes_built > 0 && !ggml_backend_buffer_is_host(t->buffer)) {
        // GPU-op 模式下，对于 device 张量不在回调里做任何统计。
        return true;
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
    if (is_topk_tensor(t->name)) {
        if (t->type != GGML_TYPE_I32) {
            GGML_LOG_WARN("跳过张量 '%s'，因为其类型不是 I32 (而是 %s)，无法解析为专家索引。\n", t->name,
                          ggml_type_name(t->type));
            return true;
        }

        // 获取数据
        uint8_t *    data_ptr = nullptr;
        const size_t n_bytes  = ggml_nbytes(t);

        // GPU 模式下已保证 is_host；非 GPU 模式才允许做 backend get
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

    } else if (is_weights_tensor(t->name)) {
        // 处理被选中的权重 (Effective Weights)

        if (t->type != GGML_TYPE_F32) {
            GGML_LOG_WARN("跳过张量 '%s'，因为其类型不是 F32 (而是 %s)。\n", t->name, ggml_type_name(t->type));
            return true;
        }

        // 获取数据
        uint8_t *    data_ptr = nullptr;
        const size_t n_bytes  = ggml_nbytes(t);

        // GPU 模式下已保证 is_host；非 GPU 模式才允许做 backend get
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
    if (!counter) {
        return;
    }

    if (!counter->enabled) {
        return;
    }

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

    // 如果启用了 GPU op，则先一次性把 device 累加结果拷回 host
    std::vector<uint64_t> dev_counts;
    std::vector<double>   dev_weights;

#if defined(LLAMA_MOE_HAVE_CUDA_RUNTIME)
    if (counter->use_gpu_op && counter->d_counts && counter->d_weights) {
        const size_t n = (size_t) counter->num_layers * (size_t) counter->num_experts;
        dev_counts.resize(n);
        dev_weights.resize(n);

        if (moe_counter_debug_take()) {
            GGML_LOG_INFO("[moe-counter dbg] save: use_gpu_op=1 device=%d d_counts=%p d_weights=%p n=%zu\n",
                          counter->cuda_device,
                          (void *) counter->d_counts,
                          (void *) counter->d_weights,
                          n);
        }

        cudaError_t cerr = cudaSetDevice(counter->cuda_device);
        if (cerr != cudaSuccess) {
            GGML_LOG_ERROR("[moe-counter dbg] cudaSetDevice(%d) failed: %s\n", counter->cuda_device, cudaGetErrorString(cerr));
        }
        cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess) {
            GGML_LOG_ERROR("[moe-counter dbg] cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cerr));
        }

        cerr = cudaMemcpy(dev_counts.data(),  counter->d_counts,  n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        if (cerr != cudaSuccess) {
            GGML_LOG_ERROR("[moe-counter dbg] cudaMemcpy(dev_counts) failed: %s\n", cudaGetErrorString(cerr));
            dev_counts.clear();
        }

        cerr = cudaMemcpy(dev_weights.data(), counter->d_weights, n * sizeof(double),   cudaMemcpyDeviceToHost);
        if (cerr != cudaSuccess) {
            GGML_LOG_ERROR("[moe-counter dbg] cudaMemcpy(dev_weights) failed: %s\n", cudaGetErrorString(cerr));
            dev_weights.clear();
        }

        if (!dev_counts.empty() && moe_counter_debug_take()) {
            unsigned long long dev_total = 0;
            for (size_t i = 0; i < dev_counts.size(); ++i) {
                dev_total += dev_counts[i];
            }
            GGML_LOG_INFO("[moe-counter dbg] save: dev_total_activations=%llu\n", dev_total);
        }
    }
#endif

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
            if (!dev_counts.empty()) {
                cnt += dev_counts[(size_t) layer * (size_t) counter->num_experts + (size_t) expert];
            }
            file << "," << cnt;
            total_activations += cnt;
        }
        file << "\n";
    }

    file.close();

    // 保存权重报告
    const std::string weights_filepath = work_dir + "expert_weights.csv";
    std::ofstream     wfile(weights_filepath);
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
                double w = counter->expert_activation_weights[layer][expert];
                if (!dev_weights.empty()) {
                    w += dev_weights[(size_t) layer * (size_t) counter->num_experts + (size_t) expert];
                }
                wfile << "," << w;
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
    if (moe_counter_debug_enabled()) {
        GGML_LOG_INFO("[moe-counter dbg] save: enabled=%d use_gpu_op=%d gpu_op_nodes_built=%d\n",
                      (int) counter->enabled,
                      (int) counter->use_gpu_op,
                      (int) counter->gpu_op_nodes_built);
    }

    GGML_LOG_INFO("==============================\n");
}

// llama.moe: build GGML_OP_MOE_COUNTER node
struct ggml_tensor * ggml_moe_counter(
        struct ggml_context * ctx,
        struct ggml_tensor  * selected_experts,
        struct ggml_tensor  * weights,
        MoeActivationCounter * counter,
        int32_t layer_idx) {
    GGML_ASSERT(ctx);
    GGML_ASSERT(selected_experts);
    GGML_ASSERT(weights);

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    result->op     = GGML_OP_MOE_COUNTER;
    result->src[0] = selected_experts;
    result->src[1] = weights;

    ggml_set_op_params_i32(result, 0, layer_idx);
    ggml_set_op_params_i32(result, 1, counter ? counter->num_experts : 0);

    uintptr_t p_counts  = 0;
    uintptr_t p_weights = 0;

#if defined(LLAMA_MOE_HAVE_CUDA_RUNTIME)
    if (counter && counter->enabled && counter->use_gpu_op) {
        p_counts  = (uintptr_t) counter->d_counts;
        p_weights = (uintptr_t) counter->d_weights;
    }
#endif

    ggml_set_op_params_i32(result, 2, (int32_t) (p_counts & 0xffffffffu));
    ggml_set_op_params_i32(result, 3, (int32_t) ((uint64_t) p_counts >> 32));
    ggml_set_op_params_i32(result, 4, (int32_t) (p_weights & 0xffffffffu));
    ggml_set_op_params_i32(result, 5, (int32_t) ((uint64_t) p_weights >> 32));

    if (counter) {
        counter->gpu_op_nodes_built++;
    }

    return result;
}
