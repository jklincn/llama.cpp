#include "ggml-moe.h"

#include "ggml-impl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// llama.moe: CUDA-only
// We intentionally avoid including <cuda_runtime.h> here to keep this translation unit
// independent of CUDA include paths (some build setups don't propagate them to C++ units).
// Instead, declare the minimal CUDA Runtime APIs we use; symbols are resolved via cudart.
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

/**
 * @struct MoeActivationCounter
 * 用于收集和统计MoE模型中专家激活次数的C++实现。
 * 此定义对C代码隐藏。
 */
struct MoeActivationCounter {
    int num_layers  = 0;
    int num_experts = 0;

    // 默认为禁用，直到 setup 被调用。防止在 setup 之前（如 warmup）触发回调报错
    bool enabled = false;

    // 是否启用 GGML_OP_MOE_COUNTER 路径（GPU侧累加，save时一次性回传）
    bool use_gpu_op = false;

    // 构图阶段是否实际插入过 GGML_OP_MOE_COUNTER（用于 sanity check）
    int gpu_op_nodes_built = 0;

    int cuda_device = 0;
    uint64_t * d_counts  = nullptr; // [num_layers * num_experts]
    double   * d_weights = nullptr; // [num_layers * num_experts]

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

    cudaError_t cerr = cudaGetDevice(&counter->cuda_device);
    if (cerr != cudaSuccess) {
        GGML_LOG_ERROR("setup_moe_activation_counter: cudaGetDevice 失败: %s\n", cudaGetErrorString(cerr));
        counter->use_gpu_op = false;
        return false;
    }

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

    if (!counter->d_counts || !counter->d_weights) {
        if (counter->d_counts) {
            cudaFree(counter->d_counts);
            counter->d_counts = nullptr;
        }
        if (counter->d_weights) {
            cudaFree(counter->d_weights);
            counter->d_weights = nullptr;
        }
        counter->use_gpu_op = false;
        return false;
    }

    cerr = cudaMemset(counter->d_counts,  0, n * sizeof(uint64_t));
    if (cerr != cudaSuccess) {
        GGML_LOG_ERROR("setup_moe_activation_counter: cudaMemset(d_counts) 失败: %s\n", cudaGetErrorString(cerr));
        return false;
    }
    cerr = cudaMemset(counter->d_weights, 0, n * sizeof(double));
    if (cerr != cudaSuccess) {
        GGML_LOG_ERROR("setup_moe_activation_counter: cudaMemset(d_weights) 失败: %s\n", cudaGetErrorString(cerr));
        return false;
    }

    counter->use_gpu_op = true;
    GGML_LOG_INFO("MoE激活计数器 GPU 模式已启用\n");
    return true;
}

bool moe_activation_counter_use_gpu_op(MoeActivationCounter * counter) {
    return counter && counter->enabled && counter->use_gpu_op;
}

void destroy_moe_activation_counter(MoeActivationCounter * counter) {
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
    delete counter;
}

// --- Helper function prototypes (internal to this file) ---

// --- Function Implementations ---

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

    // 清理：不再使用 CPU 回调统计（避免任何逐步 D2H）。
    // 计数由 GGML_OP_MOE_COUNTER 在 CUDA backend 中完成。
    GGML_UNUSED(t);
    if (ask) {
        return false;
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

    if (counter->use_gpu_op && counter->d_counts && counter->d_weights) {
        const size_t n = (size_t) counter->num_layers * (size_t) counter->num_experts;
        dev_counts.resize(n);
        dev_weights.resize(n);

        cudaError_t cerr = cudaSetDevice(counter->cuda_device);
        if (cerr != cudaSuccess) {
            GGML_LOG_ERROR("cudaSetDevice(%d) failed: %s\n", counter->cuda_device, cudaGetErrorString(cerr));
        }
        cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess) {
            GGML_LOG_ERROR("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cerr));
        }

        cerr = cudaMemcpy(dev_counts.data(),  counter->d_counts,  n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        if (cerr != cudaSuccess) {
            GGML_LOG_ERROR("cudaMemcpy(dev_counts) failed: %s\n", cudaGetErrorString(cerr));
            dev_counts.clear();
        }

        cerr = cudaMemcpy(dev_weights.data(), counter->d_weights, n * sizeof(double),   cudaMemcpyDeviceToHost);
        if (cerr != cudaSuccess) {
            GGML_LOG_ERROR("cudaMemcpy(dev_weights) failed: %s\n", cudaGetErrorString(cerr));
            dev_weights.clear();
        }
    }

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
            uint64_t cnt = !dev_counts.empty() ? dev_counts[(size_t) layer * (size_t) counter->num_experts + (size_t) expert] : 0;
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
                double w = !dev_weights.empty() ? dev_weights[(size_t) layer * (size_t) counter->num_experts + (size_t) expert] : 0.0;
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

    if (counter->enabled && counter->use_gpu_op && counter->gpu_op_nodes_built == 0) {
        GGML_LOG_WARN("MoE激活计数器已启用，但未构建任何 MOE_COUNTER 节点；报告可能为 0。\n");
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

    if (counter && counter->enabled && counter->use_gpu_op) {
        p_counts  = (uintptr_t) counter->d_counts;
        p_weights = (uintptr_t) counter->d_weights;
    }

    ggml_set_op_params_i32(result, 2, (int32_t) (p_counts & 0xffffffffu));
    ggml_set_op_params_i32(result, 3, (int32_t) ((uint64_t) p_counts >> 32));
    ggml_set_op_params_i32(result, 4, (int32_t) (p_weights & 0xffffffffu));
    ggml_set_op_params_i32(result, 5, (int32_t) ((uint64_t) p_weights >> 32));

    if (counter) {
        counter->gpu_op_nodes_built++;
    }

    return result;
}
