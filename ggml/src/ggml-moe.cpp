#include "ggml-moe.h"

// C++ standard library headers
#include <cmath>
#include <cstdio>
#include <cstdlib>  // For system()
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// ggml internal headers
#include "ggml-backend.h"
#include "ggml-impl.h"

/**
 * @struct MoeTopkCollector
 * C++ implementation of the MoE TopK tensor data collector.
 * This definition is hidden from C code.
 */
struct MoeTopkCollector {
    std::vector<uint8_t> buffer;  // Temporary data buffer
    std::string          output_dir = "moe_topk_data";
    std::ofstream        metadata_file;
    int                  tensor_counter = 0;
    bool                 initialized    = false;

    // Statistics
    int    total_collected   = 0;
    size_t total_bytes_saved = 0;

    // Destructor to ensure metadata_file is closed
    ~MoeTopkCollector() {
        if (metadata_file.is_open()) {
            metadata_file.close();
        }
    }
};

// C-compatible API implementations

MoeTopkCollector * create_moe_topk_collector(void) {
    return new (std::nothrow) MoeTopkCollector();
}

void destroy_moe_topk_collector(MoeTopkCollector * collector) {
    if (!collector) {
        return;
    }
    // Print final summary before destroying
    print_collection_summary(collector);
    delete collector;
}

// --- Helper function prototypes (internal to this file) ---

static bool        is_target_tensor(const char * tensor_name);
static std::string ggml_ne_string(const ggml_tensor * t);
static bool        save_tensor_npy(const std::string & filepath, ggml_tensor * t, uint8_t * data);
static void        save_metadata(MoeTopkCollector * collector, ggml_tensor * t, const std::string & filename);

// --- Function Implementations ---

/**
 * 检查张量名称是否匹配目标前缀
 */
static bool is_target_tensor(const char * tensor_name) {
    if (!tensor_name) {
        return false;
    }
    return strncmp(tensor_name, "ffn_moe_topk", 12) == 0;
}

/**
 * 获取张量维度字符串（实现版本）
 */
static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        // 只添加存在的维度
        if (t->ne[i] > 0) {
            if (!str.empty()) {
                str += ", ";
            }
            str += std::to_string(t->ne[i]);
        }
    }
    return str;
}

/**
 * 保存张量为NPY格式文件
 */
static bool save_tensor_npy(const std::string & filepath, ggml_tensor * t, uint8_t * data) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        GGML_LOG_ERROR("无法创建文件: %s\n", filepath.c_str());
        return false;
    }

    try {
        // NPY文件头：魔数 + 版本
        file.write("\x93NUMPY", 6);
        file.write("\x01\x00", 2);  // 版本 1.0

        // 构造数据类型字符串
        std::string dtype;
        switch (t->type) {
            case GGML_TYPE_F32:
                dtype = "'<f4'";
                break;
            case GGML_TYPE_F16:
                dtype = "'<f2'";
                break;
            case GGML_TYPE_I64:
                dtype = "'<i8'";
                break;
            case GGML_TYPE_I32:
                dtype = "'<i4'";
                break;
            case GGML_TYPE_I16:
                dtype = "'<i2'";
                break;
            case GGML_TYPE_I8:
                dtype = "'<i1'";
                break;
            default:
                GGML_LOG_ERROR("不支持的数据类型: %s\n", ggml_type_name(t->type));
                return false;
        }

        // 构造shape字符串
        std::ostringstream shape_stream;
        shape_stream << "(";
        bool first = true;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (t->ne[i] > 1 || (i == 0 && first)) {
                if (!first) {
                    shape_stream << ", ";
                }
                shape_stream << t->ne[i];
                first = false;
            }
        }
        if (!first) {
            shape_stream << ",";  // Python tuple需要逗号
        }
        shape_stream << ")";

        // 构造完整的头部
        std::ostringstream header_stream;
        header_stream << "{'descr': " << dtype << ", 'fortran_order': False" << ", 'shape': " << shape_stream.str()
                      << ", }";

        std::string header = header_stream.str();

        // 计算填充，使头部总长度对齐到16字节
        size_t total_header_size = 8 + 2 + header.size() + 1;  // 魔数+版本+头长度+头内容+换行
        size_t padding           = (16 - (total_header_size % 16)) % 16;
        header += std::string(padding, ' ') + "\n";

        // 写入头部长度
        uint16_t header_len = header.size();
        file.write(reinterpret_cast<const char *>(&header_len), 2);

        // 写入头部内容
        file.write(header.c_str(), header.size());

        // 写入张量数据
        size_t data_size = ggml_nbytes(t);
        file.write(reinterpret_cast<const char *>(data), data_size);

        file.close();
        return file.good();

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("保存NPY文件时出错: %s\n", e.what());
        return false;
    }
}

/**
 * 保存张量元数据到CSV文件
 */
static void save_metadata(MoeTopkCollector * collector, ggml_tensor * t, const std::string & filename) {
    if (!collector->metadata_file.is_open()) {
        std::string meta_path = collector->output_dir + "/metadata.csv";
        collector->metadata_file.open(meta_path);
        if (!collector->metadata_file.is_open()) {
            GGML_LOG_ERROR("无法创建元数据文件: %s\n", meta_path.c_str());
            return;
        }

        // 写入CSV头部
        collector->metadata_file << "序号,张量名称,文件名,形状,数据类型,元素数量,字节大小,操作类型\n";
    }

    // 计算元素总数
    size_t total_elements = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] > 1 || i == 0) {
            total_elements *= t->ne[i];
        }
    }

    // 写入当前张量的元数据
    collector->metadata_file << collector->tensor_counter << "," << "\"" << (t->name[0] != '\0' ? t->name : "unnamed")
                             << "\","
                             << "\"" << filename << "\","
                             << "\"" << ggml_ne_string(t) << "\","
                             << "\"" << ggml_type_name(t->type) << "\"," << total_elements << "," << ggml_nbytes(t)
                             << ","
                             << "\"" << ggml_op_desc(t) << "\"\n";

    collector->metadata_file.flush();
}

/**
 * MoE TopK张量采集回调函数
 */
bool moe_topk_collector_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * collector = (MoeTopkCollector *) user_data;

    if (ask) {
        // 只对目标张量感兴趣
        return is_target_tensor(t->name);
    }

    // 再次确认是否为目标张量（双重检查）
    if (!is_target_tensor(t->name)) {
        return true;
    }

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    // 输出基本信息
    char src1_str[128] = { 0 };
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    GGML_LOG_INFO("🎯 [MoE TopK] %s: %s = (%s) %s(%s{%s}, %s}) = {%s}\n", __func__, t->name, ggml_type_name(t->type),
                  ggml_op_desc(t), src0->name, ggml_ne_string(src0).c_str(), src1 ? src1_str : "",
                  ggml_ne_string(t).c_str());

    // 处理数据获取（GPU -> CPU 如果需要）
    const bool is_host  = ggml_backend_buffer_is_host(t->buffer);
    uint8_t *  data_ptr = nullptr;

    if (!is_host) {
        // 从GPU复制数据到CPU
        size_t n_bytes = ggml_nbytes(t);
        collector->buffer.resize(n_bytes);
        ggml_backend_tensor_get(t, collector->buffer.data(), 0, n_bytes);
        data_ptr = collector->buffer.data();
        GGML_LOG_INFO("📥 从GPU复制了 %zu 字节数据\n", n_bytes);
    } else {
        // 数据已经在CPU上
        data_ptr = (uint8_t *) t->data;
        GGML_LOG_INFO("📋 数据已在CPU内存中\n");
    }

    // 只处理非量化张量
    if (!ggml_is_quantized(t->type)) {
        // 构造文件名
        std::ostringstream filename_stream;
        filename_stream << std::setfill('0') << std::setw(4) << collector->tensor_counter << "_"
                        << (t->name[0] != '\0' ? t->name : "unnamed") << ".npy";
        std::string filename = filename_stream.str();
        std::string filepath = collector->output_dir + "/" + filename;

        // 保存NPY文件
        if (save_tensor_npy(filepath, t, data_ptr)) {
            size_t file_size = ggml_nbytes(t);
            collector->total_bytes_saved += file_size;
            collector->total_collected++;

            GGML_LOG_INFO("💾 已保存: %s (%.2f KB)\n", filename.c_str(), file_size / 1024.0);

            // 保存元数据
            save_metadata(collector, t, filename);

            // 显示一些基本统计信息
            size_t total_elements = 1;
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                if (t->ne[i] > 1 || i == 0) {
                    total_elements *= t->ne[i];
                }
            }

            GGML_LOG_INFO("📊 张量统计: %zu个元素, 形状=%s, 类型=%s\n", total_elements, ggml_ne_string(t).c_str(),
                          ggml_type_name(t->type));

        } else {
            GGML_LOG_ERROR("❌ 保存失败: %s\n", filepath.c_str());
        }

        collector->tensor_counter++;
    } else {
        GGML_LOG_INFO("⚠️  跳过量化张量: %s (类型: %s)\n", t->name, ggml_type_name(t->type));
    }

    return true;
}

/**
 * 初始化MoE TopK数据收集器
 */
bool init_moe_topk_collector(MoeTopkCollector * collector, const char * output_dir_c_str) {
    if (!collector) {
        return false;
    }

    collector->output_dir        = output_dir_c_str;
    collector->tensor_counter    = 0;
    collector->total_collected   = 0;
    collector->total_bytes_saved = 0;
    collector->buffer.clear();
    if (collector->metadata_file.is_open()) {
        collector->metadata_file.close();
    }

    // 创建输出目录
    std::string mkdir_cmd = "mkdir -p " + collector->output_dir;
    if (system(mkdir_cmd.c_str()) != 0) {
        GGML_LOG_ERROR("无法创建输出目录: %s\n", collector->output_dir.c_str());
        return false;
    }

    collector->initialized = true;
    GGML_LOG_INFO("🚀 MoE TopK数据收集器已初始化\n");
    GGML_LOG_INFO("📁 输出目录: %s\n", collector->output_dir.c_str());
    GGML_LOG_INFO("🎯 目标张量: ffn_moe_topk*\n");

    return true;
}

/**
 * 打印收集统计信息
 */
void print_collection_summary(const MoeTopkCollector * collector) {
    if (!collector) {
        return;
    }
    GGML_LOG_INFO("\n=== MoE TopK 数据收集报告 ===\n");
    GGML_LOG_INFO("收集的张量数量: %d\n", collector->total_collected);
    GGML_LOG_INFO("总数据大小: %.2f MB\n", collector->total_bytes_saved / 1024.0 / 1024.0);
    GGML_LOG_INFO("平均张量大小: %.2f KB\n", collector->total_collected > 0 ?
                                                 (collector->total_bytes_saved / 1024.0 / collector->total_collected) :
                                                 0);
    GGML_LOG_INFO("数据保存路径: %s/\n", collector->output_dir.c_str());
    GGML_LOG_INFO("元数据文件: %s/metadata.csv\n", collector->output_dir.c_str());

    if (collector->total_collected > 0) {
        GGML_LOG_INFO("\n💡 使用Python分析数据:\n");
        GGML_LOG_INFO("   import numpy as np\n");
        GGML_LOG_INFO("   import pandas as pd\n");
        GGML_LOG_INFO("   \n");
        GGML_LOG_INFO("   # 加载元数据\n");
        GGML_LOG_INFO("   metadata = pd.read_csv('%s/metadata.csv')\n", collector->output_dir.c_str());
        GGML_LOG_INFO("   print(metadata)\n");
        GGML_LOG_INFO("   \n");
        GGML_LOG_INFO("   # 加载第一个张量\n");
        GGML_LOG_INFO("   first_tensor = np.load('%s/' + metadata.iloc[0]['文件名'])\n", collector->output_dir.c_str());
        GGML_LOG_INFO("   print(f'Shape: {first_tensor.shape}, Data: {first_tensor}')\n");
    }

    GGML_LOG_INFO("============================\n\n");
}
