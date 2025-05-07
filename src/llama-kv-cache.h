#pragma once

#include "llama.h"
#include "llama-io.h"
#include "llama-memory.h"

#include "ggml-cpp.h"

#include <functional>
#include <set>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_ubatch;

// KV 缓存的通用接口
struct llama_kv_cache : public llama_memory_i {
    // 继承构造函数
    using llama_memory_i::llama_memory_i;

    // 如果批处理失败，调用此函数恢复缓存状态
    virtual void restore() = 0; // call if batch processing fails - restores the cache state
    // 批处理成功后调用，清除任何挂起的状态
    virtual void commit() = 0;  // call after successful batch processing - clears any pending state

    // 获取缓存中的 token 数量
    virtual int32_t get_n_tokens()   const = 0;
    virtual int32_t get_used_cells() const = 0; // TODO: remove, this is too-specific to the unified cache

    // 判断缓存是否支持高效地移除开头的 token
    virtual bool get_can_shift() const = 0;

    // 重写基类方法，编辑能力等同于移动能力
    bool get_can_edit() const override { return get_can_shift(); }
};

struct llama_kv_cache_guard {
    llama_kv_cache_guard(llama_kv_cache * kv) : kv(kv) {}

    ~llama_kv_cache_guard() {
        kv->restore();
    }

    void commit() {
        kv->commit();
    }

private:
    llama_kv_cache * kv;
};

// 代表 KV 缓存中的一个单元格 (cell)，通常对应一个 token 的 KV 状态
struct llama_kv_cell {
    llama_pos pos   = -1; // token 在序列中的绝对位置。
    llama_pos delta =  0; // 位置的增量或偏移，可能用于某些缓存策略
    int32_t   src   = -1; // 源索引，用于循环状态模型复制状态
    int32_t   tail  = -1; // 尾部索引，可能用于管理单元格链或组

    // 一个集合，存储了共享这个缓存单元格的序列 ID。这很重要，因为在批处理或 beam search 中，多个序列可能共享部分历史。
    std::set<llama_seq_id> seq_id;

    // 辅助函数，用于检查序列 ID
    bool has_seq_id(const llama_seq_id & id) const {
        return seq_id.find(id) != seq_id.end();
    }

    bool is_empty() const {
        return seq_id.empty();
    }

    bool is_same_seq(const llama_kv_cell & other) const {
        return seq_id == other.seq_id;
    }
};

// ring-buffer of cached KV data
// TODO: pimpl
// TODO: add notion of max sequences
// llama_kv_cache 的一个具体实现
class llama_kv_cache_unified : public llama_kv_cache {
public:
    // can be used to query data from the model if needed
    struct callbacks {
        std::function<ggml_tensor * (uint32_t n_ctx_per_seq, int il)> get_rope_factors;
    };

    // 构造函数接收超参数 (hparams) 和回调 (cbs)
    llama_kv_cache_unified(
            const llama_hparams & hparams,
            callbacks             cbs);

    virtual ~llama_kv_cache_unified() = default;

    // TODO: become constructor
    bool init(
            const llama_model & model,   // TODO: do not reference the model
          const llama_cparams & cparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                     uint32_t   kv_size,
                         bool   offload);

    int32_t get_n_tokens()   const override;
    int32_t get_used_cells() const override;

    size_t total_size() const;

    // TODO: better data structures to reduce the cost of this operation
    llama_pos pos_max() const;

    void clear() override;
    void defrag() override;

    virtual void restore() override;
    virtual void commit() override;

    // 序列操作 API
    // 支持删除、复制、保留、整体位移、缩放位置 等高级编辑，用于实现滑动窗口、beam search、RoPE 缩放等策略。
    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;

    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos delta) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    // 获取指定序列的最大 token 位置
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    bool get_can_shift() const override;

    // find an empty slot of size "n_tokens" in the cache
    // updates the cache head
    // Note: On success, it's important that cache.head points
    // to the first cell of the slot.
    // 为新的批处理数据在缓存中查找一个合适大小的空槽。
    // 成功后，cache.head 会指向槽的起始位置。
    bool find_slot(const llama_ubatch & batch);

    // TODO: maybe not needed
    uint32_t get_padding(const llama_cparams & cparams) const;

    // find how many cells are currently in use
    uint32_t cell_max() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    // defrag

    struct {
        std::vector<uint32_t> ids;
    } defrag_info;

    // return true if cells have been moved
    // 准备碎片整理，如果单元格被移动则返回 true
    bool defrag_prepare(int32_t n_max_nodes);

    // commit/restore cache

    struct slot_range {
        uint32_t c0 = 0; // note: these are cell indices, not sequence positions
        uint32_t c1 = 0;
    };

    // pending cell updates that are not yet committed
    struct {
        std::vector<slot_range> ranges;
    } pending;

    // state write/load

    // 将缓存状态写入
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1);

    // members

    const llama_hparams & hparams;

    callbacks cbs;

    bool has_shift = false;
    bool do_defrag = false;

    // TODO: remove this and implement llama_kv_cache_recurrent instead
    bool recurrent = false; // with recurrent state models, a cell can hold the state for more than one past token

    bool v_trans   = true;  // the value tensor is transposed
    bool can_shift = false;

    // Note: The value of head isn't only used to optimize searching
    // for a free KV slot. llama_decode_impl also uses it, so it
    // cannot be freely changed after a slot has been allocated.
    // 环形缓冲区的管理变量：head 指向下一个可写位置或当前活动区域的头部，size 是总容量，used 是已使用的单元格数
    uint32_t head = 0;
    uint32_t size = 0;
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // computed before each graph build
    uint32_t n = 0;

    // 保存每个 KV 单元的元信息
    std::vector<llama_kv_cell> cells;

    std::vector<ggml_tensor *> k_l; // per layer
    std::vector<ggml_tensor *> v_l;

private:
    // K 和 V 张量的数据类型
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    // GGML 上下文和后端缓冲区的向量，用于管理 GGML 张量的内存
    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

// TODO: temporary reusing llama_kv_cache_unified -- implement recurrent cache and simplify llama_kv_cache_unified
//class llama_kv_cache_recurrent : public llama_kv_cache_unified {
//public:
//    using llama_kv_cache_unified::llama_kv_cache_unified;
//};

//
// kv cache view
//

llama_kv_cache_view llama_kv_cache_view_init(const llama_kv_cache & kv, int32_t n_seq_max);

void llama_kv_cache_view_update(llama_kv_cache_view * view, const llama_kv_cache * kv);
