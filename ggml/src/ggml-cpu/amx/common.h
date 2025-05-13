#pragma once

#include "ggml.h"
#include "ggml-cpu-impl.h"

#include <algorithm>
#include <memory>
#include <type_traits>

#if defined(GGML_USE_OPENMP)
#include <omp.h>
#endif

// AMX 的 tile（矩阵寄存器）高度/宽度。
// AMX 规定高度 ≤ 16，宽度需是 64 B 对齐；在量化推理里常用 16×16。
#define TILE_M 16
#define TILE_N 16

// K 维块大小（乘法的累加维）。
// 32 = TILE_N*2，方便把 int8/int16 载入 tile 时对齐 64 B。
#define TILE_K 32

// 为什么是 16×16×32？
// AMX 本质上是把一个 tile register 当成小矩阵缓冲，
// 一条 tdpbssd 可以把 TMM0 × TMM1 累加到 TMM2；
// 16 行、64 B/行 在 int8 场景下刚好存 16 × 64 = 1024 B，
// 符合硬件限制且匹配 GPT 模型隐藏维度常见倍数（64、128）。


// VNNI/VNNI‑AMX 块宽度：Intel VNNI 指令要求把 4 个 int8 打包后一次性乘以 int16 accumulator。
// 这与 GPT‑Q / QBits 量化拆块逻辑紧耦合。
#define VNNI_BLK 4

// 通用对齐常量。很多内存 copy 会按 32 B 对齐，
// 因为 AMX tile 的最小 stride 恰好 64 B；而每个 int8 tile 行 16×4 = 64 B，32 B 方便拆半。
#define AMX_BLK_SIZE 32

// 把 AMX 8 组 tile 寄存器映射到更易读的名字
#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7

// parallel routines
// 并行辅助模板

// 整数向上取整除法，常用于「块划分」：把总 work x 均匀分到大小为 y 的块，需要补 1 才不会丢尾巴。
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T div_up(T x, T y) { return (x + y - 1) / y; }

// 给第 ith 个线程分配一段 [n_start, n_end) 索引区间，使得
// 所有线程负载尽量均衡，
// 不需要原子操作或锁。
// 211 平衡：来自 Intel MKL 术语“2‑1‑1”平衡（把总任务分 2 级，再 1 级，再 1 级）
template <typename T>
inline void balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#if 0
    // onednn partition pattern
    // oneDNN 的切分策略（chunk = n1/n2，先把 T1 个大块分给前面线程）
    T& n_my = n_end;
    if (nth <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } else {
        T n1 = div_up(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_end += n_start;
#else
    // pytorch aten partition pattern
    // PyTorch/Aten 的简单策略：每线程固定 ceil(n/nth)；最后线程不足时取 min
    T n_my = div_up(n, nth);
    n_start = ith * n_my;
    n_end = std::min(n_start + n_my, n);
#endif
}

// 把一个 lambda f(begin, end) 在 n 个迭代空间上并行执行。
// 在 #pragma omp parallel 里拿到线程 id、总线程数 → 调 balance211 → 每线程处理自己那一段。
template <typename func_t>
inline void parallel_for(int n, const func_t& f) {
#if defined(GGML_USE_OPENMP)
#pragma omp parallel
{
    int nth = omp_get_num_threads();
    int ith = omp_get_thread_num();
    int tbegin, tend;
    balance211(n, nth, ith, tbegin, tend);
    f(tbegin, tend);
}
#else
    f(0, n);
#endif
}

// GGML 自带的线程池回调也会把 nth / ith 填进 ggml_compute_params；
// 这里借用同一 balance211 算法，但由 GGML 的「任务并行」调度而不是 OpenMP。
template <typename func_t>
inline void parallel_for_ggml(const ggml_compute_params * params, int n, const func_t & f) {
    int tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
// 量化类型与 AMX Kernel 选择
inline bool qtype_has_amx_kernels(const enum ggml_type type) {
    // TODO: fix padding for vnni format
    return (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
