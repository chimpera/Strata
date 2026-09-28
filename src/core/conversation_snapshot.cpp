#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/kv_q4.hpp"
#include <cuda_runtime.h>

#include <array>
#include <limits>

namespace strata::core {
namespace {
struct Layout {
    int format;
    int64_t cells, pooled_rows, page_size;
    size_t data, scales, pooled;
};

bool valid_extent(const QsaState& st, int64_t upto, std::string& error) {
    const int64_t page = strata::kernels::qsa_real_shapes().page_size;
    if (upto < 0 || upto > st.max_cells || upto > std::numeric_limits<int64_t>::max() - (page - 1)) {
        error = "conversation snapshot: invalid K/V extent";
        return false;
    }
    return true;
}

Layout layout(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index) {
    const auto s = strata::kernels::qsa_real_shapes();
    const int64_t cells = ((upto + s.page_size - 1) / s.page_size) * s.page_size;
    const size_t rows = (size_t) cells * (size_t) g.n_head_kv;
    const size_t per = st.kv_q4 ? (size_t) strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                              : (size_t) g.head_dim * (st.kv_int8 ? 1 : 2);
    const int64_t pooled = index ? upto / s.idx_block : 0;
    return {qsa_kv_format(st), cells, pooled, s.page_size, rows * per,
            st.kv_int8 && !st.kv_q4 ? rows * (size_t) (g.head_dim / 64) * 2 : 0,
            (size_t) pooled * (size_t) g.idx_key_dim * sizeof(float)};
}

std::array<void*, 5> pools(const QsaState& st) {
    const bool host = st.kv_mode != 0;
    if (st.kv_q4)
        return {host ? st.host.k_q4 : st.k_q4, host ? st.host.v_q4 : st.v_q4, nullptr, nullptr, st.idx_pooled};
    if (st.kv_int8)
        return {host ? st.host.k_q : st.k_q, host ? st.host.v_q : st.v_q,
                host ? st.host.k_scale : st.k_scale, host ? st.host.v_scale : st.v_scale, st.idx_pooled};
    return {host ? st.host.k_pool : st.k_pool, host ? st.host.v_pool : st.v_pool, nullptr, nullptr, st.idx_pooled};
}

bool valid(const QsaState& st, const Layout& l, int64_t upto, std::string& error) {
    if (upto < 0 || upto > st.max_cells || l.cells > st.n_pages * l.page_size ||
        l.pooled_rows > st.idx_pooled_rows || (st.kv_mode != 0 && !st.host.present())) {
        error = "conversation snapshot: invalid K/V extent or missing authoritative host pool";
        return false;
    }
    return true;
}

bool transfer(void* dst, const void* src, size_t n, std::string& error) {
    if (!n) return true;
    if (!src || !dst) { error = "conversation snapshot: missing state buffer"; return false; }
    // Default handles both device allocations and device-mapped host pool aliases.
    const cudaError_t e = cudaMemcpy(dst, src, n, cudaMemcpyDefault);
    if (e == cudaSuccess) return true;
    error = std::string("conversation snapshot copy: ") + cudaGetErrorString(e);
    return false;
}
} // namespace

size_t conversation_kv_bytes(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index) {
    std::string error;
    if (!valid_extent(st, upto, error)) return 0;
    const auto l = layout(st, g, upto, index);
    return 2 * (l.data + l.scales) + l.pooled;
}

bool conversation_kv_save(ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                          int64_t upto, bool index, std::string& error) {
    if (!valid_extent(st, upto, error)) return false;
    const auto l = layout(st, g, upto, index);
    if (!valid(st, l, upto, error)) return false;
    image.format = l.format;
    image.cells = l.cells;
    image.heads = g.n_head_kv;
    image.head_dim = g.head_dim;
    image.page_size = l.page_size;
    image.pooled_rows = l.pooled_rows;
    image.idx_dim = g.idx_key_dim;
    const auto src = pools(st);
    const std::array<size_t,5> sizes = {l.data, l.data, l.scales, l.scales, l.pooled};
    const std::array<std::vector<uint8_t>*,5> dst = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i]->resize(sizes[i]);
        if (!transfer(dst[i]->data(), src[i], sizes[i], error)) return false;
    }
    return true;
}

bool conversation_kv_restore(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                             int64_t upto, bool index, std::string& error) {
    if (!valid_extent(st, upto, error)) return false;
    const auto l = layout(st, g, upto, index);
    if (!valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.data, l.scales, l.scales, l.pooled};
    const std::array<const std::vector<uint8_t>*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    if (image.format != l.format || image.cells != l.cells || image.heads != g.n_head_kv ||
        image.head_dim != g.head_dim || image.page_size != l.page_size || image.pooled_rows != l.pooled_rows ||
        image.idx_dim != g.idx_key_dim) {
        error = "conversation snapshot: incompatible K/V geometry";
        return false;
    }
    for (size_t i = 0; i < src.size(); ++i)
        if (src[i]->size() != sizes[i]) { error = "conversation snapshot: invalid K/V payload size"; return false; }
    const auto dst = pools(st);
    for (size_t i = 0; i < src.size(); ++i)
        if (sizes[i] && !dst[i]) { error = "conversation snapshot: missing target state buffer"; return false; }
    for (size_t i = 0; i < src.size(); ++i)
        if (!transfer(dst[i], src[i]->data(), sizes[i], error)) return false;
    // VRAM slots still contain the outgoing conversation. Resolve must refill
    // them from the restored authoritative pools before any attention reads.
    if (st.kv_mode == 1) strata::kernels::kv_stream_reset(st.map, nullptr);
    // Ring (mode 2) slots are refilled by MtpDrafter::kv_restore at the selected
    // checkpoint, which may be earlier than this snapshot's final live position.
    return true;
}

} // namespace strata::core
