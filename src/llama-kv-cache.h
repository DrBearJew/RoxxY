#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"
#include "llama-mtp-qblock-paged-state.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;
struct llama_kv_cache_direct_tx;
struct ggml_cuda_mtp_qblock_tail_page_map_v1;

// Consumer-requested V layout for get_v() overload.
// DEFAULT delegates to the legacy v_trans heuristic.
// FOR_FA forces the D-contiguous (FlashAttention-compatible) view.
// FOR_NON_FA forces the transposed (legacy ggml_mul_mat) view.
enum llama_kv_v_layout_request {
    LLAMA_KV_V_LAYOUT_DEFAULT = 0,
    LLAMA_KV_V_LAYOUT_FOR_FA,
    LLAMA_KV_V_LAYOUT_FOR_NON_FA,
};

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    llama_kv_cache(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
                         bool   is_mtp_draft = false);

    ~llama_kv_cache() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    llama_kv_cache * get_direct_kv_cache();

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    bool seq_import_physical(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, size_t * bytes_copied = nullptr, size_t * cells_copied = nullptr, const char ** reason = nullptr);
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;
    bool get_implicit_causal_mask_meta(const slot_info & sinfo, const llama_ubatch * ubatch, bool causal_attn, int32_t meta[4]) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo, llama_kv_v_layout_request layout) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    // sanctioned direct KV metadata transaction API for fail-closed direct verifiers.
    bool direct_tx_begin(const llama_ubatch & ubatch, llama_kv_cache_direct_tx & tx);
    bool direct_tx_apply_metadata(const llama_ubatch & ubatch, llama_kv_cache_direct_tx & tx);
    bool direct_tx_validate_metadata(const llama_kv_cache_direct_tx & tx) const;
    void direct_tx_rollback(llama_kv_cache_direct_tx & tx);
    void direct_tx_commit(llama_kv_cache_direct_tx & tx);

    // Opt-in QBlock tail-page consumer map publication. These helpers bind only
    // validated packed16 sidecar views and fail closed by clearing stale maps.
    bool register_mtp_qblock_tail_page_map(const ggml_cuda_mtp_qblock_tail_page_map_v1 & map) const;
    bool register_mtp_qblock_tail_page_map_from_commit(
            llama_pos logical_base_token,
            uint32_t accepted_tokens,
            uint64_t generation,
            ggml_cuda_mtp_qblock_tail_page_map_v1 * out_map = nullptr,
            const char ** reason = nullptr) const;
    bool register_mtp_qblock_tail_page_map_from_active_producer(
            llama_pos logical_base_token,
            uint32_t accepted_tokens,
            uint64_t generation,
            ggml_cuda_mtp_qblock_tail_page_map_v1 * out_map = nullptr,
            const char ** reason = nullptr) const;
    bool snapshot_mtp_qblock_tail_page_map_from_active_producer(
            llama_pos logical_base_token,
            uint32_t accepted_tokens,
            ggml_cuda_mtp_qblock_tail_page_map_v1 * out_map = nullptr,
            const char ** reason = nullptr) const;
    bool register_mtp_qblock_tail_page_map_from_producer_snapshot(
            llama_pos logical_base_token,
            uint32_t accepted_tokens,
            uint64_t generation,
            const ggml_cuda_mtp_qblock_tail_page_map_v1 & producer_map,
            ggml_cuda_mtp_qblock_tail_page_map_v1 * out_map = nullptr,
            const char ** reason = nullptr) const;
    bool register_mtp_qblock_tail_page_map_from_paged_state(
            ggml_cuda_mtp_qblock_tail_page_map_v1 * out_map = nullptr,
            const char ** reason = nullptr,
            uint32_t flags = 0) const;
    void clear_mtp_qblock_tail_page_maps(const char * reason = "unspecified", bool data_invalidates = true) const;

    // Cache-owned QBlock paged-KV metadata scaffold. This is the durable owner
    // state that future completed QBlock PagedAttention commits promote into;
    // current FA publication remains fail-closed and default-off.
    bool init_mtp_qblock_paged_state(llama_pos logical_base_token, uint32_t physical_pages, const char ** reason = nullptr) const;
    void clear_mtp_qblock_paged_state() const;
    bool get_mtp_qblock_paged_state(llama_mtp_qblock_paged_state_v1 * out_state) const;
    bool mtp_qblock_paged_state_alloc_txn_page(uint32_t * out_page, const char ** reason = nullptr) const;
    bool mtp_qblock_paged_state_claim_txn_page(uint32_t physical_page, uint32_t * out_slot = nullptr, const char ** reason = nullptr) const;
    bool mtp_qblock_paged_state_commit_pages(
            uint32_t valid_tail_tokens,
            const int32_t * block_table,
            uint32_t block_table_pages,
            uint32_t final_state_slot,
            const char ** reason = nullptr) const;
    bool mtp_qblock_paged_state_rollback_txn_pages(const char ** reason = nullptr) const;

    // Direct graph write helpers. These only build write nodes; callers own graph allocation/compute/rollback ordering.
    ggml_tensor * direct_cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const llama_kv_cache_direct_tx & tx) const;
    ggml_tensor * direct_cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const llama_kv_cache_direct_tx & tx) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask     (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_kq_mask_meta(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket  (ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        // PDMQ compressed K cache (standard: packed16_q8; q8_0 K request: packed8_q4).
        // Stores quantized K in I32 payload + F16 scales, ready for DOT4/PDMQ FA kernels.
        // packed8_q4 uses D/8 payload words per token/KV-head; packed16_q8 uses D/4.
        // packed16 physical layout is selected by GGML_CUDA_ROCM_PACKED16_K_LAYOUT: row, tile16/native, or page16_d16.
        ggml_tensor * k_payload = nullptr;  // GGML_TYPE_I32, D/8 or D/4 words per token/KV-head
        ggml_tensor * k_scales  = nullptr;  // GGML_TYPE_F16, D/32 scales per token/KV-head

        // Experimental sealed V4_K16D16 V cache (default-off, FA-only).
        ggml_tensor * v4_tail = nullptr;    // GGML_TYPE_F16, [D, 16 * n_heads]

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        // PDMQ K stream views (same view offset as k_stream/v_stream)
        std::vector<ggml_tensor *> k_payload_stream;
        std::vector<ggml_tensor *> k_scales_stream;
    };

    bool try_register_mtp_qblock_owned_tail_write_reservation(
            const slot_info & sinfo,
            const ggml_tensor * k_cur,
            const ggml_tensor * k_idxs,
            const kv_layer & layer) const;

    bool register_mtp_qblock_full_current_k_page_map(
            const ggml_tensor * k_view,
            uint32_t n_kv,
            uint32_t kv_size_total,
            const slot_info & sinfo,
            const char ** reason = nullptr) const;

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    bool attn_rot_k = false;
    bool attn_rot_v = false;

    // env: GGML_VK_TBQ4_D6_Q4K_ROT_K128
    // Experimental D6 metadata/proof mode: K cache is Q4_0 in explicit ROT_K128 domain.
    bool d6_q4k_rot_k128 = false;
    int32_t attn_rot_k_order = 0;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    std::vector<llama_kv_cells> v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    mutable llama_mtp_qblock_paged_state_v1 mtp_qblock_paged_state;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

struct llama_kv_cache_direct_tx {
    llama_seq_id seq_id = -1;
    llama_pos    p0     = -1;
    llama_pos    p1     = -1;

    llama_kv_cache::slot_info sinfo;
    std::vector<uint32_t> v_heads_old;
    std::vector<llama_kv_cells> v_cells_old;

    bool begun              = false;
    bool rollback_ready     = false;
    bool target_cells_empty = false;
    bool applied_metadata   = false;
    bool committed          = false;
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;
    bool get_implicit_causal_mask_meta(const llama_ubatch * ubatch, bool causal_attn, int32_t meta[4]) const;

    // last position recorded in the cache for this sequence; -1 if absent.
    // exposed for cross-context KV consumers (e.g. MTP draft) that need to
    // anchor the source position without owning a memory module of their own.
    llama_pos seq_pos_max(llama_seq_id seq_id) const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, llama_kv_v_layout_request layout) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift     (ggml_tensor * dst) const;
    void set_input_kq_mask     (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_kq_mask_meta(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket  (ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
};

// Packed16 K cache registry (shared between KV-cache ctor and DOT4 FA dispatch).
// Defined in ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu.  The metadata call binds
// sidecar bytes to their producer-selected physical layout/generation.
#ifndef GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_V1_DEFINED
#define GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_V1_DEFINED
static constexpr uint32_t GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_VERSION = 1;
static constexpr uint32_t GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_MAX_PAGES = 4;
static constexpr uint32_t GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_SCRATCH_OVERLAY = 1u << 0;
static constexpr uint32_t GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_OWNED_TAIL_WRITE = 1u << 1;
struct ggml_cuda_mtp_qblock_tail_page_map_v1 {
    uint32_t version = 0;
    uint32_t abi_bytes = 0;
    uint32_t active = 0;
    uint32_t flags = 0;
    uint32_t logical_base_token = 0;
    uint32_t valid_tail_tokens = 0;
    uint32_t page_tokens = 0;
    uint32_t physical_pages = 0;
    uint32_t block_table_pages = 0;
    int32_t block_table[GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_MAX_PAGES] = {};
    uint64_t generation = 0;
};
#endif
#ifndef GGML_CUDA_MTP_QBLOCK_FULL_PAGE_MAP_V1_DEFINED
#define GGML_CUDA_MTP_QBLOCK_FULL_PAGE_MAP_V1_DEFINED
static constexpr uint32_t GGML_CUDA_MTP_QBLOCK_FULL_PAGE_MAP_VERSION = 1;
static constexpr uint32_t GGML_CUDA_MTP_QBLOCK_FULL_PAGE_MAP_FLAG_IDENTITY = 1u << 0;
static constexpr uint32_t GGML_CUDA_MTP_QBLOCK_FULL_PAGE_MAP_FLAG_OWNED_TAIL_OVERLAY = 1u << 1;
struct ggml_cuda_mtp_qblock_full_page_map_v1 {
    uint32_t version = 0;
    uint32_t abi_bytes = 0;
    uint32_t active = 0;
    uint32_t flags = 0;
    uint32_t logical_base_token = 0;
    uint32_t valid_tokens = 0;
    uint32_t page_tokens = 0;
    uint32_t physical_pages = 0;
    uint32_t block_table_pages = 0;
    uint32_t non_identity_page_begin = 0;
    uint32_t non_identity_page_end = 0;
    const int32_t * block_table = nullptr;
    int32_t debug_first_pages[4] = {};
    uint64_t generation = 0;
};
#endif
#ifndef GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_DISPATCH_BIND_V1_DEFINED
#define GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_DISPATCH_BIND_V1_DEFINED
static constexpr uint32_t GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_DISPATCH_BIND_VERSION = 1;
static constexpr uint32_t GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_DISPATCH_BIND_NODE_NAME_MAX = 96;
struct ggml_cuda_mtp_qblock_tail_page_dispatch_bind_v1 {
    uint32_t version = 0;
    uint32_t abi_bytes = 0;
    uint32_t active = 0;
    uint32_t reserved = 0;
    ggml_cuda_mtp_qblock_tail_page_map_v1 map = {};
    uint64_t bind_count = 0;
    int32_t layer = -1;
    int32_t graph_inst = -1;
    int32_t nk = 0;
    char node_name[GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_DISPATCH_BIND_NODE_NAME_MAX] = {};
};
#endif
#ifdef __cplusplus
extern "C" {
#endif
struct ggml_tensor;
void llama_kv_cache_register_packed16(const void * k_view_data, struct ggml_tensor * payload, struct ggml_tensor * scales);
void llama_kv_cache_register_packed16_with_layout_info(const void * k_view_data, struct ggml_tensor * payload, struct ggml_tensor * scales, int layout_kind, uint32_t kv_capacity, uint32_t d);
void llama_kv_cache_register_pdmq_k_with_layout_info(const void * k_view_data, struct ggml_tensor * payload, struct ggml_tensor * scales, int k_format, int layout_kind, uint32_t kv_capacity, uint32_t d);
void llama_kv_cache_register_packed16_shadow(const void * k_view_data, struct ggml_tensor * payload, struct ggml_tensor * scales, struct ggml_tensor * shadow_k);
void llama_kv_cache_get_packed16_tensors(const void * k_view_data, struct ggml_tensor ** payload, struct ggml_tensor ** scales);
void llama_kv_cache_get_packed16_metadata(const void * k_view_data, int * layout_kind, unsigned long long * generation);
void llama_kv_cache_get_packed16_shadow_k(const void * k_view_data, struct ggml_tensor ** shadow_k);
void llama_kv_cache_register_v4_k16d16(const void * v_view_data, struct ggml_tensor * v_cache, struct ggml_tensor * v_tail);
void llama_kv_cache_get_v4_k16d16_tensors(const void * v_view_data, struct ggml_tensor ** v_cache, struct ggml_tensor ** v_tail);
void llama_kv_cache_register_mtp_qblock_tail_page_map(const void * k_view_data, const struct ggml_cuda_mtp_qblock_tail_page_map_v1 * map);
void llama_kv_cache_clear_mtp_qblock_tail_page_map(const void * k_view_data);
void llama_kv_cache_get_mtp_qblock_tail_page_map(const void * k_view_data, struct ggml_cuda_mtp_qblock_tail_page_map_v1 * map);
bool llama_kv_cache_get_mtp_qblock_tail_page_published_map(struct ggml_cuda_mtp_qblock_tail_page_map_v1 * map);
void llama_kv_cache_register_mtp_qblock_full_page_map_host(const void * k_view_data, const struct ggml_cuda_mtp_qblock_full_page_map_v1 * map, const int32_t * host_block_table);
void llama_kv_cache_clear_mtp_qblock_full_page_map(const void * k_view_data);
void llama_kv_cache_get_mtp_qblock_full_page_map(const void * k_view_data, struct ggml_cuda_mtp_qblock_full_page_map_v1 * map);
void llama_kv_cache_record_mtp_qblock_tail_page_dispatch_bind(const void * k_view_data, const struct ggml_cuda_mtp_qblock_tail_page_map_v1 * map, const char * node_name, int layer, int graph_inst, int nk);
void llama_kv_cache_note_mtp_qblock_tail_page_pending_dispatch_bind(const struct ggml_cuda_mtp_qblock_tail_page_map_v1 * map, uint64_t req_begin, uint64_t req_end, int slot);
void llama_kv_cache_note_mtp_qblock_tail_page_route_expected(const struct ggml_cuda_mtp_qblock_tail_page_map_v1 * map, uint32_t expected_layer_count);
bool llama_kv_cache_get_mtp_qblock_tail_page_last_dispatch_bind(struct ggml_cuda_mtp_qblock_tail_page_dispatch_bind_v1 * out);
void llama_kv_cache_reset_mtp_qblock_tail_page_lifecycle(bool data_invalidates);
void llama_kv_cache_reset_mtp_qblock_tail_page_lifecycle_preserve_snapshot(bool data_invalidates, bool keep_producer_snapshot);
#ifdef __cplusplus
}
#endif
