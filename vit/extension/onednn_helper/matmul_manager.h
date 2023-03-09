#ifndef __MATMUL_MANAGER_H
#define __MATMUL_MANAGER_H
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

#include "dnnl.hpp"
#include "dnnl_util.h"

class MatMulManager {
   private:
    union Signature {
        struct Detail {
            int m : 16;
            int n : 16;
            int k : 16;
            int ta : 3;    // type of A
            int tb : 3;    // type of B
            int tc : 3;    // type of C
            int bias : 1;  // with bias or not
            int reserved : 6;
        } detail;
        uint64_t combined;
    };

    struct CachedObjects {
        dnnl::matmul *prim;
        dnnl::matmul::primitive_desc *primDesc;

        // Reordered weights, orignal weight addr -> packed memory object
        std::unordered_map<void *, dnnl::memory *> reorderedWs;

        CachedObjects(dnnl::matmul *p, dnnl::matmul::primitive_desc *pd) : prim(p), primDesc(pd) {}
    };

    // Signature -> primitive, primitive description, and packed weights
    std::unordered_map<uint64_t, CachedObjects> caches;

    dnnl::engine eng;
    dnnl::stream stm;

   private:
    MatMulManager() : eng(dnnl::engine::kind::cpu, 0),
                      stm(eng) {}

    ~MatMulManager() {
        for (const auto &pair : caches) {
            const CachedObjects &objs = pair.second;
            delete objs.primDesc;
            delete objs.prim;
            for (const auto &p : objs.reorderedWs) {
                delete p.second;
            }
        }
        caches.clear();
    }

    MatMulManager(MatMulManager const &) = delete;
    void operator=(MatMulManager const &) = delete;

    void dumpSignature(Signature &sig) {
        printf("m,n,k = %d,%d,%d; type of a,b,c = %d,%d,%d; bias=%d\n",
               sig.detail.m, sig.detail.n, sig.detail.k, 
               sig.detail.ta, sig.detail.tb, sig.detail.tc, sig.detail.bias);
    }

    // Get data type defined in oneDNN
    template <typename T>
    dnnl::memory::data_type getDataType() {
        if (std::is_floating_point<T>::value) {
            return dnnl::memory::data_type::f32;
        } else if (std::is_same<T, int32_t>::value) {
            return dnnl::memory::data_type::s32;
        } else if (std::is_same<T, uint8_t>::value) {
            return dnnl::memory::data_type::u8;
        } else if (std::is_same<T, int8_t>::value) {
            return dnnl::memory::data_type::s8;
        } else {
            return dnnl::memory::data_type::bf16;
        }
    }

    // Get signature according to data types and shapes
    template <typename T_input, typename T_wei, typename T_output>
    Signature getSignature(const T_input *A, const T_wei *B, T_output *C, int M, int N, int K) {
        Signature sig;

        sig.detail.m = M;
        sig.detail.n = N;
        sig.detail.k = K;
        sig.detail.ta = (int)getDataType<T_input>();
        sig.detail.tb = (int)getDataType<T_wei>();
        sig.detail.tc = (int)getDataType<T_output>();
        sig.detail.bias = 0;
        sig.detail.reserved = 0;

        return sig;
    }

    template <typename T_input, typename T_wei, typename T_output>
    CachedObjects &getCachedObjects(
        Signature &sig, const T_input *A, const T_wei *B, T_output *C, int M, int N, int K) {
        auto it = caches.find(sig.combined);

        // Create the primitive if not found
        if (it == caches.end()) {
            dnnl::memory::dims src_tz = {M, K};
            dnnl::memory::dims weights_tz = {K, N};
            dnnl::memory::dims dst_tz = {M, N};

            dnnl::memory::data_type src_dt = getDataType<T_input>();
            dnnl::memory::data_type weights_dt = getDataType<T_wei>();
            dnnl::memory::data_type dst_dt = getDataType<T_output>();

            // Create memory descriptor primitive
            auto src_md = dnnl::memory::desc({src_tz}, src_dt, dnnl::memory::format_tag::ab);
            auto weights_md = dnnl::memory::desc({weights_tz}, weights_dt, dnnl::memory::format_tag::any);
            auto dst_md = dnnl::memory::desc({dst_tz}, dst_dt, dnnl::memory::format_tag::ab);

            // auto desc = dnnl::matmul::desc();
            auto prim_desc = new dnnl::matmul::primitive_desc(eng, src_md, weights_md, dst_md);
            auto prim = new dnnl::matmul(*prim_desc);

            CachedObjects bundle(prim, prim_desc);

            auto insertedIt = caches.insert({sig.combined, bundle}).first;

            // Do not return bundle as we need a reference
            return insertedIt->second;
        }

        // Found
        return it->second;
    }

    template <typename T_input, typename T_wei, typename T_output>
    dnnl::memory *getPackedWeight(
        CachedObjects &bundle, const T_input *A, const T_wei *B, T_output *C,
        int M, int N, int K, bool transB = false) {
        auto it = bundle.reorderedWs.find((void *)B);

        // The weight is not packed yet
        if (it == bundle.reorderedWs.end()) {
            T_wei *regularB = (T_wei *)B;

            // Transpose the weight to make it in the non-transposed layout
            // TODO: to optimize
            if (transB) {
                regularB = (T_wei *)aligned_alloc(64, K * N * sizeof(T_wei));
                for (int i = 0; i < K; ++i) {
                    for (int j = 0; j < N; ++j) {
                        // regularB[i,j] = B[j,i]
                        regularB[i * N + j] = B[j * K + i];
                    }
                }
            }

            dnnl::memory::dims weights_tz = {K, N};
            dnnl::memory::data_type weights_dt = getDataType<T_wei>();

            auto user_weights_md = dnnl::memory::desc(weights_tz, weights_dt, dnnl::memory::format_tag::ab);
            auto user_weights_memory = dnnl::memory(user_weights_md, eng, (void *)regularB);

            auto weights_memory = new dnnl::memory(bundle.primDesc->weights_desc(), eng);

            // reorder the weight if needed
            if (bundle.primDesc->weights_desc() != user_weights_memory.get_desc()) {
                auto reorder_weights = dnnl::reorder(user_weights_memory, *weights_memory);
                reorder_weights.execute(stm, {{DNNL_ARG_FROM, user_weights_memory},
                                              {DNNL_ARG_TO, *weights_memory}});
            } else {
                write_to_dnnl_memory((void *)regularB, *weights_memory);
            }

            bundle.reorderedWs.insert({(void *)B, weights_memory});

            // Free the temporary buffer
            if (transB) {
                free(regularB);
            }

            return weights_memory;
        }

        return it->second;
    }

   public:
    static MatMulManager &instance() {
        static MatMulManager manager;
        return manager;
    }

    // Get primitive and primitive desc, will firstly try to get from cache, create one if not found
    template <typename T_input, typename T_wei, typename T_output>
    std::pair<dnnl::matmul *, dnnl::matmul::primitive_desc *> getPrimitive(
        const T_input *A, const T_wei *B, T_output *C, int M, int N, int K) {
        Signature sig = getSignature(A, B, C, M, N, K);
        auto &bundle = getCachedObjects(sig, A, B, C, M, N, K);
        return std::make_pair(bundle.prim, bundle.primDesc);
    }

    // Get the reordered weight memory, will firstly try to get from cache, reorder it if not found
    template <typename T_input, typename T_wei, typename T_output>
    dnnl::memory *getWeightMemory(const T_input *A, const T_wei *B, T_output *C, int M, int N, int K) {
        Signature sig = getSignature(A, B, C, M, N, K);
        auto &bundle = getCachedObjects(sig, A, B, C, M, N, K);
        auto weightMem = getPackedWeight(bundle, A, B, C, M, N, K);
        return weightMem;
    }

    // Execute the matmul by providing primitive and reordered weight memory
    template <typename T_input, typename T_output>
    void execute(dnnl::matmul *prim, dnnl::matmul::primitive_desc *primDesc,
                 const T_input *A, dnnl::memory *weightMem, T_output *C, int M, int N, int K) {
        auto src_memory = dnnl::memory(primDesc->src_desc(), eng, (void *)A);
        auto dst_memory = dnnl::memory(primDesc->dst_desc(), eng, (void *)C);

        // Input is not reordered as we assume our primitive was created with the same params

        prim->execute(stm, {{DNNL_ARG_SRC, src_memory},
                            {DNNL_ARG_WEIGHTS, *weightMem},
                            {DNNL_ARG_DST, dst_memory}});

        stm.wait();
    }

    // Do matmul w/ BLAS-like interface
    template <typename T_input, typename T_wei, typename T_output>
    void doMatMul(const T_input *A, const T_wei *B, T_output *C, int M, int N, int K, bool transB = false) {
        Signature sig = getSignature(A, B, C, M, N, K);
        auto &bundle = getCachedObjects(sig, A, B, C, M, N, K);
        auto weightMem = getPackedWeight(bundle, A, B, C, M, N, K, transB);
        execute(bundle.prim, bundle.primDesc, A, weightMem, C, M, N, K);
    }
};

#endif