#ifndef __BERT_UTIL_H
#define __BERT_UTIL_H
#include <immintrin.h>
#include <cstdio>
#include <type_traits>

#define likely(x)       __builtin_expect((x), 1)
#define unlikely(x)     __builtin_expect((x), 0)

#define REQUIRES(assertion, message)   \
    do {                               \
        if (unlikely(!(assertion))) {  \
            printf("%s\n", (message)); \
            exit(-1);                  \
        }                              \
    } while (0)

extern "C" {
// There is no ldb, as B is packed in compact format
void igemm(uint8_t *A, int8_t *packedB, int32_t *C, int M, int N, int K, int lda, int ldc);
}

// A class for forced loop unrolling at compile time
template <int i>
struct compile_time_for {
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda& function, Args... args) {
        compile_time_for<i-1>::op(function, args...);
        function(std::integral_constant<int, i-1>{}, args...);
    }
};
template <>
struct compile_time_for<1> {
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda& function, Args... args) {
        function(std::integral_constant<int, 0>{}, args...);
    }
};
template <>
struct compile_time_for<0> {
    // 0 loops, do nothing
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda& function, Args... args) {
    }
};

class BertUtil {
   public:
    static inline __m512 vexp(const __m512& _x) {
        __m512 p16f_1 = _mm512_set1_ps(1.0f);
        __m512 p16f_half = _mm512_set1_ps(0.5f);
        __m512 p16f_127 = _mm512_set1_ps(127.f);
        __m512 p16f_exp_hi = _mm512_set1_ps(88.3762626647950f);
        __m512 p16f_exp_lo = _mm512_set1_ps(-88.3762626647949f);

        __m512 p16f_cephes_LOG2EF = _mm512_set1_ps(1.44269504088896341f);

        __m512 p16f_cephes_exp_p0 = _mm512_set1_ps(1.9875691500E-4f);
        __m512 p16f_cephes_exp_p1 = _mm512_set1_ps(1.3981999507E-3f);
        __m512 p16f_cephes_exp_p2 = _mm512_set1_ps(8.3334519073E-3f);
        __m512 p16f_cephes_exp_p3 = _mm512_set1_ps(4.1665795894E-2f);
        __m512 p16f_cephes_exp_p4 = _mm512_set1_ps(1.6666665459E-1f);
        __m512 p16f_cephes_exp_p5 = _mm512_set1_ps(5.0000001201E-1f);

        // Clamp x.
        __m512 x = _mm512_max_ps(_mm512_min_ps(_x, p16f_exp_hi), p16f_exp_lo);

        // Express exp(x) as exp(m*ln(2) + r), start by extracting
        // m = floor(x/ln(2) + 0.5).
        __m512 m = _mm512_floor_ps(_mm512_fmadd_ps(x, p16f_cephes_LOG2EF, p16f_half));

        // Get r = x - m*ln(2). If no FMA instructions are available, m*ln(2) is
        // subtracted out in two parts, m*C1+m*C2 = m*ln(2), to avoid accumulating
        // truncation errors. Note that we don't use the "pmadd" function here to
        // ensure that a precision-preserving FMA instruction is used.
        __m512 p16f_nln2 = _mm512_set1_ps(-0.6931471805599453f);
        __m512 r = _mm512_fmadd_ps(m, p16f_nln2, x);

        __m512 r2 = _mm512_mul_ps(r, r);

        // TODO(gonnet): Split into odd/even polynomials and try to exploit
        //               instruction-level parallelism.
        __m512 y = p16f_cephes_exp_p0;
        y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p1);
        y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p2);
        y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p3);
        y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p4);
        y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p5);
        y = _mm512_fmadd_ps(y, r2, r);
        y = _mm512_add_ps(y, p16f_1);

        // Build emm0 = 2^m.
        __m512i emm0 = _mm512_cvttps_epi32(_mm512_add_ps(m, p16f_127));
        emm0 = _mm512_slli_epi32(emm0, 23);

        // Return 2^m * exp(r).
        return _mm512_max_ps(_mm512_mul_ps(y, _mm512_castsi512_ps(emm0)), _x);
    }
};

#endif