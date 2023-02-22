#ifndef __BERT_QUANTIZE_H_
#define __BERT_QUANTIZE_H_
#include <immintrin.h>

#include <cstdint>

#define unlikely(x)     __builtin_expect((x), 0)

struct AffineQuantizeParam {
    double scale;
    int32_t zp;
};

template <typename QT>
struct TypeUtil {};

template <>
struct TypeUtil<uint8_t> {
    static float lowest() {
        return 0;
    }
};

template <>
struct TypeUtil<int8_t> {
    static float lowest() {
        return -128;
    }
};

class QuantizeUtil {
   private:
    static float abs_max(float *x, int size) {
        __m512 vmax = _mm512_setzero_ps();

        for (int i = 0; i < size; i += 16) {
            int remain = size - i;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
            __m512 vx = _mm512_abs_ps(_mm512_maskz_loadu_ps(mask, x + i));
            vmax = _mm512_max_ps(vx, vmax);
        }

        return _mm512_reduce_max_ps(vmax);
    }

    // Compute abs_max, and sum
    static void stats(float *x, int size, float &absmax, float &sum) {
        __m512 vmax = _mm512_setzero_ps();
        __m512 vsum = _mm512_setzero_ps();

        for (int i = 0; i < size; i += 16) {
            int remain = size - i;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
            __m512 vx = _mm512_maskz_loadu_ps(mask, x + i);
            vsum = _mm512_add_ps(vx, vsum);
            __m512 vabs = _mm512_abs_ps(vx);
            vmax = _mm512_max_ps(vabs, vmax);
        }

        absmax = _mm512_reduce_max_ps(vmax);
        sum = _mm512_reduce_add_ps(vsum);
    }

    // Stats the max & min
    // Follow some implementation in PyTorch:
    // We extend the [min, max] interval to ensure that it contains 0.
    // Otherwise, we would not meet the requirement that 0 be an exactly
    // representable value. Below code already include code like:
    // min = std::min(min, 0.f);
    // max = std::max(max, 0.f);
    static void find_max_min(float *x, int size, float &x_max, float &x_min) {
        __m512 vmax = _mm512_set1_ps(x_max);
        __m512 vmin = _mm512_set1_ps(x_min);

        for (int i = 0; i < size; i += 16) {
            int remain = size - i;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
            __m512 vx = _mm512_maskz_loadu_ps(mask, x + i);
            vmax = _mm512_max_ps(vx, vmax);
            vmin = _mm512_min_ps(vx, vmin);
        }

        x_max = _mm512_reduce_max_ps(vmax);
        x_min = _mm512_reduce_min_ps(vmin);
    }

   public:
    // Q(r) = round(r/scale)
    // Quantize without zero point (symmetric)
    static void quantize_row(float *px, int8_t *py, int size, float scale) {
        float inv_scale = 1.0f / scale;
        for (int i = 0; i < size; ++i) {
            auto v = px[i] * inv_scale;
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            py[i] = std::nearbyintf(v);
        }
    }

    // https://pytorch.org/blog/quantization-in-practice/#quantization-parameters
    // Q(r) = round(r/scale + zp)
    static void quantize_row(float *px, uint8_t *py, int size, const AffineQuantizeParam &param) {
        // float factor = 1.0f / param.scale;
        // for (int i = 0; i < size; ++i) {
        //     py[i] = std::nearbyintf(px[i] * factor + param.zp);
        // }
        float factor = 1.0f / param.scale;
        __m512 vfactor = _mm512_set1_ps(factor);
        __m512 vzp = _mm512_set1_ps((float)param.zp);
        __m512i vzero = _mm512_set1_epi32(0);

        for (int col = 0; col < size; col += 16) {
            int remain = size - col;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
            __m512 vf = vx * vfactor + vzp;
            __m512i vq = _mm512_cvt_roundps_epi32 (vf,
                                                   _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
            vq = _mm512_max_epi32(vq, vzero);
            __m128i vres = _mm512_cvtepi32_epi8(vq);

            // Store
            _mm_mask_storeu_epi8(py + col, mask, vres);
        }
    }

    // Quantize the matrix to u8 per tensor (w/ zero point)
    // Q(r) = round(r/scale + zp)
    static void quantize_input(float *x, uint8_t *y, float *scale, int32_t *zp,
                               int rows, int cols, int xstride, int ystride, bool reduce_range = true) {
        float rmax = 0, rmin = 0;

#pragma omp parallel for reduction(max : rmax) reduction(min : rmin)
        for (int r = 0; r < rows; ++r) {
            float *px = x + r * xstride;
            find_max_min(px, cols, rmax, rmin);
        }
// #pragma omp parallel
//         {
//             float rmax_private = 0;
//             float rmin_private = 0;
// #pragma omp for nowait
//             for (int r = 0; r < rows; ++r) {
//                 float *px = x + r * xstride;
//                 find_max_min(px, cols, rmax_private, rmin_private);
//             }
// #pragma omp critical
//             {
//                 if (rmax_private > rmax) rmax = rmax_private;
//                 if (rmin_private < rmin) rmin = rmin_private;
//             }
//         }

        AffineQuantizeParam param = affine_quantize_param(rmax, rmin, reduce_range);
        *scale = param.scale;
        *zp = param.zp;

#pragma omp parallel for
        for (int r = 0; r < rows; ++r) {
            float *px = x + r * xstride;
            uint8_t *py = y + r * ystride;
            quantize_row(px, py, cols, param);
        }
    }

    static AffineQuantizeParam affine_quantize_param(float max, float min, bool reduce_range = false) {
        int32_t qmax = 255;
        int32_t qmin = 0;

        if (reduce_range) {
            qmin = qmin / 2;
            qmax = qmax / 2;
        }

        // Use double precision for intermediate computation but use single precision
        // in final number to reflect the actual number used during quantization.
        double scale = (static_cast<double>(max) - min) / (qmax - qmin);

        // If scale is 0 or too small so its reciprocal is infinity, we arbitrary
        // adjust the scale to 0.1 . (Copied from PyTorch)
        if (float(scale) == 0.0f || std::isinf(1.0f / float(scale))) {
            scale = 0.1;
        }

        // Cut off small scale
        constexpr float SMALL_SCALE_THRESHOLD = 6.1e-5f;
        if (unlikely(scale < SMALL_SCALE_THRESHOLD)) {
            float org_scale = scale;
            scale = SMALL_SCALE_THRESHOLD;
            // Adjust the min and max based on the new scale
            if (min == 0.0f) {
                max = SMALL_SCALE_THRESHOLD * (qmax - qmin);
            } else if (max == 0.0f) {
                min = -SMALL_SCALE_THRESHOLD * (qmax - qmin);
            } else {
                float amplifier = SMALL_SCALE_THRESHOLD / org_scale;
                min *= amplifier;
                max *= amplifier;
            }
        }

        // Zero-point computation.
        double zero_point_from_min = qmin - min / static_cast<double>(scale);
        double zero_point_from_max = qmax - max / static_cast<double>(scale);
        double zero_point_from_min_error =
            std::abs(qmin) - std::abs(min / static_cast<double>(scale));
        double zero_point_from_max_error =
            std::abs(qmax) - std::abs(max / static_cast<double>(scale));
        double initial_zero_point =
            zero_point_from_min_error < zero_point_from_max_error
            ? zero_point_from_min
            : zero_point_from_max;

        // Now we need to nudge the zero point to be an integer
        int32_t nudged_zero_point;
        if (initial_zero_point < qmin) {
            nudged_zero_point = qmin;
        } else if (initial_zero_point > qmax) {
            nudged_zero_point = qmax;
        } else {
            nudged_zero_point = nearbyint(initial_zero_point);
        }
        
        AffineQuantizeParam param;
        param.scale = scale;
        param.zp = nudged_zero_point;

        return param;
    }

    // Code like below in PyTorch is referred:
    // max_val_pos = torch.max(-min_val_neg, max_val_pos)
    // scale = max_val_pos / (float(quant_max - quant_min) / 2)
    // scale = torch.max(scale, self.eps)
    //
    // Symmetric quantization to s8 per tensor or channel (PyTorch default is per tensor)
    // Note: the weights should always be transposed to call this func when quantize per channel
    static void quantize_weight(float *x, int8_t *y, float *scales, float *compensation,
                                int rows, int cols, int xstride, int ystride,
                                bool per_channel = false) {
        constexpr float quant_max = 127;
        constexpr float quant_min = -128;

        double scale = 1.0;

        if (!per_channel) { // quantization per tensor
            float absmax = 0;

            // TODO: make it in parallel
            for (int r = 0; r < rows; ++r) {
                float *px = x + r * xstride;
                float t = abs_max(px, cols);
                if (t > absmax) absmax = t;
            }

            scale = (absmax != 0.f ? absmax / ((quant_max - quant_min) / 2) : 1.f);

            scales[0] = (float)scale;
        }

#pragma omp parallel for
        for (int r = 0; r < rows; ++r) {
            float *px = x + r * xstride;
            int8_t *py = y + r * ystride;

            // Compute the scale for each row
            if (per_channel) {
                float absmax = abs_max(px, cols);
                scale = (absmax != 0.f ? absmax / ((quant_max - quant_min) / 2) : 1.f);

                scales[r] = (float)scale;
            }

            quantize_row(px, py, cols, scale);

            // Compensation equals the sum of all the quantized value in same row/channel
            int sum = 0;
            for (int c = 0; c < cols; ++c) {
                sum += py[c];
            }

            compensation[r] = sum;
        }
    }

    // Special version of quantize_weight
    // Make the tensor has 3 scales (thus not quantize per tensor nor per channel)
    static void quantize_weight_qkv(float *x, int8_t *y, float *scales, float *compensation,
                                    int rows, int cols, int xstride, int ystride) {
        constexpr float quant_max = 127;
        constexpr float quant_min = -128;

        // Scale value for query/key/value tensors
        double qkv_scales[3] = { 1.0 };

        for (int i = 0; i < 3; ++i) {
            const int start = i * (rows / 3);
            const int end = (i + 1) * (rows / 3);

            float absmax = 0;
            for (int r = start; r < end; ++r) {
                float *px = x + r * xstride;
                float t = abs_max(px, cols);
                if (t > absmax) absmax = t;
            }

            qkv_scales[i] = (absmax != 0.f ? absmax / ((quant_max - quant_min) / 2) : 1.f);
        }

#pragma omp parallel for
        for (int r = 0; r < rows; ++r) {
            float *px = x + r * xstride;
            int8_t *py = y + r * ystride;

            // Record the scale for each row
            int idx = r / (rows / 3);
            scales[r] = (float)qkv_scales[idx];

            quantize_row(px, py, cols, (float)qkv_scales[idx]);

            // Compensation equals the sum of all the quantized value in same row/channel
            int sum = 0;
            for (int c = 0; c < cols; ++c) {
                sum += py[c];
            }

            compensation[r] = sum;
        }
    }

    // Inplace dequantize (int -> float)
    // xs: input scale
    // ws: weight scales
    static void dequantize_row(void *data, const int size,
                               const float xs, const int32_t zp, 
                               const float *ws, const float *compensation) {
        int *pi = reinterpret_cast<int *>(data);

        __m512 vs1 = _mm512_set1_ps(xs);
        __m512 vs2 = _mm512_set1_ps(ws[0]);
        __m512 vzp = _mm512_set1_ps((float)zp);

        for (int col = 0; col < size; col += 16) {
            int remain = size - col;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            // Apply compensation and dequantize
            __m512i vx = _mm512_maskz_loadu_epi32(mask, pi + col);
            __m512 vf = _mm512_cvt_roundepi32_ps(vx,
                                                 _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512 vcomp = _mm512_maskz_loadu_ps(mask, compensation + col);
            //__m512 vs2 = _mm512_maskz_loadu_ps(mask, ws + col);
            __m512 vres = (vf - vzp * vcomp) * vs1 * vs2;

            // Store
            _mm512_mask_storeu_ps(pi + col, mask, vres);
        }
    }
};

#endif