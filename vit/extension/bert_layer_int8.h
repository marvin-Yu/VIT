/**
 * To get better accuracy, we use per-row scale for input tensor,
 * and per-column scale for weight tensor, the computing is like:
 * r = SUM_i[(s1*xi + 128) * (s2*wi)]
 * r = s1s2*SUM_i(xi*yi) + 128SUM_i(s2*wi)
 * SUM_i(xi*yi) = [r - 128SUM_i(s2*wi)] / s1s2
 * compensation = 128SUM_i(s2*wi) = 128SUM_i(quantizedW)
 *
 */
#ifndef BERT_LAYER_INT8_H_
#define BERT_LAYER_INT8_H_

#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <mkl.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <limits>
#include <new>
#include <string>

#include "bert_context.h"
#include "bert_quantize.h"
#include "bert_util.h"
#include "my_types.h"

class Int8BertLayer {
   public:
    Int8BertLayer(BertContext *ctx, int layerIdx) {
        this->ctx = ctx;
        this->layerIdx = layerIdx;
    }

    virtual ~Int8BertLayer() {
    }

    // Please note: the weights are transposed, and need to be transposed
    void setWeights(const float *_queryWeight, const float *_queryBias,
                    const float *_keyWeight, const float *_keyBias,
                    const float *_valueWeight, const float *_valueBias,
                    const float *_attOutWeight, const float *_attOutBias,
                    const float *_gamma1, const float *_beta1,
                    const float *_intermediateWeight, const float *_intermediateBias,
                    const float *_outputWeight, const float *_outputBias,
                    const float *_gamma2, const float *_beta2) {
        int hiddenSize = ctx->hiddenSize;
        int intermediateSize = ctx->intermediateSize;

        // Merged weights (in transposed format), dimension is like: 768*(768*3)
        hpj::Matrix<float> merged;
        merged.Resize(hiddenSize * 3, hiddenSize);
        copyWeights(merged, 0, hiddenSize, _queryWeight);
        copyWeights(merged, hiddenSize, hiddenSize * 2, _keyWeight);
        copyWeights(merged, hiddenSize * 2, hiddenSize * 3, _valueWeight);
        quantizeWeightQKV(qkvWeight, merged, qkvCompensation);

        // Merged bias
        qkvBias.Resize(hiddenSize * 3);
        memcpy(qkvBias.Data(), _queryBias, sizeof(float) * hiddenSize);
        memcpy(qkvBias.Data() + hiddenSize, _keyBias, sizeof(float) * hiddenSize);
        memcpy(qkvBias.Data() + hiddenSize * 2, _valueBias, sizeof(float) * hiddenSize);

        // Weights for attention output
        hpj::Matrix<float> attW((float *)_attOutWeight, hiddenSize, hiddenSize, hiddenSize);
        quantizeWeightT(attOutWeight, attW, attOutCompensation);
        attOutBias.Resize(hiddenSize);
        memcpy(attOutBias.Data(), _attOutBias, sizeof(float) * hiddenSize);

        // gamma and beta for batchnorm after self attention
        gamma1.Resize(hiddenSize);
        beta1.Resize(hiddenSize);
        memcpy(gamma1.Data(), _gamma1, sizeof(float) * hiddenSize);
        memcpy(beta1.Data(), _beta1, sizeof(float) * hiddenSize);

        // intermediate weight and bias
        hpj::Matrix<float> imW((float *)_intermediateWeight,
                               intermediateSize, hiddenSize, hiddenSize);
        quantizeWeightT(intermediateWeight, imW, imCompensation);
        intermediateBias.Resize(intermediateSize);
        memcpy(intermediateBias.Data(), _intermediateBias, sizeof(float) * intermediateSize);

        // output dense weight and bias
        hpj::Matrix<float> outW((float *)_outputWeight,
                                hiddenSize, intermediateSize, intermediateSize);
        quantizeWeightT(outputWeight, outW, outputCompensation);
        outputBias.Resize(hiddenSize);
        memcpy(outputBias.Data(), _outputBias, sizeof(float) * hiddenSize);

        // gamma and beta for the last batchnorm
        gamma2.Resize(hiddenSize);
        beta2.Resize(hiddenSize);
        memcpy(gamma2.Data(), _gamma2, sizeof(float) * hiddenSize);
        memcpy(beta2.Data(), _beta2, sizeof(float) * hiddenSize);

#if !defined(USE_MKL)
        packWeight(qkvWeight);
        packWeight(attOutWeight);
        packWeight(intermediateWeight);
        packWeight(outputWeight);
#endif
    }

    // Note: the weight is transposed
    static void packWeight(hpj::Matrix<s8> &weight) {
        int N = weight.Rows();
        int K = weight.Cols();
        int segs = (N + 63) / 64;

        REQUIRES(K % 4 == 0, "K must be multiple of 4");

        s8 *pw = weight.Data();
        int *buf = (int *)aligned_alloc(64, 64 * K);

        for (int s = 0; s < segs; ++s) {
            const int nstart = s * 64;
            const int nend = (nstart + 64 > N ? N : nstart + 64);

            int idx = 0;
            for (int i = 0; i < K / 4; ++i) {
                for (int j = nstart; j < nend; ++j) {
                    buf[idx] = ((int *)weight.Row(j))[i];
                    idx += 1;
                }
            }

            // Copy to weight, can do it because there is no overlap
            memcpy(pw + nstart * K, buf, (nend - nstart) * K);
        }

        free(buf);
    }

    // Do the forward computing for the whole BERT layer
    // input: (batchSize * maxTokenSize) x hidden_size
    // mask: attention mask
    // doQuant: whether quantize the output
    void forward(hpj::Matrix<float> &inputBuffer,
                 hpj::Matrix<float> &outBuffer,
                 hpj::Matrix<u8> &quantInput,
                 hpj::Matrix<u8> &quantOutput,
                 float *mask, bool doQuant = true) {
        auto hiddenSize = ctx->hiddenSize;
        auto &qkvMatMul = ctx->qkvMatMul;
        auto &resultBuffer1 = outBuffer;
        auto &resultBuffer2 = ctx->tmpBuffer;
        auto &intermediateBuffer = ctx->intermediateBuffer;

        auto &embQuantBuffer = ctx->embQuantBuffer;
        auto &imQuantBuffer = ctx->imQuantBuffer;

        // Query, Key, Value computed together
        int8_gemm(quantInput, qkvWeight, qkvMatMul);

        dequantizeAndAdd(qkvMatMul, qkvBias,
                         quantInput.Scales(), quantInput.ZeroPoint(),
                         qkvWeight.Scales(), qkvCompensation.Data());

        // BatchMatMul
        hpj::Matrix<float> query(qkvMatMul, 0, qkvMatMul.Rows(), 0, hiddenSize);
        hpj::Matrix<float> key(qkvMatMul, 0, qkvMatMul.Rows(), hiddenSize, hiddenSize);
        hpj::Matrix<float> value(qkvMatMul, 0, qkvMatMul.Rows(), hiddenSize * 2, hiddenSize);

#ifdef DEBUG
        printf("[Layer %d] self_attention_query:\n", layerIdx);
        dumpMatrix(query);
        printf("[Layer %d] self_attention_value:\n", layerIdx);
        dumpMatrix(value);
#endif

        batchMatMul(query, key, ctx->qk_result);
#ifdef DEBUG
        printf("[Layer %d] qk_result:\n", layerIdx);
        for (int i = 0; i < ctx->attHeadNum; ++i) {
            float *p = ctx->qk_result[i];
            printf("%f, %f, ..., %f\n", p[0], p[1], p[ctx->tokenSize - 1]);
        }
#endif

        // Softmax
        computeSoftmax(mask);

#ifdef DEBUG
        printf("bert/encoder/layer_%d/attention/self/Softmax:\n", layerIdx);
        printf("%f, %f, ...\n", ctx->qk_result[0][0], ctx->qk_result[0][1]);
        printf("%f, %f, ...\n", ctx->qk_result[1][0], ctx->qk_result[1][1]);
#endif

        // BatchMatMul
        batchMatMul(ctx->qk_result, value, resultBuffer1);
#ifdef DEBUG
        printf("[Layer %d] self attention:\n", layerIdx);
        dumpMatrix(resultBuffer1);
#endif

        QuantizeUtil::quantize_input(resultBuffer1.Data(), (uint8_t *)embQuantBuffer.Data(),
                                     embQuantBuffer.Scales(), embQuantBuffer.ZeroPoint(),
                                     resultBuffer1.Rows(), resultBuffer1.Cols(),
                                     resultBuffer1.Stride(), embQuantBuffer.Stride());

        // Self output (dense + LayerNorm + Quantization)
        denseWithSum_LN_Quant(embQuantBuffer, attOutWeight, attOutCompensation, attOutBias,
                              inputBuffer, resultBuffer2, embQuantBuffer,
                              gamma1, beta1);
#ifdef DEBUG
        printf("[Layer %d] self output:\n", layerIdx);
        dumpMatrix(resultBuffer2);
#endif

        // float debugFF = 0, debugFI = 0, debugIF = 0;
        // int debugII = 0;
        // uint8_t *pQuant = (uint8_t *)embQuantBuffer.Row(0);
        // for (int i = 0; i < hiddenSize; ++i) {
        //     debugFF += resultBuffer2(0, i) * debugW[i];
        //     debugFI += resultBuffer2(0, i) * intermediateWeight(0, i);
        //     debugIF += ((int)pQuant[i] - 128) * debugW[i];
        //     debugII += pQuant[i] * intermediateWeight(0, i);
        //     if (i % 64 == 0) {
        //         printf("%f(%d) * %f(%d), \n", resultBuffer2(0, i), (int)pQuant[i] - 128, debugW[i], intermediateWeight.Data()[i]);
        //     }
        // }
        // printf("IM FF, FI, IF, II: %f (%f*%f=%f) (%f*%f=%f) %d->%f\n", debugFF,
        //        debugFI, intermediateWeight.Scales()[0], debugFI * intermediateWeight.Scales()[0],
        //        debugIF, embQuantBuffer.Scales()[0], debugIF * embQuantBuffer.Scales()[0],
        //        debugII, (debugII - imCompensation.Data()[0]) * embQuantBuffer.Scales()[0] * intermediateWeight.Scales()[0]);

        // intermediate
        intermediate(embQuantBuffer, intermediateBuffer, imQuantBuffer);
#ifdef DEBUG
        printf("[Layer %d] intermediate:\n", layerIdx);
        dumpMatrix(intermediateBuffer);
#endif

        // float debugFF = 0, debugFI = 0, debugIF = 0;
        // int debugII = 0;
        // uint8_t *pQuant = (uint8_t *)imQuantBuffer.Row(0);
        // for (int i = 0; i < ctx->intermediateSize; ++i) {
        //     debugFF += intermediateBuffer(0, i) * debugW[i];
        //     debugFI += intermediateBuffer(0, i) * outputWeight(0, i);
        //     debugIF += ((int)pQuant[i] - 128) * debugW[i];
        //     debugII += pQuant[i] * outputWeight(0, i);
        //     if (i % 64 == 0) {
        //         printf("%f(%d) * %f(%d), \n", intermediateBuffer(0, i), (int)pQuant[i] - 128, debugW[i], outputWeight.Data()[i]);
        //     }
        // }
        // printf("IM FF, FI, IF, II: %f (%f*%f=%f) (%f*%f=%f) %d->%f\n", debugFF,
        //        debugFI, outputWeight.Scales()[0], debugFI * outputWeight.Scales()[0],
        //        debugIF, imQuantBuffer.Scales()[0], debugIF * imQuantBuffer.Scales()[0],
        //        debugII, (debugII - outputCompensation.Data()[0]) * imQuantBuffer.Scales()[0] * outputWeight.Scales()[0]);

        // dense in output
        denseWithSum_LN_Quant(imQuantBuffer, outputWeight, outputCompensation, outputBias,
                              resultBuffer2, outBuffer, quantOutput, gamma2, beta2);
#ifdef DEBUG
        printf("[Layer %d] output:\n", layerIdx);
        dumpMatrix(outBuffer);
#endif
    }

   private:
    void copyWeights(hpj::Matrix<float> &w, int startRow, int endRow, const float *data) {
        hpj::Matrix<float> subW(w, startRow, endRow - startRow, 0, w.Cols());
        copyWeights(subW, data);
    }

    void copyWeights(hpj::Matrix<float> &w, const float *data) {
        for (int i = 0; i < w.Rows(); ++i) {
            for (int j = 0; j < w.Cols(); ++j) {
                w(i, j) = *data++;
            }
        }
    }

    template <typename T>
    void copyTransposed(hpj::Matrix<T> &dst, hpj::Matrix<T> &src) {
        dst.Resize(src.Cols(), src.Rows());
        for (int i = 0; i < dst.Rows(); ++i) {
            for (int j = 0; j < dst.Cols(); ++j) {
                dst(i, j) = src(j, i);
            }
        }
    }

    // Quantize the weight per row (as it is already transposed, thus indeed per column)
    void quantizeWeightT(hpj::Matrix<s8> &dst, hpj::Matrix<float> &src,
                         hpj::Vector<float> &compensation, const char *debug_file = NULL) {
        dst.SetQScheme(hpj::per_tensor_symmetric);
        dst.Resize(src.Rows(), src.Cols());
        compensation.Resize(dst.Rows());

        QuantizeUtil::quantize_weight(src.Data(), dst.Data(), dst.Scales(),
                                      compensation.Data(), src.Rows(), src.Cols(),
                                      src.Stride(), dst.Stride());

        if (debug_file != NULL) {
            FILE *fp = fopen(debug_file, "w");
            if (fp) {
                dumpQTensor(fp, dst.Data(), dst.Rows(), dst.Cols(), dst.Stride());
                fclose(fp);
            }
        }
    }

    static void dumpQTensor(FILE *fp, int8_t *px, int rows, int cols, int stride) {
        for (int i = 0; i < rows; ++i) {
            int8_t *p = px + i * stride;
            for (int off = 0; off < cols; off += 12) {
                fprintf(fp, "%5d,%5d,%5d,%5d,%5d,%5d,%5d,%5d,%5d,%5d,%5d,%5d,\n", 
                    p[off], p[off + 1], p[off + 2], p[off + 3], p[off + 4], p[off + 5], 
                    p[off + 6], p[off + 7], p[off + 8], p[off + 9], p[off + 10], p[off + 11]);
            }
        }
    }

    // Special version of quantizeWeightT
    // As the qkvWeight is merged w/ 3 tensors, thus cannot use per_tensor quantization, 
    // We use per_channel quantization instead (but only has 3 scales)
    void quantizeWeightQKV(hpj::Matrix<s8> &dst, hpj::Matrix<float> &src,
                           hpj::Vector<float> &compensation) {
        dst.SetQScheme(hpj::per_channel_symmetric);
        dst.Resize(src.Rows(), src.Cols());
        compensation.Resize(dst.Rows());

        QuantizeUtil::quantize_weight_qkv(src.Data(), dst.Data(), dst.Scales(),
                                          compensation.Data(), src.Rows(), src.Cols(),
                                          src.Stride(), dst.Stride());
    }

    void dumpMatrix(hpj::Matrix<float> &m, bool dumpAll = false) {
        int cols = m.Cols();

        int i = 0;
        while (i < m.Rows()) {
            if (i > 2 && i < m.Rows() - 3) {
                printf("...\n");
                i = m.Rows() - 3;
                continue;
            }
            if (m.Cols() < 10 || dumpAll) {
                for (int j = 0; j < m.Cols(); ++j) {
                    // At most 12 elements in each line
                    if (j > 0 && j % 12 == 0) {
                        printf("\n");
                    }
                    printf("%13.5f,", m(i, j));
                }
            } else {
                std::cout << m(i, 0) << " " << m(i, 1) << " " << m(i, 2) << " ... " << m(i, cols - 3) << " " << m(i, cols - 2) << " " << m(i, cols - 1);
            }
            std::cout << std::endl;

            i += 1;
        }
    }

    // C = A * B, A need to be uint8_t, B need to be int8_t
    // Please note: B need to be transposed
#if defined(USE_MKL)
    template <typename T>
    void int8_gemm(hpj::Matrix<u8> &A, hpj::Matrix<s8> &B, hpj::Matrix<T> &C) {
        assert(B.Cols() == A.Cols());
        assert(sizeof(T) == 4);

        CBLAS_TRANSPOSE bTranspose = CblasTrans;
        int m = A.Rows();
        int k = A.Cols();
        int n = B.Rows();
        float alpha = 1;
        float beta = 0;
        int oc = 0;

        cblas_gemm_s8u8s32(CblasRowMajor,
                           CblasNoTrans, bTranspose, CblasFixOffset,
                           m, n, k, alpha,
                           A.Data(), A.Stride(), 0,
                           B.Data(), B.Stride(), 0, beta,
                           reinterpret_cast<int *>(C.Data()), C.Stride(), &oc);
    }
#else
    template <typename T>
    void int8_gemm(hpj::Matrix<u8> &A, hpj::Matrix<s8> &B, hpj::Matrix<T> &C) {
        int m = A.Rows();
        int k = A.Cols();
        int n = C.Cols();
        igemm((uint8_t *)A.Data(), (int8_t *)B.Data(),
              (int *)C.Data(), m, n, k, A.Stride(), C.Stride());
    }
#endif

    // result = LayerNorm(x * weight + bias + input)
    // quantRet = Quantize_u8(result)
    void denseWithSum_LN_Quant(hpj::Matrix<u8> &x,
                               hpj::Matrix<s8> &weight,
                               hpj::Vector<float> &comp,
                               hpj::Vector<float> &bias,
                               hpj::Matrix<float> &input,
                               hpj::Matrix<float> &result,
                               hpj::Matrix<u8> &quantRet,
                               hpj::Vector<float> &gamma,
                               hpj::Vector<float> &beta) {
        assert(input.Rows() == result.Rows());
        assert(input.Cols() == result.Cols());

        // After gemm, the result is int32
        int8_gemm(x, weight, result);

        float *pbias = bias.Data();
        const int size = result.Cols();
        float rmax = 0, rmin = 0;

#pragma omp parallel for reduction(max : rmax) reduction(min : rmin)
        for (int r = 0; r < result.Rows(); ++r) {
            float *presult = result.Row(r);
            float *pinput = input.Row(r);

            __m512 vs1 = _mm512_set1_ps(x.Scales()[0]);
            __m512 vzp = _mm512_set1_ps((float)(x.ZeroPoint()[0]));
            __m512 vs2 = _mm512_set1_ps(weight.Scales()[0]);
            float *compensation = comp.Data();

            for (int col = 0; col < size; col += 16) {
                int remain = size - col;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                // Apply compensation and dequantize
                __m512i vx = _mm512_maskz_loadu_epi32(mask, presult + col);
                __m512 vf = _mm512_cvt_roundepi32_ps(vx,
                                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                __m512 vcomp = _mm512_maskz_loadu_ps(mask, compensation + col);
                //__m512 vs2 = _mm512_maskz_loadu_ps(mask, inv_scalew + col);
                __m512 vres = (vf - vzp * vcomp) * vs1 * vs2;

                // Add input and bias
                __m512 vinput = _mm512_maskz_loadu_ps(mask, pinput + col);
                __m512 vbias = _mm512_maskz_loadu_ps(mask, pbias + col);
                vres = vres + vbias + vinput;

                // Store
                _mm512_mask_storeu_ps(presult + col, mask, vres);
            }

#ifdef DEBUG
            if (r == 0) {
                printf("[Layer %d] value before layernorm (dense & dense+input):\n", layerIdx);
            }
            if (r < 3) {
                printf("%10.6f %10.6f %10.6f ... %10.6f  |  %10.6f %10.6f %10.6f ... %10.6f\n", 
                presult[0] - pinput[0], presult[1] - pinput[1], presult[2] - pinput[2], presult[size - 1] - pinput[size - 1],
                presult[0], presult[1], presult[2], presult[size - 1]);
            } else if (r >= result.Rows() - 3) {
                printf("%10.6f %10.6f %10.6f ... %10.6f  |  %10.6f %10.6f %10.6f ... %10.6f\n", 
                presult[0] - pinput[0], presult[1] - pinput[1], presult[2] - pinput[2], presult[size - 1] - pinput[size - 1],
                presult[0], presult[1], presult[2], presult[size - 1]);
            } else if (r == 3) {
                printf("...\n");
            }
#endif

            batchnormStats(presult, gamma.Data(), beta.Data(), size, rmax, rmin);
        }  // end for r

        // Get quantization param
        auto param = QuantizeUtil::affine_quantize_param(rmax, rmin, true);
        *quantRet.Scales() = param.scale;
        *quantRet.ZeroPoint() = param.zp;

        // Do quantization
        #pragma omp parallel for
        for (int r = 0; r < result.Rows(); ++r) {
            float *px = result.Row(r);
            u8 *py = quantRet.Row(r);
            QuantizeUtil::quantize_row(px, py, size, param);
        }
    }

#if __AVX512F__
    // LayerNorm one row
    void batchnormRow(float *px, float *pgamma, float *pbeta, int size) {
        float sum = 0;
        float squareSum = 0;

        __m512 vsum = _mm512_set1_ps(0);
        __m512 vsqare = _mm512_set1_ps(0);

        for (int col = 0; col < size; col += 16) {
            int remain = size - col;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            // SUM(x)
            __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
            vsum = _mm512_add_ps(vsum, vx);

            // SUM(x*x)
            __m512 tmp = _mm512_mul_ps(vx, vx);
            vsqare = _mm512_add_ps(vsqare, tmp);
        }

        sum = _mm512_reduce_add_ps(vsum);
        squareSum = _mm512_reduce_add_ps(vsqare);

        // Mean
        float mean = sum / size;
        __m512 vmean = _mm512_set1_ps(mean);

        // Variance
        const float epsilon = 9.999999960041972e-13;
        float var = 1 / sqrt(squareSum / size - mean * mean + epsilon);
        __m512 vvar = _mm512_set1_ps(var);

        for (int col = 0; col < size; col += 16) {
            int remain = size - col;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
            __m512 vgamma = _mm512_maskz_loadu_ps(mask, pgamma + col);
            __m512 vbeta = _mm512_maskz_loadu_ps(mask, pbeta + col);
            __m512 vy = (vx - vmean) * vgamma * vvar + vbeta;
            _mm512_mask_storeu_ps(px + col, mask, vy);
        }
    }

    // LayerNorm + Stats max/min value (note: max/min values are extended to include 0)
    // px = LayerNorm(px)
    void batchnormStats(float *px, float *pgamma, float *pbeta, int size, float &rmax, float &rmin) {
        float sum = 0;
        float squareSum = 0;

        __m512 vsum = _mm512_set1_ps(0);
        __m512 vsqaresum = _mm512_set1_ps(0);
        //__m512 vsum_delta = _mm512_set1_ps(0);
        //__m512 vsqare_delta = _mm512_set1_ps(0);

        for (int col = 0; col < size; col += 16) {
            int remain = size - col;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            // SUM(x), https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            // Looks Kahan alg. is not needed, you could try it
            __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
            vsum += vx;
            // __m512 y = vx + vsum_delta;
            // __m512 s = vsum + y;
            // __m512 z = s - vsum;
            // vsum_delta = y - z;
            // vsum = s;

            // SUM(x*x)
            vx = _mm512_mul_ps(vx, vx);
            vsqaresum += vx;
            // y = vx + vsqare_delta;
            // s = vsqaresum + y;
            // z = s - vsqaresum;
            // vsqare_delta = y - z;
            // vsqaresum = s;
        }

        sum = _mm512_reduce_add_ps(vsum);
        squareSum = _mm512_reduce_add_ps(vsqaresum);

        // Mean
        float mean = sum / size;
        __m512 vmean = _mm512_set1_ps(mean);

        // Variance
        const float epsilon = 1e-12;
        float var = 1 / sqrt(squareSum / size - mean * mean + epsilon);
        __m512 vvar = _mm512_set1_ps(var);

        __m512 vmax = _mm512_set1_ps(rmax);
        __m512 vmin = _mm512_set1_ps(rmin);

        for (int col = 0; col < size; col += 16) {
            int remain = size - col;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
            __m512 vgamma = _mm512_maskz_loadu_ps(mask, pgamma + col);
            __m512 vbeta = _mm512_maskz_loadu_ps(mask, pbeta + col);
            __m512 vy = (vx - vmean) * vvar * vgamma  + vbeta;
            _mm512_mask_storeu_ps(px + col, mask, vy);

            vmax = _mm512_max_ps(vmax, vy);
            vmin = _mm512_min_ps(vmin, vy);
        }

        // Reduce the max and min value
        rmax = _mm512_reduce_max_ps(vmax);
        rmin = _mm512_reduce_min_ps(vmin);
    }

    void batchnormStatsRef(float *px, float *pgamma, float *pbeta, int size, float &rmax, float &rmin) {
        float sum = 0;

#pragma omp simd
        for (int j = 0; j < size; ++j) {
            sum += px[j];
        }
        float mean = sum / size;

        sum = 0;
#pragma omp simd
        for (int j = 0; j < size; ++j) {
            float delta = (px[j] - mean);
            sum += delta * delta;
        }
        float tmp = sum / size + 1e-12;
        float rvariance = 1.0f / sqrt(tmp);
        const float scale = rvariance;
        const float bias = -rvariance * mean;

#pragma omp simd
        for (int j = 0; j < size; ++j) {
            //px[j] = (px[j] - mean) * rvariance * pgamma[j] + pbeta[j];
            px[j] = (px[j]  * scale + bias) * pgamma[j] + pbeta[j];
            if (px[j] > rmax) rmax = px[j];
            if (px[j] < rmin) rmin = px[j];
        }
    }

    void batchnorm(hpj::Matrix<float> &x, hpj::Vector<float> &gamma, hpj::Vector<float> &beta) {
        float *pgamma = gamma.Data();
        float *pbeta = beta.Data();
        int size = x.Cols();

#pragma omp parallel for
        for (int r = 0; r < x.Rows(); ++r) {
            float *px = x.Row(r);
            batchnormRow(px, pgamma, pbeta, size);
        }
    }

#else
    void batchnorm(hpj::Matrix<float> &x, hpj::Vector<float> &gamma, hpj::Vector<float> &beta) {
        assert(x.Rows() == ctx->batchSize * ctx->tokenSize);
        assert(x.Cols() == ctx->hiddenSize);

        float *pgamma = gamma.Data();
        float *pbeta = beta.Data();

#pragma omp parallel for
        for (int i = 0; i < x.Rows(); ++i) {
            float sum = 0;
            float *px = x.Row(i);
#pragma omp simd
            for (int j = 0; j < x.Cols(); ++j) {
                sum += px[j];
            }
            float mean = sum / ctx->hiddenSize;

            sum = 0;
#pragma omp simd
            for (int j = 0; j < x.Cols(); ++j) {
                float delta = (px[j] - mean);
                sum += delta * delta;
            }
            float tmp = sum / ctx->hiddenSize + 9.999999960041972e-13;
            float rvariance = 1.0f / sqrt(tmp);

#pragma omp simd
            for (int j = 0; j < x.Cols(); ++j) {
                px[j] = (px[j] - mean) * rvariance * pgamma[j] + pbeta[j];
            }
        }
    }
#endif

    void intermediate_tanh(hpj::Matrix<u8> &input, hpj::Matrix<float> &output, hpj::Matrix<u8> &quantOut) {
        int8_gemm(input, intermediateWeight, output);

        float *pbias = intermediateBias.Data();
        float factor = 0.7978845608;  // np.sqrt(2 / np.pi)

#pragma omp parallel for
        for (int i = 0; i < output.Rows(); ++i) {
            int tid = omp_get_thread_num();

            float *pout = output.Row(i);
            const int size = output.Cols();

            // Dequantize
            float xs = input.Scales()[0];
            int32_t zp = input.ZeroPoint()[0];
            float *ws = intermediateWeight.Scales();
            QuantizeUtil::dequantize_row(pout, size, xs, zp, ws, imCompensation.Data());

#pragma omp simd
            for (int j = 0; j < size; ++j) {
                float x = pout[j] + pbias[j];
                ctx->erf_buffer[tid][j] = x;
                pout[j] = factor * (x + 0.044715f * x * x * x);
            }
            vsTanh(size, pout, pout);
#pragma omp simd
            for (int j = 0; j < size; ++j) {
                pout[j] = ctx->erf_buffer[tid][j] * 0.5f * (1 + pout[j]);
            }
        }

        // Quantize
        quantOut.SetQScheme(hpj::per_tensor_affine);
        QuantizeUtil::quantize_input(output.Data(), quantOut.Data(), quantOut.Scales(), quantOut.ZeroPoint(),
                                     output.Rows(), output.Cols(), output.Stride(), quantOut.Stride());
    }

    void intermediate(hpj::Matrix<u8> &input, hpj::Matrix<float> &output, hpj::Matrix<u8> &quantOut) {
        int8_gemm(input, intermediateWeight, output);

        float *pbias = intermediateBias.Data();
        const float factor = sqrt(0.5f);
        const float scale = 0.5f / factor;

#pragma omp parallel for
        for (int i = 0; i < output.Rows(); ++i) {
            int tid = omp_get_thread_num();

            float *pout = output.Row(i);
            const int size = output.Cols();

            // Dequantize
            float xs = input.Scales()[0];
            int32_t zp = input.ZeroPoint()[0];
            float *ws = intermediateWeight.Scales();
            QuantizeUtil::dequantize_row(pout, size, xs, zp, ws, imCompensation.Data());

#pragma omp simd
            for (int j = 0; j < output.Cols(); ++j) {
                pout[j] = (pout[j] + pbias[j]) * factor;
            }
            vsErf(output.Cols(), pout, ctx->erf_buffer[tid]);
#pragma omp simd
            for (int j = 0; j < output.Cols(); ++j) {
                pout[j] = pout[j] * scale * (ctx->erf_buffer[tid][j] + 1);
            }
        }

        // Quantize
        quantOut.SetQScheme(hpj::per_tensor_affine);
        QuantizeUtil::quantize_input(output.Data(), quantOut.Data(), quantOut.Scales(), quantOut.ZeroPoint(),
                                     output.Rows(), output.Cols(), output.Stride(), quantOut.Stride());
    }

    // The first BatchMatMul inside self attention
    void batchMatMul(hpj::Matrix<float> &A, hpj::Matrix<float> &B, float **c_array) {
#define GRP_COUNT 1
        MKL_INT m[GRP_COUNT] = {ctx->tokenSize};
        MKL_INT k[GRP_COUNT] = {ctx->attHeadSize};
        MKL_INT n[GRP_COUNT] = {ctx->tokenSize};

        MKL_INT lda[GRP_COUNT] = {A.Stride()};
        MKL_INT ldb[GRP_COUNT] = {B.Stride()};
        MKL_INT ldc[GRP_COUNT] = {ctx->tokenSize};

        CBLAS_TRANSPOSE transA[GRP_COUNT] = {CblasNoTrans};
        CBLAS_TRANSPOSE transB[GRP_COUNT] = {CblasTrans};

        float alpha[GRP_COUNT] = {1.0};
        float beta[GRP_COUNT] = {0.0};

        const int group_count = ctx->attHeadNum * ctx->batchSize;
        const MKL_INT size_per_grp[GRP_COUNT] = {group_count};

        // Total number of multiplications: attHeadNum * batchSize
        const float **a_array = new ConstFloatPointer[group_count];
        const float **b_array = new ConstFloatPointer[group_count];
        for (int b = 0; b < ctx->batchSize; ++b) {
            for (int i = 0; i < ctx->attHeadNum; ++i) {
                a_array[b * ctx->attHeadNum + i] = A.Row(b * ctx->tokenSize) + i * ctx->attHeadSize;
                b_array[b * ctx->attHeadNum + i] = B.Row(b * ctx->tokenSize) + i * ctx->attHeadSize;
            }
        }

        // Call cblas_sgemm_batch
        cblas_sgemm_batch(
            CblasRowMajor,
            transA,
            transB,
            m,
            n,
            k,
            alpha,
            a_array,
            lda,
            b_array,
            ldb,
            beta,
            c_array,
            ldc,
            GRP_COUNT,
            size_per_grp);
        delete[] a_array;
        delete[] b_array;
    }

    // The second BatchMatMul inside self attention
    void batchMatMul(float *a_array[], hpj::Matrix<float> &B, hpj::Matrix<float> &C) {
#define GRP_COUNT 1
        MKL_INT m[GRP_COUNT] = {ctx->tokenSize};
        MKL_INT k[GRP_COUNT] = {ctx->tokenSize};
        MKL_INT n[GRP_COUNT] = {ctx->attHeadSize};

        MKL_INT lda[GRP_COUNT] = {ctx->tokenSize};
        MKL_INT ldb[GRP_COUNT] = {B.Stride()};
        MKL_INT ldc[GRP_COUNT] = {C.Stride()};

        CBLAS_TRANSPOSE transA[GRP_COUNT] = {CblasNoTrans};
        CBLAS_TRANSPOSE transB[GRP_COUNT] = {CblasNoTrans};

        float alpha[GRP_COUNT] = {1.0};
        float beta[GRP_COUNT] = {0.0};

        const int group_count = ctx->attHeadNum * ctx->batchSize;
        const MKL_INT size_per_grp[GRP_COUNT] = {group_count};

        // Total number of multiplications: attHeadNum * batchSize
        const float **b_array = new ConstFloatPointer[group_count];
        float **c_array = new FloatPointer[group_count];
        for (int b = 0; b < ctx->batchSize; ++b) {
            for (int i = 0; i < ctx->attHeadNum; ++i) {
                b_array[b * ctx->attHeadNum + i] = B.Row(b * ctx->tokenSize) + i * ctx->attHeadSize;
                c_array[b * ctx->attHeadNum + i] = C.Row(b * ctx->tokenSize) + i * ctx->attHeadSize;
            }
        }

        // Call cblas_sgemm_batch
        cblas_sgemm_batch(
            CblasRowMajor,
            transA,
            transB,
            m,
            n,
            k,
            alpha,
            (const float **)a_array,
            lda,
            (const float **)b_array,
            ldb,
            beta,
            c_array,
            ldc,
            GRP_COUNT,
            size_per_grp);

        delete[] b_array;
        delete[] c_array;
    }

    // Dequantize and do BiasAdd
    // This function is tricky, the first parameter looks like float, but it is indeed int
    // scalex: input_tensor_scale (only have one value)
    // zp, zero point of the input tensor (only have one value)
    // scalew: weight_tensor_scale
    // compensation: refer to the comments at the beginning of this file
    void dequantizeAndAdd(hpj::Matrix<float> &m, hpj::Vector<float> &bias,
                          float *scalex, int32_t *zp, float *scalew, float *compensation) {
        float *pbias = bias.Data();

#pragma omp parallel for
        for (int r = 0; r < m.Rows(); ++r) {
            float *pf = m.Row(r);
            int *pi = reinterpret_cast<int *>(pf);

            __m512 vs1 = _mm512_set1_ps(scalex[0]);
            __m512 vzp = _mm512_set1_ps((float)zp[0]);

            const int size = m.Cols();
            for (int col = 0; col < size; col += 16) {
                int remain = size - col;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                // Apply compensation and dequantize
                __m512i vx = _mm512_maskz_loadu_epi32(mask, pi + col);
                __m512 vf = _mm512_cvt_roundepi32_ps(vx,
                                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                __m512 vcomp = _mm512_maskz_loadu_ps(mask, compensation + col);
                __m512 vs2 = _mm512_maskz_loadu_ps(mask, scalew + col);
                __m512 vres = (vf - vzp * vcomp) * vs1 * vs2;

                // Add bias
                __m512 vbias = _mm512_maskz_loadu_ps(mask, pbias + col);
                vres = vres + vbias;

                // Store
                _mm512_mask_storeu_ps(pf + col, mask, vres);
            }
        }
    }

    // input and output are both in qk_result
    void computeSoftmaxRef(float *mask) {
#pragma omp parallel for collapse(2)
        for (int b = 0; b < ctx->batchSize; ++b) {
            for (int i = 0; i < ctx->attHeadNum; ++i) {
                int tid = omp_get_thread_num();
                float *pbuffer = ctx->exp_buffer[tid];
                float *result = ctx->qk_result[b * ctx->attHeadNum + i];

                for (int row = 0; row < ctx->tokenSize; ++row) {
                    float sum = 0;

                    // max_val is used to avoid exp(x) = inf
                    float max_val = std::numeric_limits<float>::lowest();
#pragma omp simd
                    for (int j = 0; j < ctx->tokenSize; ++j) {
                        auto t = result[j] * ctx->attFactor + mask[b * ctx->tokenSize + j];
                        pbuffer[j] = t;
                        if (t > max_val) {
                            max_val = t;
                        }
                    }

#pragma omp simd
                    for (int j = 0; j < ctx->tokenSize; ++j) {
                        pbuffer[j] = result[j] * ctx->attFactor + mask[b * ctx->tokenSize + j] - max_val;
                    }
                    vsExp(ctx->tokenSize, pbuffer, pbuffer);
#pragma omp simd
                    for (int j = 0; j < ctx->tokenSize; ++j) {
                        sum += pbuffer[j];
                    }

                    float r_sum = 1.0f / sum;
#pragma omp simd
                    for (int j = 0; j < ctx->tokenSize; ++j) {
                        result[j] = pbuffer[j] * r_sum;
                    }

                    result += ctx->tokenSize;
                }
            }
        }
    }

    // att_mask is like [0, 0, ... 0, -10000, -10000, ..., -10000]
    // tail_mask is the mask to load the last vector
    template<int VEC_SIZE>
    void computeSoftmax(float *data, float *att_mask, __mmask16 tail_mask) {
        __m512 vx[VEC_SIZE];
        __m512 vsum = _mm512_set1_ps(0);

        // max_val is used to avoid exp(x) = inf
        float max_val = std::numeric_limits<float>::lowest();
        __m512 vmax = _mm512_set1_ps(max_val);

        compile_time_for<VEC_SIZE>::op([&] (auto i) {
            if constexpr (i == VEC_SIZE - 1) {
                vx[i] = _mm512_maskz_loadu_ps(tail_mask, data + i * 16);
            } else {
                vx[i] = _mm512_loadu_ps(data + i * 16);
            }
            vmax = _mm512_max_ps(vmax, vx[i]);
        });

        max_val = _mm512_reduce_max_ps(vmax);
        vmax = _mm512_set1_ps(max_val * ctx->attFactor);
        __m512 vfactor = _mm512_set1_ps(ctx->attFactor);

        // Compute vexp(vx - vmax) and sum it
        compile_time_for<VEC_SIZE>::op([&] (auto i) {
            if constexpr (i == VEC_SIZE - 1) {
                __m512 vmask = _mm512_maskz_loadu_ps(tail_mask, att_mask + i * 16);
                vx[i] = BertUtil::vexp(vx[i] * vfactor + vmask - vmax);
                vsum = _mm512_mask_add_ps(vsum, tail_mask, vsum, vx[i]);
            } else {
                __m512 vmask = _mm512_loadu_ps(att_mask + i * 16);
                vx[i] = BertUtil::vexp(vx[i] * vfactor + vmask - vmax);
                vsum = vsum + vx[i];
            }
        });

        float sum = _mm512_reduce_add_ps(vsum);
        float r_sum = 1.0f / sum;
        __m512 vrsum = _mm512_set1_ps(r_sum);

        // Compute exp/sum(exp) and store
        compile_time_for<VEC_SIZE>::op([&] (auto i) {
            vx[i] = vx[i] * vrsum;
            if constexpr (i == VEC_SIZE - 1) {
                _mm512_mask_storeu_ps(data + i * 16, tail_mask, vx[i]);
            } else {
                _mm512_storeu_ps(data + i * 16, vx[i]);
            }
        });
    }

    // General version
    void computeSoftmax(float *data, float *att_mask) {
        __m512 vsum = _mm512_set1_ps(0);

        // max_val is used to avoid exp(x) = inf
        float max_val = std::numeric_limits<float>::lowest();
        __m512 vmax = _mm512_set1_ps(max_val);

        for (int off = 0; off < ctx->tokenSize; off += 16) {
            int remain = ctx->tokenSize - off;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            __m512 vx = _mm512_maskz_loadu_ps(mask, data + off);
            vmax = _mm512_mask_max_ps(vmax, mask, vmax, vx);
        }

        max_val = _mm512_reduce_max_ps(vmax);
        vmax = _mm512_set1_ps(max_val * ctx->attFactor);
        __m512 vfactor = _mm512_set1_ps(ctx->attFactor);

        // Compute vexp(vx - vmax) and sum it
        for (int off = 0; off < ctx->tokenSize; off += 16) {
            int remain = ctx->tokenSize - off;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            __m512 vx = _mm512_maskz_loadu_ps(mask, data + off);
            __m512 vmask = _mm512_maskz_loadu_ps(mask, att_mask + off);
            vx = BertUtil::vexp(vx * vfactor + vmask - vmax);

            _mm512_mask_storeu_ps(data + off, mask, vx);

            vsum = _mm512_mask_add_ps(vsum, mask, vsum, vx);
        }

        float sum = _mm512_reduce_add_ps(vsum);
        __m512 vrsum = _mm512_set1_ps(1.0f / sum);

        // Compute exp/sum(exp) and store
        for (int off = 0; off < ctx->tokenSize; off += 16) {
            int remain = ctx->tokenSize - off;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            __m512 vx = _mm512_maskz_loadu_ps(mask, data + off);
            vx = vx * vrsum;

            _mm512_mask_storeu_ps(data + off, mask, vx);
        }
    }

    // input and output are both in qk_result
    void computeSoftmax(float *mask) {
#pragma omp parallel for collapse(2)
        for (int b = 0; b < ctx->batchSize; ++b) {
            for (int i = 0; i < ctx->attHeadNum; ++i) {

                float *result = ctx->qk_result[b * ctx->attHeadNum + i];
                __mmask16 tail_mask = (ctx->tokenSize % 16 == 0 ? 0xffff : (1 << (ctx->tokenSize % 16)) - 1);

                for (int row = 0; row < ctx->tokenSize; ++row) {
                    computeSoftmax(result, &mask[b * ctx->tokenSize]);
                    result += ctx->tokenSize;
                }
            }
        }
    }

   private:
    BertContext *ctx;

    // For debug usage
    int layerIdx;

    typedef float *FloatPointer;
    typedef const float *ConstFloatPointer;

    // Merged query, key, value weighs
    hpj::Matrix<s8> qkvWeight;
    // Compensation on qkvWeight
    hpj::Vector<float> qkvCompensation;
    // Merged query, key, value bias
    hpj::Vector<float> qkvBias;

    hpj::Matrix<s8> attOutWeight;
    hpj::Vector<float> attOutCompensation;
    hpj::Vector<float> attOutBias;

    // batchnorm param
    hpj::Vector<float> gamma1, beta1;
    hpj::Vector<float> gamma2, beta2;

    hpj::Matrix<s8> intermediateWeight;
    hpj::Vector<float> imCompensation;
    hpj::Vector<float> intermediateBias;

    hpj::Matrix<s8> outputWeight;
    hpj::Vector<float> outputCompensation;
    hpj::Vector<float> outputBias;
};

#endif
