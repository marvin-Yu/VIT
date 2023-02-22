#ifndef BERT_LAYER_BATCH_H_
#define BERT_LAYER_BATCH_H_

#include <new>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mkl.h>
#include <omp.h>
#include <iostream>
#include <immintrin.h>
#include "my_types.h"
#include "bert_context.h"
#include "bert_util.h"
#include "sgemm.h"

class BatchBertLayer
{
public:
    BatchBertLayer(BertContext *ctx, int layerIdx) {
        this->ctx = ctx;
        this->layerIdx = layerIdx;
    }

    virtual ~BatchBertLayer() {
    }

    void setWeights(const float *_queryWeight, const float *_queryBias,
                    const float *_keyWeight, const float *_keyBias,
                    const float *_valueWeight, const float *_valueBias,
                    const float *_attentionOutputWeight, const float *_attentionOutputBias,
                    const float *_gamma1, const float *_beta1,
                    const float *_intermediateWeight, const float *_intermediateBias,
                    const float *_outputWeight, const float *_outputBias,
                    const float *_gamma2, const float *_beta2) {
        int hiddenSize = ctx->hiddenSize;
        int intermediateSize = ctx->intermediateSize;

        // Merged weights, dimension is like: 768*(768*3)
        hpj::Matrix<float> tmp;

        tmp.Resize(hiddenSize, hiddenSize * 3);
        copyWeights(tmp, 0, hiddenSize, _queryWeight);
        copyWeights(tmp, hiddenSize, hiddenSize*2, _keyWeight);
        copyWeights(tmp, hiddenSize*2, hiddenSize*3, _valueWeight);
#if USE_MKL
        copyTransposed(qkvWeight, tmp);
#else
        hpj::Matrix<float> mergedW;
        copyTransposed(mergedW, tmp);

        qkvWeight.Resize(hiddenSize, hiddenSize * 3);
        ig_sgemm_packb(mergedW.Data(), qkvWeight.Data(), 
                       hiddenSize, 3 * hiddenSize, mergedW.Stride(), true);
#endif
        /*
        qkvWeight.Resize(hiddenSize, hiddenSize * 3);
        copyWeights(qkvWeight, 0, hiddenSize, _queryWeight);
        copyWeights(qkvWeight, hiddenSize, hiddenSize*2, _keyWeight);
        copyWeights(qkvWeight, hiddenSize*2, hiddenSize*3, _valueWeight);
        */

        // Merged bias
        qkvBias.Resize(hiddenSize * 3);
        memcpy(qkvBias.Data(), _queryBias, sizeof(float) * hiddenSize);
        memcpy(qkvBias.Data() + hiddenSize, _keyBias, sizeof(float) * hiddenSize);
        memcpy(qkvBias.Data() + hiddenSize*2, _valueBias, sizeof(float) * hiddenSize);

        // Weights for attention output
        attentionOutputWeight.Resize(hiddenSize, hiddenSize);
#if USE_MKL
        copyWeights(attentionOutputWeight, _attentionOutputWeight);
#else
        ig_sgemm_packb((float *)_attentionOutputWeight, attentionOutputWeight.Data(),
                       hiddenSize, hiddenSize, hiddenSize, true);
#endif
        attentionOutputBias.Resize(hiddenSize);
        memcpy(attentionOutputBias.Data(), _attentionOutputBias, sizeof(float) * hiddenSize);

        // gamma and beta for batchnorm after self attention
        gamma1.Resize(hiddenSize);
        beta1.Resize(hiddenSize);
        memcpy(gamma1.Data(), _gamma1, sizeof(float) * hiddenSize);
        memcpy(beta1.Data(), _beta1, sizeof(float) * hiddenSize);

        // intermediate weight and bias
        intermediateWeight.Resize(hiddenSize, intermediateSize);
#if USE_MKL
        copyWeights(intermediateWeight, _intermediateWeight);
#else
        ig_sgemm_packb((float *)_intermediateWeight, intermediateWeight.Data(),
                       hiddenSize, intermediateSize, hiddenSize, true);
#endif
        intermediateBias.Resize(intermediateSize);
        memcpy(intermediateBias.Data(), _intermediateBias, sizeof(float) * intermediateSize);

        // output dense weight and bias
        outputWeight.Resize(intermediateSize, hiddenSize);
#if USE_MKL
        copyWeights(outputWeight, _outputWeight);
#else
        ig_sgemm_packb((float *)_outputWeight, outputWeight.Data(),
                       intermediateSize, hiddenSize, intermediateSize, true);
#endif
        outputBias.Resize(hiddenSize);
        memcpy(outputBias.Data(), _outputBias, sizeof(float) * hiddenSize);

        // gamma and beta for the last batchnorm
        gamma2.Resize(hiddenSize);
        beta2.Resize(hiddenSize);
        memcpy(gamma2.Data(), _gamma2, sizeof(float) * hiddenSize);
        memcpy(beta2.Data(), _beta2, sizeof(float) * hiddenSize);
    }

    // Do the forward computing for the whole BERT layer
    // input: (batchSize * maxTokenSize) x hidden_size
    // mask: attention mask
    void forward(hpj::Matrix<float> &inputBuffer, 
                 hpj::Matrix<float> &outBuffer, float *mask) {
        auto hiddenSize = ctx->hiddenSize;
        auto& qkvMatMul = ctx->qkvMatMul;
        auto& resultBuffer1 = outBuffer;
        auto& resultBuffer2 = ctx->tmpBuffer;
        auto& intermediateBuffer = ctx->intermediateBuffer;

        // Query, Key, Value computed together
        dense(inputBuffer, qkvWeight, qkvBias, qkvMatMul);

        // BatchMatMul
        hpj::Matrix<float> query(qkvMatMul, 0, qkvMatMul.Rows(), 0, hiddenSize);
        hpj::Matrix<float> key(qkvMatMul, 0, qkvMatMul.Rows(), hiddenSize, hiddenSize);
        hpj::Matrix<float> value(qkvMatMul, 0, qkvMatMul.Rows(), hiddenSize*2, hiddenSize);

        batchMatMul(query, key, ctx->qk_result);
       
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
        printf("bert/encoder/layer_%d/attention/self/Reshape_3:\n", layerIdx);
        dumpMatrix(resultBuffer1);
#endif

        // dense
        denseWithSum(resultBuffer1, attentionOutputWeight, attentionOutputBias, inputBuffer, resultBuffer2);
#ifdef DEBUG
        printf("bert/encoder/layer_%d/attention/output/add:\n", layerIdx);
        dumpMatrix(resultBuffer2);
#endif

        // batchmorm
        batchnorm(resultBuffer2, gamma1, beta1);
#ifdef DEBUG
        printf("bert/encoder/layer_%d/attention/output/LayerNorm/batchnorm/add_1:\n", layerIdx);
        dumpMatrix(resultBuffer2);
#endif
        
        // intermediate
        intermediate(resultBuffer2, intermediateBuffer);
#ifdef DEBUG
        printf("intermediate(bert/encoder/layer_%d/intermediate/dense/mul_1):\n", layerIdx);
        dumpMatrix(intermediateBuffer);
#endif

        // dense in output
        denseWithSum(intermediateBuffer, outputWeight, outputBias, resultBuffer2, resultBuffer1);
#ifdef DEBUG
        printf("bert/encoder/layer_%d/output/add:\n", layerIdx);
        dumpMatrix(resultBuffer1);
#endif
        
        // batchnorm
        batchnorm(resultBuffer1, gamma2, beta2);
#ifdef DEBUG
        printf("bert/encoder/layer_%d/output/LayerNorm/batchnorm/add_1:\n", layerIdx);
        dumpMatrix(resultBuffer1);
#endif
    }

private:
    void copyWeights(hpj::Matrix<float> &w, int start_col, int end_col, const float *data) {
        hpj::Matrix<float> subW(w, 0, w.Rows(), start_col, end_col - start_col);
        copyWeights(subW, data);
    }

    void copyWeights(hpj::Matrix<float> &w, const float *data) {
        for (int j = 0; j < w.Cols(); ++j) {
            for (int i = 0; i < w.Rows(); ++i) {
                w(i, j) = *data++;
            }
        }
    }

    void copyTransposed(hpj::Matrix<float> &dst, hpj::Matrix<float> &src) {
        dst.Resize(src.Cols(), src.Rows());
        for (int i = 0; i < dst.Rows(); ++i) {
            for (int j = 0; j < dst.Cols(); ++j) {
                dst(i, j) = src(j, i);
            }
        }
    }

    void dumpMatrix(hpj::Matrix<float> &m) {
        int cols = m.Cols();
        for (int i = 0; i < m.Rows(); ++i) {
            if (m.Cols() < 10) {
                for (int j = 0; j < m.Cols(); ++j) {
                    std::cout << m(i, j) << " ";
                }
            } else {
                std::cout << m(i, 0) << " " << m(i, 1) << " " << m(i, 2) << " ... " << m(i, cols-3) << " " <<  m(i, cols-2) << " " <<  m(i, cols-1);
            }
            std::cout << std::endl;
        }
    }

    // C = A * B
    // bTranspose: B need to be transposed or not
    void sgemm(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C) {
        bool bTranspose = (A.Cols() != B.Rows());
        int m = A.Rows();
        int k = A.Cols();
        int n = (bTranspose ? B.Rows() : B.Cols());
        float alpha = 1;
        float beta = 0;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, (bTranspose ? CblasTrans : CblasNoTrans), 
                    m, n, k, alpha,
                    A.Data(), A.Stride(), 
                    B.Data(), B.Stride(), beta,
                    C.Data(), C.Stride());
    }

    void dense(hpj::Matrix<float> &x, hpj::Matrix<float> &weight, hpj::Vector<float> &bias, hpj::Matrix<float> &result) {
#if USE_MKL
        sgemm(x, weight, result);
        biasAdd(result, bias);
#else
        ig_sgemm_biasadd(x.Data(), weight.Data(), result.Data(), bias.Data(),
                         x.Rows(), weight.Cols(), x.Cols(), x.Stride(), result.Stride(), 0);
#endif
    }

    // result = x * weight + bias + input
    void denseWithSum(hpj::Matrix<float> &x, hpj::Matrix<float> &weight, hpj::Vector<float> &bias, hpj::Matrix<float> &input, hpj::Matrix<float> &result) {
        assert(input.Rows() == result.Rows());
        assert(input.Cols() == result.Cols());
#if USE_MKL
        sgemm(x, weight, result);

        float *pbias = bias.Data();

        #pragma omp parallel for
        for (int i = 0; i < result.Rows(); ++i) {
            float *presult = result.Row(i);
            float *pinput = input.Row(i);
            #pragma omp simd
            for (int j = 0; j < result.Cols(); ++j) {
                presult[j] += pinput[j] + pbias[j];
            }
        }
#else
        ig_sgemm_residential(x.Data(), weight.Data(), result.Data(),
                             bias.Data(), input.Data(), x.Rows(), weight.Cols(), x.Cols(), 
                             x.Stride(), result.Stride(), input.Stride(), 0);
#endif
    }

#if __AVX512F__
    void batchnorm(hpj::Matrix<float> &x, hpj::Vector<float> &gamma, hpj::Vector<float> &beta) {
        float *pgamma = gamma.Data();
        float *pbeta = beta.Data();
        int size = x.Cols();

        #pragma omp parallel for
        for (int r = 0; r < x.Rows(); ++r) {
            float *px = x.Row(r);

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

    void intermediate(hpj::Matrix<float> &input, hpj::Matrix<float> &output) {
#if USE_MKL
        sgemm(input, intermediateWeight, output);
#else
        ig_sgemm(input.Data(), intermediateWeight.Data(), output.Data(),
                 input.Rows(), output.Cols(), input.Cols(),
                 input.Stride(), output.Stride(), 0);
#endif

        float *pbias = intermediateBias.Data();
        float factor = 0.7978845608; // np.sqrt(2 / np.pi)

        #pragma omp parallel for
        for (int i = 0; i < output.Rows(); ++i) {
            // int tid = omp_get_thread_num();
            // float *pout = output.Row(i);
            // #pragma omp simd
            // for (int j = 0; j < output.Cols(); ++j) {
            //     float x = pout[j] + pbias[j];
            //     ctx->erf_buffer[tid][j] = x;
            //     pout[j] = factor * (x + 0.044715f * x * x * x);
            // }
            // vsTanh(output.Cols(), pout, pout);
            // #pragma omp simd
            // for (int j = 0; j < output.Cols(); ++j) {
            //     pout[j] = ctx->erf_buffer[tid][j] * 0.5f * (1 + pout[j]);
            // }
            float *pout = output.Row(i);
            __m512 c1 = _mm512_set1_ps(0.044715f);
            __m512 c2 = _mm512_set1_ps(factor);
            __m512 vone = _mm512_set1_ps(1);
            __m512 vtwo = _mm512_set1_ps(2);
            __m512 vhalf = _mm512_set1_ps(0.5f);

            for (int off = 0; off < output.Cols(); off += 16) {
                int remain = output.Cols() - off;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                __m512 vx = _mm512_maskz_loadu_ps(mask, pout + off);
                vx = vx + _mm512_maskz_loadu_ps(mask, pbias + off);

                __m512 vt = c2 * (vx + c1 * vx * vx * vx);
                vt = BertUtil::vexp(vt * vtwo);
                vt = vone - vtwo * _mm512_rcp14_ps(vt + vone); // tanh
                __m512 vy = vx * (vone + vt) * vhalf;

                _mm512_mask_storeu_ps(pout + off, mask, vy);
            }
        }
    }

    /*
    void intermediate(hpj::Matrix<float> &input, hpj::Matrix<float> &output) {
        sgemm(input, intermediateWeight, output);

        float *pbias = intermediateBias.Data();
        const float factor = sqrt(0.5f);
        const float scale = 0.5f / factor;

#ifdef __INTEL_COMPILER
        #pragma omp parallel for
        for (int i = 0; i < output.Rows(); ++i) {
            float *pout = output.Row(i);
            #pragma omp simd
            for (int j = 0; j < output.Cols(); ++j) {
                float with_bias = pout[j] + pbias[j];
                pout[j] = with_bias * 0.5f * (erf(with_bias * factor) + 1);
            }
        }
#else
        #pragma omp parallel for
        for (int i = 0; i < output.Rows(); ++i) {
            int tid = omp_get_thread_num();
            float *pout = output.Row(i);
            #pragma omp simd
            for (int j = 0; j < output.Cols(); ++j) {
                pout[j] = (pout[j] + pbias[j]) * factor;
            }
            vsErf(output.Cols(), pout, erf_buffer[tid]);
            #pragma omp simd
            for (int j = 0; j < output.Cols(); ++j) {
                pout[j] = pout[j] * scale * (erf_buffer[tid][j] + 1);
            }
        }
#endif
    }
    */
                                                                                                               
    // The first BatchMatMul inside self attention
    void batchMatMul(hpj::Matrix<float> &A, hpj::Matrix<float> &B, float **c_array){
        #define GRP_COUNT 1
        MKL_INT    m[GRP_COUNT] = {ctx->tokenSize};
        MKL_INT    k[GRP_COUNT] = {ctx->attHeadSize};
        MKL_INT    n[GRP_COUNT] = {ctx->tokenSize};
        
        MKL_INT    lda[GRP_COUNT] = {A.Stride()};
        MKL_INT    ldb[GRP_COUNT] = {B.Stride()};
        MKL_INT    ldc[GRP_COUNT] = {ctx->tokenSize};
        
        CBLAS_TRANSPOSE    transA[GRP_COUNT] = { CblasNoTrans };
        CBLAS_TRANSPOSE    transB[GRP_COUNT] = { CblasTrans };
        
        float    alpha[GRP_COUNT] = {1.0};
        float    beta[GRP_COUNT] = {0.0};
        
        const int group_count = ctx->attHeadNum * ctx->batchSize;
        const MKL_INT    size_per_grp[GRP_COUNT] = {group_count};
        
        // Total number of multiplications: attHeadNum * batchSize
        const float **a_array = new ConstFloatPointer[group_count];
        const float **b_array = new ConstFloatPointer[group_count];
        for (int b = 0; b < ctx->batchSize; ++b) {
            for (int i = 0; i < ctx->attHeadNum; ++i) {
                a_array[b*ctx->attHeadNum + i] = A.Row(b*ctx->tokenSize) + i * ctx->attHeadSize;
                b_array[b*ctx->attHeadNum + i] = B.Row(b*ctx->tokenSize) + i * ctx->attHeadSize;
            }
        }
         
        // Call cblas_sgemm_batch
        cblas_sgemm_batch (
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
        MKL_INT    m[GRP_COUNT] = {ctx->tokenSize};
        MKL_INT    k[GRP_COUNT] = {ctx->tokenSize};
        MKL_INT    n[GRP_COUNT] = {ctx->attHeadSize};
        
        MKL_INT    lda[GRP_COUNT] = {ctx->tokenSize};
        MKL_INT    ldb[GRP_COUNT] = {B.Stride()};
        MKL_INT    ldc[GRP_COUNT] = {C.Stride()};
        
        CBLAS_TRANSPOSE    transA[GRP_COUNT] = { CblasNoTrans };
        CBLAS_TRANSPOSE    transB[GRP_COUNT] = { CblasNoTrans };
        
        float    alpha[GRP_COUNT] = {1.0};
        float    beta[GRP_COUNT] = {0.0};
        
        const int group_count = ctx->attHeadNum * ctx->batchSize;
        const MKL_INT    size_per_grp[GRP_COUNT] = {group_count};
        
        // Total number of multiplications: attHeadNum * batchSize
        const float **b_array = new ConstFloatPointer[group_count];
        float **c_array = new FloatPointer[group_count];
        for (int b = 0; b < ctx->batchSize; ++b) {
            for (int i = 0; i < ctx->attHeadNum; ++i) {
                b_array[b*ctx->attHeadNum + i] = B.Row(b*ctx->tokenSize) + i * ctx->attHeadSize;
                c_array[b*ctx->attHeadNum + i] = C.Row(b*ctx->tokenSize) + i * ctx->attHeadSize;
            }
        }
        
        // Call cblas_sgemm_batch
        cblas_sgemm_batch (
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

    // Add bias to matrix
    void biasAdd(hpj::Matrix<float> &m, hpj::Vector<float> &bias) {
        float *pbias = bias.Data();
        #pragma omp parallel for
        for (int i = 0; i < m.Rows(); ++i) {
            float *p = m.Row(i);
            #pragma omp simd
            for (int j = 0; j < m.Cols(); ++j) {
                p[j] += pbias[j];
            }
        }
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

    typedef float * FloatPointer;
    typedef const float * ConstFloatPointer;

    // Merged query, key, value weighs
    hpj::Matrix<float> qkvWeight;
    // Merged query, key, value bias
    hpj::Vector<float> qkvBias;

    hpj::Matrix<float> attentionOutputWeight;
    hpj::Vector<float> attentionOutputBias;

    // batchnorm param
    hpj::Vector<float> gamma1, beta1;
    hpj::Vector<float> gamma2, beta2;

    hpj::Matrix<float> intermediateWeight;
    hpj::Vector<float> intermediateBias;

    hpj::Matrix<float> outputWeight;
    hpj::Vector<float> outputBias;
};

#endif


