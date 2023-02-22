#ifndef __BERT_CONTEXT_H
#define __BERT_CONTEXT_H
#include <cmath>
#include <cstdio>
#include <omp.h>
#include "my_types.h"

struct BertContext {
    // # of mini-batch
    int batchSize;
    // # of tokens
    int tokenSize;
    // Max supported tokens each buffer has prepared
    int supportedTokenSize;

    // For BERT-base, hidden_size=768
    const int hiddenSize;
    // For BERT-base, intermediate_size=3072
    const int intermediateSize;
    // For BERT-base, attHeadNum=12
    const int attHeadNum;
    // attHeadSize = hiddenSize / attHeadNum
    const int attHeadSize;
    // attFactor = 1 / sqrtf(attHeadSize)
    const float attFactor;

    // # of thread
    int numThreads;

    // Store the result of input*qkvWeight
    hpj::Matrix<float> qkvMatMul;
    // Buffer like the dimesion of 128x768
    hpj::Matrix<float> tmpBuffer, outBuffer1, outBuffer2;
    // Buffer to store the result of intermediate
    hpj::Matrix<float> intermediateBuffer;

    // Store the BatchMatMul result of query and key
    float **qk_result;
    int qk_result_count;
    // Store the result of exp for each line
    float **exp_buffer;
    // Temp buffer in intermediate
    float **erf_buffer;

    // Below params are for INT8
    bool is_int8;
    // Extra buffer for INT8: Quant buffer w/ size like 128x768
    hpj::Matrix<u8> embQuantBuffer;
    // Extra buffer for INT8: Buffer for intermediate result, size like 128x3072
    hpj::Matrix<u8> imQuantBuffer;

    BertContext(int hiddenSize, int attHeadNum, int intermediateSize, 
                bool is_int8 = false, int numThreads = 0):
        hiddenSize(hiddenSize),
        intermediateSize(intermediateSize),
        attHeadNum(attHeadNum),
        attHeadSize(hiddenSize / attHeadNum),
        attFactor(1 / sqrtf(attHeadSize)) {
        this->is_int8 = is_int8;

        // Set the default value (don't worry, it can be changed later)
        this->batchSize = 1;
        this->tokenSize = 64;
        this->supportedTokenSize = tokenSize;
        this->numThreads = numThreads;

        if (numThreads == 0) {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                if (tid == 0) { this->numThreads = omp_get_num_threads(); }
            }
        }

        qkvMatMul.Resize(batchSize * tokenSize, hiddenSize*3);
        tmpBuffer.Resize(batchSize * tokenSize, hiddenSize);
        outBuffer1.Resize(batchSize * tokenSize, hiddenSize);
        outBuffer2.Resize(batchSize * tokenSize, hiddenSize);
        intermediateBuffer.Resize(batchSize * tokenSize, intermediateSize);

        if (is_int8) {
            embQuantBuffer.Resize(batchSize * tokenSize, hiddenSize);
            imQuantBuffer.Resize(batchSize * tokenSize, intermediateSize);
        }

        // Prepare buffer of exp_buffer
        exp_buffer = new float*[this->numThreads];
        for (int i = 0; i < this->numThreads; ++i) {
            exp_buffer[i] = (float *)aligned_alloc(64, sizeof(float) * supportedTokenSize);
        }

        // Prepare buffer of erf_buffer
        erf_buffer = new float*[this->numThreads];
        for (int i = 0; i < this->numThreads; ++i) {
            erf_buffer[i] = (float *)aligned_alloc(64, sizeof(float) * intermediateSize);
        }

        this->qk_result_count = attHeadNum * batchSize;
        this->qk_result = new float*[qk_result_count];
        for (int i = 0; i < qk_result_count; ++i) {
            qk_result[i] = (float *)aligned_alloc(64, sizeof(float) * supportedTokenSize * supportedTokenSize);
        }
    }

    void dump() {
        printf("batch_size=%d\n", batchSize);
        printf("tokenSize=%d\n", tokenSize);

        printf("hiddenSize=%d\n", hiddenSize);
        printf("intermediateSize=%d\n", intermediateSize);
        printf("attHeadNum=%d\n", attHeadNum);
        printf("attHeadSize=%d\n", attHeadSize);
        printf("attFactor=%f\n", attFactor);
        
        printf("numThreads=%d\n", numThreads);
    }

    // Resize to make sure the intermediate buffer is enough
    void resize(int batchSize, int tokenSize) {
        const int rowsNeeded = batchSize * tokenSize;

        qkvMatMul.Resize(rowsNeeded, hiddenSize*3);
        tmpBuffer.Resize(rowsNeeded, hiddenSize);
        outBuffer1.Resize(rowsNeeded, hiddenSize);
        outBuffer2.Resize(rowsNeeded, hiddenSize);
        intermediateBuffer.Resize(rowsNeeded, intermediateSize);

        if (is_int8) {
            embQuantBuffer.Resize(rowsNeeded, hiddenSize);
            imQuantBuffer.Resize(rowsNeeded, intermediateSize);
        }

        // Re-allocate exp_buffer
        int oriSupportedSize = supportedTokenSize;
        if (tokenSize > oriSupportedSize) {
            supportedTokenSize = tokenSize + 8; // reserve some buffer
            for (int i = 0; i < this->numThreads; ++i) {
                free(exp_buffer[i]);
                exp_buffer[i] = (float *)aligned_alloc(64, sizeof(float) * supportedTokenSize);
            }
        }

        if (tokenSize > oriSupportedSize || attHeadNum * batchSize > qk_result_count) {
            // Need to enlarge the array size
            if (attHeadNum * batchSize > qk_result_count) {
                for (int i = 0; i < qk_result_count; ++i) {
                    free(qk_result[i]);
                }

                delete[] qk_result;

                this->qk_result_count = attHeadNum * batchSize;
                this->qk_result = new float*[qk_result_count];
            }

            for (int i = 0; i < qk_result_count; ++i) {
                qk_result[i] = (float *)aligned_alloc(64, sizeof(float) * supportedTokenSize * supportedTokenSize);
            }
        }

        this->batchSize = batchSize;
        this->tokenSize = tokenSize;
    }

    ~BertContext() {
        // exp_buffer, erf_buffer
        for (int i = 0; i < numThreads; ++i) {
            free(exp_buffer[i]);
            free(erf_buffer[i]);
        }
        delete[] exp_buffer;
        delete[] erf_buffer;

        // qk_result
        for (int i = 0; i < qk_result_count; ++i) {
            free(qk_result[i]);
        }
        delete[] qk_result;
    }
};

#endif
