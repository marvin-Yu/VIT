#ifndef __SGEMM_SIMPLE_H
#define __SGEMM_SIMPLE_H

extern "C" {
// To pack matrix B
// trans_b: if the input 'b' transposed or not
// B is in K x N if transB=false
// B is in N x K if transB=true
void ig_sgemm_packb(float *B, float *packedB, int K, int N, int ldb, bool transB);

// To compute sgemm: C = A * B + beta * C
// Note: there is no ldb, as B is packed in compact format
void ig_sgemm(float *A, float *packedB, float *C, int M, int N, int K, int lda, int ldc, float beta);

// To compute sgemm w/ bias_add: C = A * B + beta * C + bias
void ig_sgemm_biasadd(float *A, float *packedB, float *C, float *bias, int M, int N, int K, int lda, int ldc, float beta);

// C = A * B + beta * C + bias + res
// ldres, redidential matrix stride
void ig_sgemm_residential(float *A, float *packedB, float *C, float *bias, float *res, int M, int N, int K, int lda, int ldc, int ldres, float beta);
}

#endif